#include "../../../include/Thot.h"
#include <torch/torch.h>
#include <chrono>

/*
 * 1) (Ehler Loops->Zscore) -> Spectral Sub Manifold (Maybe SDE-solver)
 * 2) Hanxel Decomp ([1]Trend, [2]Cycle, [3]Residual) ->
 */

struct SSAGroup {
    at::Tensor trend;
    at::Tensor cycle;
    at::Tensor residual;
};


at::Tensor autocorrelation_fft(const at::Tensor& x_in, int max_lag = -1) {
    TORCH_CHECK(x_in.dim() == 2, "x must be 2D (T, d)");

    auto x = x_in.to(torch::kFloat32);
    const auto T = x.size(0);
    const auto D = x.size(1);

    auto mean = x.mean(0, /*keepdim=*/true);
    x = x - mean;

    const int64_t Nfft = 2 * T;
    auto fft_x = torch::fft_fft(x, Nfft, /*dim=*/0);

    // Power spectrum (|FFT|^2)
    auto power = fft_x * torch::conj(fft_x);

    auto ifft_result = torch::fft_ifft(power, Nfft, /*dim=*/0);
    auto acf = at::real(ifft_result);

    acf = acf.index({torch::indexing::Slice(0, T)});

    auto var = x.var(0, /*unbiased=*/false, /*keepdim=*/true);
    acf = acf / var;

    // Force ACF[0] = 1
    auto acf0 = acf.index({0}).unsqueeze(0);  // shape (1, d)
    acf = acf / acf0;

    if (max_lag > 0 && max_lag < T)
        acf = acf.index({torch::indexing::Slice(0, max_lag + 1)});

    return acf;
}

#include <vector>
#include <cmath>

std::vector<int64_t> decorrelation_lags(const at::Tensor& p) {
    // p: (T, D)
    const auto abs_p = p.abs();
    const double threshold = 1.0 / std::exp(1.0);
    const auto T = abs_p.size(0);
    const auto D = abs_p.size(1);

    std::vector<int64_t> taus(D, T - 1); // default = max lag if never below threshold

    auto p_cpu = abs_p.to(torch::kCPU); // ensure host access
    const auto* data = p_cpu.data_ptr<float>();

    for (int64_t d = 0; d < D; ++d) {
        for (int64_t t = 1; t < T; ++t) { // skip t=0 (ρ(0)=1)
            float val = data[d * T + t];
            if (val < threshold) {
                taus[d] = t;
                break;
            }
        }
    }
    return taus;
}

std::vector<at::Tensor> hankel_decomposition(const at::Tensor& x, const std::vector<int64_t>& taus) {
    TORCH_CHECK(x.dim() == 2, "x must be (T, D)");
    const auto T = x.size(0);
    const auto D = x.size(1);

    TORCH_CHECK(static_cast<int64_t>(taus.size()) == D, "taus.size() must match number of dimensions");

    std::vector<at::Tensor> hankels;
    hankels.reserve(D);

    for (int64_t d = 0; d < D; ++d) {
        int64_t L = taus[d];
        if (L >= T) L = T - 1;
        int64_t K = T - L + 1;

        // Extract the series (T,)
        auto x_d = x.index({torch::indexing::Slice(), d});

        // Build the Hankel matrix (L, K)
        std::vector<at::Tensor> columns;
        columns.reserve(K);
        for (int64_t k = 0; k < K; ++k) {
            columns.push_back(x_d.index({torch::indexing::Slice(k, k + L)}).unsqueeze(1));
        }
        auto H = torch::cat(columns, 1);
        hankels.push_back(H);
    }
    return hankels;
}
SSAGroup ssa_from_hankel(const at::Tensor& H, int trend_rank = 1, int cycle_rank = 2) {
    // Compute thin SVD
    auto svd = torch::linalg_svd(H, /*full_matrices=*/false);
    auto U = std::get<0>(svd);
    auto S = std::get<1>(svd);
    auto V = std::get<2>(svd);

    auto make_part = [&](int start, int end) {
        if (start >= S.size(0)) return torch::zeros_like(H);
        end = std::min<int64_t>(end, S.size(0));
        auto Usub = U.index({torch::indexing::Slice(), torch::indexing::Slice(start, end)});
        auto Ssub = torch::diag(S.index({torch::indexing::Slice(start, end)}));
        auto Vsub = V.index({torch::indexing::Slice(start, end)});
        return Usub.matmul(Ssub).matmul(Vsub);
    };

    auto H_trend = make_part(0, trend_rank);
    auto H_cycle = make_part(trend_rank, trend_rank + cycle_rank);
    auto H_resid = H - H_trend - H_cycle;

    return {H_trend, H_cycle, H_resid};
}

at::Tensor diagonal_average(const at::Tensor& H) {
    const auto L = H.size(0);
    const auto K = H.size(1);
    const int64_t T = L + K - 1;

    auto y = torch::zeros({T}, H.options());
    auto counts = torch::zeros({T}, H.options());

    auto H_cpu = H.to(torch::kCPU);
    const float* data = H_cpu.data_ptr<float>();

    for (int64_t i = 0; i < L; ++i)
        for (int64_t j = 0; j < K; ++j) {
            int64_t t = i + j;
            y[t] += data[i * K + j];
            counts[t] += 1.0f;
        }

    y = y / counts;
    return y;
}

std::vector<at::Tensor> extract_trend_cycle_residual(const at::Tensor& x, const std::vector<int64_t>& taus) {
    auto hankels = hankel_decomposition(x, taus);
    std::vector<at::Tensor> results;
    results.reserve(hankels.size());

    for (const auto& H : hankels) {
        auto parts = ssa_from_hankel(H);
        auto trend = diagonal_average(parts.trend);
        auto cycle = diagonal_average(parts.cycle);
        auto resid = diagonal_average(parts.residual);

        // Stack along last dim → (T, 3)
        // Trim to same length (all have T' = L+K-1)
        auto T = trend.size(0);
        auto combined = torch::stack({trend, cycle, resid}, /*dim=*/1); // (T, 3)
        results.push_back(combined);
    }
    return results; // vector of (T, 3) tensors, one per series
}




int main() {
    auto t1= std::chrono::high_resolution_clock::now();

    auto [x1, y1, x2, y2] = Thot::Data::Load::ETTh("/home/moonfloww/Projects/DATASETS/ETT/ETTh1/ETTh1.csv", 0.5, 0.1f, true); // extract 60%
    Thot::Data::Check::Size(x1, "Raw");


    auto acf = autocorrelation_fft(x1);
    auto tau_c = decorrelation_lags(acf);
    for (int i =0; i<tau_c.size(); i++)
        std::cout << "tau_c["<< i << "]: " << tau_c[i] << std::endl;
    auto ssa_components = extract_trend_cycle_residual(x1, tau_c);

    for (size_t i = 0; i < ssa_components.size(); ++i) {
        auto comp = ssa_components[i]; // shape (T, 3)
        std::cout << "Series " << i << " components shape = " << comp.sizes() << std::endl;

        Thot::Plot::Data::Timeserie{comp};

    }


    std::cout << (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now()-t1)).count() << "ms" << std::endl;
    return 0;
}
