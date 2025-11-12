#include "../../../include/Thot.h"
#include <torch/torch.h>
#include <chrono>

/*
 * 1) (Ehler Loops->Zscore) -> Spectral Sub Manifold (Maybe SDE-solver)
 * 2) Hanxel Decomp ([1]Trend, [2]Cycle, [3]Residual) ->
 */



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



int main() {
    auto t1= std::chrono::high_resolution_clock::now();

    auto [x1, y1, x2, y2] = Thot::Data::Load::ETTh("/home/moonfloww/Projects/DATASETS/ETT/ETTh1/ETTh1.csv", 0.5, 0.1f, true); // extract 60%
    Thot::Data::Check::Size(x1, "Raw");

    Thot::Plot::Data::Timeserie{x2};
    auto acf = autocorrelation_fft(x2);
    Thot::Plot::Data::Timeserie{acf};
    auto tau_c = decorrelation_lags(acf);
    for (size_t i = 0; i < tau_c.size(); ++i)
        std::cout << "Series " << i << " τ_c = " << tau_c[i] << std::endl;


    (void)hankel_decomposition(x2, tau_c);
    std::cout << (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-t1)).count() << "ms" << std::endl;
    return 0;
}
