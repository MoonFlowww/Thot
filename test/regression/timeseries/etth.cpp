#include "../../../include/Thot.h"
#include <torch/torch.h>
#include <chrono>

/*
 * 1) (Ehler Loops->Zscore) -> Spectral Sub Manifold (Maybe SDE-solver)
 * 2) Hanxel Decomp ([1]Trend, [2]Cycle, [3]Residual) ->
 */

#include <torch/torch.h>
#include <vector>
#include <cmath>
#include <limits>

namespace SSA {
    struct TauOptions {
        double  exp_threshold = 1.0 / std::exp(1.0); // for first-cross
        bool    use_unbiased_acf = true; // inflate large lags by T/(T-k)
        bool    smooth_acf = true; // 5-tap triangular MA
        int     smooth_win = 5; // x%2==1
        double  seasonal_gate = 0.20; // ACF at period > gate
        std::vector<int64_t> seasonal_periods = {24, 168}; // ETTh1 hourly seasonality
    };
    struct SSAGroup { at::Tensor trend, cycle, residual; };


    inline at::Tensor smooth_triangular(const at::Tensor& r, int win) {
        if (!win || win <= 1 || win % 2 == 0) return r;
        auto kvec   = torch::arange(1, win/2 + 2, r.options().dtype(torch::kFloat32));
        auto kernel = torch::cat({kvec, kvec.index({torch::indexing::Slice(0, -1)}).flip(0)});
        kernel = kernel / kernel.sum();
        auto x = r.transpose(0,1).unsqueeze(1); // (D,1,T)
        // weight: (C_out=1, C_in=1, K=win)
        auto y = at::conv1d(
            x, kernel.view({1,1,-1}), c10::nullopt,
            /*stride=*/{1}, /*padding=*/{win/2}, /*dilation=*/{1}, /*groups=*/1
        );
        return y.squeeze(1).transpose(0,1);
    }

    inline at::Tensor first_below_threshold(const at::Tensor& r, double thr) {
        auto below = (r.abs() < thr);
        below.index_put_({0}, false);
        auto idx = below.to(torch::kFloat32).argmax(0);
        auto none = ~below.any(0);
        auto T = r.size(0);
        auto fb = torch::full_like(idx, T/4, idx.options());
        idx = torch::where(none, fb, idx).clamp(2, T-1);
        return idx.to(torch::kLong);
    }

    inline at::Tensor first_zero_crossing(const at::Tensor& r) {
        auto nonpos = (r <= 0);
        nonpos.index_put_({0}, false);
        auto idx = nonpos.to(torch::kFloat32).argmax(0);
        auto none = ~nonpos.any(0);
        auto T = r.size(0);
        auto fb = torch::full_like(idx, T/4, idx.options());
        idx = torch::where(none, fb, idx).clamp(2, T-1);
        return idx.to(torch::kLong);
    }


    //Sokal IACT with Geyer IMS
    inline at::Tensor sokal_iact(const at::Tensor& r) {
        const int64_t T = r.size(0);
        const int64_t D = r.size(1);
        const int64_t M = (T - 1) / 2;
        if (M <= 0) return torch::ones({D}, r.options().dtype(torch::kFloat32));

        auto rho = r.clone();
        rho.index_put_({0}, 0);
        auto p = rho.index({torch::indexing::Slice(1, 1 + 2*M, 2)}) +
                 rho.index({torch::indexing::Slice(2, 2 + 2*M, 2)}); // (M, D)

        auto cm = std::get<0>(torch::cummin(p, /*dim=*/0));
        auto positive = (cm > 0).to(cm.dtype());
        auto sump = (cm * positive).sum(/*dim=*/0);
        auto tau  = 1.0 + 2.0 * sump;
        return tau.clamp_min(1.0);
    }



    inline at::Tensor ar1_time_constant(const at::Tensor& r) {
        const int64_t T = r.size(0);
        (void)T;
        auto rho1 = r.index({1}).clamp(-0.999f, 0.999f).abs() + 1e-12f;
        return -1.0 / torch::log(rho1);
    }
    inline at::Tensor cleaned_acf(const at::Tensor& acf, int64_t T, const TauOptions& opt) {
        auto r = acf.clone();
        if (opt.use_unbiased_acf) {
            auto k = torch::arange(T, acf.options().dtype(torch::kFloat32)).unsqueeze(1); // (T,1)
            auto denom = (float)T - k;
            denom = denom.clamp_min(1.0f);
            auto infl = (float)T / denom;   // (T,1)
            r = r * infl;
            r.index_put_({0}, acf.index({0}));
        }
        if (opt.smooth_acf) {
            r = smooth_triangular(r, opt.smooth_win);
            auto r0 = r.index({0}).unsqueeze(0);
            r = r / (r0 + 1e-12f);
            r.index_put_({0}, 1.0f);
        }
        return r;
    }

    //  Dominant period per channel from RFFT no padding, Hann taper
    inline std::pair<at::Tensor, at::Tensor> dominant_periods_rfft(
        const at::Tensor& x_in, int64_t Pmin=8, int64_t Pmax=2000, float power_gate=2.0f) {
        TORCH_CHECK(x_in.dim() == 2, "x must be (T, D)");
        auto x = x_in.to(torch::kFloat32);
        const int64_t T = x.size(0);
        const int64_t D = x.size(1);

        // Hann taper reduces leakage
        auto n = torch::arange(T, x.options());
        auto hann = 0.5f - 0.5f * torch::cos(2.0f * M_PI * n / (float)(T - 1));
        x = (x - x.mean(0, true)) * hann.unsqueeze(1);

        auto X = torch::fft_rfft(x, /*n=*/c10::nullopt, /*dim=*/0); // (F, D)
        auto Pxx = at::abs(X).pow(2);
        const int64_t F = Pxx.size(0);

        auto kmin = std::max<int64_t>(1, (int64_t)std::ceil((double)T / (double)Pmax));
        auto kmax = std::min<int64_t>(F - 1, (int64_t)std::floor((double)T / (double)Pmin));
        if (kmin >= kmax) {
            return { torch::full({D}, (long)Pmin, x.options().dtype(torch::kLong)),
                     torch::zeros({D}, x.options().dtype(torch::kBool)) };
        }

        auto band = Pxx.index({torch::indexing::Slice(kmin, kmax + 1)});
        auto idx_rel = band.argmax(0);
        auto idx = idx_rel + kmin;
        auto Pstar = torch::round((double)T / idx.to(torch::kFloat32)).to(torch::kLong)
                        .clamp(Pmin, Pmax);

        // gate: peak must exceed (power_gate × median band power)
        auto medpow = std::get<0>(at::median(band, /*dim=*/0));
        auto cols = torch::arange(D, Pxx.options().dtype(torch::kLong));
        auto peakpow= Pxx.index({idx, cols});
        auto strong= peakpow > (power_gate * (medpow + 1e-12f));

        return {Pstar, strong};
    }

    inline at::Tensor choose_tau(const at::Tensor& acf, int64_t T, const at::Tensor& x_in, const TauOptions& opt = {}, bool snap_to_24 = true) {
        // ACF-based candidates (your existing logic)
        auto r = cleaned_acf(acf, T, opt);
        auto tau_e = first_below_threshold(r, opt.exp_threshold).to(torch::kFloat32);
        auto tau_z = first_zero_crossing(r).to(torch::kFloat32);
        auto tau_i = torch::ceil(sokal_iact(r));
        auto tau_a = torch::ceil(ar1_time_constant(r));
        auto z_cap = torch::min(tau_z, torch::full_like(tau_z, (float)(T/2)));

        // Spectral candidate: ~1.5 × dominant period
        auto [Pstar, strong] = dominant_periods_rfft(x_in);
        auto tau_spec = (1.5f * Pstar.to(torch::kFloat32)).round();

        auto stack = torch::stack({tau_e, tau_i, tau_a, z_cap, tau_spec}, 0);   // (5,D)
        auto tau   = std::get<0>(at::median(stack, /*dim=*/0));                 // (D)

        auto lower = Pstar.to(torch::kFloat32);
        auto upper = 4.0f * lower;
        tau = torch::where(strong, tau.clamp(lower, upper), tau);

        if (snap_to_24) {
            // snap when P* is near common seasonal quanta (24 or 168)
            auto near24  = ( (Pstar - 24).abs()  <= 2 );
            auto near168 = ( (Pstar - 168).abs() <= 4 );
            auto q24  = 24.0f;
            auto q168 = 168.0f;
            auto snapped24  = ( (tau / q24).round()  * q24 );
            auto snapped168 = ( (tau / q168).round() * q168 );
            tau = torch::where(near168, snapped168, tau);
            tau = torch::where(near24,  snapped24,  tau);
        }

        // casu SSA bounds
        tau = tau.clamp(2.0f, (float)(T/2));
        return tau.to(torch::kLong).contiguous();
    }



    // --- Autocorrelation via FFT (column-wise) ---
    inline at::Tensor autocorrelation_fft(const at::Tensor& x_in, int max_lag = -1) {
        TORCH_CHECK(x_in.dim() == 2, "x must be 2D (T, D)");
        auto x = x_in.to(torch::kFloat32);
        const int64_t T = x.size(0);

        x = x - x.mean(0, /*keepdim=*/true);

        const int64_t Nfft = 2 * T;
        auto X = torch::fft_fft(x, Nfft, /*dim=*/0);
        auto P = X * torch::conj(X);
        auto acf_c = torch::fft_ifft(P, Nfft, /*dim=*/0);
        auto acf = at::real(acf_c).index({torch::indexing::Slice(0, T)}); // (T, D)

        // Normalize so acf[0] == 1 per column
        auto a0 = acf.index({0}).unsqueeze(0);
        // guard against zero variance columns
        a0 = a0 + 1e-12;
        acf = acf / a0;

        if (max_lag > 0 && max_lag < T) {
            acf = acf.index({torch::indexing::Slice(0, max_lag + 1)});
        }
        return acf.contiguous();
    }

    inline at::Tensor decorrelation_lags(const at::Tensor& acf, double threshold = 1.0 / std::exp(1.0)) {
        const int64_t T = acf.size(0);
        auto below = (acf.abs() < threshold);
        below.index_put_({0}, false);

        auto idx = below.to(torch::kFloat32).argmax(0); // if no true, returns 0
        auto no_cross = ~below.any(0);
        auto fallback = torch::full_like(idx, T / 4, idx.options());
        idx = torch::where(no_cross, fallback, idx);
        idx = idx.clamp(2, T - 1);
        return idx.to(torch::kLong);
    }

    inline at::Tensor hankel_matrix(const at::Tensor& x_in, int64_t L) {
        TORCH_CHECK(x_in.dim() == 1, "x must be 1D for hankel_matrix");
        const int64_t T = x_in.size(0);
        TORCH_CHECK(L >= 2 && L < T, "Invalid L for SSA window");
        const int64_t K = T - L + 1;

        auto x = x_in;
        const int64_t s = x.stride(0);
        return x.as_strided({L, K}, {s, s});
    }


    inline SSAGroup decompose(const at::Tensor& H, int trend_rank = 1, int cycle_rank = 2) {
        // H: (L, K)
        auto svd = torch::linalg_svd(H, /*full_matrices=*/false);
        auto U  = std::get<0>(svd);  // (L, rmin)
        auto S  = std::get<1>(svd);  // (rmin)
        auto Vh = std::get<2>(svd);  // (rmin, K)

        const int64_t rmin = S.size(0);
        const int64_t r_trend = std::min<int64_t>(trend_rank, rmin);
        const int64_t r_cycle = std::min<int64_t>(cycle_rank, rmin - r_trend);

        auto part = [&](int64_t start, int64_t end) -> at::Tensor {
            if (start >= end) return torch::zeros_like(H);
            const int64_t r = end - start;
            auto U_  = U.index({torch::indexing::Slice(), torch::indexing::Slice(start, end)});      // (L, r)
            auto S_  = S.index({torch::indexing::Slice(start, end)});                                 // (r)
            auto Vh_ = Vh.index({torch::indexing::Slice(start, end), torch::indexing::Slice()});      // (r, K)
            return U_.matmul(at::diag(S_)).matmul(Vh_);
        };

        auto Ht = part(0, r_trend);
        auto Hc = part(r_trend, r_trend + r_cycle);
        auto Hr = H - Ht - Hc;
        return {Ht, Hc, Hr};
    }

    // --- Proper diagonal averaging (anti-diagonals), O(T) offsets, no NaNs ---
    inline at::Tensor diagonal_average(const at::Tensor& H) {
        const int64_t L = H.size(0);
        const int64_t K = H.size(1);
        const int64_t T = L + K - 1;

        auto y = torch::empty({T}, H.options());
        auto Hflip = H.flip(1);

        for (int64_t offset = -(L - 1); offset <= (K - 1); ++offset) {
            auto d = Hflip.diagonal(offset);      // vector along anti-diagonal
            int64_t t = offset + (L - 1);         // map offset → [0, T)
            y.index_put_({t}, d.mean());
        }
        return y;
    }

    // --- Extract (per column) -> at::Tensor(T,3) for each input series ---
    inline std::vector<at::Tensor> extract(const at::Tensor& x_in, const at::Tensor& taus,
                                           int trend_rank = 1, int cycle_rank = 2) {
        TORCH_CHECK(x_in.dim() == 2, "x must be (T, D)");
        const int64_t T = x_in.size(0);
        const int64_t D = x_in.size(1);

        std::vector<at::Tensor> out;
        out.reserve(D);

        for (int64_t d = 0; d < D; ++d) {
            auto x = x_in.index({torch::indexing::Slice(), d}).to(torch::kFloat32);
            x = x.contiguous();
            int64_t tau = taus[d].item<int64_t>();
            int64_t L = std::max<int64_t>(2, std::min<int64_t>(tau, T / 2));

            auto H = hankel_matrix(x, L);
            auto parts = decompose(H, trend_rank, cycle_rank);

            auto trend = diagonal_average(parts.trend);
            auto cycle = diagonal_average(parts.cycle);
            auto resid = diagonal_average(parts.residual);

            out.push_back(torch::stack({trend, cycle, resid}, /*dim=*/1)); // (T,3)
        }
        return out;
    }
}

int main() {
    auto t1 = std::chrono::high_resolution_clock::now();

    auto [x1, y1, x2, y2] =
        Thot::Data::Load::ETTh("/home/moonfloww/Projects/DATASETS/ETT/ETTh1/ETTh1.csv", 0.5, 0.1f, true);

    auto acf   = SSA::autocorrelation_fft(x1);
    auto tau_c = SSA::choose_tau(acf, x1.size(0), x1);

    std::cout << "Tau_c = " << tau_c << std::endl;

    auto comps = SSA::extract(x1, tau_c);
    for (size_t i = 0; i < comps.size(); ++i) {
        std::cout << "Series " << i << " -> " << comps[i].sizes() << std::endl;
        Thot::Plot::Data::Timeserie{comps[i]};
    }

    std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - t1).count()<< " ms" << std::endl;
}
