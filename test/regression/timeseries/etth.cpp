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

    Thot::Model model("Etth1_Hanxel");
    model.to_device(torch::cuda::is_available());


    model.add(Thot::Layer::FC({x1.size(0), 128, true},
                              Thot::Activation::SiLU,
                              Thot::Initialization::HeNormal),
              "token_projection");
    model.add(Thot::Layer::HardDropout({.probability = 0.05f}), "proj_dropout");
    model.add(Thot::Layer::FC({128, 128, true},
                              Thot::Activation::SiLU,
                              Thot::Initialization::HeNormal),
              "token_mixer");
    model.add(Thot::Layer::HardDropout({.probability = 0.05f}), "mixer_dropout");

    Thot::Block::Details::Transformer::Classic::EncoderOptions encoder_options{};
    encoder_options.layers = 4;
    encoder_options.embed_dim = 128;
    encoder_options.attention.num_heads = 8;
    encoder_options.attention.dropout = 0.1;
    encoder_options.feed_forward.mlp_ratio = 3.5;
    encoder_options.feed_forward.activation = Thot::Activation::SiLU;
    encoder_options.dropout = 0.1;
    encoder_options.positional_encoding.type = Thot::Layer::Details::PositionalEncodingType::Sinusoidal;
    encoder_options.positional_encoding.max_length = static_cast<std::size_t>(x1.size(1));
    encoder_options.positional_encoding.dropout = 0.05;

    model.add(Thot::Block::Transformer::Classic::Encoder(encoder_options), "encoder");
    model.add(Thot::Layer::Reduce({.op = Thot::Layer::ReduceOp::Mean, .dims = {1}, .keep_dim = false}),
              "token_pool");

    std::vector<Thot::Layer::Descriptor> residual_branch{
        Thot::Layer::FC({128, 512, true}, Thot::Activation::SiLU, Thot::Initialization::HeNormal),
        Thot::Layer::HardDropout({.probability = 0.2f}),
        Thot::Layer::FC({512, 128, true}, Thot::Activation::Identity, Thot::Initialization::XavierUniform),
    };
    model.add(Thot::Block::Residual(residual_branch,
                                    /*repeats=*/3,
                                    {},
                                    {.final_activation = Thot::Activation::SiLU, .dropout = 0.1}),
              "residual_core");

    model.add(Thot::Layer::FC({128, 256, true},
                              Thot::Activation::SiLU,
                              Thot::Initialization::HeNormal),
              "head_fc1");
    model.add(Thot::Layer::HardDropout({.probability = 0.2f}), "head_dropout");
    model.add(Thot::Layer::FC({256, 1, true},
                              Thot::Activation::Identity,
                              Thot::Initialization::XavierUniform),
              "out");

    model.links({
        {Thot::Port::Module("@input"), Thot::Port::Module("token_projection")},
        {Thot::Port::Module("token_projection"), Thot::Port::Module("proj_dropout")},
        {Thot::Port::Module("proj_dropout"), Thot::Port::Module("token_mixer")},
        {Thot::Port::Module("token_mixer"), Thot::Port::Module("mixer_dropout")},
        {Thot::Port::Module("mixer_dropout"), Thot::Port::Module("encoder")},
        {Thot::Port::Module("encoder"), Thot::Port::Module("token_pool")},
        {Thot::Port::Module("token_pool"), Thot::Port::Module("residual_core")},
        {Thot::Port::Module("residual_core"), Thot::Port::Module("head_fc1")},
        {Thot::Port::Module("head_fc1"), Thot::Port::Module("head_dropout")},
        {Thot::Port::Module("head_dropout"), Thot::Port::Module("out")},
        {Thot::Port::Module("out"), Thot::Port::Module("@output")},
    });

    const auto mse_descriptor = Thot::Loss::MSE({});
    model.set_loss(mse_descriptor);
    model.set_optimizer(Thot::Optimizer::AdamW({.learning_rate = 1e-3, .weight_decay = 1e-4}));

    model.train();

    const int64_t epochs = 200;
    const int64_t batch_size = 128;
    const auto sample_count = train_sequence.size(0);

    for (int64_t epoch = 0; epoch < epochs; ++epoch) {
        auto permutation = torch::randperm(sample_count, torch::TensorOptions().dtype(torch::kLong)).to(device);
        double epoch_loss = 0.0;
        int64_t batch_counter = 0;

        for (int64_t start = 0; start < sample_count; start += batch_size) {
            const auto end = std::min<int64_t>(start + batch_size, sample_count);
            auto indices = permutation.index({torch::indexing::Slice(start, end)});

            auto batch_inputs = train_sequence.index_select(0, indices);
            auto batch_targets = train_targets.index_select(0, indices);

            model.zero_grad();
            auto predictions = model.forward(batch_inputs);
            auto loss = Thot::Loss::Details::compute(mse_descriptor, predictions, batch_targets);
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model.parameters(), 1.0);
            model.step();

            epoch_loss += loss.item<double>();
            ++batch_counter;
        }

        if ((epoch + 1) % 20 == 0 || epoch == 0) {
            auto train_pred = model.forward(train_sequence).squeeze(1);
            auto train_target_flat = train_targets.squeeze(1);
            auto train_mse = torch::mse_loss(train_pred, train_target_flat).item<double>();
            auto train_mae = torch::linalg_vector_norm(train_pred - train_target_flat, 1).item<double>() /
                             static_cast<double>(train_target_flat.size(0));

            std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                      << " | Avg batch MSE: " << (epoch_loss / static_cast<double>(batch_counter))
                      << " | Train MSE: " << train_mse
                      << " | Train MAE: " << train_mae << std::endl;

            if (test_sequence.defined() && test_sequence.size(0) > 0) {
                torch::NoGradGuard no_grad;
                model.eval();
                auto eval_predictions = model.forward(test_sequence).squeeze(1);
                auto eval_mse = torch::mse_loss(eval_predictions, test_targets.squeeze(1)).item<double>();
                auto eval_mae = torch::linalg_vector_norm(eval_predictions - test_targets.squeeze(1), 1).item<double>() /
                                 static_cast<double>(test_targets.size(0));
                std::cout << "    Test MSE: " << eval_mse << " | Test MAE: " << eval_mae << std::endl;
                model.train();
            }
        }
    }

    if (test_sequence.defined() && test_sequence.size(0) > 0) {
        torch::NoGradGuard no_grad;
        model.eval();
        auto eval_predictions = model.forward(test_sequence).squeeze(1);
        auto eval_mse = torch::mse_loss(eval_predictions, test_targets.squeeze(1)).item<double>();
        auto eval_rmse = std::sqrt(eval_mse);
        auto eval_mae = torch::linalg_vector_norm(eval_predictions - test_targets.squeeze(1), 1).item<double>() /
                         static_cast<double>(test_targets.size(0));

        std::cout << "Final test metrics -> MSE: " << eval_mse << ", RMSE: " << eval_rmse
                  << ", MAE: " << eval_mae << std::endl;
    }
    std::cout << (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now()-t1)).count() << "ms" << std::endl;
    return 0;
}
