#include <torch/torch.h>
#include <../include/Thot.h>
#include <chrono>
#include <limits>
#include <string>
#include <algorithm>
#include <vector>
#include <numeric>
#include <iostream>

struct NetImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr}, conv5{nullptr};
    torch::nn::MaxPool2d pool{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    NetImpl() {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).stride(1).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 3).stride(1).padding(1)));
        pool  = register_module("pool",  torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2, 2}).stride({2, 2})));

        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)));
        conv4 = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)));
        conv5 = register_module("conv5", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)));

        fc1 = register_module("fc1", torch::nn::Linear(1152, 524));
        fc2 = register_module("fc2", torch::nn::Linear(524, 126));
        fc3 = register_module("fc3", torch::nn::Linear(126, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1->forward(x));
        x = torch::relu(conv2->forward(x));
        x = pool->forward(x);

        x = torch::relu(conv3->forward(x));
        x = torch::relu(conv4->forward(x));
        x = pool->forward(x);

        x = torch::relu(conv5->forward(x));
        x = pool->forward(x);

        x = x.view({x.size(0), -1});
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }
};
TORCH_MODULE(Net);
namespace LatencyUtils {
    struct StepLatencyStats {
        std::string title;
        double total_ms = 0.0;
        double min_ms   = std::numeric_limits<double>::max();
        double max_ms   = 0.0;
        std::size_t steps = 0;

        // Online central moments
        double mean = 0.0;
        double M2   = 0.0; // sum of squared deviations
        double M3   = 0.0; // sum of cubed deviations

        double median_ms = 0.0;
        double p10_ms    = 0.0;
        double p90_ms    = 0.0;
        double p98_ms    = 0.0;
        double mode_ms   = 0.0;

        void record(double ms, std::size_t count = 1) {
            if (count == 0) return;

            for (std::size_t c = 0; c < count; ++c) {
                double n1 = static_cast<double>(steps);
                double n  = n1 + 1.0;

                double delta    = ms - mean;
                double delta_n  = delta / n;
                double delta_n2 = delta_n * delta_n;
                double term1    = delta * delta_n * n1;

                // Pebay update for central moments (up to 3rd)
                double newM3   = M3 + term1 * delta_n * (n1 - 1.0) - 3.0 * delta_n * M2;
                double newM2   = M2 + term1;
                double newMean = mean + delta_n;

                mean = newMean;
                M2   = newM2;
                M3   = newM3;

                ++steps;
            }

            total_ms += ms * static_cast<double>(count);
            min_ms = std::min(min_ms, ms);
            max_ms = std::max(max_ms, ms);
        }

        double average_ms() const {
            return steps == 0 ? 0.0 : mean;
        }

        double variance_ms() const {
            if (steps < 2) return 0.0;
            return M2 / static_cast<double>(steps - 1);
        }

        double std_ms() const {
            return std::sqrt(variance_ms());
        }

        // Fisher's unbiased sample skewness
        double skew() const {
            if (steps < 3) return 0.0;

            const double n  = static_cast<double>(steps);
            const double m2 = M2 / n;
            const double m3 = M3 / n;
            if (m2 <= 0.0) return 0.0;

            const double g1 = m3 / std::pow(m2, 1.5); // moment skewness
            return std::sqrt(n * (n - 1.0)) / (n - 2.0) * g1;
        }

        double coeff_of_variation() const {
            const double mu  = average_ms();
            const double sig = std_ms();
            if (mu == 0.0) return 0.0;
            return sig / mu;
        }

        double steps_per_second() const {
            const double mu = average_ms();
            if (mu <= 0.0) return 0.0;
            return 1000.0 / mu;
        }
    };


    void print_stats(const StepLatencyStats& stats) {
        const double min_ms = stats.steps == 0 ? 0.0 : stats.min_ms;
        const double max_ms = stats.steps == 0 ? 0.0 : stats.max_ms;
        const double mu     = stats.average_ms();
        const double std    = stats.std_ms();
        const double skew   = stats.skew();

        std::cout << "\n" << stats.title << "\n";
        std::cout
            << "  Steps (after filters): " << stats.steps << "\n"
            << "  Avg latency:           " << mu   << " ms\n"
            << "  Std latency:           " << std  << " ms\n"
            << "  Skew latency:          " << skew << "\n";

        if (stats.steps > 0) {
            std::cout
                << "  P10 / P50 / P90 / P98: "
                << stats.p10_ms    << " / "
                << stats.median_ms << " / "
                << stats.p90_ms    << " / "
                << stats.p98_ms    << " ms\n"
                << "  Mode latency (hist):   " << stats.mode_ms << " ms\n"
                << "  Coeff. of variation:   " << stats.coeff_of_variation() << "\n"
                << "  Throughput:            " << stats.steps_per_second() << " steps/s\n";
        }

        std::cout
            << "  ~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            << "  Min latency:            " << min_ms << " ms\n"
            << "  Max latency:            " << max_ms << " ms\n"
            << std::endl;
    }




    struct TukeyFence {
        double lower = 0.0;
        double upper = 0.0;
    };

    static double quantile_linear(const std::vector<double>& sorted, double p) {
        if (sorted.empty()) return 0.0;
        if (sorted.size() == 1) return sorted[0];

        const double idx = p * static_cast<double>(sorted.size() - 1);
        const std::size_t i = static_cast<std::size_t>(idx);
        const double frac = idx - static_cast<double>(i);

        if (i + 1 < sorted.size())
            return sorted[i] * (1.0 - frac) + sorted[i + 1] * frac;
        return sorted[i];
    }

    static TukeyFence compute_tukey_fence(const std::vector<double>& sorted_vals, double k) {
        if (sorted_vals.empty()) return {};

        const double q1 = quantile_linear(sorted_vals, 0.25);
        const double q3 = quantile_linear(sorted_vals, 0.75);
        const double iqr = q3 - q1;

        TukeyFence f;
        f.lower = q1 - k * iqr;
        f.upper = q3 + k * iqr;
        return f;
    }

    static double estimate_mode_from_sorted(const std::vector<double>& sorted) {
        if (sorted.empty()) return 0.0;
        if (sorted.size() == 1) return sorted[0];

        const double minv = sorted.front();
        const double maxv = sorted.back();
        if (minv == maxv) return minv;

        const std::size_t n = sorted.size();

        const double q1  = quantile_linear(sorted, 0.25);
        const double q3  = quantile_linear(sorted, 0.75);
        const double iqr = q3 - q1;

        // Freedman–Diaconis bin width
        double bin_width = 0.0;
        if (iqr > 0.0) {
            bin_width = 2.0 * iqr * std::pow(static_cast<double>(n), -1.0 / 3.0);
        }
        if (bin_width <= 0.0) {
            bin_width = (maxv - minv) / std::min<std::size_t>(n, 32);
        }

        std::size_t bin_count = static_cast<std::size_t>(std::ceil((maxv - minv) / bin_width));
        bin_count = std::clamp<std::size_t>(bin_count, 4, 128);

        std::vector<std::size_t> hist(bin_count, 0);
        const double inv_width = static_cast<double>(bin_count) / (maxv - minv);

        for (double v : sorted) {
            std::size_t idx = static_cast<std::size_t>((v - minv) * inv_width);
            if (idx >= bin_count) idx = bin_count - 1;
            hist[idx]++;
        }

        std::size_t best_bin   = 0;
        std::size_t best_count = hist[0];
        for (std::size_t i = 1; i < bin_count; ++i) {
            if (hist[i] > best_count) {
                best_count = hist[i];
                best_bin   = i;
            }
        }

        const double width = (maxv - minv) / static_cast<double>(bin_count);
        return minv + (static_cast<double>(best_bin) + 0.5) * width;
    }



    StepLatencyStats build_stats_from_samples(const std::string& title, const std::vector<double>& samples, std::size_t warmup_steps = 100, double tukey_k = 0.98) {
        StepLatencyStats stats{title};

        if (samples.size() <= warmup_steps)
            return stats;

        auto begin = samples.begin() + static_cast<std::ptrdiff_t>(warmup_steps);
        std::vector<double> tail(begin, samples.end());

        if (tail.empty()) {
            return stats;
        }

        // For Tukey fence, we use a sorted copy of the tail
        std::vector<double> sorted = tail;
        std::sort(sorted.begin(), sorted.end());

        const TukeyFence fence = compute_tukey_fence(sorted, tukey_k);

        // Keep only inliers and feed them to the online stats
        std::vector<double> filtered;
        filtered.reserve(sorted.size());

        for (double v : tail) {
            if (v < fence.lower || v > fence.upper)
                continue;
            stats.record(v);
            filtered.push_back(v);
        }

        if (!filtered.empty()) {
            std::sort(filtered.begin(), filtered.end());
            stats.median_ms = quantile_linear(filtered, 0.50);
            stats.p10_ms    = quantile_linear(filtered, 0.10);
            stats.p90_ms    = quantile_linear(filtered, 0.90);
            stats.p98_ms    = quantile_linear(filtered, 0.98);
            stats.mode_ms   = estimate_mode_from_sorted(filtered);
        }

        return stats;
    }

}
inline Thot::Loss::Details::CrossEntropyDescriptor set(Thot::Model& model) { // Used to reset learnable parameters between Thot PreBuild and Thot Custom
    model.add(Thot::Layer::Conv2d({.in_channels = 1,  .out_channels = 32,  .kernel_size = {3,3}, .stride={1,1}, .padding={1,1}, .dilation={1,1}}, Thot::Activation::ReLU, Thot::Initialization::HeUniform));
    model.add(Thot::Layer::Conv2d({.in_channels = 32, .out_channels = 32,  .kernel_size = {3,3}, .stride={1,1}, .padding={1,1}, .dilation={1,1}}, Thot::Activation::ReLU, Thot::Initialization::HeUniform));
    model.add(Thot::Layer::MaxPool2d({.kernel_size={2,2}, .stride={2,2}, .padding={0,0}}));

    model.add(Thot::Layer::Conv2d({.in_channels = 32, .out_channels = 64,  .kernel_size = {3,3}, .stride={1,1}, .padding={1,1}, .dilation={1,1}}, Thot::Activation::ReLU, Thot::Initialization::HeUniform));
    model.add(Thot::Layer::Conv2d({.in_channels = 64, .out_channels = 64,  .kernel_size = {3,3}, .stride={1,1}, .padding={1,1}, .dilation={1,1}}, Thot::Activation::ReLU, Thot::Initialization::HeUniform));
    model.add(Thot::Layer::MaxPool2d({.kernel_size={2,2}, .stride={2,2}, .padding={0,0}}));

    model.add(Thot::Layer::Conv2d({.in_channels = 64, .out_channels = 128, .kernel_size = {3,3}, .stride={1,1}, .padding={1,1}, .dilation={1,1}}, Thot::Activation::ReLU, Thot::Initialization::HeUniform));
    model.add(Thot::Layer::MaxPool2d({.kernel_size={2,2}, .stride={2,2}, .padding={0,0}}));

    model.add(Thot::Layer::Flatten());
    model.add(Thot::Layer::FC({1152, 524}, Thot::Activation::ReLU,  Thot::Initialization::HeUniform));
    model.add(Thot::Layer::FC({524, 126},  Thot::Activation::ReLU,  Thot::Initialization::HeUniform));
    model.add(Thot::Layer::FC({126, 10},   Thot::Activation::Identity, Thot::Initialization::HeUniform));

    model.set_optimizer(Thot::Optimizer::SGD({.learning_rate = 1e-3}));
    const auto ce = Thot::Loss::CrossEntropy({.label_smoothing = 0.02f});
    model.set_loss(ce);
    model.use_cuda(false);
    return ce;
}


int _main() {
    auto [x1, y1, x2, y2] = Thot::Data::Load::MNIST("/home/moonfloww/Projects/DATASETS/MNIST", .1f, 1.f, true);

    const int64_t epochs= 10;
    const int64_t B= 64;
    const int64_t N= x1.size(0);
    Thot::Data::Check::Size(x1);

    const int64_t steps_per_epoch = (N + B - 1) / B;
    const std::size_t total_steps_estimate = static_cast<std::size_t>(epochs * steps_per_epoch);

    std::vector<double> thot_prebuilt_samples;
    std::vector<double> thot_custom_samples;
    std::vector<double> libtorch_samples;

    thot_prebuilt_samples.reserve(total_steps_estimate);
    thot_custom_samples.reserve(total_steps_estimate);
    libtorch_samples.reserve(total_steps_estimate);

    // Thot built-in train()
    std::cout << "training 100% Thot" << std::endl;
    Thot::Model model1("");
    set(model1); // define the network
    model1.clear_training_telemetry(); // not necessary
    model1.train(x1, y1, {.epoch = epochs, .batch_size = B, .monitor = false, .enable_amp = false});

    const auto& telemetry = model1.training_telemetry();
    for (const auto& epoch : telemetry.epochs()) {
        const double step_latency_sec = epoch.step_latency_value();
        if (step_latency_sec <= 0.0) {
            continue;
        }

        std::size_t epoch_steps = 0;
        if (epoch.duration_seconds > 0.0) {
            const double estimated_steps = epoch.duration_seconds / step_latency_sec;
            if (estimated_steps > 0.0) {
                epoch_steps = static_cast<std::size_t>(std::llround(estimated_steps));
            }
        }
        if (epoch_steps == 0) {
            epoch_steps = static_cast<std::size_t>(std::max<int64_t>(1, steps_per_epoch));
        }

        const double step_latency_ms = step_latency_sec * 1000.0;
        for (std::size_t s = 0; s < epoch_steps; ++s) {
            thot_prebuilt_samples.push_back(step_latency_ms);
        }
    }

    // 50% Thot (manual training loop using Thot model
    std::cout << "training 50% Thot" << std::endl;
    Thot::Model model2("");
    const auto ce = set(model2); // define the network
    model2.clear_training_telemetry(); // not necessary
    for (int64_t e = 0; e < epochs; ++e) {
        for (int64_t i = 0; i < N; i += B) { auto step_start = std::chrono::high_resolution_clock::now();
            const int64_t end = std::min(i + B, N);

            auto inputs  = x1.index({torch::indexing::Slice(i, end)}).to(model2.device());
            auto targets = y1.index({torch::indexing::Slice(i, end)}).to(model2.device());

            model2.zero_grad();
            auto logits = model2.forward(inputs);
            auto loss   = Thot::Loss::Details::compute(ce, logits, targets);
            loss.backward();
            model2.step();

            auto step_end = std::chrono::high_resolution_clock::now();
            double step_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(step_end - step_start).count();
            thot_custom_samples.push_back(step_ms);
        }
    }

    //libtorch
    std::cout << "training 100% LibTorch" << std::endl;

    torch::Device device = torch::kCPU;//torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    Net net;
    net->to(device);
    torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(1e-3));
    torch::nn::CrossEntropyLoss criterion(torch::nn::CrossEntropyLossOptions().label_smoothing(0.02));

    net->train();
    for (int64_t e = 0; e < epochs; ++e) {
        for (int64_t i = 0; i < N; i += B) { auto step_start = std::chrono::high_resolution_clock::now();
            const int64_t end = std::min(i + B, N);

            auto inputs  = x1.index({torch::indexing::Slice(i, end)}).to(device);
            auto targets = y1.index({torch::indexing::Slice(i, end)}).to(device);

            optimizer.zero_grad();
            auto logits = net->forward(inputs);
            auto loss   = criterion(logits, targets);
            loss.backward();
            optimizer.step();

            auto step_end = std::chrono::high_resolution_clock::now();
            double step_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(step_end - step_start).count();
            libtorch_samples.push_back(step_ms);
        }
    }

    // Compute stats
    std::cout << "\n\n\n";

    const LatencyUtils::StepLatencyStats thot_prebuilt_stats =
        LatencyUtils::build_stats_from_samples("Thot Train() (warmup+Tukey 0.98)", thot_prebuilt_samples, 200, 0.98);

    const LatencyUtils::StepLatencyStats thot_custom_stats =
        LatencyUtils::build_stats_from_samples("Thot + Custom Train() (warmup+Tukey 0.98)", thot_custom_samples, 200, 0.98);

    const LatencyUtils::StepLatencyStats libtorch_stats =
        LatencyUtils::build_stats_from_samples("Libtorch Raw (warmup+Tukey 0.98)", libtorch_samples, 200, 0.98);

    print_stats(thot_prebuilt_stats);
    print_stats(thot_custom_stats);
    print_stats(libtorch_stats);

    auto compute_overhead = [](double lhs, double rhs) {
        if (rhs == 0.0) return 0.0;
        return (lhs - rhs) / rhs * 100.0;
    };

    const double ThotPreBuild = thot_prebuilt_stats.average_ms();
    const double Thotcustom = thot_custom_stats.average_ms();
    const double LibTorchRaw  = libtorch_stats.average_ms();


    std::cout << "\n\nOverhead %" << std::endl;
    std::cout << "   Thot vs Libtorch Overhead: " << compute_overhead(ThotPreBuild, LibTorchRaw) << "%\n";
    std::cout << "   Thot PreBuild Train() vs Custom Train() Overhead: " << compute_overhead(ThotPreBuild, Thotcustom) << "%\n";
    std::cout << "   Thot Custom Train() vs Libtorch Overhead: " << compute_overhead(Thotcustom, LibTorchRaw) << "%\n";

    return 0;
}
