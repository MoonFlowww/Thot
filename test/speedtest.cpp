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
        double min_ms = std::numeric_limits<double>::max();
        double max_ms = 0.0;
        std::size_t steps = 0;

        void record(double ms, std::size_t count = 1) {
            if (count == 0) return;
            total_ms += ms * static_cast<double>(count);
            min_ms = std::min(min_ms, ms);
            max_ms = std::max(max_ms, ms);
            steps += count;
        }

        double average_ms() const {
            return steps == 0 ? 0.0 : total_ms / static_cast<double>(steps);
        }
    };

    void print_stats(const StepLatencyStats& stats) {
        const double min_ms = stats.steps == 0 ? 0.0 : stats.min_ms;
        const double max_ms = stats.steps == 0 ? 0.0 : stats.max_ms;

        std::cout << stats.title << "\n"
                  << "  Steps (after filters): " << stats.steps << "\n"
                  << "  Avg latency: " << stats.average_ms() << " ms\n"
                  << "  Min latency: " << min_ms << " ms\n"
                  << "  Max latency: " << max_ms << " ms\n"
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

    StepLatencyStats build_stats_from_samples(const std::string& title, const std::vector<double>& samples, std::size_t warmup_steps = 100, double tukey_k = 0.95) {
        StepLatencyStats stats{title};

        if (samples.size() <= warmup_steps)
            return stats;

        auto begin = samples.begin() + static_cast<std::ptrdiff_t>(warmup_steps);
        std::vector<double> tail(begin, samples.end());

        if (tail.empty()) {
            return stats;
        }

        std::vector<double> sorted = tail;
        std::sort(sorted.begin(), sorted.end());

        const TukeyFence fence = compute_tukey_fence(sorted, tukey_k);

        for (double v : tail) {
            if (v < fence.lower || v > fence.upper)
                continue;
            stats.record(v);
        }

        return stats;
    }
}

int main() {
    Thot::Model model("SpeedTestMNIST");
    model.use_cuda(torch::cuda::is_available());

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

    auto [x1, y1, x2, y2] = Thot::Data::Load::MNIST("/home/moonfloww/Projects/DATASETS/MNIST", 1.f, 1.f, true);
    auto [xvalid, yvalid] = Thot::Data::Manipulation::Fraction(x2, y2, 0.1f);

    const int64_t epochs= 10;
    const int64_t B= 128;
    const int64_t N= x1.size(0);
    Thot::Data::Check::Size(x1);

    const int64_t steps_per_epoch = (N + B - 1) / B;
    const std::size_t total_steps_estimate = static_cast<std::size_t>(epochs * steps_per_epoch);

    std::vector<double> thot_prebuilt_samples;
    std::vector<double> thot_homemade_samples;
    std::vector<double> libtorch_samples;

    thot_prebuilt_samples.reserve(total_steps_estimate);
    thot_homemade_samples.reserve(total_steps_estimate);
    libtorch_samples.reserve(total_steps_estimate);

    // Thot (built-in train() + telemetry)
    std::cout << "training 100% Thot" << std::endl;
    model.clear_training_telemetry();
    model.train(x1, y1, {.epoch = epochs, .batch_size = B, .monitor = false, .enable_amp = true});

    const auto& telemetry = model.training_telemetry();
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
    for (int64_t e = 0; e < epochs; ++e) {
        for (int64_t i = 0; i < N; i += B) { auto step_start = std::chrono::high_resolution_clock::now();
            const int64_t end = std::min(i + B, N);

            auto inputs  = x1.index({torch::indexing::Slice(i, end)}).to(model.device());
            auto targets = y1.index({torch::indexing::Slice(i, end)}).to(model.device());

            model.zero_grad();
            auto logits = model.forward(inputs);
            auto loss   = Thot::Loss::Details::compute(ce, logits, targets);
            loss.backward();
            model.step();

            auto step_end = std::chrono::high_resolution_clock::now();
            double step_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(step_end - step_start).count();
            thot_homemade_samples.push_back(step_ms);
        }
    }

    //libtorch
    std::cout << "training 100% LibTorch" << std::endl;

    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
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
        LatencyUtils::build_stats_from_samples("Thot Train() (telemetry, warmup+Tukey 0.95)", thot_prebuilt_samples, 100, 0.95);

    const LatencyUtils::StepLatencyStats thot_homemade_stats =
        LatencyUtils::build_stats_from_samples("Thot + homemade Train() (warmup+Tukey 0.95)", thot_homemade_samples, 100, 0.95);

    const LatencyUtils::StepLatencyStats libtorch_stats =
        LatencyUtils::build_stats_from_samples("Libtorch Raw (warmup+Tukey 0.95)", libtorch_samples, 100, 0.95);

    print_stats(thot_prebuilt_stats);
    print_stats(thot_homemade_stats);
    print_stats(libtorch_stats);

    auto compute_overhead = [](double lhs, double rhs) {
        if (rhs == 0.0) return 0.0;
        return (lhs - rhs) / rhs * 100.0;
    };

    const double ThotPreBuild = thot_prebuilt_stats.average_ms();
    const double ThotHomeMade = thot_homemade_stats.average_ms();
    const double LibTorchRaw  = libtorch_stats.average_ms();

    std::cout << "Thot vs Libtorch Step Overhead: " << compute_overhead(ThotPreBuild, LibTorchRaw) << "%\n";
    std::cout << "Thot PreBuild Step Overhead vs Homemade: " << compute_overhead(ThotPreBuild, ThotHomeMade) << "%\n";
    std::cout << "Thot Homemade Step vs Libtorch Overhead: " << compute_overhead(ThotHomeMade, LibTorchRaw) << "%\n";

    return 0;
}
