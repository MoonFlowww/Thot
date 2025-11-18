#include <torch/torch.h>
#include "../../include/Thot.h"
struct NetImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    NetImpl(int64_t in_channels, int64_t num_classes) {
        conv1 = register_module("conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 32, 3).stride(1).padding(1)));
        conv2 = register_module("conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)));

        // For MNIST 1x28x28 with 2x2 pooling twice:
        // 28 -> 14 -> 7  so spatial 7x7, channels 64 => 64*7*7 = 3136
        fc1 = register_module("fc1", torch::nn::Linear(64 * 7 * 7, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, num_classes));
    }

    torch::Tensor forward(torch::Tensor x) {
        // x: [N, C, H, W]
        x = torch::relu(conv1->forward(x));
        x = torch::max_pool2d(x, 2); // 28->14

        x = torch::relu(conv2->forward(x));
        x = torch::max_pool2d(x, 2); // 14->7

        x = x.view({x.size(0), -1}); // [N, 64*7*7]
        x = torch::relu(fc1->forward(x));
        return fc2->forward(x); // [N, num_classes]
    }
};
TORCH_MODULE(Net);


torch::Tensor compute_ntk_features(Net& model, const torch::Tensor& X) {
    // X: [N, in_dim, ...] – we'll only assume batch in dim 0
    TORCH_CHECK(X.requires_grad() == false,
                "We differentiate w.r.t. params, inputs don't need gradients.");

    model->eval();

    // Precompute total number of parameters
    int64_t P = 0;
    for (const auto& p : model->parameters()) {
        if (p.requires_grad()) P += p.numel();
    }
    TORCH_CHECK(P > 0, "Model has no trainable parameters.");

    const auto N = X.size(0);
    std::vector<torch::Tensor> feature_rows;
    feature_rows.reserve(N);

    for (int64_t i = 0; i < N; ++i) {
        // Single sample with batch dim
        auto xi = X[i].unsqueeze(0);  // [1, ...]

        // Zero all param grads
        for (auto& p : model->parameters()) {
            if (p.grad().defined()) {
                p.grad().zero_();
            }
        }

        // Forward
        auto yi = model->forward(xi);  // e.g. [1, out_dim]

        // Make it scalar: sum over outputs
        auto scalar = yi.sum(); // scalar Tensor

        // Backward w.r.t. params
        torch::autograd::backward({scalar}, {torch::Tensor()});

        // Collect grads into one big vector
        std::vector<torch::Tensor> grads_flat;
        grads_flat.reserve(model->parameters().size());

        for (const auto& p : model->parameters()) {
            if (!p.requires_grad()) continue;
            auto g = p.grad();
            TORCH_CHECK(g.defined(),
                        "Param gradient not defined. Did you accidentally disable grad?");
            grads_flat.push_back(g.view({-1}));
        }

        auto phi_i = torch::cat(grads_flat); // [P]
        TORCH_CHECK(phi_i.size(0) == P);

        feature_rows.push_back(phi_i.unsqueeze(0)); // [1, P]
    }

    // Φ: [N, P]
    return torch::cat(feature_rows, /*dim=*/0);
}


// Φ: [N, P]
torch::Tensor compute_ntk_from_features(const torch::Tensor& Phi) {
    // K = Φ Φ^T  -> [N, N]
    return torch::matmul(Phi, Phi.t());
}

// Convenience wrapper: directly from model + data
torch::Tensor compute_ntk(Net& model, const torch::Tensor& X) {
    auto Phi = compute_ntk_features(model, X);
    return compute_ntk_from_features(Phi);
}


// Φ1: [N1, P], Φ2: [N2, P]
torch::Tensor compute_cross_ntk(const torch::Tensor& Phi1,
                                const torch::Tensor& Phi2) {
    // K(X1, X2) = Φ1 Φ2^T  -> [N1, N2]
    return torch::matmul(Phi1, Phi2.t());
}



struct NTKRegressor {
    torch::Tensor Phi_train; // [N, P]
    torch::Tensor alpha;     // [N, C]
    double lambda = 0.0;

    void fit(const torch::Tensor& Phi, const torch::Tensor& y,
             double reg_lambda = 1e-6) {
        TORCH_CHECK(Phi.dim() == 2, "Phi must be [N, P]");
        TORCH_CHECK(y.size(0) == Phi.size(0),
                    "y and Phi must have the same N in dim 0");

        auto device = Phi.device();
        auto dtype  = Phi.dtype();

        auto Phi_dev = Phi;
        auto y_dev   = y.to(device).to(dtype);

        Phi_train = Phi_dev.detach().clone();
        lambda = reg_lambda;

        auto N = Phi_dev.size(0);
        auto K = torch::matmul(Phi_dev, Phi_dev.t()); // [N, N] on `device`
        auto I = torch::eye(N, K.options());
        auto K_reg = K + lambda * I;

        // A X = B  => X = solve(A,B)
        alpha = torch::linalg_solve(K_reg, y_dev);
    }

    torch::Tensor predict(const torch::Tensor& Phi_test) const {
        TORCH_CHECK(Phi_test.dim() == 2, "Phi_test must be [N*, P]");
        TORCH_CHECK(Phi_test.size(1) == Phi_train.size(1),
                    "Feature dimension mismatch between train and test.");

        auto device = Phi_train.device();

        auto Phi_test_dev = Phi_test.to(device);

        auto K_test = torch::matmul(Phi_test_dev, Phi_train.t()); // [N*, N]
        return torch::matmul(K_test, alpha); // [N*, C]
    }
};
double estimate_max_lr_from_ntk(Net& model,
                                const torch::Tensor& X,
                                int64_t N_subset = 512,
                                double safety = 0.25) {
    TORCH_CHECK(X.size(0) >= N_subset, "Not enough samples for subset");

    // Take a subset to keep K small enough
    auto X_sub = X.narrow(0, 0, N_subset);

    // Compute NTK features and kernel
    auto Phi = compute_ntk_features(model, X_sub); // [N_subset, P]
    auto K   = torch::matmul(Phi, Phi.t());        // [N_subset, N_subset]

    // For safety / speed, often easier on CPU for eigs
    auto K_cpu = K.to(torch::kCPU);

    // Eigenvalues of symmetric matrix
    auto evals = torch::linalg_eigvalsh(K_cpu);   // [N_subset]
    double lambda_max = evals.max().item<double>();

    double eta_crit = 2.0 / lambda_max;
    double eta_safe = safety * eta_crit;

    std::cout << "lambda_max = " << lambda_max
              << ", eta_crit = " << eta_crit
              << ", eta_safe = " << eta_safe << std::endl;

    return eta_safe;
}



int main() {
    auto [x1, y1, x2, y2] =
        Thot::Data::Load::MNIST("/home/moonfloww/Projects/DATASETS/Image/MNIST",
                                0.1f, 0.1f, true);

    torch::Device device = torch::kCUDA;

    // For a normal CNN, you don't need to downsample to 8x8; let it be 28x28
    x1 = x1.to(device);
    x2 = x2.to(device);
    y1 = y1.to(device, torch::kLong);
    y2 = y2.to(device, torch::kLong);

    int64_t in_channels = x1.size(1); // should be 1 for MNIST
    int64_t num_classes = 10;

    Net model(in_channels, num_classes);
    model->to(device);

    // Estimate LR from NTK spectrum (using a subset)
    int64_t N_subset = std::min<int64_t>(512, x1.size(0));
    double lr_ntk = estimate_max_lr_from_ntk(model, x1, N_subset, 0.25);

    std::cout << "Suggested NTK-based LR = " << lr_ntk << std::endl;

    // Now you can plug this LR into a real training loop
    torch::optim::SGD optimizer(model->parameters(),
                                torch::optim::SGDOptions(lr_ntk).momentum(0.9));

    // Standard training loop (cross entropy, mini-batch)
    model->train();
    const int64_t epochs = 10;
    const int64_t batch_size = 64;

    for (int64_t epoch = 0; epoch < epochs; ++epoch) {
        // you can wrap x1,y1 into a Thot::DataLoader or manual mini-batch loop
        for (int64_t i = 0; i < x1.size(0); i += batch_size) {
            auto end = std::min<int64_t>(i + batch_size, x1.size(0));
            auto xb = x1.narrow(0, i, end - i);
            auto yb = y1.narrow(0, i, end - i);

            optimizer.zero_grad();
            auto logits = model->forward(xb);
            auto loss = torch::nn::functional::cross_entropy(logits, yb);
            loss.backward();
            optimizer.step();
        }
        std::cout << "Epoch " << epoch << " done\n";
    }

    return 0;
}

