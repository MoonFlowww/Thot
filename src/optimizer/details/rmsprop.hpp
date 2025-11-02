#ifndef THOT_RMSPROP_HPP
#define THOT_RMSPROP_HPP

#include <torch/torch.h>
#include <tuple>

namespace Thot::Optimizer::Details {

    struct RMSpropOptions {
        double learning_rate{1e-2};
        double alpha{0.99};
        double eps{1e-8};
        double weight_decay{0.0};
        double momentum{0.0};
        bool centered{false};
    };

    struct RMSpropDescriptor {
        RMSpropOptions options{};
    };

    inline torch::optim::RMSpropOptions to_torch_options(const RMSpropOptions& options) {
        torch::optim::RMSpropOptions torch_options(options.learning_rate);
        torch_options = torch_options.alpha(options.alpha);
        torch_options = torch_options.eps(options.eps);
        torch_options = torch_options.weight_decay(options.weight_decay);
        torch_options = torch_options.momentum(options.momentum);
        torch_options = torch_options.centered(options.centered);
        return torch_options;
    }

}
#endif // THOT_RMSPROP_HPP
