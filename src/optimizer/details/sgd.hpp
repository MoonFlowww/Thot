#ifndef THOT_SGD_HPP
#define THOT_SGD_HPP

#include <torch/torch.h>

namespace Thot::Optimizer::Details {

struct SGDOptions {
    double learning_rate{1e-2};
    double momentum{0.0};
    double dampening{0.0};
    double weight_decay{0.0};
    bool nesterov{false};
    bool maximize{false};
};

struct SGDDescriptor {
    SGDOptions options{};
};

inline torch::optim::SGDOptions to_torch_options(const SGDOptions& options) {
    torch::optim::SGDOptions torch_options(options.learning_rate);
    torch_options = torch_options.momentum(options.momentum);
    torch_options = torch_options.dampening(options.dampening);
    torch_options = torch_options.weight_decay(options.weight_decay);
    torch_options = torch_options.nesterov(options.nesterov);
    torch_options = torch_options.maximize(options.maximize);
    return torch_options;
}

}  // namespace Thot::Optimizer::Details

#endif //THOT_SGD_HPP
