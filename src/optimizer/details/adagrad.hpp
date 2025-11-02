#ifndef THOT_ADAGRAD_HPP
#define THOT_ADAGRAD_HPP

#include <torch/torch.h>

namespace Thot::Optimizer::Details {

    struct AdagradOptions {
        double learning_rate{1e-2};
        double lr_decay{0.0};
        double weight_decay{0.0};
        double initial_accumulator_value{0.0};
        double eps{1e-10};
    };

    struct AdagradDescriptor {
        AdagradOptions options{};
    };

    inline torch::optim::AdagradOptions to_torch_options(const AdagradOptions& options) {
        torch::optim::AdagradOptions torch_options(options.learning_rate);
        torch_options = torch_options.lr_decay(options.lr_decay);
        torch_options = torch_options.weight_decay(options.weight_decay);
        torch_options = torch_options.initial_accumulator_value(options.initial_accumulator_value);
        torch_options = torch_options.eps(options.eps);
        return torch_options;
    }

}

#endif // THOT_ADAGRAD_HPP