#ifndef THOT_ADAM_HPP
#define THOT_ADAM_HPP
// Adam: https://arxiv.org/pdf/1412.6980
// Adan: https://arxiv.org/pdf/2208.06677

#include <tuple>
#include <functional>
#include <type_traits>
#include <torch/torch.h>

namespace Thot::Optimizer::Details {

    struct AdamWOptions {
        double learning_rate{1e-3};
        double beta1{0.9};
        double beta2{0.999};
        double eps{1e-8};
        double weight_decay{1e-2};
        bool amsgrad{false};
    };

    struct AdamWDescriptor {
        AdamWOptions options{};
    };

    inline torch::optim::AdamWOptions to_torch_options(const AdamWOptions& options) {
        torch::optim::AdamWOptions torch_options(options.learning_rate);
        torch_options = torch_options.betas(std::make_tuple(options.beta1, options.beta2));
        torch_options = torch_options.eps(options.eps);
        torch_options = torch_options.weight_decay(options.weight_decay);
        torch_options = torch_options.amsgrad(options.amsgrad);
        return torch_options;
    }

}

#endif //THOT_ADAM_HPP