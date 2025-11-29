#ifndef OMNI_L2_HPP
#define OMNI_L2_HPP

#include <torch/torch.h>

namespace Omni::Regularization::Details {

    struct L2Options {
        double coefficient{0.0};
    };

    struct L2Descriptor {
        L2Options options{};
    };

    [[nodiscard]] inline torch::Tensor penalty(const L2Descriptor& descriptor, const torch::Tensor& params) {
        const auto& options = descriptor.options;
        if (options.coefficient == 0.0) {
            return params.new_zeros({});
        }

        auto penalty = params.pow(2).sum();
        return penalty.mul(options.coefficient);
    }

}

#endif //OMNI_L2_HPP
