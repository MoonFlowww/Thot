#ifndef THOT_OPTIMIZER_REGISTRY_HPP
#define THOT_OPTIMIZER_REGISTRY_HPP


#include <memory>
#include <type_traits>

#include <torch/torch.h>

#include "details/adam.hpp"
#include "details/sgd.hpp"

namespace Thot::Optimizer::Details {
    template <class Owner, class Descriptor>
    std::unique_ptr<torch::optim::Optimizer> build_optimizer(Owner&, const Descriptor&) {
        static_assert(sizeof(Descriptor) == 0, "Unsupported optimizer descriptor provided to build_optimizer.");
        return nullptr;
    }

    template <class Owner>
    std::unique_ptr<torch::optim::Optimizer> build_optimizer(Owner& owner, const SGDDescriptor& descriptor) {
        auto options = to_torch_options(descriptor.options);
        return std::make_unique<torch::optim::SGD>(owner.parameters(), options);
    }

    template <class Owner>
    std::unique_ptr<torch::optim::Optimizer> build_optimizer(Owner& owner, const AdamWDescriptor& descriptor) {
        auto options = to_torch_options(descriptor.options);
        return std::make_unique<torch::optim::AdamW>(owner.parameters(), options);
    }
}

#endif // THOT_OPTIMIZER_REGISTRY_HPP