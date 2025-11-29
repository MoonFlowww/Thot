#ifndef Nott_LRSCHEDULER_REGISTRY_HPP
#define Nott_LRSCHEDULER_REGISTRY_HPP

#include <memory>
#include <type_traits>

#include <torch/torch.h>

#include "details/cosineannealing.hpp"

namespace Nott::LrScheduler::Details {
    template <class Owner, class Descriptor>
    std::unique_ptr<Scheduler> build_scheduler(Owner&, torch::optim::Optimizer&, const Descriptor&) {
        static_assert(sizeof(Descriptor) == 0, "Unsupported scheduler descriptor provided to build_scheduler.");
        return nullptr;
    }

    template <class Owner>
    std::unique_ptr<Scheduler> build_scheduler(Owner&, torch::optim::Optimizer& optimizer, const CosineAnnealingDescriptor& descriptor) {
        return std::make_unique<CosineAnnealingScheduler>(optimizer, descriptor.options);
    }
}

#endif //Nott_LRSCHEDULER_REGISTRY_HPP