#ifndef OMNI_OPTIMIZER_REGISTRY_HPP
#define OMNI_OPTIMIZER_REGISTRY_HPP


#include <memory>
#include <type_traits>

#include <torch/torch.h>

#include "details/adam.hpp"
#include "details/sgd.hpp"
#include "details/sophia.hpp"
#include "details/muon.hpp"
#include "details/adagrad.hpp"
#include "details/adafactor.hpp"
#include "details/lamb.hpp"
#include "details/lion.hpp"
#include "details/rmsprop.hpp"

namespace Omni::Optimizer::Details {
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
    std::unique_ptr<torch::optim::Optimizer> build_optimizer(Owner& owner, const RMSpropDescriptor& descriptor) {
        auto options = to_torch_options(descriptor.options);
        return std::make_unique<torch::optim::RMSprop>(owner.parameters(), options);
    }

    template <class Owner>
    std::unique_ptr<torch::optim::Optimizer> build_optimizer(Owner& owner, const AdagradDescriptor& descriptor) {
        auto options = to_torch_options(descriptor.options);
        return std::make_unique<torch::optim::Adagrad>(owner.parameters(), options);
    }

    template <class Owner>
    std::unique_ptr<torch::optim::Optimizer> build_optimizer(Owner& owner, const AdamWDescriptor& descriptor) {
        auto options = to_torch_options(descriptor.options);
        return std::make_unique<torch::optim::AdamW>(owner.parameters(), options);
    }

    template <class Owner>
    std::unique_ptr<torch::optim::Optimizer> build_optimizer(Owner& owner, const AdamDescriptor& descriptor) {
        auto options = to_torch_options(descriptor.options);
        return std::make_unique<torch::optim::Adam>(owner.parameters(), options);
    }

    template <class Owner>
    std::unique_ptr<torch::optim::Optimizer> build_optimizer(Owner& owner, const LionDescriptor& descriptor) {
        return std::make_unique<Lion>(owner.parameters(), descriptor.options);
    }

    template <class Owner>
    std::unique_ptr<torch::optim::Optimizer> build_optimizer(Owner& owner, const LAMBDescriptor& descriptor) {
        return std::make_unique<LAMB>(owner.parameters(), descriptor.options);
    }

    template <class Owner>
    std::unique_ptr<torch::optim::Optimizer> build_optimizer(Owner& owner, const AdafactorDescriptor& descriptor) {
        return std::make_unique<Adafactor>(owner.parameters(), descriptor.options);
    }

    template <class Owner>
    std::unique_ptr<torch::optim::Optimizer> build_optimizer(Owner& owner, const SophiaGDescriptor& descriptor) {
        return std::make_unique<SophiaG>(owner.parameters(), descriptor.options);
    }

    template <class Owner>
    std::unique_ptr<torch::optim::Optimizer> build_optimizer(Owner& owner, const SophiaHDescriptor& descriptor) {
        return std::make_unique<SophiaH>(owner.parameters(), descriptor.options);
    }


    template <class Owner>
    std::unique_ptr<torch::optim::Optimizer> build_optimizer(Owner& owner, const MuonDescriptor& descriptor) {
        return std::make_unique<Muon>(owner.parameters(), descriptor.options);
    }

    template <class Owner>
    std::unique_ptr<torch::optim::Optimizer> build_optimizer(Owner& owner, const AdaMuonDescriptor& descriptor) {
        return std::make_unique<AdaMuon>(owner.parameters(), descriptor.options);
    }

    template <class Owner>
    std::unique_ptr<torch::optim::Optimizer> build_optimizer(Owner& owner, const MuonManifoldDescriptor& descriptor) {
        return std::make_unique<MuonManifold>(owner.parameters(), descriptor.options);
    }
}

#endif // OMNI_OPTIMIZER_REGISTRY_HPP