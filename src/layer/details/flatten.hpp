#ifndef THOT_FLATTEN_HPP
#define THOT_FLATTEN_HPP
#include <cstdint>
#include <utility>


#include <torch/nn/pimpl.h>
#include "../../activation/activation.hpp"
#include "../../common/local.hpp"
#include "../registry.hpp"

namespace Thot::Layer::Details {

    struct FlattenOptions {
        std::int64_t start_dim{1};
        std::int64_t end_dim{-1};
    };

    class FlattenImpl : public torch::nn::Module {
    public:
        FlattenImpl() = default;

        explicit FlattenImpl(FlattenOptions options)
        {
            reset(std::move(options));
        }

        void reset(FlattenOptions options)
        {
            options_ = std::move(options);
            const auto torch_options = torch::nn::FlattenOptions()
                                          .start_dim(options_.start_dim)
                                          .end_dim(options_.end_dim);
            flatten_ = register_module("flatten", torch::nn::Flatten(torch_options));
        }

        [[nodiscard]] torch::Tensor forward(torch::Tensor input)
        {
            return flatten_->forward(std::move(input));
        }

        [[nodiscard]] const FlattenOptions& options() const noexcept { return options_; }

    private:
        FlattenOptions options_{};
        torch::nn::Flatten flatten_{nullptr};
    };

    TORCH_MODULE(Flatten);


    struct FlattenDescriptor {
        FlattenOptions options{};
        ::Thot::Activation::Descriptor activation{::Thot::Activation::Identity};
        ::Thot::LocalConfig local{};
    };

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const FlattenDescriptor& descriptor, std::size_t index)
    {
        auto module = owner.register_module("flatten_" + std::to_string(index), Flatten(descriptor.options));

        RegisteredLayer registered_layer{};
        registered_layer.activation = descriptor.activation.type;
        registered_layer.module = to_shared_module_ptr(module);
        registered_layer.local = descriptor.local;
        registered_layer.bind_module_forward(module.get());
        return registered_layer;
    }

}

#endif //THOT_FLATTEN_HPP