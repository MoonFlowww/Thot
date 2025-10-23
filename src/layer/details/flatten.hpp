#ifndef THOT_FLATTEN_HPP
#define THOT_FLATTEN_HPP
#include <cstdint>
#include <utility>


#include <torch/nn/pimpl.h>
#include "../../activation/activation.hpp"

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
    };

}

#endif //THOT_FLATTEN_HPP