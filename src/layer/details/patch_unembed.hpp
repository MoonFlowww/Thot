#ifndef OMNI_PATCH_UNEMBED_HPP
#define OMNI_PATCH_UNEMBED_HPP

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <torch/nn/pimpl.h>

#include "../../activation/activation.hpp"
#include "../../common/local.hpp"
#include "../registry.hpp"

namespace Omni::Layer::Details {

    struct PatchUnembedOptions {
        std::int64_t channels{1};
        std::int64_t tokens_height{1};
        std::int64_t tokens_width{1};
        std::int64_t patch_size{1};
        std::int64_t target_height{-1};
        std::int64_t target_width{-1};
        bool align_corners{false};
    };

    class PatchUnembedImpl : public torch::nn::Module {
    public:
        PatchUnembedImpl() = default;
        explicit PatchUnembedImpl(PatchUnembedOptions options) {
            reset(std::move(options));
        }

        void reset(PatchUnembedOptions options)
        {
            if (options.channels <= 0)
                throw std::invalid_argument("Patch unembed requires positive channel count.");
            if (options.tokens_height <= 0 || options.tokens_width <= 0)
                throw std::invalid_argument("Patch unembed requires positive token grid dimensions.");
            if (options.patch_size <= 0)
                throw std::invalid_argument("Patch unembed requires positive patch size.");
            options_ = std::move(options);
        }

        [[nodiscard]] torch::Tensor forward(torch::Tensor input)
        {
            if (!input.defined()) {
                return input;
            }

            if (input.dim() != 3) {
                throw std::invalid_argument("Patch unembed expects tensor of shape (B, tokens, patch_dim).");
            }

            const auto batch = input.size(0);
            const auto tokens = input.size(1);
            const auto patch_dim = input.size(2);
            const auto expected_tokens = options_.tokens_height * options_.tokens_width;
            if (tokens != expected_tokens) {
                throw std::invalid_argument("Patch unembed received unexpected token count.");
            }

            const auto patch_area = options_.patch_size * options_.patch_size;
            const auto expected_patch_dim = options_.channels * patch_area;
            if (patch_dim != expected_patch_dim) {
                throw std::invalid_argument("Patch unembed received unexpected patch embedding dimension.");
            }

            auto decoded = input.view({batch,
                                       options_.tokens_height, options_.tokens_width,
                                       options_.channels,
                                       options_.patch_size, options_.patch_size});
            decoded = decoded.permute({0, 3, 1, 4, 2, 5}).contiguous();
            decoded = decoded.view({batch, options_.channels,
                                     options_.tokens_height * options_.patch_size,
                                     options_.tokens_width * options_.patch_size});

            if (options_.target_height > 0 && options_.target_width > 0
                && (decoded.size(2) != options_.target_height || decoded.size(3) != options_.target_width)) {
                decoded = torch::nn::functional::interpolate(
                    decoded,
                    torch::nn::functional::InterpolateFuncOptions()
                        .size(std::vector<int64_t>{options_.target_height, options_.target_width})
                        .mode(torch::kBilinear)
                        .align_corners(options_.align_corners));
            }

            return decoded;
        }

        [[nodiscard]] const PatchUnembedOptions& options() const noexcept { return options_; }

    private:
        PatchUnembedOptions options_{};
    };

    TORCH_MODULE(PatchUnembed);

    struct PatchUnembedDescriptor {
        PatchUnembedOptions options{};
        ::Omni::Activation::Descriptor activation{::Omni::Activation::Identity};
        ::Omni::LocalConfig local{};
    };

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const PatchUnembedDescriptor& descriptor, std::size_t index)
    {
        auto module = owner.register_module("patch_unembed_" + std::to_string(index), PatchUnembed(descriptor.options));

        RegisteredLayer registered_layer{};
        registered_layer.activation = descriptor.activation.type;
        registered_layer.module = to_shared_module_ptr(module);
        registered_layer.local = descriptor.local;
        registered_layer.bind_module_forward(module.get());
        return registered_layer;
    }

}

#endif //OMNI_PATCH_UNEMBED_HPP