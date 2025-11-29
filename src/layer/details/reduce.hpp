#ifndef Nott_LAYER_REDUCE_HPP
#define Nott_LAYER_REDUCE_HPP
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "../../activation/activation.hpp"
#include "../../common/local.hpp"
#include "../registry.hpp"

namespace Nott::Layer::Details {

    enum class ReduceOp {
        Sum,
        Mean,
        Max,
        Min,
    };

    struct ReduceOptions {
        ReduceOp op{ReduceOp::Mean};
        std::vector<std::int64_t> dims{};
        bool keep_dim{false};
    };

    namespace Detail {
        inline std::vector<std::int64_t> normalise_dims(const torch::Tensor& input, const std::vector<std::int64_t>& dims)
        {
            if (input.dim() == 0) {
                return {};
            }

            if (dims.empty()) {
                std::vector<std::int64_t> all_dims(static_cast<std::size_t>(input.dim()));
                std::iota(all_dims.begin(), all_dims.end(), std::int64_t{0});
                return all_dims;
            }

            std::vector<std::int64_t> normalised = dims;
            const auto total_dims = input.dim();
            for (auto& dim : normalised) {
                if (dim < 0) {
                    dim += total_dims;
                }
                TORCH_CHECK(dim >= 0 && dim < total_dims, "Reduce layer received an out-of-range dimension index.");
            }

            std::sort(normalised.begin(), normalised.end());
            normalised.erase(std::unique(normalised.begin(), normalised.end()), normalised.end());
            return normalised;
        }

        inline torch::Tensor reduce_all_dimensions(torch::Tensor input, const ReduceOptions& options)
        {
            switch (options.op) {
                case ReduceOp::Sum: return torch::sum(input);
                case ReduceOp::Mean: return torch::mean(input);
                case ReduceOp::Max: return torch::amax(input);
                case ReduceOp::Min: return torch::amin(input);
            }

            TORCH_CHECK(false, "Unsupported reduce operation.");
            return torch::Tensor{};
        }

        inline torch::Tensor reduce_tensor(torch::Tensor input, const ReduceOptions& options)
        {
            if (!input.defined()) {
                return input;
            }

            auto dims = normalise_dims(input, options.dims);
            if (dims.empty()) {
                auto reduced = reduce_all_dimensions(std::move(input), options);
                if (options.keep_dim && reduced.defined()) {
                    return reduced.unsqueeze(0);
                }
                return reduced;
            }

            torch::IntArrayRef dims_ref(dims);
            switch (options.op) {
                case ReduceOp::Sum: return torch::sum(input, dims_ref, options.keep_dim);
                case ReduceOp::Mean: return torch::mean(input, dims_ref, options.keep_dim);
                case ReduceOp::Max: return torch::amax(input, dims_ref, options.keep_dim);
                case ReduceOp::Min: return torch::amin(input, dims_ref, options.keep_dim);
            }

            TORCH_CHECK(false, "Unsupported reduce operation.");
            return torch::Tensor{};
        }
    }

    class ReduceImpl : public torch::nn::Module {
    public:
        ReduceImpl() = default;

        explicit ReduceImpl(ReduceOptions options)
        {
            reset(std::move(options));
        }

        void reset(ReduceOptions options)
        {
            options_ = std::move(options);
        }

        torch::Tensor forward(torch::Tensor input)
        {
            return Detail::reduce_tensor(std::move(input), options_);
        }

        [[nodiscard]] const ReduceOptions& options() const noexcept { return options_; }

    private:
        ReduceOptions options_{};
    };

    TORCH_MODULE(Reduce);

    struct ReduceDescriptor {
        ReduceOptions options{};
        ::Nott::Activation::Descriptor activation{::Nott::Activation::Identity};
        ::Nott::LocalConfig local{};
    };

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const ReduceDescriptor& descriptor, std::size_t index)
    {
        auto module = owner.register_module("reduce_" + std::to_string(index), Reduce(descriptor.options));

        RegisteredLayer registered_layer{};
        registered_layer.activation = descriptor.activation.type;
        registered_layer.module = to_shared_module_ptr(module);
        registered_layer.local = descriptor.local;
        registered_layer.bind_module_forward(module.get());
        return registered_layer;
    }

}
#endif //Nott_LAYER_REDUCE_HPP