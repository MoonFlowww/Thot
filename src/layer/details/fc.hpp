#ifndef THOT_FC_HPP
#define THOT_FC_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <torch/torch.h>

#include "../../initialization/initialization.hpp"

namespace thot::layer::details {
    template <std::size_t In, std::size_t Out, class Init = initialization::Xavier>
    class linear_descriptor {
    public:
        using module_type = torch::nn::Linear;
        using init_policy = Init;

        static constexpr std::size_t input_size = In;
        static constexpr std::size_t output_size = Out;
        static constexpr bool has_bias = true;

        linear_descriptor()
            : module_(torch::nn::LinearOptions(static_cast<int64_t>(In), static_cast<int64_t>(Out))) {
            initialize();
        }

        [[nodiscard]] inline torch::Tensor forward(const torch::Tensor& input) {
            return module_->forward(input);
        }

        [[nodiscard]] inline torch::Tensor backward(const torch::Tensor& grad_output) const {
            return grad_output.matmul(module_->weight);
        }

        [[nodiscard]] inline torch::Tensor weight() const {
            return module_->weight;
        }

        [[nodiscard]] inline torch::Tensor bias() const {
            if (module_->options.with_bias() && module_->bias.defined()) {
                return module_->bias;
            }
            return {};
        }

        [[nodiscard]] inline std::array<torch::Tensor, 2> parameters() const {
            torch::Tensor bias_tensor;
            if (module_->options.with_bias() && module_->bias.defined()) {
                bias_tensor = module_->bias;
            }
            return {module_->weight, bias_tensor};
        }

        [[nodiscard]] inline module_type& module() noexcept { return module_; }
        [[nodiscard]] inline const module_type& module() const noexcept { return module_; }

    private:
        void initialize() {
            if constexpr (std::is_default_constructible_v<Init>) {
                initialization::apply(Init{}, module_);
            }
        }

        module_type module_;
    };
}
#endif //THOT_FC_HPP