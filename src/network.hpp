#ifndef THOT_NETWORK_HPP
#define THOT_NETWORK_HPP
/*
 * This file correspond to the network-only function.
 * It must be if-free, or, if can't be avoided then if constexpr.
 * Network will send pointers of function (for every module that compose a Neural Network, e.g. layers/opt/loss/regularization/blocks/...)
 * Only what's needed for forward and backward of the total network will be coded in here
 * We must keep the runtime exempt of conditions such as "Do we use regularization", "Do we use kfold", "...". Or any non-necessary logic for the runtime of Forward&Backward
*/


#include <cstddef>
#include <tuple>
#include <utility>

#include <torch/torch.h>

namespace thot::network {
    template <typename... Layers>
    class Sequential {
    public:
        using layers_type = std::tuple<Layers...>;

        Sequential() : layers_(make_layers()) {}
        explicit Sequential(layers_type layers) : layers_(std::move(layers)) {}

        static constexpr std::size_t size() noexcept { return sizeof...(Layers); }

        [[nodiscard]] inline layers_type& layers() noexcept { return layers_; }
        [[nodiscard]] inline const layers_type& layers() const noexcept { return layers_; }

        [[nodiscard]] inline torch::Tensor forward(const torch::Tensor& input) {
            if constexpr (sizeof...(Layers) == 0) {
                return input;
            } else {
                torch::Tensor output = input;
                std::apply(
                    [&](auto&... layer) {
                        ((output = layer.forward(output)), ...);
                    },
                    layers_);
                return output;
            }
        }

        [[nodiscard]] inline torch::Tensor backward(const torch::Tensor& grad_output) {
            if constexpr (sizeof...(Layers) == 0) {
                return grad_output;
            } else {
                torch::Tensor grad = grad_output;
                apply_reverse([&grad](auto& layer) { grad = layer.backward(grad); });
                return grad;
            }
        }

    private:
        template <typename Fn>
        inline void apply_reverse(Fn&& fn) {
            apply_reverse_impl(std::forward<Fn>(fn), std::make_index_sequence<sizeof...(Layers)>{});
        }

        template <typename Fn, std::size_t... Is>
        inline void apply_reverse_impl(Fn&& fn, std::index_sequence<Is...>) {
            (fn(std::get<sizeof...(Layers) - 1U - Is>(layers_)), ...);
        }

        static layers_type make_layers() {
            if constexpr (sizeof...(Layers) == 0) {
                return {};
            } else {
                return layers_type{Layers{}...};
            }
        }

        layers_type layers_;
    };
}


#endif //THOT_NETWORK_HPP