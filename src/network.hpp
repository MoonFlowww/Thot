#ifndef THOT_NETWORK_HPP
#define THOT_NETWORK_HPP
/*
* Pure network translation unit – meant to be compilable in isolation.
 * ---------------------------------------------------------------------------
* Implemented responsibilities:
 *  - Provide a lightweight `Runtime` wrapper that stores modules inside a
 *    `std::tuple`, exposing constexpr accessors and utilities to inspect the
 *    compile-time topology.
 *  - Offer forwarding helpers that transparently call modules regardless of
 *    whether they expose `operator()`, `.forward(...)`, or are held by pointer.
 *  - Assemble branchless forward and backward pipelines by unrolling the tuple
 *    and returning callable closures through `make_forward_pass` and
 *    `make_backward_pass`.
 *
 * Not yet implemented:
 *  - CUDA façade types or bindings around libtorch kernels.
 *  - Optimizer/initialisation helpers beyond what is exposed by other modules.
 *  - Wider runtime integrations (logging, data loading, CLI helpers).
 */


// Upcoming API sketch (to be validated with core.hpp):
// namespace Thot::Network {
//     template <class Config>
//     struct Runtime;
//
//     template <class Config>
//     [[nodiscard]] constexpr auto make_forward_pass(const Config&) noexcept;
//
//     template <class Config>
//     [[nodiscard]] constexpr auto make_backward_pass(const Config&) noexcept;
// }

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <torch/torch.h>


namespace Thot::Network {
    namespace Details {

        template <typename...>
        inline constexpr bool dependent_false_v = false;

        template <typename Module, typename Tensor>
        [[nodiscard]] auto invoke_forward(Module& module, Tensor&& tensor)
        {
            if constexpr (requires { module(std::forward<Tensor>(tensor)); }) {
                return module(std::forward<Tensor>(tensor));
            } else if constexpr (requires { module.forward(std::forward<Tensor>(tensor)); }) {
                return module.forward(std::forward<Tensor>(tensor));
            } else if constexpr (requires { module->forward(std::forward<Tensor>(tensor)); }) {
                return module->forward(std::forward<Tensor>(tensor));
            } else {
                static_assert(dependent_false_v<Module>, "Stored module does not expose a forward entry point.");
            }
        }

        template <typename Module, typename Gradient>
        [[nodiscard]] auto invoke_backward(Module& module, Gradient&& gradient)
        {
            if constexpr (requires { module.backward(std::forward<Gradient>(gradient)); }) {
                return module.backward(std::forward<Gradient>(gradient));
            } else if constexpr (requires { module->backward(std::forward<Gradient>(gradient)); }) {
                return module->backward(std::forward<Gradient>(gradient));
            } else {
                return std::forward<Gradient>(gradient);
            }
        }

        template <typename Tuple, typename Input>
        [[nodiscard]] auto unroll_forward(Tuple& modules, Input&& input)
        {
            return std::apply(
                [&](auto&... module) {
                    auto value = std::forward<Input>(input);
                    ((value = Details::invoke_forward(module, std::move(value))), ...);
                    return value;
                },
                modules);
        }

        template <typename Tuple, typename Gradient, std::size_t... Indices>
        [[nodiscard]] auto unroll_backward_impl(Tuple& modules, Gradient&& gradient, std::index_sequence<Indices...>)
        {
            auto value = std::forward<Gradient>(gradient);
            ((value = Details::invoke_backward(std::get<sizeof...(Indices) - 1U - Indices>(modules), std::move(value))), ...);
            return value;
        }

        template <typename Tuple, typename Gradient>
        [[nodiscard]] auto unroll_backward(Tuple& modules, Gradient&& gradient)
        {
            constexpr auto size = std::tuple_size_v<std::remove_reference_t<Tuple>>;
            if constexpr (size == 0) {
                return std::forward<Gradient>(gradient);
            } else {
                return Details::unroll_backward_impl(modules,
                                                     std::forward<Gradient>(gradient),
                                                     std::make_index_sequence<size>{});
            }
        }
    }

    template <typename ModuleTuple>
    class Runtime;

    template <typename... Modules>
    class Runtime<std::tuple<Modules...>> {
    public:
        using module_tuple_type = std::tuple<Modules...>;

        constexpr Runtime() = default;

        template <typename... ModuleArgs,
                  typename = std::enable_if_t<(sizeof...(ModuleArgs) == sizeof...(Modules)) &&
                                              (std::is_constructible_v<Modules, ModuleArgs&&> && ...)>>
        constexpr explicit Runtime(ModuleArgs&&... modules) noexcept(
            (std::is_nothrow_constructible_v<Modules, ModuleArgs&&> && ...))
            : modules_(std::forward<ModuleArgs>(modules)...) {}

        constexpr explicit Runtime(module_tuple_type modules) noexcept(
            (std::is_nothrow_move_constructible_v<Modules> && ...))
            : modules_(std::move(modules)) {}

        constexpr Runtime(const Runtime&) = default;
        constexpr Runtime(Runtime&&) = default;
        constexpr Runtime& operator=(const Runtime&) = default;
        constexpr Runtime& operator=(Runtime&&) = default;

        [[nodiscard]] constexpr module_tuple_type& modules() noexcept { return modules_; }
        [[nodiscard]] constexpr const module_tuple_type& modules() const noexcept { return modules_; }


        void set_device(const torch::Device& device) { device_ = device; }

        void set_device(bool use_cuda = true)
        {
            if (use_cuda && torch::cuda::is_available()) {
                device_ = torch::Device(torch::kCUDA, /*index=*/0);
            } else {
                device_ = torch::Device(torch::kCPU, /*index=*/0);
            }
        }

        [[nodiscard]] const torch::Device& device() const noexcept { return device_; }

        template <std::size_t Index>
        [[nodiscard]] constexpr auto& module_at() noexcept
        {
            static_assert(Index < sizeof...(Modules), "Requested module index out of range.");
            return std::get<Index>(modules_);
        }

        template <std::size_t Index>
        [[nodiscard]] constexpr const auto& module_at() const noexcept
        {
            static_assert(Index < sizeof...(Modules), "Requested module index out of range.");
            return std::get<Index>(modules_);
        }

        [[nodiscard]] static constexpr std::size_t size() noexcept { return sizeof...(Modules); }

    private:
        module_tuple_type modules_{};
    };

    template <typename... Modules>
    Runtime(Modules&&...) -> Runtime<std::tuple<std::decay_t<Modules>...>>;

    template <typename Runtime>
    [[nodiscard]] constexpr auto make_forward_pass(Runtime& runtime) noexcept
    {
        return [&runtime](auto&& input) -> decltype(auto) {
            auto tensor = std::forward<decltype(input)>(input);
            if constexpr (requires { tensor.device(); }) {
                if (tensor.device() != runtime.device()) {
                    tensor = tensor.to(runtime.device());
                }
            }
            return Details::unroll_forward(runtime.modules(), std::move(tensor));
        };
    }

    template <typename Runtime>
    [[nodiscard]] constexpr auto make_backward_pass(Runtime& runtime) noexcept
    {
        return [&runtime](auto&& gradient) -> decltype(auto) {
            auto tensor = std::forward<decltype(gradient)>(gradient);
            if constexpr (requires { tensor.device(); }) {
                if (tensor.device() != runtime.device()) {
                    tensor = tensor.to(runtime.device());
                }
            }
            return Details::unroll_backward(runtime.modules(), std::move(tensor));
        };
    }
}


#endif //THOT_NETWORK_HPP