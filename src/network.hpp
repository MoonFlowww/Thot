#ifndef THOT_NETWORK_HPP
#define THOT_NETWORK_HPP
/*
* Pure network translation unit – meant to be compilable in isolation.
 * ---------------------------------------------------------------------------
 * Planned responsibilities:
 *  - Receive already-instantiated module functors (layers, activations, loss,
 *    optimizer hooks, regularisation terms, etc.) from core.hpp as raw pointers
 *    or inline objects.
 *  - Provide constexpr-driven assembly of the forward and backward pipelines
 *    using tuple/unrolled execution so that the emitted machine code contains
 *    zero runtime branching (all feature toggles handled at compile time).
 *  - Wrap libtorch CUDA kernels through thin façade types so call sites remain
 *    expression-template friendly and compatible with a lazy syntax DSL while
 *    still delegating heavy lifting to the underlying library.
 *  - Expose lightweight `constexpr` helpers to: initialise parameters, execute
 *    forward passes, accumulate gradients, and apply optimizer steps.
 *  - Keep the public API header-only for maximal inlining, while implementation
 *    details can be hidden in nested `Details` namespaces or dedicated headers
 *    if template complexity warrants separation.
 *  - Avoid any dependency on the wider runtime (logging, data loading, CLI) so
 *    that, once compiled, this TU can be lifted and reused independently.
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

    }  // namespace Details

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
            return Details::unroll_forward(runtime.modules(), std::forward<decltype(input)>(input));
        };
    }

    template <typename Runtime>
    [[nodiscard]] constexpr auto make_backward_pass(Runtime& runtime) noexcept
    {
        return [&runtime](auto&& gradient) -> decltype(auto) {
            return Details::unroll_backward(runtime.modules(), std::forward<decltype(gradient)>(gradient));
        };
    }
}


#endif //THOT_NETWORK_HPP