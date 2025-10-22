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

#include <torch/torch.h>

#endif //THOT_NETWORK_HPP