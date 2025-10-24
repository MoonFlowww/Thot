#ifndef THOT_REGULARIZATION_APPLY_HPP
#define THOT_REGULARIZATION_APPLY_HPP

#include <torch/torch.h>

#include <optional>
#include <utility>
#include <variant>

#include "regularization.hpp"

namespace Thot::Regularization {
    namespace detail {
        template <class...>
        inline constexpr bool always_false_v = false;

        template <class Descriptor, class... Args>
        [[nodiscard]] inline torch::Tensor apply_descriptor(const Descriptor& descriptor,
                                                            const torch::Tensor& params,
                                                            Args&&... args) {
            if constexpr (requires {
                              Details::penalty(descriptor, params, std::forward<Args>(args)...);
                          }) {
                return Details::penalty(descriptor, params, std::forward<Args>(args)...);
                          } else {
                              static_assert(always_false_v<Descriptor>,
                                            "No penalty implementation matches the provided descriptor/state combination.");
                          }
        }
    } // namespace detail

    template <class... Args>
    [[nodiscard]] inline torch::Tensor apply(const Descriptor& descriptor,
                                             const torch::Tensor& params,
                                             Args&&... args) {
        return std::visit(
            [&](const auto& concrete) -> torch::Tensor {
                return detail::apply_descriptor(concrete, params, std::forward<Args>(args)...);
            },
            descriptor);
    }

    template <class TensorContainer, class... Args>
    [[nodiscard]] inline torch::Tensor accumulate(const Descriptor& descriptor,
                                                  const TensorContainer& parameters,
                                                  std::optional<torch::TensorOptions> fallback_options = std::nullopt,
                                                  Args&&... args) {
        torch::Tensor total;
        bool initialized = false;

        for (const auto& parameter : parameters) {
            auto penalty = apply(descriptor, parameter, std::forward<Args>(args)...);
            if (!initialized) {
                total = penalty;
                initialized = true;
            } else {
                total = total + penalty;
            }
        }

        if (!initialized) {
            auto options = fallback_options.value_or(torch::TensorOptions().dtype(torch::kFloat32));
            return torch::zeros({}, options);
        }

        return total;
    }
}
#endif // THOT_REGULARIZATION_APPLY_HPP