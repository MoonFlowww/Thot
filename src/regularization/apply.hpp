#ifndef THOT_REGULARIZATION_APPLY_HPP
#define THOT_REGULARIZATION_APPLY_HPP

#include <torch/torch.h>
#include <type_traits>
#include <optional>
#include <utility>

#include "regularization.hpp"

namespace Thot::Regularization {
    namespace detail {
        template <class...>
        inline constexpr bool always_false_v = false;

        template <class Descriptor, class... Args>
        concept SupportsPenalty = requires(const Descriptor& descriptor,
                                   const torch::Tensor& params,
                                   Args&&... args) {
            Details::penalty(descriptor, params, std::forward<Args>(args)...);
                                   };

        template <class Descriptor, class... Args>
            requires SupportsPenalty<Descriptor, Args...>
        [[nodiscard]] inline torch::Tensor apply_descriptor(const Descriptor& descriptor,
                                                            const torch::Tensor& params,
                                                            Args&&... args) {
            return Details::penalty(descriptor, params, std::forward<Args>(args)...);
        }

        template <class Descriptor, class... Args>
            requires (!SupportsPenalty<Descriptor, Args...>)
        [[nodiscard]] inline torch::Tensor apply_descriptor(const Descriptor&,
                                                            const torch::Tensor&,
                                                            Args&&...) {
            static_assert(always_false_v<Descriptor>,
                          "No penalty implementation matches the provided descriptor/state combination.");
        }
    }

    template <class DescriptorType, class... Args>
           requires (!std::is_same_v<std::decay_t<DescriptorType>, Descriptor>)
    [[nodiscard]] inline torch::Tensor apply(const DescriptorType& descriptor,
                                             const torch::Tensor& params,
                                             Args&&... args) {
        return detail::apply_descriptor(descriptor, params, std::forward<Args>(args)...);
    }

    template <class DescriptorType, class TensorContainer, class... Args>
            requires (!std::is_same_v<std::decay_t<DescriptorType>, Descriptor>)
        [[nodiscard]] inline torch::Tensor accumulate(const DescriptorType& descriptor,
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