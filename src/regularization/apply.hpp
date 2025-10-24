#ifndef THOT_REGULARIZATION_APPLY_HPP
#define THOT_REGULARIZATION_APPLY_HPP

#include <torch/torch.h>
#include <type_traits>
#include <optional>
#include <utility>
#include <algorithm>
#include <functional>
#include <memory>
#include <type_traits>
#include <variant>
#include <vector>

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
            static_assert(always_false_v<Descriptor>, "No penalty implementation matches the provided descriptor/state combination.");
            return {};
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
    using StateVariant = std::variant<std::monostate,
                                      Details::EWCState,
                                      Details::MASState,
                                      Details::SIState>;

    using ParameterList = std::vector<torch::Tensor>;

    using Accumulator = std::function<torch::Tensor(const ParameterList&, const std::optional<torch::TensorOptions>&)>;

    namespace detail {
        inline torch::Tensor combine_penalty(torch::Tensor current,
                                             torch::Tensor next,
                                             const torch::TensorOptions& fallback_options,
                                             bool& initialized)
        {
            if (!initialized) {
                if (next.defined()) {
                    initialized = true;
                    return next;
                }
                return torch::zeros({}, fallback_options);
            }

            if (!next.defined()) {
                return current;
            }

            if (next.device() != current.device()) {
                next = next.to(current.device());
            }
            if (next.scalar_type() != current.scalar_type()) {
                next = next.to(current.scalar_type());
            }
            return current + next;
        }
    }

    [[nodiscard]] inline Accumulator bind_accumulator(Descriptor descriptor,
                                                      std::shared_ptr<const std::vector<StateVariant>> states)
    {
        return std::visit(
            [&](const auto& concrete_descriptor) -> Accumulator {
                using DescriptorType = std::decay_t<decltype(concrete_descriptor)>;

                if constexpr (std::is_same_v<DescriptorType, Details::EWCDescriptor>) {
                    return [descriptor = concrete_descriptor, states = std::move(states)](const ParameterList& parameters,
                                                                                        const std::optional<torch::TensorOptions>& fallback) {
                        const auto fallback_options = fallback.value_or(torch::TensorOptions().dtype(torch::kFloat32));
                        if (!states || states->empty() || parameters.empty()) {
                            return torch::zeros({}, fallback_options);
                        }

                        torch::Tensor total;
                        bool initialized = false;
                        const auto limit = std::min(states->size(), parameters.size());

                        for (std::size_t index = 0; index < limit; ++index) {
                            const auto& state_variant = states->at(index);
                            const auto& state = std::get<Details::EWCState>(state_variant);
                            auto penalty = apply(descriptor, parameters[index], state);
                            total = detail::combine_penalty(total, penalty, fallback_options, initialized);
                        }

                        if (!initialized) {
                            return torch::zeros({}, fallback_options);
                        }

                        return total;
                    };
                } else if constexpr (std::is_same_v<DescriptorType, Details::MASDescriptor>) {
                    return [descriptor = concrete_descriptor, states = std::move(states)](const ParameterList& parameters,
                                                                                        const std::optional<torch::TensorOptions>& fallback) {
                        const auto fallback_options = fallback.value_or(torch::TensorOptions().dtype(torch::kFloat32));
                        if (!states || states->empty() || parameters.empty()) {
                            return torch::zeros({}, fallback_options);
                        }

                        torch::Tensor total;
                        bool initialized = false;
                        const auto limit = std::min(states->size(), parameters.size());

                        for (std::size_t index = 0; index < limit; ++index) {
                            const auto& state_variant = states->at(index);
                            const auto& state = std::get<Details::MASState>(state_variant);
                            auto penalty = apply(descriptor, parameters[index], state);
                            total = detail::combine_penalty(total, penalty, fallback_options, initialized);
                        }

                        if (!initialized) {
                            return torch::zeros({}, fallback_options);
                        }

                        return total;
                    };
                } else if constexpr (std::is_same_v<DescriptorType, Details::SIDescriptor>) {
                    return [descriptor = concrete_descriptor, states = std::move(states)](const ParameterList& parameters,
                                                                                        const std::optional<torch::TensorOptions>& fallback) {
                        const auto fallback_options = fallback.value_or(torch::TensorOptions().dtype(torch::kFloat32));
                        if (!states || states->empty() || parameters.empty()) {
                            return torch::zeros({}, fallback_options);
                        }

                        torch::Tensor total;
                        bool initialized = false;
                        const auto limit = std::min(states->size(), parameters.size());

                        for (std::size_t index = 0; index < limit; ++index) {
                            const auto& state_variant = states->at(index);
                            const auto& state = std::get<Details::SIState>(state_variant);
                            auto penalty = apply(descriptor, parameters[index], state);
                            total = detail::combine_penalty(total, penalty, fallback_options, initialized);
                        }

                        if (!initialized) {
                            return torch::zeros({}, fallback_options);
                        }

                        return total;
                    };
                } else {
                    return [descriptor = concrete_descriptor](const ParameterList& parameters,
                                                              const std::optional<torch::TensorOptions>& fallback) {
                        return accumulate(descriptor, parameters, fallback);
                    };
                }
            },
            std::move(descriptor));
    }
}
#endif // THOT_REGULARIZATION_APPLY_HPP