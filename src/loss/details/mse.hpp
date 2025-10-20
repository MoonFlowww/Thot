#ifndef THOT_MSE_HPP
#define THOT_MSE_HPP

#include <cstddef>
#include <optional>
#include <type_traits>

#include <torch/torch.h>

namespace thot::loss::details {
    enum class Reduction {
        None,
        Mean,
        Sum
    };

    template <typename T>
    inline constexpr bool is_tensor_v = std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, torch::Tensor>;

    template <typename...>
    inline constexpr bool always_false_v = false;

    template <Reduction ReductionMode,
              bool WithWeight = false,
              std::size_t PredictionRank = 0,
              std::size_t TargetRank = 0>
    struct MSEDescriptor {
        static_assert(PredictionRank == TargetRank || PredictionRank == 0 || TargetRank == 0,
                      "Prediction and target tensors must expose matching static ranks when provided.");

        static constexpr Reduction reduction = ReductionMode;
        static constexpr bool uses_weight = WithWeight;
        static constexpr std::size_t prediction_rank = PredictionRank;
        static constexpr std::size_t target_rank = TargetRank;

        template <typename Prediction, typename Target>
        static inline torch::Tensor value(const Prediction& prediction,
                                          const Target& target,
                                          const std::optional<torch::Tensor>& weight = std::nullopt) {
            static_assert(is_tensor_v<Prediction>, "Prediction tensor must be a torch::Tensor");
            static_assert(is_tensor_v<Target>, "Target tensor must be a torch::Tensor");

            auto options = torch::nn::functional::MSELossFuncOptions().reduction(torch::kNone);
            auto raw_loss = torch::nn::functional::mse_loss(prediction, target, options);

            if constexpr (uses_weight) {
                if (!weight.has_value()) {
                    TORCH_CHECK(false, "MSE descriptor expects a weight tensor when WithWeight = true");
                }
                const auto& weight_tensor = weight.value();
                TORCH_CHECK(weight_tensor.sizes() == raw_loss.sizes(),
                            "Weight tensor must match the element-wise shape for MSE loss");
                raw_loss = raw_loss * weight_tensor;
            } else {
                TORCH_CHECK(!weight.has_value(), "Unexpected weight tensor provided to unweighted MSE descriptor");
            }

            if constexpr (ReductionMode == Reduction::None) {
                return raw_loss;
            } else if constexpr (ReductionMode == Reduction::Mean) {
                return raw_loss.mean();
            } else if constexpr (ReductionMode == Reduction::Sum) {
                return raw_loss.sum();
            } else {
                static_assert(always_false_v<std::integral_constant<Reduction, ReductionMode>>,
                              "Unsupported reduction mode for MSE descriptor");
            }
        }

        template <typename Prediction, typename Target>
        static inline torch::Tensor pullback(const Prediction& prediction,
                                             const Target& target,
                                             const std::optional<torch::Tensor>& weight = std::nullopt) {
            static_assert(is_tensor_v<Prediction>, "Prediction tensor must be a torch::Tensor");
            static_assert(is_tensor_v<Target>, "Target tensor must be a torch::Tensor");

            auto differentiable_prediction = prediction.clone();
            differentiable_prediction.set_requires_grad(true);

            auto loss = value(differentiable_prediction, target, weight);
            if (loss.sizes().size() == 0) {
                loss.backward();
            } else {
                auto grad_output = torch::ones_like(loss);
                loss.backward(grad_output);
            }
            auto gradient = differentiable_prediction.grad();
            TORCH_CHECK(gradient.defined(), "Failed to compute gradient for MSE pullback");
            return gradient;
        }
    };

    template <typename Descriptor>
    struct is_mse_descriptor : std::false_type {};

    template <Reduction ReductionMode, bool WithWeight, std::size_t PredictionRank, std::size_t TargetRank>
    struct is_mse_descriptor<MSEDescriptor<ReductionMode, WithWeight, PredictionRank, TargetRank>> : std::true_type {};
}
#endif //THOT_MSE_HPP