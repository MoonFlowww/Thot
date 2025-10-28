#ifndef THOT_PLOT_HPP
#define THOT_PLOT_HPP

#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>
#include <cstdint>
#include <vector>
#include <torch/torch.h>

#include "details/training.hpp"
#include "details/reliability.hpp"
#include "details/reliability/reliability_det.hpp"
#include "details/reliability/reliability_pr.hpp"
#include "details/reliability/reliability_roc.hpp"
#include "details/reliability/reliability_youdens.hpp"

namespace Thot {
    class Model;
}

namespace Thot::Plot {
    namespace Training {
        struct LossOptions {
            bool learningRate{false};
            bool smoothing{false};
            std::optional<std::size_t> smoothingWindow{};
            bool logScale{false};
        };

        struct LossDescriptor {
            LossOptions options{};
        };

        [[nodiscard]] constexpr auto Loss(LossOptions options = {}) -> LossDescriptor
        {
            return LossDescriptor{options};
        }

        inline void Render(Model& model,
                           const LossDescriptor& descriptor,
                           torch::Tensor losses,
                           std::optional<torch::Tensor> validationLoss = std::nullopt,
                           std::optional<torch::Tensor> learningRates = std::nullopt)
        {
            Details::Training::RenderLoss(model,
                                          descriptor,
                                          std::move(losses),
                                          std::move(validationLoss),
                                          std::move(learningRates));
        }
    }

    namespace Reliability {
        struct DETOptions {
            bool KSTest{false};
            bool confidenceBands{false};
            bool annotateCrossing{true};
        };

        struct DETDescriptor {
            DETOptions options{};
        };

        [[nodiscard]] constexpr auto DET(DETOptions options = {}) -> DETDescriptor
        {
            return DETDescriptor{options};
        }

        struct ROCOptions {
            bool KSTest{false};
            bool thresholds{false};
            bool logScale{false};
        };

        struct ROCDescriptor {
            ROCOptions options{};
        };

        [[nodiscard]] constexpr auto ROC(ROCOptions options = {}) -> ROCDescriptor
        {
            return ROCDescriptor{options};
        }

        struct YoudensOptions {
            bool KSTest{false};
            bool annotate{true};
            bool showIsoCost{false};
        };

        struct YoudensDescriptor {
            YoudensOptions options{};
        };

        [[nodiscard]] constexpr auto Youdens(YoudensOptions options = {}) -> YoudensDescriptor
        {
            return YoudensDescriptor{options};
        }

        struct PROptions {
            bool samples{false};
            bool random{false};
            bool interpolate{true};
        };

        struct PRDescriptor {
            PROptions options{};
        };

        [[nodiscard]] constexpr auto PR(PROptions options = {}) -> PRDescriptor
        {
            return PRDescriptor{options};
        }

        struct GradCAMOptions {
            std::size_t samples{0};
            bool random{false};
            bool normalize{true};
            bool overlay{true};
        };

        struct GradCAMDescriptor {
            GradCAMOptions options{};
        };

        [[nodiscard]] constexpr auto GradCAM(GradCAMOptions options = {}) -> GradCAMDescriptor
        {
            return GradCAMDescriptor{options};
        }

        struct LIMEOptions {
            std::size_t samples{500};
            bool random{true};
            bool normalize{true};
            bool showWeights{false};
        };

        struct LIMEDescriptor {
            LIMEOptions options{};
        };

        [[nodiscard]] constexpr auto LIME(LIMEOptions options = {}) -> LIMEDescriptor
        {
            return LIMEDescriptor{options};
        }


        inline void Render(Model& model,
                           const DETDescriptor& descriptor,
                           torch::Tensor logits,
                           torch::Tensor targets)
        {
            Details::Reliability::RenderDET(model,
                                            descriptor,
                                            std::move(logits),
                                            std::move(targets));
        }

        inline void Render(Model& model,
                           const DETDescriptor& descriptor,
                           torch::Tensor trainLogits,
                           torch::Tensor trainTargets,
                           torch::Tensor testLogits,
                           torch::Tensor testTargets)
        {
            Details::Reliability::RenderDET(model,
                                            descriptor,
                                            std::move(trainLogits),
                                            std::move(trainTargets),
                                            std::move(testLogits),
                                            std::move(testTargets));
        }

        inline void Render(Model& model,
                           const DETDescriptor& descriptor,
                           const std::vector<double>& probabilities,
                           const std::vector<int64_t>& targets)
        {
            Details::Reliability::RenderDET(model,
                                            descriptor,
                                            probabilities,
                                            targets);
        }

        inline void Render(Model& model,
                           const DETDescriptor& descriptor,
                           const std::vector<double>& trainProbabilities,
                           const std::vector<int64_t>& trainTargets,
                           const std::vector<double>& testProbabilities,
                           const std::vector<int64_t>& testTargets)
        {
            Details::Reliability::RenderDET(model,
                                            descriptor,
                                            trainProbabilities,
                                            trainTargets,
                                            testProbabilities,
                                            testTargets);
        }

        inline void Render(Model& model,
                           const ROCDescriptor& descriptor,
                           torch::Tensor logits,
                           torch::Tensor targets)
        {
            Details::Reliability::RenderROC(model,
                                            descriptor,
                                            std::move(logits),
                                            std::move(targets));
        }

        inline void Render(Model& model,
                   const ROCDescriptor& descriptor,
                   torch::Tensor trainLogits,
                   torch::Tensor trainTargets,
                   torch::Tensor testLogits,
                   torch::Tensor testTargets)
        {
            Details::Reliability::RenderROC(model,
                                            descriptor,
                                            std::move(trainLogits),
                                            std::move(trainTargets),
                                            std::move(testLogits),
                                            std::move(testTargets));
        }

        inline void Render(Model& model,
                           const ROCDescriptor& descriptor,
                           const std::vector<double>& probabilities,
                           const std::vector<int64_t>& targets)
        {
            Details::Reliability::RenderROC(model,
                                            descriptor,
                                            probabilities,
                                            targets);
        }

        inline void Render(Model& model,
                           const ROCDescriptor& descriptor,
                           const std::vector<double>& trainProbabilities,
                           const std::vector<int64_t>& trainTargets,
                           const std::vector<double>& testProbabilities,
                           const std::vector<int64_t>& testTargets)
        {
            Details::Reliability::RenderROC(model,
                                            descriptor,
                                            trainProbabilities,
                                            trainTargets,
                                            testProbabilities,
                                            testTargets);
        }

        inline void Render(Model& model,
                           const YoudensDescriptor& descriptor,
                           torch::Tensor logits,
                           torch::Tensor targets)
        {
            Details::Reliability::RenderYoudens(model,
                                                descriptor,
                                                std::move(logits),
                                                std::move(targets));
        }
        inline void Render(Model& model,
                           const YoudensDescriptor& descriptor,
                           torch::Tensor trainLogits,
                           torch::Tensor trainTargets,
                           torch::Tensor testLogits,
                           torch::Tensor testTargets)
        {
            Details::Reliability::RenderYoudens(model,
                                                descriptor,
                                                std::move(trainLogits),
                                                std::move(trainTargets),
                                                std::move(testLogits),
                                                std::move(testTargets));
        }

        inline void Render(Model& model,
                           const YoudensDescriptor& descriptor,
                           const std::vector<double>& probabilities,
                           const std::vector<int64_t>& targets)
        {
            Details::Reliability::RenderYoudens(model,
                                                descriptor,
                                                probabilities,
                                                targets);
        }

        inline void Render(Model& model,
                           const YoudensDescriptor& descriptor,
                           const std::vector<double>& trainProbabilities,
                           const std::vector<int64_t>& trainTargets,
                           const std::vector<double>& testProbabilities,
                           const std::vector<int64_t>& testTargets)
        {
            Details::Reliability::RenderYoudens(model,
                                                descriptor,
                                                trainProbabilities,
                                                trainTargets,
                                                testProbabilities,
                                                testTargets);
        }

        inline void Render(Model& model,
                           const PRDescriptor& descriptor,
                           torch::Tensor logits,
                           torch::Tensor targets)
        {
            Details::Reliability::RenderPR(model,
                                           descriptor,
                                           std::move(logits),
                                           std::move(targets));
        }
        inline void Render(Model& model,
                           const PRDescriptor& descriptor,
                           torch::Tensor trainLogits,
                           torch::Tensor trainTargets,
                           torch::Tensor testLogits,
                           torch::Tensor testTargets)
        {
            Details::Reliability::RenderPR(model,
                                           descriptor,
                                           std::move(trainLogits),
                                           std::move(trainTargets),
                                           std::move(testLogits),
                                           std::move(testTargets));
        }

        inline void Render(Model& model,
                           const PRDescriptor& descriptor,
                           const std::vector<double>& probabilities,
                           const std::vector<int64_t>& targets)
        {
            Details::Reliability::RenderPR(model,
                                           descriptor,
                                           probabilities,
                                           targets);
        }

        inline void Render(Model& model,
                           const PRDescriptor& descriptor,
                           const std::vector<double>& trainProbabilities,
                           const std::vector<int64_t>& trainTargets,
                           const std::vector<double>& testProbabilities,
                           const std::vector<int64_t>& testTargets)
        {
            Details::Reliability::RenderPR(model,
                                           descriptor,
                                           trainProbabilities,
                                           trainTargets,
                                           testProbabilities,
                                           testTargets);
        }

        inline void Render(Model& model,
                           const GradCAMDescriptor& descriptor,
                           torch::Tensor inputs,
                           torch::Tensor targets,
                           std::optional<std::size_t> targetLayer = std::nullopt)
        {
            Details::Reliability::RenderGradCAM(model,
                                                descriptor,
                                                std::move(inputs),
                                                std::move(targets),
                                                std::move(targetLayer));
        }

        inline void Render(Model& model,
                           const LIMEDescriptor& descriptor,
                           torch::Tensor inputs,
                           torch::Tensor targets)
        {
            Details::Reliability::RenderLIME(model,
                                             descriptor,
                                             std::move(inputs),
                                             std::move(targets));
        }
    }

    template <class Descriptor, class... Args>
    inline auto Render(Model& model, Descriptor&& descriptor, Args&&... args)
        -> decltype(auto)
    {
        if constexpr (std::is_same_v<std::decay_t<Descriptor>, Training::LossDescriptor>) {
            return Training::Render(model,
                                    descriptor,
                                    std::forward<Args>(args)...);
        } else if constexpr (std::is_same_v<std::decay_t<Descriptor>, Reliability::DETDescriptor>) {
            return Reliability::Render(model,
                                       descriptor,
                                       std::forward<Args>(args)...);
        } else if constexpr (std::is_same_v<std::decay_t<Descriptor>, Reliability::ROCDescriptor>) {
            return Reliability::Render(model,
                                       descriptor,
                                       std::forward<Args>(args)...);
        } else if constexpr (std::is_same_v<std::decay_t<Descriptor>, Reliability::YoudensDescriptor>) {
            return Reliability::Render(model,
                                       descriptor,
                                       std::forward<Args>(args)...);
        } else if constexpr (std::is_same_v<std::decay_t<Descriptor>, Reliability::PRDescriptor>) {
            return Reliability::Render(model,
                                       descriptor,
                                       std::forward<Args>(args)...);
        } else if constexpr (std::is_same_v<std::decay_t<Descriptor>, Reliability::GradCAMDescriptor>) {
            return Reliability::Render(model,
                                       descriptor,
                                       std::forward<Args>(args)...);
        } else if constexpr (std::is_same_v<std::decay_t<Descriptor>, Reliability::LIMEDescriptor>) {
            return Reliability::Render(model,
                                       descriptor,
                                       std::forward<Args>(args)...);
        } else {
            static_assert(sizeof(Descriptor) == 0, "Unsupported plot descriptor.");
        }
    }
}

#endif //THOT_PLOT_HPP