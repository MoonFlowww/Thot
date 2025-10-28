#ifndef THOT_PLOT_DETAILS_RELIABILITY_HPP
#define THOT_PLOT_DETAILS_RELIABILITY_HPP

#include <cstddef>
#include <optional>
#include <stdexcept>

#include <torch/torch.h>

namespace Thot {
    class Model;
}

namespace Thot::Plot::Reliability {
    struct DETOptions {
        bool KSTest{false};
        bool confidenceBands{false};
        bool annotateCrossing{true};
        bool logScale{false};
        bool expScale{false};
    };
    struct DETDescriptor {
        DETOptions options{};
    };

    struct ROCOptions {
        bool KSTest{false};
        bool thresholds{false};
        bool logScale{false};
    };

    struct ROCDescriptor {
        ROCOptions options{};
    };

    struct YoudensOptions {
        bool KSTest{false};
        bool annotate{true};
        bool showIsoCost{false};
        bool logScale{false};
    };

    struct YoudensDescriptor {
        YoudensOptions options{};
    };

    struct PROptions {
        bool samples{false};
        bool random{false};
        bool interpolate{true};
        bool logScale{false};
        bool expScale{false};
    };

    struct PRDescriptor {
        PROptions options{};
    };

    struct GradCAMOptions {
        std::size_t samples{0};
        bool random{false};
        bool normalize{true};
        bool overlay{true};
    };

    struct GradCAMDescriptor {
        GradCAMOptions options{};
    };

    struct LIMEOptions {
        std::size_t samples{500};
        bool random{true};
        bool normalize{true};
        bool showWeights{false};
    };

    struct LIMEDescriptor {
        LIMEOptions options{};
    };
}

namespace Thot::Plot::Details::Reliability {
    inline void RenderGradCAM(Model& /*model*/, const Plot::Reliability::GradCAMDescriptor& /*descriptor*/,
                                  torch::Tensor /*inputs*/, torch::Tensor /*targets*/,
                                  std::optional<std::size_t> /*targetLayer*/) {
        throw std::logic_error("Plot::Details::Reliability::RenderGradCAM is not implemented yet.");
    }

    inline void RenderLIME(Model& /*model*/, const Plot::Reliability::LIMEDescriptor& /*descriptor*/, torch::Tensor /*inputs*/, torch::Tensor /*targets*/) {
        throw std::logic_error("Plot::Details::Reliability::RenderLIME is not implemented yet.");
    }
}

#endif // THOT_PLOT_DETAILS_RELIABILITY_HPP