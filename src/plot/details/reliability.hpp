#ifndef THOT_PLOT_DETAILS_RELIABILITY_HPP
#define THOT_PLOT_DETAILS_RELIABILITY_HPP

#include <cstddef>
#include <optional>

#include <torch/torch.h>

#include "../../utils/gnuplot.hpp"

namespace Thot {
    class Model;
}

namespace Thot::Plot::Reliability {
    struct DETOptions {
        bool KSTest{false};
        bool confidenceBands{false};
        bool annotateCrossing{true};
        bool adjustScale{false};
        Utils::Gnuplot::TerminalOptions terminal{};
    };
    struct DETDescriptor {
        DETOptions options{};
    };

    struct ROCOptions {
        bool KSTest{false};
        bool thresholds{false};
        bool adjustScale{false};
        Utils::Gnuplot::TerminalOptions terminal{};
    };

    struct ROCDescriptor {
        ROCOptions options{};
    };

    struct YoudensOptions {
        bool KSTest{false};
        bool annotate{true};
        bool showIsoCost{false};
        bool adjustScale{false};
        Utils::Gnuplot::TerminalOptions terminal{};
    };

    struct YoudensDescriptor {
        YoudensOptions options{};
    };

    struct PROptions {
        bool samples{false};
        bool random{false};
        bool interpolate{true};
        bool adjustScale{false};
        Utils::Gnuplot::TerminalOptions terminal{};
    };

    struct PRDescriptor {
        PROptions options{};
    };

    struct GradCAMOptions {
        std::size_t samples{0};
        bool random{false};
        bool normalize{true};
        bool overlay{true};
        Utils::Gnuplot::TerminalOptions terminal{};
    };

    struct GradCAMDescriptor {
        GradCAMOptions options{};
    };

    struct LIMEOptions {
        std::size_t samples{500};
        bool random{true};
        bool normalize{true};
        bool showWeights{false};
        Utils::Gnuplot::TerminalOptions terminal{};
    };

    struct LIMEDescriptor {
        LIMEOptions options{};
    };
}

namespace Thot::Plot::Details::Reliability {
    void RenderGradCAM(Model& model,
                       const Plot::Reliability::GradCAMDescriptor& descriptor,
                       torch::Tensor inputs,
                       torch::Tensor targets,
                       std::optional<std::size_t> targetLayer = std::nullopt);

    void RenderLIME(Model& model,
                    const Plot::Reliability::LIMEDescriptor& descriptor,
                    torch::Tensor inputs,
                    torch::Tensor targets);
}
#include "reliability/reliability_gradcam.hpp"
#include "reliability/reliability_lime.hpp"
#endif // THOT_PLOT_DETAILS_RELIABILITY_HPP