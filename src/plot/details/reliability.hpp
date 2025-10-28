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

    struct GradCAMDescriptor;
    struct LIMEDescriptor;
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