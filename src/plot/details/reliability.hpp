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
    struct DETDescriptor;
    struct ROCDescriptor;
    struct YoudensDescriptor;
    struct PRDescriptor;
    struct GradCAMDescriptor;
    struct LIMEDescriptor;
}

namespace Thot::Plot::Details::Reliability {
    inline void RenderDET(Model& /*model*/, const Plot::Reliability::DETDescriptor& /*descriptor*/, torch::Tensor /*logits*/, torch::Tensor /*targets*/)
    {
        throw std::logic_error("Plot::Details::Reliability::RenderDET is not implemented yet.");
    }

    inline void RenderROC(Model& /*model*/, const Plot::Reliability::ROCDescriptor& /*descriptor*/, torch::Tensor /*logits*/, torch::Tensor /*targets*/)
    {
        throw std::logic_error("Plot::Details::Reliability::RenderROC is not implemented yet.");
    }

    inline void RenderYoudens(Model& /*model*/, const Plot::Reliability::YoudensDescriptor& /*descriptor*/, torch::Tensor /*logits*/, torch::Tensor /*targets*/)
    {
        throw std::logic_error("Plot::Details::Reliability::RenderYoudens is not implemented yet.");
    }

    inline void RenderPR(Model& /*model*/, const Plot::Reliability::PRDescriptor& /*descriptor*/, torch::Tensor /*logits*/, torch::Tensor /*targets*/)
    {
        throw std::logic_error("Plot::Details::Reliability::RenderPR is not implemented yet.");
    }

    inline void RenderGradCAM(Model& /*model*/, const Plot::Reliability::GradCAMDescriptor& /*descriptor*/, torch::Tensor /*inputs*/, torch::Tensor /*targets*/, std::optional<std::size_t> /*targetLayer*/)
    {
        throw std::logic_error("Plot::Details::Reliability::RenderGradCAM is not implemented yet.");
    }

    inline void RenderLIME(Model& /*model*/, const Plot::Reliability::LIMEDescriptor& /*descriptor*/, torch::Tensor /*inputs*/, torch::Tensor /*targets*/)
    {
        throw std::logic_error("Plot::Details::Reliability::RenderLIME is not implemented yet.");
    }
}

#endif // THOT_PLOT_DETAILS_RELIABILITY_HPP