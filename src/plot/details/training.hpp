#ifndef THOT_PLOT_DETAILS_TRAINING_HPP
#define THOT_PLOT_DETAILS_TRAINING_HPP

#include <optional>
#include <stdexcept>

#include <torch/torch.h>

namespace Thot {
    class Model;
}

namespace Thot::Plot::Training {
    struct LossDescriptor;
}

namespace Thot::Plot::Details::Training {
    inline void RenderLoss(Model& /*model*/, const Plot::Training::LossDescriptor& /*descriptor*/, torch::Tensor /*losses*/, std::optional<torch::Tensor> /*validationLoss*/, std::optional<torch::Tensor> /*learningRates*/)
    {
        throw std::logic_error("Plot::Details::Training::RenderLoss is not implemented yet.");
    }
}

#endif // THOT_PLOT_DETAILS_TRAINING_HPP