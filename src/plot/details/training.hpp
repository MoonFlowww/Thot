#ifndef Nott_PLOT_DETAILS_TRAINING_HPP
#define Nott_PLOT_DETAILS_TRAINING_HPP

#include <optional>
#include <stdexcept>

#include <torch/torch.h>

namespace Nott {
    class Model;
}

namespace Nott::Plot::Training {
    struct LossDescriptor;
}

namespace Nott::Plot::Details::Training {
    inline void RenderLoss(Model& /*model*/, const Plot::Training::LossDescriptor& /*descriptor*/, torch::Tensor /*losses*/, std::optional<torch::Tensor> /*validationLoss*/, std::optional<torch::Tensor> /*learningRates*/)
    {
        throw std::logic_error("Plot::Details::Training::RenderLoss is not implemented yet.");
    }
}

#endif // Nott_PLOT_DETAILS_TRAINING_HPP