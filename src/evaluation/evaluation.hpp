#ifndef THOT_EVALUATION_HPP
#define THOT_EVALUATION_HPP
// This file is an factory, must exempt it from any logical-code. For functions look into "/details"
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "details/classification.hpp"
#include "details/timeserie.hpp"

namespace Thot::Evaluation {
    using Options = Details::Classification::Options;

    using ClassificationDescriptor = Details::Classification::Descriptor;
    inline constexpr ClassificationDescriptor Classification{};

    using ClassificationReport = Details::Classification::Report;

    template <class Model>
    [[nodiscard]] inline auto Evaluate(Model& model,
                                       torch::Tensor inputs,
                                       torch::Tensor targets,
                                       ClassificationDescriptor,
                                       std::vector<Metric::Classification::Descriptor> metrics,
                                       const Options& options = Options{}) -> ClassificationReport
    {
        return Details::Classification::Evaluate(
            model,
            std::move(inputs),
            std::move(targets),
            std::move(metrics),
            options);
    }
}

#endif //THOT_EVALUATION_HPP