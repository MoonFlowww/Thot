#pragma once

#include "../tensor.hpp"
#include <memory>

namespace Thot {
    namespace Regularizers {

        class Regularizer {
        public:
            virtual ~Regularizer() = default;

            // compute penalty for given parameters
            virtual float compute(const Utils::Tensor& params) = 0;

            // update any internal state after a training step
            virtual void update_step(const Utils::Tensor& params) {}
        };

        using RegularizerPtr = std::shared_ptr<Regularizer>;

    } // namespace Regularizers
} // namespace Thot