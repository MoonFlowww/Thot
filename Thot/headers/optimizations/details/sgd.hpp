#pragma once
#include "../../../cuda/cuh/optimizations/sgd.cuh"

#include "../../tensor.hpp"
#include <memory>
#include <type_traits>




namespace Thot {

    class SGD : public Optimizer {
    public:
        SGD(float learning_rate = 0.01f) : Optimizer(learning_rate) {}

        inline void update(Utils::Tensor& weights, const Utils::Tensor& gradients) override {
            if (weights.size() != gradients.size()) {
                throw std::invalid_argument("Weight and gradient dimensions don't match in SGD optimizer");
            }

            ::cuda::optimizations::launchSGDUpdate(
                static_cast<float*>(weights.data()),
                static_cast<const float*>(gradients.data()),
                this->learning_rate_,
                weights.size()
            );
        }

        static std::shared_ptr<Optimizer> create(float learning_rate = 0.01f) {
            return std::make_shared<SGD>(learning_rate);
        }
    };

} // namespace Thot
