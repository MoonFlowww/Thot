#pragma once
#include "../../../cuda/cuh/optimizations/sgd.cuh"

#include "../../tensor.hpp"
#include <memory>
#include <type_traits>
#include <string>

namespace Thot {

    class SGD : public Optimizer {
    public:
        SGD(float learning_rate = 0.01f, LearningRate lr_type = LearningRate::Constant, LrFn lr_fn = nullptr)
            : Optimizer(learning_rate, lr_type, lr_fn) {}

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

        std::string get_name() const override { return "SGD"; }
        std::string get_params() const override {
            return "Lr=" + std::to_string(learning_rate_);
        }

        static std::shared_ptr<Optimizer> create(float learning_rate = 0.01f, LearningRate lr_type = LearningRate::Constant, LrFn lr_fn = nullptr) {
            return std::make_shared<SGD>(learning_rate, lr_type, lr_fn);
        }
    };

} // namespace Thot
