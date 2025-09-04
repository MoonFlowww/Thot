#pragma once
#include "../../../cuda/cuh/optimizations/adamuon.cuh"

#include "../../tensor.hpp"
#include <string>

namespace Thot {

    class AdaMuon : public Optimizer {
    private:
        float beta1_;
        float beta2_;
        float weight_decay_;
    public:
        AdaMuon(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f,
                float weight_decay = 0.0f, LearningRate lr_type = LearningRate::Constant, LrFn lr_fn = nullptr)
            : Optimizer(learning_rate, lr_type, lr_fn),
              beta1_(beta1), beta2_(beta2), weight_decay_(weight_decay) {}

        inline void update(Utils::Tensor& weights, const Utils::Tensor& gradients) override {
            if (weights.size() != gradients.size()) {
                throw std::invalid_argument("Weight and gradient dimensions don't match in AdaMuon optimizer");
            }
            ::cuda::optimizations::launchAdaMuonUpdate(
                static_cast<float*>(weights.data()),
                static_cast<const float*>(gradients.data()),
                this->learning_rate_,
                beta1_,
                beta2_,
                weight_decay_,
                weights.size()
            );
        }

        std::string get_name() const override { return "AdaMuon"; }

        std::string get_params() const override {
            return "lr=" + std::to_string(learning_rate_) +
                   ", beta1=" + std::to_string(beta1_) +
                   ", beta2=" + std::to_string(beta2_) +
                   ", weight_decay=" + std::to_string(weight_decay_);
        }
    };

} // namespace Thot