#pragma once
#include "../../../cuda/cuh/optimizations/muon.cuh"

#include "../../tensor.hpp"
#include <string>

namespace Thot {

    class Muon : public Optimizer {
    private:
        float beta_;
        float weight_decay_;
    public:
        Muon(float learning_rate = 0.01f, float beta = 0.9f, float weight_decay = 0.0f,
             LearningRate lr_type = LearningRate::Constant, LrFn lr_fn = nullptr)
            : Optimizer(learning_rate, lr_type, lr_fn), beta_(beta), weight_decay_(weight_decay) {}

        inline void update(Utils::Tensor& weights, const Utils::Tensor& gradients) override {
            if (weights.size() != gradients.size()) {
                throw std::invalid_argument("Weight and gradient dimensions don't match in Muon optimizer");
            }
            ::cuda::optimizations::launchMuonUpdate(
                static_cast<float*>(weights.data()),
                static_cast<const float*>(gradients.data()),
                this->learning_rate_,
                beta_,
                weight_decay_,
                weights.size()
            );
        }

        std::string get_name() const override { return "Muon"; }

        std::string get_params() const override {
            return "lr=" + std::to_string(learning_rate_) +
                   ", beta=" + std::to_string(beta_) +
                   ", weight_decay=" + std::to_string(weight_decay_);
        }
    };

} // namespace Thot
