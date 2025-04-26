#pragma once
#include "../../../cuda/cuh/optimizations/sgdm.cuh"

#include "../../tensor.hpp"
#include <memory>

#include <unordered_map>
#include <string>
#include <sstream>
#include <type_traits>


namespace Thot {

    class SGDM : public Optimizer {
    private:
        float momentum_;
        std::unordered_map<std::string, Utils::Tensor> velocity_map_;

        std::string generate_tensor_id(const Utils::Tensor& tensor) {
            std::ostringstream oss;
            oss << tensor.data();
            return oss.str();
        }

    public:
        SGDM(float learning_rate = 0.01f, float momentum = 0.9f)
            : Optimizer(learning_rate), momentum_(momentum) {
        }

        inline void update(Utils::Tensor& weights, const Utils::Tensor& gradients) override {
            if (weights.size() != gradients.size())
                throw std::invalid_argument("Weight and gradient dimensions don't match in SGDM optimizations");


            std::string tensor_id = generate_tensor_id(weights);
            if (velocity_map_.find(tensor_id) == velocity_map_.end()) {
                velocity_map_[tensor_id] = Utils::Tensor(weights.shape(), true); 
            }

            Utils::Tensor& velocity = velocity_map_[tensor_id];

            ::cuda::optimizations::launchSGDMUpdate(
                static_cast<float*>(weights.data()),
                static_cast<float*>(velocity.data()),
                static_cast<const float*>(gradients.data()),
                this->learning_rate_,
                momentum_,
                weights.size()
            );
        }

        float get_momentum() const { return momentum_; }
        void set_momentum(float momentum) { momentum_ = momentum; }

        std::string get_name() const override { return "SGDM"; }
        std::string get_params() const override {
            return "Lr==" + std::to_string(learning_rate_) +
                ", Momentum=" + std::to_string(momentum_);
        }

        static std::shared_ptr<Optimizer> create(float learning_rate = 0.01f, float momentum = 0.9f) {
            return std::make_shared<SGDM>(learning_rate, momentum);
        }
    };
} // namespace Thot
