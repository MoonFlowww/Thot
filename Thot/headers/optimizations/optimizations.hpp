#pragma once
#include "../tensor.hpp"
#include <memory>
#include <unordered_map>
#include <string>
#include <sstream>

namespace Thot {
    class Optimizer {
    protected:
        float learning_rate_;

    public:
        Optimizer(float learning_rate) : learning_rate_(learning_rate) {}
        virtual ~Optimizer() = default;

        virtual void update(Utils::Tensor& weights, const Utils::Tensor& gradients) = 0;

        float get_learning_rate() const { return learning_rate_; }
        void set_learning_rate(float lr) { learning_rate_ = lr; }
    };

    class SGD;
    class SGDM;
    class Adam;

#include "details/sgd.hpp"
#include "details/sgdm.hpp"
#include "details/adam.hpp"

    namespace optimizations {
        // Create SGD optimizer
        inline std::shared_ptr<Optimizer> SGD(float learning_rate = 0.01f) {
            return std::shared_ptr<Optimizer>(new Thot::SGD(learning_rate));
        }

        // Create SGDM optimizer
        inline std::shared_ptr<Optimizer> SGDM(float learning_rate = 0.01f, float momentum = 0.9f) {
            return std::shared_ptr<Optimizer>(new Thot::SGDM(learning_rate, momentum));
        }

        // Create Adam optimizer
        inline std::shared_ptr<Optimizer> Adam(float learning_rate = 0.001f, float beta1 = 0.9f,
            float beta2 = 0.999f, float epsilon = 1e-8f) {
            return std::shared_ptr<Optimizer>(new Thot::Adam(learning_rate, beta1, beta2, epsilon));
        }
    }
}