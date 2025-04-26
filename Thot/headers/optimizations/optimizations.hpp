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
        virtual std::string get_name() const = 0;
        virtual std::string get_params() const = 0;

        float get_learning_rate() const { return learning_rate_; }
        void set_learning_rate(float lr) { learning_rate_ = lr; }

        static std::shared_ptr<Optimizer> SGD(float learning_rate = 0.01f);
        static std::shared_ptr<Optimizer> SGDM(float learning_rate = 0.01f, float momentum = 0.9f);
        static std::shared_ptr<Optimizer> Adam(float learning_rate = 0.001f, float beta1 = 0.9f,
            float beta2 = 0.999f, float epsilon = 1e-8f);
    };

    class SGD;
    class SGDM;
    class Adam;
}

#include "details/sgd.hpp"
#include "details/sgdm.hpp"
#include "details/adam.hpp"

namespace Thot {
    inline std::shared_ptr<Optimizer> Optimizer::SGD(float learning_rate) {
        return std::make_shared<Thot::SGD>(learning_rate);
    }

    inline std::shared_ptr<Optimizer> Optimizer::SGDM(float learning_rate, float momentum) {
        return std::make_shared<Thot::SGDM>(learning_rate, momentum);
    }

    inline std::shared_ptr<Optimizer> Optimizer::Adam(float learning_rate, float beta1, float beta2, float epsilon) {
        return std::make_shared<Thot::Adam>(learning_rate, beta1, beta2, epsilon);
    }
}