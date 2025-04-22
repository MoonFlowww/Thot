#pragma once

#include "../../tensor.hpp"
#include <memory>
#include <unordered_map>
#include <string>
#include <sstream>
#include <cmath>
#include <type_traits>

// Include CUDA implementation
#include "../../../cuda/cuh/optimizations/adam.cuh"


namespace Thot {

    class Adam : public Optimizer {
    private:
        float beta1_;
        float beta2_;
        float epsilon_;
        int t_; // timestep counter

        std::unordered_map<std::string, Utils::Tensor> m_map_; // first moment vectors
        std::unordered_map<std::string, Utils::Tensor> v_map_; // second moment vectors

        std::string generate_tensor_id(const Utils::Tensor& tensor) {
            std::ostringstream oss;
            oss << tensor.data();
            return oss.str();
        }

    public:
        Adam(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
            : Optimizer(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {
        }

        inline void update(Utils::Tensor& weights, const Utils::Tensor& gradients) override {
            if (weights.size() != gradients.size()) {
                throw std::invalid_argument("Weight and gradient dimensions don't match in Adam optimizer");
            }

            t_++;

            std::string tensor_id = generate_tensor_id(weights);

            if (m_map_.find(tensor_id) == m_map_.end()) 
                m_map_[tensor_id] = Utils::Tensor(weights.shape(), true);


            if (v_map_.find(tensor_id) == v_map_.end()) 
                v_map_[tensor_id] = Utils::Tensor(weights.shape(), true);



            Utils::Tensor& m = m_map_[tensor_id]; // first moment
            Utils::Tensor& v = v_map_[tensor_id]; // second moment

            float correction1 = 1.0f - std::pow(beta1_, t_);
            float correction2 = 1.0f - std::pow(beta2_, t_);

            cuda::optimizations::launchAdamUpdate(
                static_cast<float*>(weights.data()),
                static_cast<float*>(m.data()),
                static_cast<float*>(v.data()),
                static_cast<const float*>(gradients.data()),
                this->learning_rate_,
                beta1_,
                beta2_,
                epsilon_,
                correction1,
                correction2,
                weights.size()
            );
        }

        float get_beta1() const { return beta1_; }
        void set_beta1(float beta1) { beta1_ = beta1; }

        float get_beta2() const { return beta2_; }
        void set_beta2(float beta2) { beta2_ = beta2; }

        float get_epsilon() const { return epsilon_; }
        void set_epsilon(float epsilon) { epsilon_ = epsilon; }

        static std::shared_ptr<Optimizer> create(float learning_rate = 0.001f, float beta1 = 0.9f,
            float beta2 = 0.999f, float epsilon = 1e-8f) {
            return std::make_shared<Adam>(learning_rate, beta1, beta2, epsilon);
        }
    };
} // namespace Thot