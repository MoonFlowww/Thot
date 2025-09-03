#pragma once
#pragma once

#include "../../tensor.hpp"
#include "../../initializations/initializations.hpp"

#include "../../../cuda/cuh/layers/fc.cuh"
#include "../../../cuda/cuh/optimizations/sgd.cuh"
#include "../../activations/activations.hpp"

#include <random>
#include <cmath>
#include <iostream>
#include <utility> 

namespace Thot {

    class Layer;
    class Network;

    class FCLayer : public Layer {
    private:
        friend class Network;
        int input_size_;
        int output_size_;
        Activation activation_type_;
        Initialization initialization_type_;


        Utils::Tensor weights_;
        Utils::Tensor bias_;
        Utils::Tensor grad_weights_;
        Utils::Tensor grad_bias_;

        Utils::Tensor pre_act_output_; // Output before activation
        Utils::Tensor activation_output_; // Storage for output after activation

    public:
        FCLayer(int input_size, int output_size, Activation activation_type = Activation::ReLU, Initialization weight_init = Initialization::Xavier, const std::string& name = "FC") : Layer(name), input_size_(input_size), output_size_(output_size), activation_type_(activation_type), initialization_type_(weight_init) {



            weights_ = Utils::Tensor({ output_size, input_size });

            Initializers::initialize_tensor(weights_, weight_init, input_size, output_size);

            grad_weights_ = Utils::Tensor({ output_size, input_size }, true); // Init to zeros

            bias_ = Utils::Tensor({ output_size });
            Initializers::zeros(bias_);

            grad_bias_ = Utils::Tensor({ output_size }, true); // Init to zeros
        }


        Utils::Tensor forward(const Utils::Tensor& input) override {
            this->input_cache_ = Utils::Tensor(input.shape());

            float* src_ptr = static_cast<float*>(input.data());
            float* dst_ptr = static_cast<float*>(this->input_cache_.data());
            size_t size = input.size() * sizeof(float);
            ::cudaMemcpy(dst_ptr, src_ptr, size, ::cudaMemcpyDeviceToDevice);

            int batch_size = input.shape()[0];

            Utils::Tensor pre_activation({ batch_size, output_size_ });
            Utils::Tensor output({ batch_size, output_size_ });

            float* input_ptr = static_cast<float*>(this->input_cache_.data());
            float* weights_ptr = static_cast<float*>(weights_.data());
            float* bias_ptr = static_cast<float*>(bias_.data());
            float* output_ptr = static_cast<float*>(pre_activation.data());

            ::cuda::layers::launchFCForward(input_ptr, weights_ptr, bias_ptr, output_ptr, batch_size, input_size_, output_size_);



            pre_act_output_ = std::move(pre_activation);

            Activations::apply_activation(pre_act_output_, output, activation_type_);

            activation_output_ = Utils::Tensor(output.shape());
            float* act_src_ptr = static_cast<float*>(output.data());
            float* act_dst_ptr = static_cast<float*>(activation_output_.data());
            size_t act_size = output.size() * sizeof(float);
            ::cudaMemcpy(act_dst_ptr, act_src_ptr, act_size, ::cudaMemcpyDeviceToDevice);

            return std::move(output);
        }

        Utils::Tensor backward(const Utils::Tensor& grad_output) override {
            int batch_size = grad_output.shape()[0];

            Utils::Tensor grad_pre_activation({ batch_size, output_size_ });
            Utils::Tensor grad_input({ batch_size, input_size_ });

            if (activation_type_ == Activation::Linear) {
                float* src_ptr = static_cast<float*>(grad_output.data());
                float* dst_ptr = static_cast<float*>(grad_pre_activation.data());
                size_t size = grad_output.size() * sizeof(float);
                ::cudaMemcpy(dst_ptr, src_ptr, size, ::cudaMemcpyDeviceToDevice);
            }
            else {
                Utils::Tensor temp_input({ batch_size, output_size_ });
                Utils::Tensor temp_output({ batch_size, output_size_ });
                Utils::Tensor temp_grad_output({ batch_size, output_size_ });

                float* pre_act_ptr = static_cast<float*>(pre_act_output_.data());
                float* act_ptr = static_cast<float*>(activation_output_.data());
                float* grad_out_ptr = static_cast<float*>(grad_output.data());

                float* temp_in_ptr = static_cast<float*>(temp_input.data());
                float* temp_out_ptr = static_cast<float*>(temp_output.data());
                float* temp_grad_ptr = static_cast<float*>(temp_grad_output.data());

                size_t size = batch_size * output_size_ * sizeof(float);

                ::cudaMemcpy(temp_in_ptr, pre_act_ptr, size, ::cudaMemcpyDeviceToDevice);
                ::cudaMemcpy(temp_out_ptr, act_ptr, size, ::cudaMemcpyDeviceToDevice);
                ::cudaMemcpy(temp_grad_ptr, grad_out_ptr, size, ::cudaMemcpyDeviceToDevice);

                Activations::apply_activation_gradient(
                    temp_input,          // input to activation (pre-activation)
                    temp_output,         // output from activation
                    temp_grad_output,    // gradient coming from next layer
                    grad_pre_activation, // where to store gradient
                    activation_type_);   // activation type
            }

            float* grad_pre_activation_ptr = static_cast<float*>(grad_pre_activation.data());
            float* weights_ptr = static_cast<float*>(weights_.data());
            float* input_ptr = static_cast<float*>(this->input_cache_.data());
            float* grad_input_ptr = static_cast<float*>(grad_input.data());
            float* grad_weights_ptr = static_cast<float*>(grad_weights_.data());
            float* grad_bias_ptr = static_cast<float*>(grad_bias_.data());

            // 1. Compute gradients for input
            ::cuda::layers::launchFCBackwardInput(
                grad_pre_activation_ptr, weights_ptr,
                grad_input_ptr, batch_size,
                input_size_, output_size_
            );


            // 2. Compute gradients for weights
            ::cuda::layers::launchFCBackwardWeights(
                input_ptr, grad_pre_activation_ptr,
                grad_weights_ptr, batch_size,
                input_size_, output_size_
            );

            // 3. Compute gradients for bias
            ::cuda::layers::launchFCBackwardBias(
                grad_pre_activation_ptr,
                grad_bias_ptr, batch_size,
                output_size_
            );


            if (this->optimizer_) {
                this->optimizer_->update(weights_, grad_weights_);
                this->optimizer_->update(bias_, grad_bias_);
            }
            else { // SGD by default
                // Apply basic gradient descent: w = w - lr * grad
                const int weights_size = weights_.size();
                const int bias_size = bias_.size();

                for (int i = 0; i < weights_size; ++i) {
                    float* w_ptr = static_cast<float*>(weights_.data());
                    float* gw_ptr = static_cast<float*>(grad_weights_.data());
                    w_ptr[i] -= this->optimizer_->get_learning_rate() * gw_ptr[i];
                }

                for (int i = 0; i < bias_size; ++i) {
                    float* b_ptr = static_cast<float*>(bias_.data());
                    float* gb_ptr = static_cast<float*>(grad_bias_.data());
                    b_ptr[i] -= this->optimizer_->get_learning_rate() * gb_ptr[i];
                }
            }

            return std::move(grad_input);
        }



        size_t get_flops(int batch_size = 1) const {
            // Matrix multiplication: 2 * input_size * output_size FLOPs per sample
            // Bias addition: output_size FLOPs per sample
            return batch_size * (2 * input_size_ * output_size_ + output_size_);
        }
        Activation get_activation() const override {
            return activation_type_;
        }

        Initialization get_initialization() const override {
            return initialization_type_;
        }

        int get_input_size() const override { return input_size_; }
        int get_output_size() const override { return output_size_; }

        Utils::Tensor& weights() { return weights_; }
        const Utils::Tensor& weights() const { return weights_; }
        Utils::Tensor& bias() { return bias_; }
        const Utils::Tensor& bias() const { return bias_; }
    };
} // namespace Thot