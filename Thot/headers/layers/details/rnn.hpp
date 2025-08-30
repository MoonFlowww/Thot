#pragma once

#include "../../tensor.hpp"
#include "../../initializations/initializations.hpp"
#include "../../../cuda/cuh/layers/rnn.cuh"
#include "../../activations/activations.hpp"

#include <random>
#include <cmath>
#include <iostream>
#include <utility> 

namespace Thot {

    class Layer;

    class RNNLayer : public Layer {
    private:
        int input_size_;
        int hidden_size_;
        int seq_length_;
        Activation activation_type_;
        Initialization initialization_type_;

        Utils::Tensor weights_ih_;    // Input to hidden weights
        Utils::Tensor weights_hh_;    // Hidden to hidden weights
        Utils::Tensor bias_;
        Utils::Tensor grad_weights_ih_;
        Utils::Tensor grad_weights_hh_;
        Utils::Tensor grad_bias_;

        Utils::Tensor hidden_state_;  // Current hidden state
        Utils::Tensor prev_hidden_state_; // Previous hidden state
        Utils::Tensor output_;        // Output after activation

    public:
        RNNLayer(int input_size, int hidden_size, int seq_length = 1,
         Activation activation_type = Activation::Tanh,
         Initialization param_init = Initialization::Xavier,
         const std::string& name = "RNN")
    : Layer(name),
        input_size_(input_size),
        hidden_size_(hidden_size),
        seq_length_(seq_length),
        activation_type_(activation_type),
        initialization_type_(param_init) {

            // Input-to-hidden weights
            weights_ih_ = Utils::Tensor({ hidden_size, input_size });
            Initializers::initialize_tensor(weights_ih_, param_init, input_size, hidden_size);
            grad_weights_ih_ = Utils::Tensor({ hidden_size, input_size }, true);

            // Hidden-to-hidden weights
            weights_hh_ = Utils::Tensor({ hidden_size, hidden_size });
            Initializers::initialize_tensor(weights_hh_, param_init, hidden_size, hidden_size);
            grad_weights_hh_ = Utils::Tensor({ hidden_size, hidden_size }, true);

            // Bias
            bias_ = Utils::Tensor({ hidden_size });
            Initializers::initialize_tensor(bias_, param_init, hidden_size, 1);
            grad_bias_ = Utils::Tensor({ hidden_size }, true);

            // Hidden state
            hidden_state_ = Utils::Tensor({ 1, hidden_size }, true);
            prev_hidden_state_ = Utils::Tensor({ 1, hidden_size }, true);
        }

        Utils::Tensor& W_ih() { return weights_ih_; }
        Utils::Tensor& W_hh() { return weights_hh_; }
        Utils::Tensor& bias() { return bias_; }

        void reset_hidden_state() {
            // Reset hidden state between sequences
            std::vector<float> zeros(hidden_size_, 0.0f);
            hidden_state_.upload(zeros);
            prev_hidden_state_.upload(zeros);
        }

        size_t get_flops(int batch_size = 1) const override {
            // Input to hidden: 2 * input_size * hidden_size FLOPs per timestep
            // Hidden to hidden: 2 * hidden_size * hidden_size FLOPs per timestep
            // Bias addition: hidden_size FLOPs per timestep
            // Multiply by sequence length and batch size
            return batch_size * seq_length_ *
                (2 * input_size_ * hidden_size_ +
                    2 * hidden_size_ * hidden_size_ +
                    hidden_size_);
        }

        Activation get_activation() const override {
            return activation_type_;
        }

        Initialization get_initialization() const override {
            return initialization_type_;
        }

        Utils::Tensor forward(const Utils::Tensor& input) override {
            int batch_size = input.shape()[0];


            /*
            this->input_cache_ = Utils::Tensor(input.shape(), true);
            cudaMemcpy(this->input_cache_.data(),
                       input.data(),
                       input.size() * sizeof(float),
                       cudaMemcpyDeviceToDevice);
            */


            this->input_cache_ = Utils::Tensor(input.shape());
            float* src_ptr = static_cast<float*>(input.data());
            float* dst_ptr = static_cast<float*>(this->input_cache_.data());
            size_t size = input.size() * sizeof(float);
            ::cudaMemcpy(dst_ptr, src_ptr, size, ::cudaMemcpyDeviceToDevice);



            output_ = Utils::Tensor({ batch_size, hidden_size_ });
            Utils::Tensor output_copy({ batch_size, hidden_size_ });


            float* h_src_ptr = static_cast<float*>(hidden_state_.data());
            float* h_dst_ptr = static_cast<float*>(prev_hidden_state_.data());
            size_t h_size = hidden_state_.size() * sizeof(float);
            ::cudaMemcpy(h_dst_ptr, h_src_ptr, h_size, ::cudaMemcpyDeviceToDevice);


            float* input_ptr = static_cast<float*>(this->input_cache_.data());
            float* weights_ih_ptr = static_cast<float*>(weights_ih_.data());
            float* weights_hh_ptr = static_cast<float*>(weights_hh_.data());
            float* bias_ptr = static_cast<float*>(bias_.data());
            float* prev_hidden_ptr = static_cast<float*>(prev_hidden_state_.data());
            float* hidden_ptr = static_cast<float*>(hidden_state_.data());
            float* output_ptr = static_cast<float*>(output_.data());

            // TODO: comment debug
            /*
            auto h_input = input.download();
            std::cout << "Input[0] host after upload: " << h_input[0] << std::endl;

            auto h_input_cache = this->input_cache_.download();
            std::cout << "Input_cache[0] before kernel: " << h_input_cache[0] << std::endl;

            auto h_wih = weights_ih_.download();
            std::cout << "W_ih[0]: " << h_wih[0] << std::endl;
            */

            ::cuda::layers::launchRNNForward(
                input_ptr, weights_ih_ptr, weights_hh_ptr, bias_ptr,
                prev_hidden_ptr, hidden_ptr, output_ptr,
                batch_size, seq_length_, input_size_, hidden_size_);

            float* out_src_ptr = static_cast<float*>(output_.data());
            float* out_dst_ptr = static_cast<float*>(output_copy.data());
            size_t out_size = output_.size() * sizeof(float);
            ::cudaMemcpy(out_dst_ptr, out_src_ptr, out_size, ::cudaMemcpyDeviceToDevice);

            return output_copy;
        }

        Utils::Tensor backward(const Utils::Tensor& grad_output) override {
            int batch_size = grad_output.shape()[0];

            // Create tensors for gradients
            Utils::Tensor grad_input({ batch_size, input_size_ });
            Utils::Tensor grad_hidden({ batch_size, hidden_size_ });

            float* grad_output_ptr = static_cast<float*>(grad_output.data());
            float* input_ptr = static_cast<float*>(this->input_cache_.data());
            float* weights_ih_ptr = static_cast<float*>(weights_ih_.data());
            float* weights_hh_ptr = static_cast<float*>(weights_hh_.data());
            float* hidden_ptr = static_cast<float*>(hidden_state_.data());
            float* prev_hidden_ptr = static_cast<float*>(prev_hidden_state_.data());
            float* grad_input_ptr = static_cast<float*>(grad_input.data());
            float* grad_hidden_ptr = static_cast<float*>(grad_hidden.data());
            float* grad_weights_ih_ptr = static_cast<float*>(grad_weights_ih_.data());
            float* grad_weights_hh_ptr = static_cast<float*>(grad_weights_hh_.data());
            float* grad_bias_ptr = static_cast<float*>(grad_bias_.data());

            // Calculate gradients for input
            ::cuda::layers::launchRNNBackwardInput(
                grad_output_ptr, weights_ih_ptr,
                grad_input_ptr, batch_size, seq_length_,
                input_size_, hidden_size_
            );

            // Calculate gradients for hidden state
            ::cuda::layers::launchRNNBackwardHidden(
                grad_output_ptr, weights_hh_ptr,
                grad_hidden_ptr, batch_size, hidden_size_
            );

            // Calculate gradients for input-to-hidden weights
            ::cuda::layers::launchRNNBackwardWeightsIH(
                input_ptr, grad_hidden_ptr,
                grad_weights_ih_ptr, batch_size, seq_length_,
                input_size_, hidden_size_
            );

            // Calculate gradients for hidden-to-hidden weights
            ::cuda::layers::launchRNNBackwardWeightsHH(
                prev_hidden_ptr, grad_hidden_ptr,
                grad_weights_hh_ptr, batch_size, hidden_size_
            );

            // Calculate gradients for bias
            ::cuda::layers::launchRNNBackwardBias(
                grad_hidden_ptr, grad_bias_ptr,
                batch_size, hidden_size_
            );

            // Apply gradients
            if (this->optimizer_) {
                this->optimizer_->update(weights_ih_, grad_weights_ih_);
                this->optimizer_->update(weights_hh_, grad_weights_hh_);
                this->optimizer_->update(bias_, grad_bias_);
            }
            else { // SGD by default
                // Update input-to-hidden weights
                const int weights_ih_size = weights_ih_.size();
                for (int i = 0; i < weights_ih_size; ++i) {
                    float* w_ptr = static_cast<float*>(weights_ih_.data());
                    float* gw_ptr = static_cast<float*>(grad_weights_ih_.data());
                    w_ptr[i] -= this->optimizer_->get_learning_rate() * gw_ptr[i];
                }

                // Update hidden-to-hidden weights
                const int weights_hh_size = weights_hh_.size();
                for (int i = 0; i < weights_hh_size; ++i) {
                    float* w_ptr = static_cast<float*>(weights_hh_.data());
                    float* gw_ptr = static_cast<float*>(grad_weights_hh_.data());
                    w_ptr[i] -= this->optimizer_->get_learning_rate() * gw_ptr[i];
                }

                // Update bias
                const int bias_size = bias_.size();
                for (int i = 0; i < bias_size; ++i) {
                    float* b_ptr = static_cast<float*>(bias_.data());
                    float* gb_ptr = static_cast<float*>(grad_bias_.data());
                    b_ptr[i] -= this->optimizer_->get_learning_rate() * gb_ptr[i];
                }
            }

            return std::move(grad_input);
        }

        int get_input_size() const override { return input_size_; }
        int get_output_size() const override { return hidden_size_; }
    };
} // namespace Thot
