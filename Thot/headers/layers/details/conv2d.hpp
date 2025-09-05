#pragma once

#include "../../tensor.hpp"
#include "../../initializations/initializations.hpp"
#include "../../../cuda/cuh/layers/conv2d.cuh"
#include "../../activations/activations.hpp"

#include <random>
#include <cmath>
#include <iostream>
#include <utility>
#include <chrono>

namespace Thot {

    class Layer;
    class Network;

    class Conv2DLayer : public Layer {
    private:
        friend class Network;
        int in_channels_;
        int in_height_;
        int in_width_;
        int out_channels_;
        int kernel_size_;
        int stride_;
        int padding_;
        int out_height_;
        int out_width_;
        Activation activation_type_;
        Initialization initialization_type_;
        ::cuda::layers::ConvAlgo conv_algo_;

        Utils::Tensor weights_;
        Utils::Tensor bias_;
        Utils::Tensor grad_weights_;
        Utils::Tensor grad_bias_;

        Utils::Tensor pre_act_output_; // Output before activation
        Utils::Tensor activation_output_; // Storage for output after activation

        std::chrono::nanoseconds total_init_;

    public:
        Conv2DLayer(int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride = 1, int padding = 0,
            Activation activation_type = Activation::ReLU,
            Initialization weight_init = Initialization::Xavier,
            ::cuda::layers::ConvAlgo conv_algo = ::cuda::layers::ConvAlgo::Auto,
            const std::string& name = "Conv2D")
            : Layer(name),
            in_channels_(in_channels),
            in_height_(in_height),
            in_width_(in_width),
            out_channels_(out_channels),
            kernel_size_(kernel_size),
            stride_(stride),
            padding_(padding),
            activation_type_(activation_type),
            initialization_type_(weight_init),
            conv_algo_(conv_algo) {

            auto t1 = std::chrono::high_resolution_clock::now();
            out_height_ = (in_height_ + 2 * padding_ - kernel_size_) / stride_ + 1;
            out_width_ = (in_width_ + 2 * padding_ - kernel_size_) / stride_ + 1;

            weights_ = Utils::Tensor({ out_channels_, in_channels_, kernel_size_, kernel_size_ });

            // For convolution layers, fan_in = in_channels * kernel_size * kernel_size
            // fan_out = out_channels * kernel_size * kernel_size
            int fan_in = in_channels_ * kernel_size_ * kernel_size_;
            int fan_out = out_channels_ * kernel_size_ * kernel_size_;

            Initializations::initialize_tensor(weights_, weight_init, fan_in, fan_out);

            grad_weights_ = Utils::Tensor({ out_channels_, in_channels_, kernel_size_, kernel_size_ }, true);

            // Initialize bias
            bias_ = Utils::Tensor({ out_channels_ });
            Initializations::zeros(bias_);
            grad_bias_ = Utils::Tensor({ out_channels_ }, true);
            auto t2 = std::chrono::high_resolution_clock::now();
            total_init_ = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1);
        }

        size_t get_flops(int batch_size = 1) const override {
            // Each output element requires kernel_size^2 * in_channels MACs (2 FLOPs each)
            // Plus bias addition (1 FLOP per output element)
            // Total output elements = batch_size * out_channels * out_height * out_width
            return batch_size * out_channels_ * out_height_ * out_width_ *
                (2 * kernel_size_ * kernel_size_ * in_channels_ + 1);
        }

        size_t get_parameters() const override {
            return out_channels_ * in_channels_ * kernel_size_ * kernel_size_ + out_channels_;
        }

        Activation get_activation() const override {
            return activation_type_;
        }

        Initialization get_initialization() const override {
            return initialization_type_;
        }

        float get_latency() const override {
            return total_init_.count();
        }

        Utils::Tensor forward(const Utils::Tensor& input) override {
            this->input_cache_ = Utils::Tensor(input.shape());
            float* src_ptr = static_cast<float*>(input.data());
            float* dst_ptr = static_cast<float*>(this->input_cache_.data());
            size_t size = input.size() * sizeof(float);
            ::cudaMemcpy(dst_ptr, src_ptr, size, ::cudaMemcpyDeviceToDevice);

            int batch_size = input.shape()[0];

            Utils::Tensor pre_activation({ batch_size, out_channels_, out_height_, out_width_ });

            // Create output tensor for post-activation values
            Utils::Tensor output({ batch_size, out_channels_, out_height_, out_width_ });

            float* input_ptr = static_cast<float*>(this->input_cache_.data());
            float* weights_ptr = static_cast<float*>(weights_.data());
            float* bias_ptr = static_cast<float*>(bias_.data());
            float* pre_act_ptr = static_cast<float*>(pre_activation.data());

            ::cuda::layers::launchConv2DForward(
                input_ptr, weights_ptr, bias_ptr, pre_act_ptr,
                batch_size, in_channels_, in_height_, in_width_,
                out_channels_, kernel_size_, stride_, padding_,
                out_height_, out_width_, conv_algo_
            );

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

            Utils::Tensor grad_pre_activation({ batch_size, out_channels_, out_height_, out_width_ });
            Utils::Tensor grad_input({ batch_size, in_channels_, in_height_, in_width_ });

            if (activation_type_ == Activation::Linear) {
                float* src_ptr = static_cast<float*>(grad_output.data());
                float* dst_ptr = static_cast<float*>(grad_pre_activation.data());
                size_t size = grad_output.size() * sizeof(float);
                ::cudaMemcpy(dst_ptr, src_ptr, size, ::cudaMemcpyDeviceToDevice);
            }
            else {
                Utils::Tensor temp_input({ batch_size, out_channels_, out_height_, out_width_ });
                Utils::Tensor temp_output({ batch_size, out_channels_, out_height_, out_width_ });
                Utils::Tensor temp_grad_output({ batch_size, out_channels_, out_height_, out_width_ });

                float* pre_act_ptr = static_cast<float*>(pre_act_output_.data());
                float* act_ptr = static_cast<float*>(activation_output_.data());
                float* grad_out_ptr = static_cast<float*>(grad_output.data());

                float* temp_in_ptr = static_cast<float*>(temp_input.data());
                float* temp_out_ptr = static_cast<float*>(temp_output.data());
                float* temp_grad_ptr = static_cast<float*>(temp_grad_output.data());

                size_t size = grad_output.size() * sizeof(float);

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

            // Compute gradients for input
            ::cuda::layers::launchConv2DBackwardInput(
                grad_pre_activation_ptr, weights_ptr, grad_input_ptr,
                batch_size, in_channels_, in_height_, in_width_,
                out_channels_, kernel_size_, stride_, padding_,
                out_height_, out_width_, conv_algo_
            );

            // Compute gradients for weights
            ::cuda::layers::launchConv2DBackwardWeights(
                input_ptr, grad_pre_activation_ptr, grad_weights_ptr,
                batch_size, in_channels_, in_height_, in_width_,
                out_channels_, kernel_size_, stride_, padding_,
                out_height_, out_width_, conv_algo_
            );

            // Compute gradients for bias
            ::cuda::layers::launchConv2DBackwardBias(
                grad_pre_activation_ptr, grad_bias_ptr,
                batch_size, out_channels_, out_height_, out_width_
            );

            if (this->optimizer_) {
                this->optimizer_->update(weights_, grad_weights_);
                this->optimizer_->update(bias_, grad_bias_);
            }
            else { // SGD by default
                // Update w
                const int weights_size = weights_.size();
                for (int i = 0; i < weights_size; ++i) {
                    float* w_ptr = static_cast<float*>(weights_.data());
                    float* gw_ptr = static_cast<float*>(grad_weights_.data());
                    w_ptr[i] -= this->optimizer_->get_learning_rate() * gw_ptr[i];
                }

                // Update b
                const int bias_size = bias_.size();
                for (int i = 0; i < bias_size; ++i) {
                    float* b_ptr = static_cast<float*>(bias_.data());
                    float* gb_ptr = static_cast<float*>(grad_bias_.data());
                    b_ptr[i] -= this->optimizer_->get_learning_rate() * gb_ptr[i];
                }
            }

            return std::move(grad_input);
        }
        int get_input_size() const override { return in_channels_ * in_height_ * in_width_; } // color * height * width
        int get_output_size() const override { return out_channels_ * out_height_ * out_width_; }

        int in_channels() const { return in_channels_; }
        int in_height() const { return in_height_; }
        int in_width() const { return in_width_; }
        int out_channels() const { return out_channels_; }
        int kernel_size() const { return kernel_size_; }
        int stride() const { return stride_; }
        int padding() const { return padding_; }

        Utils::Tensor& weights() { return weights_; }
        const Utils::Tensor& weights() const { return weights_; }
        Utils::Tensor& bias() { return bias_; }
        const Utils::Tensor& bias() const { return bias_; }
    };
} // namespace Thot