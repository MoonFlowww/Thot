#pragma once
#include "../../cuda/cuh/activations/activations.cuh"
#include "../tensor.hpp"
#include <iostream>

namespace Thot {
    enum class Activation {
        Linear,
        ReLU,
        Sigmoid,
        Tanh,
        LeakyReLU,
        ELU,
        GELU,
        Softmax
    };

    namespace Activations {

        inline void apply_activation(Utils::Tensor& input, Utils::Tensor& output, Activation type) {
            int size = input.size();
            float* input_ptr = static_cast<float*>(input.data());
            float* output_ptr = static_cast<float*>(output.data());

            switch (type) {

            case Activation::ReLU:
                ::cuda::activations::launchReluForward(input_ptr, output_ptr, size);
                break;

            case Activation::Sigmoid:
                ::cuda::activations::launchSigmoidForward(input_ptr, output_ptr, size);
                break;

            case Activation::Tanh:
                ::cuda::activations::launchTanhForward(input_ptr, output_ptr, size);
                break;

            case Activation::LeakyReLU:
                ::cuda::activations::launchLeakyReluForward(input_ptr, output_ptr, 0.01f, size);
                break;

            case Activation::ELU:
                ::cuda::activations::launchEluForward(input_ptr, output_ptr, 1.0f, size);
                break;

            case Activation::GELU:
                ::cuda::activations::launchGeluForward(input_ptr, output_ptr, size);
                break;

            case Activation::Softmax:
                int batch_size = input.shape()[0];
                int feature_dim = size / batch_size;
                ::cuda::activations::launchSoftmaxForward(input_ptr, output_ptr, batch_size, feature_dim);
                break;
            }
        }

        inline void apply_activation_gradient(Utils::Tensor& input, Utils::Tensor& output, Utils::Tensor& grad_output, Utils::Tensor& grad_input, Activation type) {
            int size = input.size();
            float* input_ptr = static_cast<float*>(input.data());
            float* output_ptr = static_cast<float*>(output.data());
            float* grad_output_ptr = static_cast<float*>(grad_output.data());
            float* grad_input_ptr = static_cast<float*>(grad_input.data());

            switch (type) {
            case Activation::Linear:
                ::cudaMemcpy(grad_input_ptr, grad_output_ptr, size * sizeof(float), ::cudaMemcpyDeviceToDevice);
                break;

            case Activation::ReLU:
                ::cuda::activations::launchReluBackward(grad_output_ptr, input_ptr, grad_input_ptr, size);
                break;

            case Activation::Sigmoid:
                ::cuda::activations::launchSigmoidBackward(grad_output_ptr, output_ptr, grad_input_ptr, size);
                break;

            case Activation::Tanh:
                ::cuda::activations::launchTanhBackward(grad_output_ptr, output_ptr, grad_input_ptr, size);
                break;

            case Activation::LeakyReLU:
                ::cuda::activations::launchLeakyReluBackward(grad_output_ptr, input_ptr, grad_input_ptr, 0.01f, size);
                break;

            case Activation::ELU:
                ::cuda::activations::launchEluBackward(grad_output_ptr, output_ptr, input_ptr, grad_input_ptr, 1.0f, size);
                break;

            case Activation::GELU:
                ::cuda::activations::launchGeluBackward(grad_output_ptr, input_ptr, grad_input_ptr, size);
                break;

            case Activation::Softmax:
                std::cout << "CUDA: Computing Softmax activation gradient" << std::endl;
                int batch_size = input.shape()[0];
                int feature_dim = size / batch_size;
                ::cuda::activations::launchSoftmaxBackward(grad_output_ptr, output_ptr, grad_input_ptr, batch_size, feature_dim);
                break;
            }
        }
    } // namespace Activations
} // namespace Thot 