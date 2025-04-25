#include <math_constants.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include "../../cuh/activations/activations.cuh"

namespace cuda {
    const int BLOCK_SIZE = 256;
    namespace activations {

        // ReLU Activation
        __global__ void relu_forward(const float* input, float* output, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                output[idx] = fmaxf(0.0f, input[idx]);
            }
        }

        __global__ void relu_backward(const float* grad_output, const float* input, float* grad_input, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
            }
        }

        // Sigmoid Activation
        __global__ void sigmoid_forward(const float* input, float* output, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                output[idx] = 1.0f / (1.0f + expf(-input[idx]));
            }
        }

        __global__ void sigmoid_backward(const float* grad_output, const float* output, float* grad_input, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                // Sigmoid gradient is sigmoid(x) * (1 - sigmoid(x))
                // output already contains sigmoid(x)
                grad_input[idx] = grad_output[idx] * output[idx] * (1.0f - output[idx]);
            }
        }

        // Tanh Activation
        __global__ void tanh_forward(const float* input, float* output, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                output[idx] = tanhf(input[idx]);
            }
        }

        __global__ void tanh_backward(const float* grad_output, const float* output, float* grad_input, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                // Tanh gradient is (1 - tanh^2(x))
                // output already contains tanh(x)
                grad_input[idx] = grad_output[idx] * (1.0f - output[idx] * output[idx]);
            }
        }

        // Leaky ReLU Activation
        __global__ void leaky_relu_forward(const float* input, float* output, float negative_slope, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float x = input[idx];
                output[idx] = x > 0.0f ? x : x * negative_slope;
            }
        }

        __global__ void leaky_relu_backward(const float* grad_output, const float* input, float* grad_input, float negative_slope, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float x = input[idx];
                grad_input[idx] = grad_output[idx] * (x > 0.0f ? 1.0f : negative_slope);
            }
        }

        // ELU Activation
        __global__ void elu_forward(const float* input, float* output, float alpha, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float x = input[idx];
                output[idx] = x > 0.0f ? x : alpha * (expf(x) - 1.0f);
            }
        }

        __global__ void elu_backward(const float* grad_output, const float* output, const float* input, float* grad_input, float alpha, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float x = input[idx];
                float grad = x > 0.0f ? 1.0f : output[idx] + alpha;
                grad_input[idx] = grad_output[idx] * grad;
            }
        }

        // GELU Activation (Gaussian Error Linear Unit)
        __global__ void gelu_forward(const float* input, float* output, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float x = input[idx];
                // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                float sqrt2_over_pi = 0.7978845608028654f;  // sqrt(2/pi)
                float y = sqrt2_over_pi * (x + 0.044715f * x * x * x);
                output[idx] = 0.5f * x * (1.0f + tanhf(y));
            }
        }

        __global__ void gelu_backward(const float* grad_output, const float* input, float* grad_input, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float x = input[idx];
                float sqrt2_over_pi = 0.7978845608028654f;
                float y = sqrt2_over_pi * (x + 0.044715f * x * x * x);
                float tanh_y = tanhf(y);

                float inner_term = 0.044715f * x * x;
                float outer_term = 1.0f - tanh_y * tanh_y;
                float derivative = 0.5f * (1.0f + tanh_y) + 0.5f * x * outer_term * sqrt2_over_pi * (1.0f + 3.0f * inner_term);

                grad_input[idx] = grad_output[idx] * derivative;
            }
        }

        // Softmax Activation
        __global__ void softmax_forward(const float* input, float* output, int batch_size, int feature_dim) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < batch_size) {
                // Find max value for numerical stability
                float max_val = -CUDART_INF_F;
                for (int i = 0; i < feature_dim; ++i) {
                    int input_idx = idx * feature_dim + i;
                    max_val = fmaxf(max_val, input[input_idx]);
                }

                // Compute sum of exp(input - max_val)
                float sum_exp = 0.0f;
                for (int i = 0; i < feature_dim; ++i) {
                    int input_idx = idx * feature_dim + i;
                    sum_exp += expf(input[input_idx] - max_val);
                }

                // Compute softmax: exp(input - max_val) / sum_exp
                for (int i = 0; i < feature_dim; ++i) {
                    int input_idx = idx * feature_dim + i;
                    output[input_idx] = expf(input[input_idx] - max_val) / sum_exp;
                }
            }
        }

        __global__ void softmax_backward(const float* grad_output, const float* output, float* grad_input, int batch_size, int feature_dim) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < batch_size) {
                for (int i = 0; i < feature_dim; ++i) {
                    int current_idx = idx * feature_dim + i;
                    float grad_sum = 0.0f;

                    // Compute gradient for each element in the softmax
                    for (int j = 0; j < feature_dim; ++j) {
                        int j_idx = idx * feature_dim + j;
                        float kronecker_delta = (i == j) ? 1.0f : 0.0f;
                        float softmax_grad = output[current_idx] * (kronecker_delta - output[j_idx]);
                        grad_sum += grad_output[j_idx] * softmax_grad;
                    }

                    grad_input[current_idx] = grad_sum;
                }
            }
        }

        // Wrapper functions for launching kernels
        void launchReluForward(const float* input, float* output, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            relu_forward << <num_blocks, BLOCK_SIZE, 0, stream >> > (input, output, size);
        }

        void launchReluBackward(const float* grad_output, const float* input, float* grad_input, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            relu_backward << <num_blocks, BLOCK_SIZE, 0, stream >> > (grad_output, input, grad_input, size);
        }

        void launchSigmoidForward(const float* input, float* output, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            sigmoid_forward << <num_blocks, BLOCK_SIZE, 0, stream >> > (input, output, size);
        }

        void launchSigmoidBackward(const float* grad_output, const float* output, float* grad_input, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            sigmoid_backward << <num_blocks, BLOCK_SIZE, 0, stream >> > (grad_output, output, grad_input, size);
        }

        void launchTanhForward(const float* input, float* output, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            tanh_forward << <num_blocks, BLOCK_SIZE, 0, stream >> > (input, output, size);
        }

        void launchTanhBackward(const float* grad_output, const float* output, float* grad_input, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            tanh_backward << <num_blocks, BLOCK_SIZE, 0, stream >> > (grad_output, output, grad_input, size);
        }

        void launchLeakyReluForward(const float* input, float* output, float negative_slope, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            leaky_relu_forward << <num_blocks, BLOCK_SIZE, 0, stream >> > (input, output, negative_slope, size);
        }

        void launchLeakyReluBackward(const float* grad_output, const float* input, float* grad_input, float negative_slope, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            leaky_relu_backward << <num_blocks, BLOCK_SIZE, 0, stream >> > (grad_output, input, grad_input, negative_slope, size);
        }

        void launchEluForward(const float* input, float* output, float alpha, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            elu_forward << <num_blocks, BLOCK_SIZE, 0, stream >> > (input, output, alpha, size);
        }

        void launchEluBackward(const float* grad_output, const float* output, const float* input, float* grad_input, float alpha, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            elu_backward << <num_blocks, BLOCK_SIZE, 0, stream >> > (grad_output, output, input, grad_input, alpha, size);
        }

        void launchGeluForward(const float* input, float* output, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            gelu_forward << <num_blocks, BLOCK_SIZE, 0, stream >> > (input, output, size);
        }

        void launchGeluBackward(const float* grad_output, const float* input, float* grad_input, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            gelu_backward << <num_blocks, BLOCK_SIZE, 0, stream >> > (grad_output, input, grad_input, size);
        }

        void launchSoftmaxForward(const float* input, float* output, int batch_size, int feature_dim, cudaStream_t stream) {
            softmax_forward << <batch_size, 1, 0, stream >> > (input, output, batch_size, feature_dim);
        }

        void launchSoftmaxBackward(const float* grad_output, const float* output, float* grad_input, int batch_size, int feature_dim, cudaStream_t stream) {
            softmax_backward << <batch_size, 1, 0, stream >> > (grad_output, output, grad_input, batch_size, feature_dim);
        }

    }  // namespace activations
}  // namespace KernelThot
