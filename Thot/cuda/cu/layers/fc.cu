#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cmath>
#include "../../cuh/layers/fc.cuh"

namespace cuda {
    namespace layers {

        __global__ void fc_forward(const float* input, const float* weights, const float* bias,
            float* output, int batch_size, int input_size, int output_size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * output_size) return;

            int batch_idx = idx / output_size;
            int output_idx = idx % output_size;

            float sum = 0.0f;
            for (int i = 0; i < input_size; ++i) {
                sum += input[batch_idx * input_size + i] * weights[i * output_size + output_idx];
            }
            if (bias != nullptr) sum += bias[output_idx];
            output[idx] = sum;
        }

        __global__ void fc_backward_input(const float* grad_output, const float* weights,
            float* grad_input, int batch_size, int input_size, int output_size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * input_size) return;

            int batch_idx = idx / input_size;
            int input_idx = idx % input_size;

            float sum = 0.0f;
            for (int o = 0; o < output_size; ++o) {
                sum += grad_output[batch_idx * output_size + o] * weights[input_idx * output_size + o];
            }
            grad_input[idx] = sum;
        }

        __global__ void fc_backward_weights(const float* input, const float* grad_output,
            float* grad_weights, int batch_size, int input_size, int output_size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= input_size * output_size) return;

            int input_idx = idx / output_size;
            int output_idx = idx % output_size;

            float sum = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                sum += input[b * input_size + input_idx] * grad_output[b * output_size + output_idx];
            }
            grad_weights[idx] = sum;
        }

        __global__ void fc_backward_bias(const float* grad_output, float* grad_bias,
            int batch_size, int output_size) {
            int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (output_idx >= output_size) return;

            float sum = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                sum += grad_output[b * output_size + output_idx];
            }
            grad_bias[output_idx] = sum;
        }



        //
        // Launch functions
        //

        void launchFCForward(const float* input, const float* weights, const float* bias,
            float* output, int batch_size, int input_size, int output_size,
            cudaStream_t stream) {
            const int blockSize = 256;
            const int numBlocks = (batch_size * output_size + blockSize - 1) / blockSize;


            fc_forward << <numBlocks, blockSize, 0, stream >> > (input, weights, bias, output, batch_size, input_size, output_size);


        }

        void launchFCBackwardInput(const float* grad_output, const float* weights,
            float* grad_input, int batch_size, int input_size, int output_size,
            cudaStream_t stream) {
            const int blockSize = 256;
            const int numBlocks = (batch_size * input_size + blockSize - 1) / blockSize;


            fc_backward_input << <numBlocks, blockSize, 0, stream >> > (grad_output, weights, grad_input, batch_size, input_size, output_size);

        }

        void launchFCBackwardWeights(const float* input, const float* grad_output,
            float* grad_weights, int batch_size, int input_size, int output_size,
            cudaStream_t stream) {
            const int blockSize = 256;
            const int numBlocks = (input_size * output_size + blockSize - 1) / blockSize;


            fc_backward_weights << <numBlocks, blockSize, 0, stream >> > (input, grad_output, grad_weights, batch_size, input_size, output_size);

        }

        void launchFCBackwardBias(const float* grad_output, float* grad_bias, int batch_size, int output_size, cudaStream_t stream) {
            const int blockSize = 256;
            const int numBlocks = (output_size + blockSize - 1) / blockSize;

            fc_backward_bias << <numBlocks, blockSize, 0, stream >> > (grad_output, grad_bias, batch_size, output_size);

        }





    }
}