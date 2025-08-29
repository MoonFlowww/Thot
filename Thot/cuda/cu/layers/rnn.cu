#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cmath>
#include "../../cuh/layers/rnn.cuh"

namespace cuda::layers {

    __global__ void rnn_forward(const float* input, const float* weights_ih, const float* weights_hh, const float* bias, const float* prev_hidden_state,
                                float* hidden_state, float* output, int batch_size, int seq_length, int input_size, int hidden_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= batch_size * hidden_size) return;

        int batch_idx  = idx / hidden_size;
        int hidden_idx = idx % hidden_size;

        float sum = 0.0f;

        // input contribution
        for (int i = 0; i < input_size; ++i) {
            sum += input[batch_idx * input_size + i] *
                    weights_ih[hidden_idx * input_size + i];
        }

        // recurrent contribution (fix: include batch_idx offset)
        const float* prev_row = prev_hidden_state + batch_idx * hidden_size;
        const float* w_hh_row = weights_hh + hidden_idx * hidden_size;
        for (int h = 0; h < hidden_size; ++h) {
            sum += prev_row[h] * w_hh_row[h];
        }

        if (bias != nullptr) sum += bias[hidden_idx];

        float activated = tanhf(sum);

        // fix: include batch offset when writing state
        hidden_state[batch_idx * hidden_size + hidden_idx] = activated;
        output[batch_idx * hidden_size + hidden_idx] = activated;

    }




    // Backward pass kernels
    __global__ void rnn_backward_input(const float* grad_output, const float* weights_ih, float* grad_input, int batch_size, int seq_length, int input_size, int hidden_size) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= batch_size * input_size) return;

        int batch_idx = idx / input_size;
        int input_idx = idx % input_size;

        float sum = 0.0f;
        for (int h = 0; h < hidden_size; ++h) {
            sum += grad_output[batch_idx * hidden_size + h] * weights_ih[h * input_size + input_idx];
        }

        grad_input[idx] = sum;
    }

    __global__ void rnn_backward_hidden(const float* grad_output, const float* weights_hh, float* grad_hidden, int batch_size, int hidden_size) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= batch_size * hidden_size) return;

        int batch_idx = idx / hidden_size;
        int hidden_idx = idx % hidden_size;

        float sum = 0.0f;
        for (int h = 0; h < hidden_size; ++h) {
            sum += grad_output[batch_idx * hidden_size + h] * weights_hh[h * hidden_size + hidden_idx];
        }

        grad_hidden[idx] = sum;
    }

    __global__ void rnn_backward_weights_ih(const float* input, const float* grad_hidden, float* grad_weights_ih, int batch_size, int seq_length, int input_size, int hidden_size) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= hidden_size * input_size) return;

        int hidden_idx = idx / input_size;
        int input_idx = idx % input_size;

        float sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            sum += input[b * input_size + input_idx] * grad_hidden[b * hidden_size + hidden_idx];
        }

        grad_weights_ih[idx] += sum;  // Accumulate gradient
    }

    __global__ void rnn_backward_weights_hh(const float* hidden_state, const float* grad_hidden, float* grad_weights_hh, int batch_size, int hidden_size) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= hidden_size * hidden_size) return;

        int h1_idx = idx / hidden_size;
        int h2_idx = idx % hidden_size;

        float sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            sum += hidden_state[b * hidden_size + h2_idx] * grad_hidden[b * hidden_size + h1_idx];
        }

        grad_weights_hh[idx] += sum;  // Accumulate gradient
    }

    __global__ void rnn_backward_bias(const float* grad_hidden, float* grad_bias, int batch_size, int hidden_size) {

        int hidden_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (hidden_idx >= hidden_size) return;

        float sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            sum += grad_hidden[b * hidden_size + hidden_idx];
        }

        grad_bias[hidden_idx] += sum;  // Accumulate gradient
    }




    void launchRNNForward(const float* input, const float* weights_ih, const float* weights_hh, const float* bias, const float* prev_hidden_state, float* hidden_state, float* output, int batch_size,
                          int seq_length, int input_size, int hidden_size, cudaStream_t stream) {
        int blockSize = 256;
        int numBlocks = (batch_size * hidden_size + blockSize - 1) / blockSize;

        rnn_forward<<<numBlocks, blockSize, 0, stream>>>(
            input, weights_ih, weights_hh, bias,
            prev_hidden_state, hidden_state, output,
            batch_size, seq_length, input_size, hidden_size);

        //cudaError_t err = cudaGetLastError();
        //if (err != cudaSuccess) printf("Kernel launch error in RNN: %s\n", cudaGetErrorString(err));
        cudaDeviceSynchronize();
        float host_first = 0.0f;
        cudaMemcpy(&host_first, output, sizeof(float), cudaMemcpyDeviceToHost);
        //printf("rnn_forward output[0] = %f\n", host_first);
    }




    void launchRNNBackwardInput(const float* grad_output, const float* weights_ih,
                                float* grad_input, int batch_size, int seq_length, int input_size, int hidden_size,
                                cudaStream_t stream) {

        const int blockSize = 256;
        const int numBlocks = (batch_size * input_size + blockSize - 1) / blockSize;

        rnn_backward_input << <numBlocks, blockSize, 0, stream >> > (
            grad_output, weights_ih, grad_input,
            batch_size, seq_length, input_size, hidden_size
        );
    }

    void launchRNNBackwardHidden(const float* grad_output, const float* weights_hh,
                                 float* grad_hidden, int batch_size, int hidden_size,
                                 cudaStream_t stream) {

        const int blockSize = 256;
        const int numBlocks = (batch_size * hidden_size + blockSize - 1) / blockSize;

        rnn_backward_hidden << <numBlocks, blockSize, 0, stream >> > (
            grad_output, weights_hh, grad_hidden,
            batch_size, hidden_size
        );
    }

    void launchRNNBackwardWeightsIH(const float* input, const float* grad_hidden,
                                    float* grad_weights_ih, int batch_size, int seq_length, int input_size, int hidden_size,
                                    cudaStream_t stream) {

        const int blockSize = 256;
        const int numBlocks = (hidden_size * input_size + blockSize - 1) / blockSize;

        rnn_backward_weights_ih << <numBlocks, blockSize, 0, stream >> > (
            input, grad_hidden, grad_weights_ih,
            batch_size, seq_length, input_size, hidden_size
        );
    }

    void launchRNNBackwardWeightsHH(const float* hidden_state, const float* grad_hidden,
                                    float* grad_weights_hh, int batch_size, int hidden_size,
                                    cudaStream_t stream) {

        const int blockSize = 256;
        const int numBlocks = (hidden_size * hidden_size + blockSize - 1) / blockSize;

        rnn_backward_weights_hh << <numBlocks, blockSize, 0, stream >> > (
            hidden_state, grad_hidden, grad_weights_hh,
            batch_size, hidden_size
        );
    }

    void launchRNNBackwardBias(const float* grad_hidden, float* grad_bias,
                               int batch_size, int hidden_size,
                               cudaStream_t stream) {

        const int blockSize = 256;
        const int numBlocks = (hidden_size + blockSize - 1) / blockSize;

        rnn_backward_bias << <numBlocks, blockSize, 0, stream >> > (
            grad_hidden, grad_bias,
            batch_size, hidden_size
        );
    }
}
