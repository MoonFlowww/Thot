#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cmath>
#include "../../cuh/layers/rnn.cuh"
#include "../../cuh/layers/rnn_kernels.cuh"

namespace cuda::layers {





    void launchRNNForward(const float* input, const float* weights_ih, const float* weights_hh, const float* bias, const float* prev_hidden_state, float* hidden_state, float* output, int batch_size,
                          int seq_length, int input_size, int hidden_size, cudaStream_t stream) {
        int blockSize = 256;
        int numBlocks = (batch_size * hidden_size + blockSize - 1) / blockSize;

        rnn_forward<<<numBlocks, blockSize, 0, stream>>>(
            input, weights_ih, weights_hh, bias,
            prev_hidden_state, hidden_state, output,
            batch_size, seq_length, input_size, hidden_size);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch error in RNN: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();
        float host_first = 0.0f;
        cudaMemcpy(&host_first, output, sizeof(float), cudaMemcpyDeviceToHost);
        printf("rnn_forward output[0] = %f\n", host_first);
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
