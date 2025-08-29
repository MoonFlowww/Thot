#pragma once

#include <cuda_runtime.h>

namespace cuda::layers {

    void launchRNNForward(const float* input, const float* weights_ih, const float* weights_hh,
                          const float* bias, const float* prev_hidden_state, float* hidden_state, float* output,
                          int batch_size, int seq_length, int input_size, int hidden_size, cudaStream_t stream = 0);

    void launchRNNBackwardInput(const float* grad_output, const float* weights_ih,
                                float* grad_input, int batch_size, int seq_length, int input_size, int hidden_size,
                                cudaStream_t stream = 0);

    void launchRNNBackwardHidden(const float* grad_output, const float* weights_hh,
                                 float* grad_hidden, int batch_size, int hidden_size,
                                 cudaStream_t stream = 0);

    void launchRNNBackwardWeightsIH(const float* input, const float* grad_hidden,
                                    float* grad_weights_ih, int batch_size, int seq_length, int input_size, int hidden_size,
                                    cudaStream_t stream = 0);

    void launchRNNBackwardWeightsHH(const float* hidden_state, const float* grad_hidden,
                                    float* grad_weights_hh, int batch_size, int hidden_size,
                                    cudaStream_t stream = 0);

    void launchRNNBackwardBias(const float* grad_hidden, float* grad_bias,
                               int batch_size, int hidden_size,
                               cudaStream_t stream = 0);
}
