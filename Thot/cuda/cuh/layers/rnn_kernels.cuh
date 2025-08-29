//
// Created by moonfloww on 29/08/2025.
//

#ifndef THOT_RNN_KERNELS_CUH
#define THOT_RNN_KERNELS_CUH
#include <cuda_runtime.h>

namespace cuda::layers {

    __global__ void rnn_forward(const float* input, const float* weights_ih, const float* weights_hh,
                            const float* bias, const float* prev_hidden_state, float* hidden_state, float* output,
                            int batch_size, int seq_length, int input_size, int hidden_size);

    // Backward pass kernels
    __global__ void rnn_backward_input(const float* grad_output, const float* weights_ih,
                                       float* grad_input, int batch_size, int seq_length, int input_size, int hidden_size);

    __global__ void rnn_backward_hidden(const float* grad_output, const float* weights_hh,
                                        float* grad_hidden, int batch_size, int hidden_size);

    __global__ void rnn_backward_weights_ih(const float* input, const float* grad_hidden,
                                            float* grad_weights_ih, int batch_size, int seq_length, int input_size, int hidden_size);

    __global__ void rnn_backward_weights_hh(const float* hidden_state, const float* grad_hidden,
                                            float* grad_weights_hh, int batch_size, int hidden_size);

    __global__ void rnn_backward_bias(const float* grad_hidden, float* grad_bias,
                                      int batch_size, int hidden_size);


}
#endif //THOT_RNN_KERNELS_CUH