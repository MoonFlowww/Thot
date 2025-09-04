#pragma once

#include <cuda_runtime.h>

namespace cuda {
    namespace layers {
        __global__ void spike_forward(const float* input, float* membrane,
                                      float* output, float threshold, int total);

        __global__ void spike_backward(const float* spikes, const float* grad_output,
                                       float* grad_input, int total);

        void launchSpikeForward(const float* input, float* membrane, float* output,
                                int batch_size, int neurons, float threshold,
                                cudaStream_t stream = 0);

        void launchSpikeBackward(const float* spikes, const float* grad_output,
                                 float* grad_input, int batch_size, int neurons,
                                 cudaStream_t stream = 0);
    } // namespace layers
} // namespace cuda