#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cuda {
    namespace optimizations {
        __global__ void sgd_update(float* weights, const float* gradients, float learning_rate, int size);

        void launchSGDUpdate(float* weights, const float* gradients, float learning_rate, int size, cudaStream_t stream = 0);
    }
}
