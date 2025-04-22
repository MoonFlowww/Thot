#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cuda {
    namespace optimizations {
        __global__ void sgdm_update(float* weights, float* velocity, const float* gradients,
            float learning_rate, float momentum, int size);

        void launchSGDMUpdate(float* weights, float* velocity, const float* gradients, 
            float learning_rate, float momentum, int size,
            cudaStream_t stream = 0);
    }
}


