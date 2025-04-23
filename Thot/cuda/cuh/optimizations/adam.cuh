#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cuda {
    namespace optimizations {
        __global__ void adam_update(float* weights, float* m, float* v, const float* gradients,
            float learning_rate, float beta1, float beta2, float epsilon,
            float correction1, float correction2, int size);

        void launchAdamUpdate(float* weights, float* m, float* v, const float* gradients,
            float learning_rate, float beta1, float beta2, float epsilon,
            float correction1, float correction2, int size,
            cudaStream_t stream = 0);
    }
}


