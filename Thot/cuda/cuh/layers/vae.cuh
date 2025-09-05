#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace cuda {
    namespace layers {

        __global__ void vae_sample_kernel(const float* mean, const float* logvar, float* z, float* eps, int size, unsigned long long seed);
        __global__ void vae_backward_kernel(const float* grad_z, const float* eps, const float* logvar, float* grad_mean, float* grad_logvar, int size);

        void launchVAESample(const float* mean, const float* logvar, float* z, float* eps, int size, cudaStream_t stream = 0);

        void launchVAESampleBackward(const float* grad_z, const float* eps, const float* logvar, float* grad_mean, float* grad_logvar, int size, cudaStream_t stream = 0);

    } // namespace layers
} // namespace cuda