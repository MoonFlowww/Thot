#include "../../cuh/layers/vae.cuh"

#include <curand_kernel.h>
#include <curand_normal.h>

namespace cuda {
    namespace layers {

        __global__ void vae_sample_kernel(const float* mean, const float* logvar, float* z, float* eps, int size, unsigned long long seed) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return;
            curandState state;
            curand_init(seed, idx, 0, &state);
            float e = curand_normal(&state);
            eps[idx] = e;
            float std = expf(0.5f * logvar[idx]);
            z[idx] = mean[idx] + std * e;
        }

        __global__ void vae_backward_kernel(const float* grad_z, const float* eps, const float* logvar, float* grad_mean, float* grad_logvar, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return;
            grad_mean[idx] = grad_z[idx];
            float std = expf(0.5f * logvar[idx]);
            grad_logvar[idx] = grad_z[idx] * 0.5f * std * eps[idx];
        }
        void launchVAESample(const float* mean, const float* logvar, float* z, float* eps, int size, cudaStream_t stream) {
            int block = 256;
            int grid = (size + block - 1) / block;
            unsigned long long seed = 1234ULL;
            vae_sample_kernel<<<grid, block, 0, stream>>>(mean, logvar, z, eps, size, seed);
        }

        void launchVAESampleBackward(const float* grad_z, const float* eps, const float* logvar, float* grad_mean, float* grad_logvar, int size, cudaStream_t stream) {
            int block = 256;
            int grid = (size + block - 1) / block;
            vae_backward_kernel<<<grid, block, 0, stream>>>(grad_z, eps, logvar, grad_mean, grad_logvar, size);
        }

    } // namespace layers
} // namespace cuda