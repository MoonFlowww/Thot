#include "../../cuh/optimizations/sgdm.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cuda {
    namespace optimizations {
        const int blockSize = 256;

        __global__ void sgdm_update(float* weights, float* velocity, const float* gradients,
            float learning_rate, float momentum, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx < size) {
                // Update velocity: v = momentum * v - learning_rate * gradient
                velocity[idx] = momentum * velocity[idx] - learning_rate * gradients[idx];

                // Update weights: w = w + v
                weights[idx] += velocity[idx];
            }
        }

        void launchSGDMUpdate(float* weights, float* velocity, const float* gradients,
            float learning_rate, float momentum, int size,
            cudaStream_t stream) {
            const int numBlocks = (size + blockSize - 1) / blockSize;

            // Launch kernel
            sgdm_update<<<numBlocks, blockSize, 0, stream>>>(
                weights, velocity, gradients, learning_rate, momentum, size);
        }
    }
}
