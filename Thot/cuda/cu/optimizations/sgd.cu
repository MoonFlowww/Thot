#pragma once
#include "../../cuh/optimizations/sgd.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cuda {
    namespace optimizations {

        __global__ void sgd_update(float* weights, const float* gradients, float learning_rate, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                weights[idx] -= learning_rate * gradients[idx];
            }
        }

        void launchSGDUpdate(float* weights, const float* gradients, float learning_rate, int size, cudaStream_t stream) {
            const int blockSize = 256;
            const int numBlocks = (size + blockSize - 1) / blockSize;
            sgd_update<<<numBlocks, blockSize, 0, stream>>>(weights, gradients, learning_rate, size);
        }

    } // namespace optimizations
} // namespace cuda