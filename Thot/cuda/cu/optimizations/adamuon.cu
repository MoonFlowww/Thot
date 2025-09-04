#include "../../cuh/optimizations/adamuon.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

namespace cuda {
    namespace optimizations {

        __global__ void adamuon_update(float* weights, const float* gradients,
                                       float learning_rate, float beta1, float beta2,
                                       float weight_decay, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float g = gradients[idx];
                float wd = 1.0f - learning_rate * weight_decay;
                // simplified adaptive update: scale by sqrt(|grad|)
                float scaled = g / (sqrtf(fabsf(g)) + 1e-8f);
                weights[idx] = wd * weights[idx] - learning_rate * (beta1 * scaled + (1 - beta1) * scaled);
            }
        }

        void launchAdaMuonUpdate(float* weights, const float* gradients,
                                 float learning_rate, float beta1, float beta2,
                                 float weight_decay, int size, cudaStream_t stream) {
            const int blockSize = 256;
            const int numBlocks = (size + blockSize - 1) / blockSize;
            adamuon_update<<<numBlocks, blockSize, 0, stream>>>(weights, gradients,
                                                                learning_rate, beta1, beta2,
                                                                weight_decay, size);
        }

    } // namespace optimizations
} // namespace cuda