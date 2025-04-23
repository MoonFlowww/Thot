#include "../../cuh/optimizations/adam.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

namespace cuda{
    namespace optimizations {
        const int blockSize = 256;
        
        __global__ void adam_update(float* weights, float* m, float* v, const float* gradients, float learning_rate, float beta1, float beta2, float epsilon, float correction1, float correction2, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx < size) {
                // Update biased first moment estimate
                m[idx] = beta1 * m[idx] + (1.0f - beta1) * gradients[idx];

                // Update biased second raw moment estimate
                v[idx] = beta2 * v[idx] + (1.0f - beta2) * gradients[idx] * gradients[idx];

                // Compute bias-corrected learning rate
                float corrected_lr = learning_rate * sqrtf(correction2) / correction1;

                weights[idx] -= corrected_lr * m[idx] / (sqrtf(v[idx]) + epsilon);
            }
        }

        void launchAdamUpdate(float* weights, float* m, float* v, const float* gradients,
            float learning_rate, float beta1, float beta2, float epsilon,
            float correction1, float correction2, int size,
            cudaStream_t stream) {

            const int numBlocks = (size + blockSize - 1) / blockSize;

            adam_update<<<numBlocks, blockSize, 0, stream>>>(weights, m, v, gradients, learning_rate, beta1, beta2, epsilon, correction1, correction2, size);
        }
    }
}
