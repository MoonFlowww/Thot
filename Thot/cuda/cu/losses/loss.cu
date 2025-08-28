#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../../cuh/losses/loss.cuh"

namespace cuda {
    namespace losses {



        // Mean Squared Error (MSE)
        __global__ void mse(const float* predictions, const float* targets, float* loss, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float diff = predictions[idx] - targets[idx];
                loss[idx] = 0.5f * diff * diff;
            }
        }

        __global__ void mseGradient(const float* predictions, const float* targets, float* gradients, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                gradients[idx] = predictions[idx] - targets[idx];
            }
        }

        // Mean Absolute Error (MAE)
        __global__ void mae(const float* predictions, const float* targets, float* loss, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                loss[idx] = fabsf(predictions[idx] - targets[idx]);
            }
        }
        __global__ void maeGradient(const float* predictions, const float* targets, float* gradients, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float diff = predictions[idx] - targets[idx];
                gradients[idx] = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
            }
        }

        // Binary Cross-Entropy
        __global__ void binaryCrossEntropy(const float* predictions, const float* targets, float* loss, int size, float epsilon) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float p = fmaxf(fminf(predictions[idx], 1.0f - epsilon), epsilon);
                float t = targets[idx];
                loss[idx] = -(t * logf(p) + (1.0f - t) * logf(1.0f - p));
            }
        }

        __global__ void binaryCrossEntropyGradient(const float* predictions, const float* targets, float* gradients, int size, float epsilon) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float p = fmaxf(fminf(predictions[idx], 1.0f - epsilon), epsilon);
                float t = targets[idx];
                gradients[idx] = -t / p + (1.0f - t) / (1.0f - p);
            }
        }

        // Categorical Cross-Entropy
        __global__ void categoricalCrossEntropy(const float* predictions, const float* targets, float* loss, int batch_size, int num_classes, float epsilon) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < batch_size) {
                loss[idx] = 0.0f;
                for (int c = 0; c < num_classes; ++c) {
                    int i = idx * num_classes + c;
                    float p = fmaxf(predictions[i], epsilon);
                    float t = targets[i];
                    loss[idx] -= t * logf(p);
                }
            }
        }

        __global__ void categoricalCrossEntropyGradient(const float* predictions, const float* targets, float* gradients, int batch_size, int num_classes, float epsilon) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < batch_size * num_classes) {
                int b = idx / num_classes;
                int c = idx % num_classes;
                int i = b * num_classes + c;
                float p = fmaxf(predictions[i], epsilon);
                float t = targets[i];
                gradients[i] = -t / p;
            }
        }

        // Sparse Categorical Cross-Entropy
        __global__ void sparseCategoricalCrossEntropy(const float* predictions, const float* targets, float* loss, int batch_size, int num_classes, float epsilon) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < batch_size) {
                int target_class = targets[idx];
                if (target_class >= 0 && target_class < num_classes) {
                    float p = fmaxf(predictions[idx * num_classes + target_class], epsilon);
                    loss[idx] = -logf(p);
                }
                else {
                    loss[idx] = 0.0f;
                }
            }
        }

        __global__ void sparseCategoricalCrossEntropyGradient(const float* predictions, const float* targets, float* gradients, int batch_size, int num_classes, float epsilon) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < batch_size * num_classes) {
                int b = idx / num_classes;
                int c = idx % num_classes;
                int target_class = targets[b];

                if (c == target_class) {
                    float p = fmaxf(predictions[idx], epsilon);
                    gradients[idx] = -1.0f / p;
                }
                else {
                    gradients[idx] = 0.0f;
                }
            }
        }

        // Hinge
        __global__ void hinge(const float* predictions, const float* targets, float* loss, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float margin = 1.0f - predictions[idx] * targets[idx];
                loss[idx] = fmaxf(0.0f, margin);
            }
        }

        __global__ void hingeGradient(const float* predictions, const float* targets, float* gradients, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float margin = 1.0f - predictions[idx] * targets[idx];
                gradients[idx] = (margin > 0.0f) ? -targets[idx] : 0.0f;
            }
        }

        // Huber
        __global__ void huber(const float* predictions, const float* targets, float* loss, int size, float delta) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float diff = fabsf(predictions[idx] - targets[idx]);
                loss[idx] = (diff <= delta) ?
                    0.5f * diff * diff :
                    delta * (diff - 0.5f * delta);
            }
        }

        __global__ void huberGradient(const float* predictions, const float* targets, float* gradients, int size, float delta) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float diff = predictions[idx] - targets[idx];
                gradients[idx] = (fabsf(diff) <= delta) ?
                    diff :
                    delta * ((diff > 0.0f) ? 1.0f : -1.0f);
            }
        }

        // KL Divergence
        __global__ void klDivergence(const float* predictions, const float* targets, float* loss, int size, float epsilon) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float p = fmaxf(predictions[idx], epsilon);
                float q = fmaxf(targets[idx], epsilon);
                loss[idx] = q * logf(q / p);
            }
        }

        __global__ void klDivergenceGradient(const float* predictions, const float* targets, float* gradients, int size, float epsilon) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float p = fmaxf(predictions[idx], epsilon);
                float q = fmaxf(targets[idx], epsilon);
                gradients[idx] = -q / p;
            }
        }

        // Wrapper functions for launching 
        void launchMSE(const float* predictions, const float* targets, float* loss, int size, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            mse << <numBlocks, blockSize, 0, stream >> > (predictions, targets, loss, size);
        }

        void launchMSEGradient(const float* predictions, const float* targets, float* gradients, int size, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            mseGradient << <numBlocks, blockSize, 0, stream >> > (predictions, targets, gradients, size);
        }

        void launchMAE(const float* predictions, const float* targets, float* loss, int size, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            mae << <numBlocks, blockSize, 0, stream >> > (predictions, targets, loss, size);
        }

        void launchMAEGradient(const float* predictions, const float* targets, float* gradients, int size, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            maeGradient << <numBlocks, blockSize, 0, stream >> > (predictions, targets, gradients, size);
        }

        void launchBinaryCrossEntropy(const float* predictions, const float* targets, float* loss, int size, float epsilon, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            binaryCrossEntropy << <numBlocks, blockSize, 0, stream >> > (predictions, targets, loss, size, epsilon);
        }

        void launchBinaryCrossEntropyGradient(const float* predictions, const float* targets, float* gradients, int size, float epsilon, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            binaryCrossEntropyGradient << <numBlocks, blockSize, 0, stream >> > (predictions, targets, gradients, size, epsilon);
        }

        void launchCategoricalCrossEntropy(const float* predictions, const float* targets, float* loss, int batch_size, int num_classes, float epsilon, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (batch_size + blockSize - 1) / blockSize;
            categoricalCrossEntropy << <numBlocks, blockSize, 0, stream >> > (predictions, targets, loss, batch_size, num_classes, epsilon);
        }

        void launchCategoricalCrossEntropyGradient(const float* predictions, const float* targets, float* gradients, int batch_size, int num_classes, float epsilon, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (batch_size * num_classes + blockSize - 1) / blockSize;
            categoricalCrossEntropyGradient << <numBlocks, blockSize, 0, stream >> > (predictions, targets, gradients, batch_size, num_classes, epsilon);
        }

        void launchSparseCategoricalCrossEntropy(const float* predictions, const float* targets, float* loss, int batch_size, int num_classes, float epsilon, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (batch_size + blockSize - 1) / blockSize;
            sparseCategoricalCrossEntropy << <numBlocks, blockSize, 0, stream >> > (predictions, targets, loss, batch_size, num_classes, epsilon);
        }

        void launchSparseCategoricalCrossEntropyGradient(const float* predictions, const float* targets, float* gradients, int batch_size, int num_classes, float epsilon, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (batch_size * num_classes + blockSize - 1) / blockSize;
            sparseCategoricalCrossEntropyGradient << <numBlocks, blockSize, 0, stream >> > (predictions, targets, gradients, batch_size, num_classes, epsilon);
        }

        void launchHinge(const float* predictions, const float* targets, float* loss, int size, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            hinge << <numBlocks, blockSize, 0, stream >> > (predictions, targets, loss, size);
        }

        void launchHingeGradient(const float* predictions, const float* targets, float* gradients, int size, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            hingeGradient << <numBlocks, blockSize, 0, stream >> > (predictions, targets, gradients, size);
        }

        void launchHuber(const float* predictions, const float* targets, float* loss, int size, float delta, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            huber << <numBlocks, blockSize, 0, stream >> > (predictions, targets, loss, size, delta);
        }

        void launchHuberGradient(const float* predictions, const float* targets, float* gradients, int size, float delta, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            huberGradient << <numBlocks, blockSize, 0, stream >> > (predictions, targets, gradients, size, delta);
        }

        void launchKLDivergence(const float* predictions, const float* targets, float* loss, int size, float epsilon, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            klDivergence << <numBlocks, blockSize, 0, stream >> > (predictions, targets, loss, size, epsilon);
        }

        void launchKLDivergenceGradient(const float* predictions, const float* targets, float* gradients, int size, float epsilon, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            klDivergenceGradient << <numBlocks, blockSize, 0, stream >> > (predictions, targets, gradients, size, epsilon);
        }



    }  // namespace losses

}  // namespace cuda 