#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "../../cuh/losses/loss.cuh"

namespace cuda {
    namespace losses {


        __device__ bool verbose = false;


        // Mean Squared Error (MSE)
        __global__ void mse(const float* predictions, const float* targets, float* loss, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float diff = predictions[idx] - targets[idx];
                loss[idx] = 0.5f * diff * diff;
                if (verbose && loss[idx]!=0) printf("MSE loss[%d] = %f\n", idx, loss[idx]);
            }
        }

        __global__ void mseGradient(const float* predictions, const float* targets, float* gradients, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                gradients[idx] = predictions[idx] - targets[idx];
                if (verbose && gradients[idx]!=0) printf("MSE grad[%d] = %f\n", idx, gradients[idx]);
            }
        }

        // Mean Absolute Error (MAE)
        __global__ void mae(const float* predictions, const float* targets, float* loss, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                loss[idx] = fabsf(predictions[idx] - targets[idx]);
                if (verbose && loss[idx]!=0) printf("MAE loss[%d] = %f\n", idx, loss[idx]);
            }
        }

        __global__ void maeGradient(const float* predictions, const float* targets, float* gradients, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float diff = predictions[idx] - targets[idx];
                gradients[idx] = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
                if (verbose && gradients[idx]!=0) printf("MAE grad[%d] = %f\n", idx, gradients[idx]);
            }
        }

        // Binary Cross-Entropy
        __global__ void binaryCrossEntropy(const float* predictions, const float* targets, float* loss, int size, float epsilon) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float p = fmaxf(fminf(predictions[idx], 1.0f - epsilon), epsilon);
                float t = targets[idx];
                loss[idx] = -(t * logf(p) + (1.0f - t) * logf(1.0f - p));
                if (verbose && loss[idx]!=0) printf("BCE loss[%d] = %f\n", idx, loss[idx]);
            }
        }

        __global__ void binaryCrossEntropyGradient(const float* predictions, const float* targets, float* gradients, int size, float epsilon) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float p = fmaxf(fminf(predictions[idx], 1.0f - epsilon), epsilon);
                float t = targets[idx];
                gradients[idx] = -t / p + (1.0f - t) / (1.0f - p);
                if (verbose && gradients[idx]!=0) printf("BCE grad[%d] = %f\n", idx, gradients[idx]);
            }
        }


        __global__ void crossEntropy(const float* predictions, const float* targets, float* loss, int size, float epsilon) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float p = fmaxf(predictions[idx], epsilon);
                float t = targets[idx];
                loss[idx] = -t * logf(p);
                if (verbose && loss[idx]!=0) printf("CE loss[%d] = %f\n", idx, loss[idx]);
            }
        }

        __global__ void crossEntropyGradient(const float* predictions, const float* targets, float* gradients, int size, float epsilon) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float p = fmaxf(predictions[idx], epsilon);
                float t = targets[idx];
                gradients[idx] = -t / p;
                if (verbose && gradients[idx]!=0) printf("CE grad[%d] = %f\n", idx, gradients[idx]);
            }
        }



        // Categorical Cross-Entropy
        __global__ void categoricalCrossEntropy(const float* predictions,
                                                const float* targets,
                                                float* loss,
                                                int num_classes,
                                                float epsilon) {
            extern __shared__ float sdata[];
            int b = blockIdx.x;
            int tid = threadIdx.x;
            const float* p_row = predictions + b * num_classes;
            const float* t_row = targets + b * num_classes;

            float local = 0.0f;
            int vec_stride = blockDim.x * 4;
            int limit = num_classes & ~3;
            for (int i = tid * 4; i < limit; i += vec_stride) {
                float4 p = reinterpret_cast<const float4*>(p_row)[i / 4];
                float4 t = reinterpret_cast<const float4*>(t_row)[i / 4];
                local += -t.x * logf(fmaxf(p.x, epsilon))
                       + -t.y * logf(fmaxf(p.y, epsilon))
                       + -t.z * logf(fmaxf(p.z, epsilon))
                       + -t.w * logf(fmaxf(p.w, epsilon));
            }
            for (int i = limit + tid; i < num_classes; i += blockDim.x) {
                float p = fmaxf(p_row[i], epsilon);
                float t = t_row[i];
                local += -t * logf(p);
            }

            sdata[tid] = local;
            __syncthreads();
            for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
                if (tid < offset)
                    sdata[tid] += sdata[tid + offset];
                __syncthreads();
            }
            if (tid == 0) {
                loss[b] = sdata[0];
                if (verbose && loss[b] != 0) printf("CCE loss[%d] = %f\n", b, loss[b]);
            }
        }

        __global__ void categoricalCrossEntropyGradient(const float* predictions,
                                                        const float* targets,
                                                        float* gradients,
                                                        int num_classes,
                                                        float epsilon) {
            int b = blockIdx.x;
            int tid = threadIdx.x;
            const float* p_row = predictions + b * num_classes;
            const float* t_row = targets + b * num_classes;
            float* g_row = gradients + b * num_classes;

            int vec_stride = blockDim.x * 4;
            int limit = num_classes & ~3;
            for (int i = tid * 4; i < limit; i += vec_stride) {
                float4 p = reinterpret_cast<const float4*>(p_row)[i / 4];
                float4 t = reinterpret_cast<const float4*>(t_row)[i / 4];
                float4 g;
                g.x = -t.x / fmaxf(p.x, epsilon);
                g.y = -t.y / fmaxf(p.y, epsilon);
                g.z = -t.z / fmaxf(p.z, epsilon);
                g.w = -t.w / fmaxf(p.w, epsilon);
                reinterpret_cast<float4*>(g_row)[i / 4] = g;
            }
            for (int i = limit + tid; i < num_classes; i += blockDim.x) {
                float p = fmaxf(p_row[i], epsilon);
                float t = t_row[i];
                g_row[i] = -t / p;
            }
            if (verbose && tid == 0)
                for (int i = 0; i < num_classes; ++i)
                    if (g_row[i] != 0)
                        printf("CCE grad[%d] = %f\n", b * num_classes + i, g_row[i]);
        }


        // Sparse Categorical Cross-Entropy
        __global__ void sparseCategoricalCrossEntropy(const float* predictions, const float* targets, float* loss, int batch_size, int num_classes, float epsilon) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < batch_size) {
                int target_class = targets[idx];
                if (target_class >= 0 && target_class < num_classes) {
                    float p = fmaxf(predictions[idx * num_classes + target_class], epsilon);
                    loss[idx] = -logf(p);
                } else {
                    loss[idx] = 0.0f;
                }
                if (verbose && loss[idx]!=0) printf("Sparse CCE loss[%d] = %f\n", idx, loss[idx]);
            }
        }

        __global__ void sparseCategoricalCrossEntropyGradient(const float* predictions,
                                                              const float* targets,
                                                              float* gradients,
                                                              int num_classes,
                                                              float epsilon) {
            int b = blockIdx.x;
            int target = static_cast<int>(targets[b]);
            const float* p_row = predictions + b * num_classes;
            float* g_row = gradients + b * num_classes;
            int tid = threadIdx.x;
            for (int i = tid; i < num_classes; i += blockDim.x) {
                if (i == target)
                    g_row[i] = -1.0f / fmaxf(p_row[i], epsilon);
                else
                    g_row[i] = 0.0f;
            }
            if (verbose && tid == 0 && target >= 0 && target < num_classes)
                printf("Sparse CCE grad[%d] = %f\n", b * num_classes + target,
                       g_row[target]);
        }

        // Hinge
        __global__ void hinge(const float* predictions, const float* targets, float* loss, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float margin = 1.0f - predictions[idx] * targets[idx];
                loss[idx] = fmaxf(0.0f, margin);
                if (verbose && loss[idx]!=0) printf("Hinge loss[%d] = %f\n", idx, loss[idx]);
            }
        }

        __global__ void hingeGradient(const float* predictions, const float* targets, float* gradients, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float margin = 1.0f - predictions[idx] * targets[idx];
                gradients[idx] = (margin > 0.0f) ? -targets[idx] : 0.0f;
                if (verbose && gradients[idx]!=0) printf("Hinge grad[%d] = %f\n", idx, gradients[idx]);
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
                if (verbose && loss[idx]!=0) printf("Huber loss[%d] = %f\n", idx, loss[idx]);
            }
        }

        __global__ void huberGradient(const float* predictions, const float* targets, float* gradients, int size, float delta) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float diff = predictions[idx] - targets[idx];
                gradients[idx] = (fabsf(diff) <= delta) ?
                    diff :
                    delta * ((diff > 0.0f) ? 1.0f : -1.0f);
                if (verbose && gradients[idx]!=0) printf("Huber grad[%d] = %f\n", idx, gradients[idx]);
            }
        }

        // KL Divergence
        __global__ void klDivergence(const float* predictions, const float* targets, float* loss, int size, float epsilon) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float p = fmaxf(predictions[idx], epsilon);
                float q = fmaxf(targets[idx], epsilon);
                loss[idx] = q * logf(q / p);
                if (verbose && loss[idx]!=0) printf("KL loss[%d] = %f\n", idx, loss[idx]);
            }
        }

        __global__ void klDivergenceGradient(const float* predictions, const float* targets, float* gradients, int size, float epsilon) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float p = fmaxf(predictions[idx], epsilon);
                float q = fmaxf(targets[idx], epsilon);
                gradients[idx] = -q / p;
                if (verbose && gradients[idx]!=0) printf("KL grad[%d] = %f\n", idx, gradients[idx]);
            }
        }

        // Wrapper functions for launching 
        void launchMSE(const float* predictions, const float* targets, float* loss, int size, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            mse << <numBlocks, blockSize, 0, stream >> > (predictions, targets, loss, size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) printf("Kernel launch error in launchMSE: %s\n", cudaGetErrorString(err));
            cudaDeviceSynchronize();
        }

        void launchMSEGradient(const float* predictions, const float* targets, float* gradients, int size, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            mseGradient << <numBlocks, blockSize, 0, stream >> > (predictions, targets, gradients, size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) printf("Kernel launch error in launchMSEGradient: %s\n", cudaGetErrorString(err));
            cudaDeviceSynchronize();
        }

        void launchMAE(const float* predictions, const float* targets, float* loss, int size, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            mae << <numBlocks, blockSize, 0, stream >> > (predictions, targets, loss, size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) printf("Kernel launch error in launchMAE: %s\n", cudaGetErrorString(err));
            cudaDeviceSynchronize();
        }

        void launchMAEGradient(const float* predictions, const float* targets, float* gradients, int size, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            maeGradient << <numBlocks, blockSize, 0, stream >> > (predictions, targets, gradients, size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) printf("Kernel launch error in launchMAEGradient: %s\n", cudaGetErrorString(err));
            cudaDeviceSynchronize();
        }

        void launchBinaryCrossEntropy(const float* predictions, const float* targets, float* loss, int size, float epsilon, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            binaryCrossEntropy << <numBlocks, blockSize, 0, stream >> > (predictions, targets, loss, size, epsilon);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) printf("Kernel launch error in launchBinaryCrossEntropy: %s\n", cudaGetErrorString(err));
            cudaDeviceSynchronize();
        }

        void launchBinaryCrossEntropyGradient(const float* predictions, const float* targets, float* gradients, int size, float epsilon, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            binaryCrossEntropyGradient << <numBlocks, blockSize, 0, stream >> > (predictions, targets, gradients, size, epsilon);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) printf("Kernel launch error in launchBinaryCrossEntropyGradient: %s\n", cudaGetErrorString(err));
            cudaDeviceSynchronize();
        }

        void launchCrossEntropy(const float* predictions, const float* targets, float* loss, int size, float epsilon, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            crossEntropy<<<numBlocks, blockSize, 0, stream>>>(predictions, targets, loss, size, epsilon);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) printf("Kernel launch error in launchCrossEntropy: %s\n", cudaGetErrorString(err));
            cudaDeviceSynchronize();
        }

        void launchCrossEntropyGradient(const float* predictions, const float* targets, float* gradients, int size, float epsilon, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            crossEntropyGradient<<<numBlocks, blockSize, 0, stream>>>(predictions, targets, gradients, size, epsilon);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) printf("Kernel launch error in launchCrossEntropyGradient: %s\n", cudaGetErrorString(err));
            cudaDeviceSynchronize();
        }

        void launchCategoricalCrossEntropy(const float* predictions, const float* targets, float* loss, int batch_size, int num_classes, float epsilon, cudaStream_t stream) {
            int blockSize = 256;
            size_t shared = blockSize * sizeof(float);
            categoricalCrossEntropy<<<batch_size, blockSize, shared, stream>>>( predictions, targets, loss, num_classes, epsilon);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                printf("Kernel launch error in launchCategoricalCrossEntropy: %s\n", cudaGetErrorString(err));
            cudaDeviceSynchronize();
        }

        void launchCategoricalCrossEntropyGradient(const float* predictions, const float* targets, float* gradients, int batch_size, int num_classes, float epsilon, cudaStream_t stream) {
            int blockSize = 256;
            categoricalCrossEntropyGradient<<<batch_size, blockSize, 0, stream>>>(predictions, targets, gradients, num_classes, epsilon);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                printf("Kernel launch error in launchCategoricalCrossEntropyGradient: %s\n", cudaGetErrorString(err));
            cudaDeviceSynchronize();
        }

        void launchSparseCategoricalCrossEntropy(const float* predictions, const float* targets, float* loss, int batch_size, int num_classes, float epsilon, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (batch_size + blockSize - 1) / blockSize;
            sparseCategoricalCrossEntropy<<<numBlocks, blockSize, 0, stream>>>(predictions, targets, loss, batch_size, num_classes, epsilon);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                printf("Kernel launch error in launchSparseCategoricalCrossEntropy: %s\n", cudaGetErrorString(err));
            cudaDeviceSynchronize();
        }

        void launchSparseCategoricalCrossEntropyGradient(const float* predictions, const float* targets, float* gradients, int batch_size, int num_classes, float epsilon, cudaStream_t stream) {
            int blockSize = 256;
            sparseCategoricalCrossEntropyGradient<<<batch_size, blockSize, 0, stream>>>(predictions, targets, gradients, num_classes, epsilon);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                printf("Kernel launch error in launchSparseCategoricalCrossEntropyGradient: %s\n", cudaGetErrorString(err));
            cudaDeviceSynchronize();
        }

        void launchHinge(const float* predictions, const float* targets, float* loss, int size, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            hinge << <numBlocks, blockSize, 0, stream >> > (predictions, targets, loss, size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) printf("Kernel launch error in launchHinge: %s\n", cudaGetErrorString(err));
            cudaDeviceSynchronize();
        }

        void launchHingeGradient(const float* predictions, const float* targets, float* gradients, int size, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            hingeGradient << <numBlocks, blockSize, 0, stream >> > (predictions, targets, gradients, size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) printf("Kernel launch error in launchHingeGradient: %s\n", cudaGetErrorString(err));
            cudaDeviceSynchronize();
        }

        void launchHuber(const float* predictions, const float* targets, float* loss, int size, float delta, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            huber << <numBlocks, blockSize, 0, stream >> > (predictions, targets, loss, size, delta);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) printf("Kernel launch error in launchHuber: %s\n", cudaGetErrorString(err));
            cudaDeviceSynchronize();
        }

        void launchHuberGradient(const float* predictions, const float* targets, float* gradients, int size, float delta, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            huberGradient << <numBlocks, blockSize, 0, stream >> > (predictions, targets, gradients, size, delta);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) printf("Kernel launch error in launchHuberGradient: %s\n", cudaGetErrorString(err));
            cudaDeviceSynchronize();
        }

        void launchKLDivergence(const float* predictions, const float* targets, float* loss, int size, float epsilon, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            klDivergence << <numBlocks, blockSize, 0, stream >> > (predictions, targets, loss, size, epsilon);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) printf("Kernel launch error in launchKLDivergence: %s\n", cudaGetErrorString(err));
            cudaDeviceSynchronize();
        }

        void launchKLDivergenceGradient(const float* predictions, const float* targets, float* gradients, int size, float epsilon, cudaStream_t stream) {
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            klDivergenceGradient << <numBlocks, blockSize, 0, stream >> > (predictions, targets, gradients, size, epsilon);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) printf("Kernel launch error in launchKLDivergenceGradient: %s\n", cudaGetErrorString(err));
            cudaDeviceSynchronize();
        }


        float reduceLoss(float* loss, int size, cudaStream_t stream) {
            thrust::device_ptr<float> loss_ptr(loss);
            return thrust::reduce(thrust::cuda::par.on(stream), loss_ptr, loss_ptr + size, 0.0f);
        }


    }  // namespace losses

}  // namespace cuda 