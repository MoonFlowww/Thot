#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <random>
#include <cstdio>
#include <cmath>

#include "../../cuh/initializations/random.cuh"
#ifdef THOT_CUDA_DEBUG_SYNC
#define CUDA_DEBUG_SYNC() cudaDeviceSynchronize()
#else
#define CUDA_DEBUG_SYNC() ((void)0)
#endif


namespace cuda {
    const int BLOCK_SIZE = 256;
    namespace initializations {

        __global__ void random_uniform(float* data, int size, float low, float high, unsigned long long seed) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                curandState state;
                curand_init(seed, idx, 0, &state);
                float r = curand_uniform(&state); // (0,1]
                data[idx] = low + (high - low) * r;
            }
        }

        __global__ void random_normal(float* data, int size, float mean, float stddev, unsigned long long seed) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                curandState state;
                curand_init(seed, idx, 0, &state);
                float r = curand_normal(&state);
                data[idx] = mean + stddev * r;
            }
        }

        __global__ void random_truncated_normal(float* data, int size, float mean, float stddev, unsigned long long seed) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                curandState state;
                curand_init(seed, idx, 0, &state);
                float r;
                do {
                    r = curand_normal(&state);
                } while (fabsf(r) > 2.0f);
                data[idx] = mean + stddev * r;
            }
        }

        __global__ void fill_constant(float* data, int size, float value) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                data[idx] = value;
            }
        }

        __global__ void dirac_init(float* data, int rows, int cols) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int size = rows * cols;
            if (idx < size) {
                int r = idx / cols;
                int c = idx % cols;
                data[idx] = (r == c) ? 1.0f : 0.0f;
            }
        }

        __global__ void scale_values(float* data, int size, float scale) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                data[idx] *= scale;
            }
        }

        void launchRandomUniform(float* data, int size, float low, float high, cudaStream_t stream) {
            unsigned long long seed = std::random_device{}();
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            random_uniform<<<num_blocks, BLOCK_SIZE, 0, stream>>>(data, size, low, high, seed);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchRandomUniform: %s\n", cudaGetErrorString(err));
            }
            CUDA_DEBUG_SYNC();
        }

        void launchRandomNormal(float* data, int size, float mean, float stddev, cudaStream_t stream) {
            unsigned long long seed = std::random_device{}();
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            random_normal<<<num_blocks, BLOCK_SIZE, 0, stream>>>(data, size, mean, stddev, seed);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchRandomNormal: %s\n", cudaGetErrorString(err));
            }
            CUDA_DEBUG_SYNC();
        }

        void launchRandomTruncatedNormal(float* data, int size, float mean, float stddev, cudaStream_t stream) {
            unsigned long long seed = std::random_device{}();
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            random_truncated_normal<<<num_blocks, BLOCK_SIZE, 0, stream>>>(data, size, mean, stddev, seed);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchRandomTruncatedNormal: %s\n", cudaGetErrorString(err));
            }
            CUDA_DEBUG_SYNC();
        }

        void launchFill(float* data, int size, float value, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            fill_constant<<<num_blocks, BLOCK_SIZE, 0, stream>>>(data, size, value);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchFill: %s\n", cudaGetErrorString(err));
            }
            CUDA_DEBUG_SYNC();
        }

        void launchDirac(float* data, int rows, int cols, cudaStream_t stream) {
            int size = rows * cols;
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            dirac_init<<<num_blocks, BLOCK_SIZE, 0, stream>>>(data, rows, cols);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchDirac: %s\n", cudaGetErrorString(err));
            }
            CUDA_DEBUG_SYNC();
        }

        void launchScale(float* data, int size, float scale, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            scale_values<<<num_blocks, BLOCK_SIZE, 0, stream>>>(data, size, scale);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchScale: %s\n", cudaGetErrorString(err));
            }
            CUDA_DEBUG_SYNC();
        }

        void launchLyapunov(float* data, int rows, int cols, cudaStream_t stream) {
            int size = rows * cols;
            // Step 1: random normal init
            launchRandomNormal(data, size, 0.0f, 1.0f, stream);

            if (rows <= 0 || cols <= 0) return;

            cublasHandle_t handle;
            cublasCreate(&handle);
            cublasSetStream(handle, stream);

            float* v; float* v_new;
            cudaMalloc(&v, cols * sizeof(float));
            cudaMalloc(&v_new, rows * sizeof(float));
            launchFill(v, cols, 1.0f, stream);

            float alpha = 1.0f; float beta = 0.0f;
            for (int i = 0; i < 20; ++i) {
                cublasSgemv(handle, CUBLAS_OP_N, rows, cols, &alpha, data, rows, v, 1, &beta, v_new, 1);
                float norm;
                cublasSnrm2(handle, rows, v_new, 1, &norm);
                float inv = (norm > 0.0f) ? 1.0f / norm : 1.0f;
                cublasSscal(handle, rows, &inv, v_new, 1);
                std::swap(v, v_new);
            }

            cublasSgemv(handle, CUBLAS_OP_N, rows, cols, &alpha, data, rows, v, 1, &beta, v_new, 1);
            float spectral = 0.0f;
            cublasSdot(handle, rows, v, 1, v_new, 1, &spectral);
            spectral = fabsf(spectral);
            if (spectral > 0.0f) {
                float scale = 0.95f / spectral;
                launchScale(data, size, scale, stream);
            }

            cudaFree(v);
            cudaFree(v_new);
            cublasDestroy(handle);
            CUDA_DEBUG_SYNC();
        }

    } // namespace initializations
} // namespace cuda