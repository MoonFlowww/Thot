#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../../cuh/LowRankCuda/lowrank.cuh"

namespace cuda {
    namespace low_rank {

        // Elementwise addition kernel: C = A + B
        __global__ void add_kernel(const float* A, const float* B, float* C, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                C[idx] = A[idx] + B[idx];
            }
        }



        // Elementwise multiplication kernel: C = A * B
        __global__ void multiply_kernel(const float* A, const float* B, float* C, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                C[idx] = A[idx] * B[idx];
            }
        }



        // Scalar addition kernel: B = A + scalar
        __global__ void add_scalar_kernel(const float* A, float scalar, float* B, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                B[idx] = A[idx] + scalar;
            }
        }


        // Scalar multiplication kernel: B = A * scalar
        __global__ void multiply_scalar_kernel(const float* A, float scalar, float* B, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                B[idx] = A[idx] * scalar;
            }
        }



        void launchAdd(const float* A, const float* B, float* C, int size, cudaStream_t stream) {
            const int blockSize = 256;
            const int numBlocks = (size + blockSize - 1) / blockSize;
            add_kernel << <numBlocks, blockSize, 0, stream >> > (A, B, C, size);
        }



        void launchMultiply(const float* A, const float* B, float* C, int size, cudaStream_t stream) {
            const int blockSize = 256;
            const int numBlocks = (size + blockSize - 1) / blockSize;
            multiply_kernel << <numBlocks, blockSize, 0, stream >> > (A, B, C, size);
        }



        void launchAddScalar(const float* A, float scalar, float* B, int size, cudaStream_t stream) {
            const int blockSize = 256;
            const int numBlocks = (size + blockSize - 1) / blockSize;
            add_scalar_kernel << <numBlocks, blockSize, 0, stream >> > (A, scalar, B, size);
        }


        void launchMultiplyScalar(const float* A, float scalar, float* B, int size, cudaStream_t stream) {
            const int blockSize = 256;
            const int numBlocks = (size + blockSize - 1) / blockSize;
            multiply_scalar_kernel << <numBlocks, blockSize, 0, stream >> > (A, scalar, B, size);
        }


    } // low_rank
} // cuda

