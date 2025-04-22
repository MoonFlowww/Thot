#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


namespace cuda {
    namespace low_rank {

        // Elementwise addition kernel: C = A + B
        __global__ void add_kernel(const float* A, const float* B, float* C, int size);

        // Elementwise multiplication kernel: C = A * B
        __global__ void multiply_kernel(const float* A, const float* B, float* C, int size);

        // Scalar addition kernel: B = A + scalar
        __global__ void add_scalar_kernel(const float* A, float scalar, float* B, int size);

        // Scalar multiplication kernel: B = A * scalar
        __global__ void multiply_scalar_kernel(const float* A, float scalar, float* B, int size);

        void launchAdd(const float* A, const float* B, float* C, int size, cudaStream_t stream = 0);

        void launchMultiply(const float* A, const float* B, float* C, int size, cudaStream_t stream = 0);

        void launchAddScalar(const float* A, float scalar, float* B, int size, cudaStream_t stream = 0);

        void launchMultiplyScalar(const float* A, float scalar, float* B, int size, cudaStream_t stream = 0);


    } // low_rank
} // cuda

using cuda::low_rank::launchAdd;
using cuda::low_rank::launchAddScalar;
using cuda::low_rank::launchMultiply;
using cuda::low_rank::launchMultiplyScalar;
