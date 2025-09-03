#pragma once

#include <cuda_runtime.h>

namespace cuda {
    namespace initializations {
        __global__ void random_uniform(float* data, int size, float low, float high, unsigned long long seed);
        __global__ void random_normal(float* data, int size, float mean, float stddev, unsigned long long seed);
        __global__ void random_truncated_normal(float* data, int size, float mean, float stddev, unsigned long long seed);
        __global__ void fill_constant(float* data, int size, float value);
        __global__ void dirac_init(float* data, int rows, int cols);
        __global__ void scale_values(float* data, int size, float scale);

        void launchRandomUniform(float* data, int size, float low, float high, cudaStream_t stream = 0);
        void launchRandomNormal(float* data, int size, float mean, float stddev, cudaStream_t stream = 0);
        void launchRandomTruncatedNormal(float* data, int size, float mean, float stddev, cudaStream_t stream = 0);
        void launchFill(float* data, int size, float value, cudaStream_t stream = 0);
        void launchDirac(float* data, int rows, int cols, cudaStream_t stream = 0);
        void launchScale(float* data, int size, float scale, cudaStream_t stream = 0);
        void launchLyapunov(float* data, int rows, int cols, cudaStream_t stream = 0);
    }
}