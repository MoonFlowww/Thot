#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

#include "../../cuh/layers/spike.cuh"

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
} } while(0)
#endif

namespace cuda::layers {

__global__ void spike_forward(const float* __restrict__ input, float* __restrict__ membrane,
                              float* __restrict__ output, float threshold, int total) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        float v = membrane[idx] + input[idx];
        if (v >= threshold) {
            output[idx] = 1.0f;
            membrane[idx] = 0.0f;
        } else {
            output[idx] = 0.0f;
            membrane[idx] = v;
        }
    }
}

__global__ void spike_backward(const float* __restrict__ spikes, const float* __restrict__ grad_output,
                               float* __restrict__ grad_input, int total) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        grad_input[idx] = spikes[idx] * grad_output[idx];
    }
}

void launchSpikeForward(const float* input, float* membrane, float* output,
                        int batch_size, int neurons, float threshold, cudaStream_t stream) {
    int total = batch_size * neurons;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    spike_forward<<<gridSize, blockSize, 0, stream>>>(input, membrane, output, threshold, total);
    CUDA_CHECK(cudaGetLastError());
}

void launchSpikeBackward(const float* spikes, const float* grad_output, float* grad_input,
                         int batch_size, int neurons, cudaStream_t stream) {
    int total = batch_size * neurons;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    spike_backward<<<gridSize, blockSize, 0, stream>>>(spikes, grad_output, grad_input, total);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda::layers