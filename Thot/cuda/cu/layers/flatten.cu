#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cmath>
#include "../../cuh/layers/flatten.cuh"
#ifdef THOT_CUDA_DEBUG_SYNC
#define CUDA_DEBUG_SYNC() cudaDeviceSynchronize()
#else
#define CUDA_DEBUG_SYNC() ((void)0)
#endif



namespace cuda {
    namespace layers {

        __global__ void flatten_forward(const float* input, float* output, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                output[idx] = input[idx];
            }
        }

        __global__ void flatten_backward(const float* grad_output, float* grad_input, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                grad_input[idx] = grad_output[idx];
            }
        }

        void launchFlattenForward(const float* input, float* output, int batch_size, int feature_size, cudaStream_t stream) {
            int size = batch_size * feature_size;
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            flatten_forward<<<numBlocks, blockSize, 0, stream>>>(input, output, size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchFlattenForward: %s\n", cudaGetErrorString(err));
            }
            CUDA_DEBUG_SYNC();
        }

        void launchFlattenBackward(const float* grad_output, float* grad_input, int batch_size, int feature_size, cudaStream_t stream) {
            int size = batch_size * feature_size;
            int blockSize = 256;
            int numBlocks = (size + blockSize - 1) / blockSize;
            flatten_backward<<<numBlocks, blockSize, 0, stream>>>(grad_output, grad_input, size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchFlattenBackward: %s\n", cudaGetErrorString(err));
            }
            CUDA_DEBUG_SYNC();
        }

    }
}