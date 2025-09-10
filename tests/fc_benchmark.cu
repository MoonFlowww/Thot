#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>

#include "../Thot/cuda/cuh/layers/fc.cuh"

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    std::cerr << "CUDA error " << cudaGetErrorString(_e) << std::endl; } } while(0)
#endif

template <typename T>
__forceinline__ __device__ T ro(const T* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

// Naive kernels copied from original implementation
__global__ void fc_forward_naive(const float* __restrict__ input, const float* __restrict__ weights,
                                 const float* __restrict__ bias, float* __restrict__ output,
                                 int batch_size, int input_size, int output_size) {
    const int64_t N = (int64_t)batch_size * output_size;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t t = idx;
        const int b = t / output_size;
        const int o = t - (int64_t)b * output_size;
        float sum = 0.0f;
        const int64_t inB = (int64_t)b * input_size;
        for (int i = 0; i < input_size; ++i) {
            sum += ro(input + inB + i) * ro(weights + (int64_t)i * output_size + o);
        }
        if (bias) sum += ro(bias + o);
        output[idx] = sum;
    }
}

__global__ void fc_backward_input_naive(const float* __restrict__ grad_output, const float* __restrict__ weights,
                                        float* __restrict__ grad_input, int batch_size, int input_size, int output_size) {
    const int64_t N = (int64_t)batch_size * input_size;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t t = idx;
        const int b = t / input_size;
        const int i = t - (int64_t)b * input_size;
        float sum = 0.0f;
        const int64_t goB = (int64_t)b * output_size;
        const int64_t wRow = (int64_t)i * output_size;
        for (int o = 0; o < output_size; ++o) {
            sum += ro(grad_output + goB + o) * ro(weights + wRow + o);
        }
        grad_input[idx] = sum;
    }
}

__global__ void fc_backward_weights_naive(const float* __restrict__ input, const float* __restrict__ grad_output,
                                          float* __restrict__ grad_weights, int batch_size, int input_size, int output_size) {
    const int64_t N = (int64_t)input_size * output_size;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t t = idx;
        const int i = t / output_size;
        const int o = t - (int64_t)i * output_size;
        float sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            const int64_t inIdx = (int64_t)b * input_size + i;
            const int64_t goIdx = (int64_t)b * output_size + o;
            sum += ro(input + inIdx) * ro(grad_output + goIdx);
        }
        grad_weights[idx] = sum / static_cast<float>(batch_size);
    }
}

// Benchmark helper
float elapsedMs(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

int main() {
    int batch = 256;
    int in = 512;
    int out = 512;
    size_t in_bytes = (size_t)batch * in * sizeof(float);
    size_t w_bytes = (size_t)in * out * sizeof(float);
    size_t out_bytes = (size_t)batch * out * sizeof(float);

    float *input, *weights, *bias, *output, *grad_output, *grad_input, *grad_weights;
    CUDA_CHECK(cudaMalloc(&input, in_bytes));
    CUDA_CHECK(cudaMalloc(&weights, w_bytes));
    CUDA_CHECK(cudaMalloc(&bias, out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output, out_bytes));
    CUDA_CHECK(cudaMalloc(&grad_output, out_bytes));
    CUDA_CHECK(cudaMalloc(&grad_input, in_bytes));
    CUDA_CHECK(cudaMalloc(&grad_weights, w_bytes));

    // Warm up data (fill with 1s)
    std::vector<float> h_in(batch * in, 1.0f);
    std::vector<float> h_w(in * out, 1.0f);
    std::vector<float> h_b(out, 1.0f);
    std::vector<float> h_go(batch * out, 1.0f);
    CUDA_CHECK(cudaMemcpy(input, h_in.data(), in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weights, h_w.data(), w_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bias, h_b.data(), out * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grad_output, h_go.data(), out_bytes, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((batch * out + block.x - 1) / block.x);
    dim3 grid_in((batch * in + block.x - 1) / block.x);
    dim3 grid_w((in * out + block.x - 1) / block.x);

    // Warm up
    fc_forward_naive<<<grid, block>>>(input, weights, bias, output, batch, in, out);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));

    // Naive
    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < 50; ++i) {
        fc_forward_naive<<<grid, block>>>(input, weights, bias, output, batch, in, out);
    }
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float naive_ms = elapsedMs(s, e);

    // cublas implementation
    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < 50; ++i) {
        cuda::layers::launchFCForward(input, weights, bias, output, batch, in, out);
    }
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float cublas_ms = elapsedMs(s, e);

    // Backward input naive
    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < 50; ++i) {
        fc_backward_input_naive<<<grid_in, block>>>(grad_output, weights, grad_input, batch, in, out);
    }
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float naive_bi_ms = elapsedMs(s, e);

    // Backward input cublas
    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < 50; ++i) {
        cuda::layers::launchFCBackwardInput(grad_output, weights, grad_input, batch, in, out);
    }
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float cublas_bi_ms = elapsedMs(s, e);

    // Backward weights naive
    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < 50; ++i) {
        fc_backward_weights_naive<<<grid_w, block>>>(input, grad_output, grad_weights, batch, in, out);
    }
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float naive_bw_ms = elapsedMs(s, e);

    // Backward weights cublas
    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < 50; ++i) {
        cuda::layers::launchFCBackwardWeights(input, grad_output, grad_weights, batch, in, out);
    }
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float cublas_bw_ms = elapsedMs(s, e);

    std::cout << "Naive forward time: " << naive_ms << " ms\n";
    std::cout << "cuBLAS forward time: " << cublas_ms << " ms\n";
    std::cout << "Naive backward input time: " << naive_bi_ms << " ms\n";
    std::cout << "cuBLAS backward input time: " << cublas_bi_ms << " ms\n";
    std::cout << "Naive backward weights time: " << naive_bw_ms << " ms\n";
    std::cout << "cuBLAS backward weights time: " << cublas_bw_ms << " ms\n";

    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(weights));
    CUDA_CHECK(cudaFree(bias));
    CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFree(grad_output));
    CUDA_CHECK(cudaFree(grad_input));
    CUDA_CHECK(cudaFree(grad_weights));
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    return 0;
}
