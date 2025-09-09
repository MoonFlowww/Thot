#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../../cuh/layernorm/layernorm.cuh"

#ifndef CUDA_CHECK
#define CUDA_CHECK(x)                                                          \
    do {                                                                         \
        cudaError_t _e = (x);                                                      \
        if (_e != cudaSuccess) {                                                   \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,                     \
            cudaGetErrorString(_e));                                          \
        }                                                                          \
    } while (0)
#endif

namespace cuda::layernorm {

    namespace {
        __device__ inline float rsqrt_approx(float x) { return rsqrtf(x); }
        } // namespace

    __global__ void rms_forward(const float *__restrict__ input,
                                float *__restrict__ output, int rows, int cols) {
        extern __shared__ float sdata[];
        int row = blockIdx.x;
        int tid = threadIdx.x;
        const float *row_in = input + row * cols;

        float sum = 0.0f;
        int cols4 = cols / 4;
        const float4 *row_in4 = reinterpret_cast<const float4 *>(row_in);
        for (int i = tid; i < cols4; i += blockDim.x) {
            float4 v = row_in4[i];
            sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        for (int i = cols4 * 4 + tid; i < cols; i += blockDim.x) {
            float v = row_in[i];
            sum += v * v;
        }
        sdata[tid] = sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s)
            sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        float mean_sq = sdata[0] / cols;
        float scale = rsqrt_approx(mean_sq + 1e-5f);

        float4 *row_out4 = reinterpret_cast<float4 *>(output + row * cols);
        for (int i = tid; i < cols4; i += blockDim.x) {
            float4 v = row_in4[i];
            v.x *= scale;
            v.y *= scale;
            v.z *= scale;
            v.w *= scale;
            row_out4[i] = v;
        }
        for (int i = cols4 * 4 + tid; i < cols; i += blockDim.x) {
            output[row * cols + i] = row_in[i] * scale;
        }
    }

    __global__ void rms_backward(const float *__restrict__ input,
                                 const float *__restrict__ grad_output,
                                 float *__restrict__ grad_input, int rows,
                                 int cols) {
        extern __shared__ float sdata[];
        int row = blockIdx.x;
        int tid = threadIdx.x;
        const float *row_in = input + row * cols;
        const float *row_go = grad_output + row * cols;

        float sum = 0.0f;
        int cols4 = cols / 4;
        const float4 *row_in4 = reinterpret_cast<const float4 *>(row_in);
        const float4 *row_go4 = reinterpret_cast<const float4 *>(row_go);
        for (int i = tid; i < cols4; i += blockDim.x) {
            float4 v = row_in4[i];
            sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        for (int i = cols4 * 4 + tid; i < cols; i += blockDim.x) {
            float v = row_in[i];
            sum += v * v;
        }
        sdata[tid] = sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s)
            sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        float mean_sq = sdata[0] / cols;
        float inv_rms = rsqrt_approx(mean_sq + 1e-5f);
        float inv_rms3 = inv_rms * inv_rms * inv_rms;

        float sum_gx = 0.0f;
        for (int i = tid; i < cols4; i += blockDim.x) {
            float4 v_in = row_in4[i];
            float4 v_go = row_go4[i];
            sum_gx +=
                v_go.x * v_in.x + v_go.y * v_in.y + v_go.z * v_in.z + v_go.w * v_in.w;
        }
        for (int i = cols4 * 4 + tid; i < cols; i += blockDim.x) {
            sum_gx += row_go[i] * row_in[i];
        }
        sdata[tid] = sum_gx;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s)
            sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        float sgx = sdata[0];

        float4 *row_gi4 = reinterpret_cast<float4 *>(grad_input + row * cols);
        for (int i = tid; i < cols4; i += blockDim.x) {
            float4 v_in = row_in4[i];
            float4 v_go = row_go4[i];
            float4 g;
            g.x = inv_rms * v_go.x - v_in.x * sgx * inv_rms3 / cols;
            g.y = inv_rms * v_go.y - v_in.y * sgx * inv_rms3 / cols;
            g.z = inv_rms * v_go.z - v_in.z * sgx * inv_rms3 / cols;
            g.w = inv_rms * v_go.w - v_in.w * sgx * inv_rms3 / cols;
            row_gi4[i] = g;
        }
        for (int i = cols4 * 4 + tid; i < cols; i += blockDim.x) {
            grad_input[row * cols + i] =
                inv_rms * row_go[i] - row_in[i] * sgx * inv_rms3 / cols;
        }
    }

__global__ void dyt_forward(const float *__restrict__ input,
                            float *__restrict__ output, int rows, int cols) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *row_in = input + row * cols;

    float sum = 0.0f;
    int cols4 = cols / 4;
    const float4 *row_in4 = reinterpret_cast<const float4 *>(row_in);
    for (int i = tid; i < cols4; i += blockDim.x) {
        float4 v = row_in4[i];
        sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (int i = cols4 * 4 + tid; i < cols; i += blockDim.x) {
        float v = row_in[i];
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
        sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
        float mean_sq = sdata[0] / cols;
        float scale = rsqrt_approx(mean_sq + 1e-5f);

        float4 *row_out4 = reinterpret_cast<float4 *>(output + row * cols);
        for (int i = tid; i < cols4; i += blockDim.x) {
            float4 v = row_in4[i];
            v.x = tanhf(v.x * scale);
            v.y = tanhf(v.y * scale);
            v.z = tanhf(v.z * scale);
            v.w = tanhf(v.w * scale);
            row_out4[i] = v;
        }
        for (int i = cols4 * 4 + tid; i < cols; i += blockDim.x) {
            output[row * cols + i] = tanhf(row_in[i] * scale);
        }
    }

    __global__ void dyt_backward(const float *__restrict__ input,
                             const float *__restrict__ output,
                             const float *__restrict__ grad_output,
                             float *__restrict__ grad_input, int rows,
                             int cols) {
        extern __shared__ float sdata[];
        int row = blockIdx.x;
        int tid = threadIdx.x;
        const float *row_in = input + row * cols;
        const float *row_out = output + row * cols;
        const float *row_go = grad_output + row * cols;

        float sum = 0.0f;
        int cols4 = cols / 4;
        const float4 *row_in4 = reinterpret_cast<const float4 *>(row_in);
        for (int i = tid; i < cols4; i += blockDim.x) {
            float4 v = row_in4[i];
            sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        for (int i = cols4 * 4 + tid; i < cols; i += blockDim.x) {
            float v = row_in[i];
            sum += v * v;
        }
        sdata[tid] = sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s)
                sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        float mean_sq = sdata[0] / cols;
        float inv_rms = rsqrt_approx(mean_sq + 1e-5f);
        float inv_rms3 = inv_rms * inv_rms * inv_rms;

        float sum_gx = 0.0f;
        const float4 *row_out4 = reinterpret_cast<const float4 *>(row_out);
        const float4 *row_go4 = reinterpret_cast<const float4 *>(row_go);
        for (int i = tid; i < cols4; i += blockDim.x) {
            float4 v_in = row_in4[i];
            float4 v_out = row_out4[i];
            float4 v_go = row_go4[i];
            float4 tgo;
            tgo.x = v_go.x * (1.0f - v_out.x * v_out.x);
            tgo.y = v_go.y * (1.0f - v_out.y * v_out.y);
            tgo.z = v_go.z * (1.0f - v_out.z * v_out.z);
            tgo.w = v_go.w * (1.0f - v_out.w * v_out.w);
            sum_gx += tgo.x * v_in.x + tgo.y * v_in.y + tgo.z * v_in.z + tgo.w * v_in.w;
        }
        for (int i = cols4 * 4 + tid; i < cols; i += blockDim.x) {
            float outv = row_out[i];
            float gov = row_go[i] * (1.0f - outv * outv);
            sum_gx += gov * row_in[i];
        }
        sdata[tid] = sum_gx;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s)
                sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        float sgx = sdata[0];

        float4 *row_gi4 = reinterpret_cast<float4 *>(grad_input + row * cols);
        for (int i = tid; i < cols4; i += blockDim.x) {
            float4 v_in = row_in4[i];
            float4 v_out = row_out4[i];
            float4 v_go = row_go4[i];
            float4 tgo;
            tgo.x = v_go.x * (1.0f - v_out.x * v_out.x);
            tgo.y = v_go.y * (1.0f - v_out.y * v_out.y);
            tgo.z = v_go.z * (1.0f - v_out.z * v_out.z);
            tgo.w = v_go.w * (1.0f - v_out.w * v_out.w);
            float4 g;
            g.x = inv_rms * tgo.x - v_in.x * sgx * inv_rms3 / cols;
            g.y = inv_rms * tgo.y - v_in.y * sgx * inv_rms3 / cols;
            g.z = inv_rms * tgo.z - v_in.z * sgx * inv_rms3 / cols;
            g.w = inv_rms * tgo.w - v_in.w * sgx * inv_rms3 / cols;
            row_gi4[i] = g;
        }
        for (int i = cols4 * 4 + tid; i < cols; i += blockDim.x) {
            float outv = row_out[i];
            float gov = row_go[i] * (1.0f - outv * outv);
            grad_input[row * cols + i] =
                inv_rms * gov - row_in[i] * sgx * inv_rms3 / cols;
        }
    }

    void launchRMSForward(const float *input, float *output, int rows, int cols,
                      cudaStream_t stream) {
        const int blockSize = 256;
        const int gridSize = rows;
        rms_forward<<<gridSize, blockSize, blockSize * sizeof(float), stream>>>(
            input, output, rows, cols);
        CUDA_CHECK(cudaGetLastError());
    }

    void launchRMSBackward(const float *input, const float *grad_output,
                       float *grad_input, int rows, int cols,
                       cudaStream_t stream) {
        const int blockSize = 256;
        const int gridSize = rows;
        rms_backward<<<gridSize, blockSize, blockSize * sizeof(float), stream>>>(
            input, grad_output, grad_input, rows, cols);
        CUDA_CHECK(cudaGetLastError());
    }

    void launchDyTForward(const float *input, float *output, int rows, int cols,
                      cudaStream_t stream) {
        const int blockSize = 256;
        const int gridSize = rows;
        dyt_forward<<<gridSize, blockSize, blockSize * sizeof(float), stream>>>(
            input, output, rows, cols);
        CUDA_CHECK(cudaGetLastError());
    }

    void launchDyTBackward(const float *input, const float *output,
                       const float *grad_output, float *grad_input, int rows,
                       int cols, cudaStream_t stream) {
        const int blockSize = 256;
        const int gridSize = rows;
        dyt_backward<<<gridSize, blockSize, blockSize * sizeof(float), stream>>>(
            input, output, grad_output, grad_input, rows, cols);
        CUDA_CHECK(cudaGetLastError());
    }

} // namespace cuda::layernorm