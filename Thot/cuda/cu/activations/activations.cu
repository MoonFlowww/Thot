#include <math_constants.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

#include "../../cuh/activations/activations.cuh"
#ifdef THOT_CUDA_DEBUG_SYNC
#define CUDA_DEBUG_SYNC() cudaDeviceSynchronize()
#else
#define CUDA_DEBUG_SYNC() ((void)0)
#endif


namespace cuda {
    const int BLOCK_SIZE = 256;
    namespace activations {

        // ReLU Activation
        __global__ void relu_forward(const float* input, float* output, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                output[idx] = fmaxf(0.0f, input[idx]);
            }
        }

        __global__ void relu_backward(const float* grad_output, const float* input, float* grad_input, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
            }
        }

        // Sigmoid Activation
        __global__ void sigmoid_forward(const float* input, float* output, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                output[idx] = 1.0f / (1.0f + expf(-input[idx]));
            }
        }

        __global__ void sigmoid_backward(const float* grad_output, const float* output, float* grad_input, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                // Sigmoid gradient is sigmoid(x) * (1 - sigmoid(x))
                // output already contains sigmoid(x)
                grad_input[idx] = grad_output[idx] * output[idx] * (1.0f - output[idx]);
            }
        }

        // Tanh Activation
        __global__ void tanh_forward(const float* input, float* output, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                output[idx] = tanhf(input[idx]);
            }
        }

        __global__ void tanh_backward(const float* grad_output, const float* output, float* grad_input, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                // Tanh gradient is (1 - tanh^2(x))
                // output already contains tanh(x)
                grad_input[idx] = grad_output[idx] * (1.0f - output[idx] * output[idx]);
            }
        }

        // Leaky ReLU Activation
        __global__ void leaky_relu_forward(const float* input, float* output, float negative_slope, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float x = input[idx];
                output[idx] = x > 0.0f ? x : x * negative_slope;
            }
        }

        __global__ void leaky_relu_backward(const float* grad_output, const float* input, float* grad_input, float negative_slope, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float x = input[idx];
                grad_input[idx] = grad_output[idx] * (x > 0.0f ? 1.0f : negative_slope);
            }
        }

        // ELU Activation
        __global__ void elu_forward(const float* input, float* output, float alpha, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float x = input[idx];
                output[idx] = x > 0.0f ? x : alpha * (expf(x) - 1.0f);
            }
        }

        __global__ void elu_backward(const float* grad_output, const float* output, const float* input, float* grad_input, float alpha, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float x = input[idx];
                float grad = x > 0.0f ? 1.0f : output[idx] + alpha;
                grad_input[idx] = grad_output[idx] * grad;
            }
        }

        // GELU Activation (Gaussian Error Linear Unit)
        __global__ void gelu_forward(const float* input, float* output, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float x = input[idx];
                // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                float sqrt2_over_pi = 0.7978845608028654f;  // sqrt(2/pi)
                float y = sqrt2_over_pi * (x + 0.044715f * x * x * x);
                output[idx] = 0.5f * x * (1.0f + tanhf(y));
            }
        }

        __global__ void gelu_backward(const float* grad_output, const float* input, float* grad_input, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float x = input[idx];
                float sqrt2_over_pi = 0.7978845608028654f;
                float y = sqrt2_over_pi * (x + 0.044715f * x * x * x);
                float tanh_y = tanhf(y);

                float inner_term = 0.044715f * x * x;
                float outer_term = 1.0f - tanh_y * tanh_y;
                float derivative = 0.5f * (1.0f + tanh_y) + 0.5f * x * outer_term * sqrt2_over_pi * (1.0f + 3.0f * inner_term);

                grad_input[idx] = grad_output[idx] * derivative;
            }
        }

        // Softmax Activation
         __global__ void softmax_forward(const float* input, float* output,
                                        int batch_size, int feature_dim) {
            extern __shared__ float shared[];
            float* smax = shared;
            float* ssum = shared + blockDim.x;

            int row = blockIdx.x;
            int tid = threadIdx.x;
            if (row >= batch_size) return;

            const float* in_row = input + row * feature_dim;
            float* out_row = output + row * feature_dim;

            // ---- compute maximum ----
            float local_max = -CUDART_INF_F;
            int vec_stride = blockDim.x * 4;
            int limit = feature_dim & ~3;

            for (int i = tid * 4; i < limit; i += vec_stride) {
                float4 v = reinterpret_cast<const float4*>(in_row)[i / 4];
                local_max = fmaxf(local_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
            }
            for (int i = limit + tid; i < feature_dim; i += blockDim.x)
                local_max = fmaxf(local_max, in_row[i]);

            smax[tid] = local_max;
            __syncthreads();
            for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
                if (tid < offset)
                    smax[tid] = fmaxf(smax[tid], smax[tid + offset]);
                __syncthreads();
            }
            float max_val = smax[0];

            // ---- compute sum of exp(x - max) ----
            float local_sum = 0.0f;
            for (int i = tid * 4; i < limit; i += vec_stride) {
                float4 v = reinterpret_cast<const float4*>(in_row)[i / 4];
                local_sum += expf(v.x - max_val) + expf(v.y - max_val) +
                             expf(v.z - max_val) + expf(v.w - max_val);
            }
            for (int i = limit + tid; i < feature_dim; i += blockDim.x)
                local_sum += expf(in_row[i] - max_val);

            ssum[tid] = local_sum;
            __syncthreads();
            for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
                if (tid < offset)
                    ssum[tid] += ssum[tid + offset];
                __syncthreads();
            }
            float sum_val = ssum[0];

            // ---- write outputs ----
            for (int i = tid * 4; i < limit; i += vec_stride) {
                float4 v = reinterpret_cast<const float4*>(in_row)[i / 4];
                float4 outv;
                outv.x = expf(v.x - max_val) / sum_val;
                outv.y = expf(v.y - max_val) / sum_val;
                outv.z = expf(v.z - max_val) / sum_val;
                outv.w = expf(v.w - max_val) / sum_val;
                reinterpret_cast<float4*>(out_row)[i / 4] = outv;
            }
            for (int i = limit + tid; i < feature_dim; i += blockDim.x)
                out_row[i] = expf(in_row[i] - max_val) / sum_val;
        }

        __global__ void softmax_backward(const float* grad_output, const float* output,
                                         float* grad_input, int batch_size,
                                         int feature_dim) {
            extern __shared__ float sdata[];

            int row = blockIdx.x;
            int tid = threadIdx.x;
            if (row >= batch_size) return;

            const float* g_row = grad_output + row * feature_dim;
            const float* o_row = output + row * feature_dim;
            float* gi_row = grad_input + row * feature_dim;

            // Compute dot = sum_j g_j * o_j
            float local_dot = 0.0f;
            int vec_stride = blockDim.x * 4;
            int limit = feature_dim & ~3;
            for (int i = tid * 4; i < limit; i += vec_stride) {
                float4 g = reinterpret_cast<const float4*>(g_row)[i / 4];
                float4 o = reinterpret_cast<const float4*>(o_row)[i / 4];
                local_dot += g.x * o.x + g.y * o.y + g.z * o.z + g.w * o.w;
            }
            for (int i = limit + tid; i < feature_dim; i += blockDim.x)
                local_dot += g_row[i] * o_row[i];

            sdata[tid] = local_dot;
            __syncthreads();
            for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
                if (tid < offset)
                    sdata[tid] += sdata[tid + offset];
                __syncthreads();
            }
            float dot = sdata[0];

            // Compute gradient
            for (int i = tid * 4; i < limit; i += vec_stride) {
                float4 g = reinterpret_cast<const float4*>(g_row)[i / 4];
                float4 o = reinterpret_cast<const float4*>(o_row)[i / 4];
                float4 gi;
                gi.x = o.x * (g.x - dot);
                gi.y = o.y * (g.y - dot);
                gi.z = o.z * (g.z - dot);
                gi.w = o.w * (g.w - dot);
                reinterpret_cast<float4*>(gi_row)[i / 4] = gi;
            }
            for (int i = limit + tid; i < feature_dim; i += blockDim.x) {
                float g = g_row[i];
                float o = o_row[i];
                gi_row[i] = o * (g - dot);
            }
        }


        // Wrapper functions for launching kernels
        void launchReluForward(const float* input, float* output, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            relu_forward << <num_blocks, BLOCK_SIZE, 0, stream >> > (input, output, size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchReluForward: %s\n", cudaGetErrorString(err));
            }
            CUDA_DEBUG_SYNC();
        }

        void launchReluBackward(const float* grad_output, const float* input, float* grad_input, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            relu_backward << <num_blocks, BLOCK_SIZE, 0, stream >> > (grad_output, input, grad_input, size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchReluBackward: %s\n", cudaGetErrorString(err));
            }
            CUDA_DEBUG_SYNC();
        }

        void launchSigmoidForward(const float* input, float* output, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            sigmoid_forward << <num_blocks, BLOCK_SIZE, 0, stream >> > (input, output, size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchSigmoidForward: %s\n", cudaGetErrorString(err));
            }
            CUDA_DEBUG_SYNC();
        }

        void launchSigmoidBackward(const float* grad_output, const float* output, float* grad_input, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            sigmoid_backward << <num_blocks, BLOCK_SIZE, 0, stream >> > (grad_output, output, grad_input, size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchSigmoidBackward: %s\n", cudaGetErrorString(err));
            }
            CUDA_DEBUG_SYNC();
        }

        void launchTanhForward(const float* input, float* output, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            tanh_forward << <num_blocks, BLOCK_SIZE, 0, stream >> > (input, output, size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchTanhForward: %s\n", cudaGetErrorString(err));
            }
            CUDA_DEBUG_SYNC();
        }

        void launchTanhBackward(const float* grad_output, const float* output, float* grad_input, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            tanh_backward << <num_blocks, BLOCK_SIZE, 0, stream >> > (grad_output, output, grad_input, size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchTanhBackward: %s\n", cudaGetErrorString(err));
            }
            CUDA_DEBUG_SYNC();
        }

        void launchLeakyReluForward(const float* input, float* output, float negative_slope, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            leaky_relu_forward << <num_blocks, BLOCK_SIZE, 0, stream >> > (input, output, negative_slope, size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchLeakyReluForward: %s\n", cudaGetErrorString(err));
            }
            CUDA_DEBUG_SYNC();
        }

        void launchLeakyReluBackward(const float* grad_output, const float* input, float* grad_input, float negative_slope, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            leaky_relu_backward << <num_blocks, BLOCK_SIZE, 0, stream >> > (grad_output, input, grad_input, negative_slope, size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchLeakyReluBackward: %s\n", cudaGetErrorString(err));
            }
            CUDA_DEBUG_SYNC();
        }

        void launchEluForward(const float* input, float* output, float alpha, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            elu_forward << <num_blocks, BLOCK_SIZE, 0, stream >> > (input, output, alpha, size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchEluForward: %s\n", cudaGetErrorString(err));
            }
            CUDA_DEBUG_SYNC();
        }

        void launchEluBackward(const float* grad_output, const float* output, const float* input, float* grad_input, float alpha, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            elu_backward << <num_blocks, BLOCK_SIZE, 0, stream >> > (grad_output, output, input, grad_input, alpha, size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchEluBackward: %s\n", cudaGetErrorString(err));
            }
            CUDA_DEBUG_SYNC();
        }

        void launchGeluForward(const float* input, float* output, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            gelu_forward << <num_blocks, BLOCK_SIZE, 0, stream >> > (input, output, size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchGeluForward: %s\n", cudaGetErrorString(err));
            }
            CUDA_DEBUG_SYNC();
        }

        void launchGeluBackward(const float* grad_output, const float* input, float* grad_input, int size, cudaStream_t stream) {
            int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            gelu_backward << <num_blocks, BLOCK_SIZE, 0, stream >> > (grad_output, input, grad_input, size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchGeluBackward: %s\n", cudaGetErrorString(err));
            }
            CUDA_DEBUG_SYNC();
        }

        void launchSoftmaxForward(const float* input, float* output, int batch_size, int feature_dim, cudaStream_t stream) {
            int blockSize = 256;
            size_t shared = 2 * blockSize * sizeof(float);
            softmax_forward<<<batch_size, blockSize, shared, stream>>>(
                input, output, batch_size, feature_dim);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchSoftmaxForward: %s\n", cudaGetErrorString(err));
            }
            CUDA_DEBUG_SYNC();
        }

        void launchSoftmaxBackward(const float* grad_output, const float* output, float* grad_input, int batch_size, int feature_dim, cudaStream_t stream) {
            int blockSize = 256;
            size_t shared = blockSize * sizeof(float);
            softmax_backward<<<batch_size, blockSize, shared, stream>>>(
                grad_output, output, grad_input, batch_size, feature_dim);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchSoftmaxBackward: %s\n", cudaGetErrorString(err));
            }
            CUDA_DEBUG_SYNC();
        }

    }  // namespace activations
}  // namespace Cuda
