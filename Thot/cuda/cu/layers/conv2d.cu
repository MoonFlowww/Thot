#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <cmath>
#include "../../cuh/layers/conv2d.cuh"

namespace cuda {
    namespace layers {

        // Simple tiled GEMM kernel for small matrices (row-major layout)
        __global__ void small_gemm(const float* A, const float* B, float* C,
            int M, int N, int K) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < M && col < N) {
                float sum = 0.0f;
                for (int e = 0; e < K; ++e) {
                    sum += A[row * K + e] * B[e * N + col];
                }
                C[row * N + col] = sum;
            }
        }

        __global__ void add_bias(float* C, const float* bias, int M, int N) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= M * N) return;
            int oc = idx / N;
            C[idx] += bias[oc];
        }


        __global__ void conv2d_forward(const float* input, const float* weights, const float* bias,
            float* output, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * out_channels * out_height * out_width) return;

            int b = idx / (out_channels * out_height * out_width);
            int oc = (idx / (out_height * out_width)) % out_channels;
            int h = (idx / out_width) % out_height;
            int w = idx % out_width;

            float sum = 0.0f;

            for (int ic = 0; ic < in_channels; ++ic) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int ih = h * stride - padding + kh;
                        int iw = w * stride - padding + kw;

                        if (ih < 0 || ih >= in_height || iw < 0 || iw >= in_width) continue;

                        // Input index
                        int in_idx = b * (in_channels * in_height * in_width) +
                            ic * (in_height * in_width) +
                            ih * in_width +
                            iw;

                        // Weight index
                        int w_idx = oc * (in_channels * kernel_size * kernel_size) +
                            ic * (kernel_size * kernel_size) +
                            kh * kernel_size +
                            kw;

                        sum += input[in_idx] * weights[w_idx];
                    }
                }
            }

            if (bias != nullptr) sum += bias[oc];


            output[idx] = sum;
        }

        __global__ void conv2d_backward_input(const float* grad_output, const float* weights,
            float* grad_input, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * in_channels * in_height * in_width) return;

            int b = idx / (in_channels * in_height * in_width);
            int ic = (idx / (in_height * in_width)) % in_channels;
            int h = (idx / in_width) % in_height;
            int w = idx % in_width;

            float sum = 0.0f;

            for (int oc = 0; oc < out_channels; ++oc) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        // Calculate output position
                        int oh = (h + padding - kh) / stride;
                        int ow = (w + padding - kw) / stride;

                        if (oh >= 0 && oh < out_height && ow >= 0 && ow < out_width &&
                            (h + padding - kh) % stride == 0 && (w + padding - kw) % stride == 0) {

                            // Grad output index
                            int go_idx = b * (out_channels * out_height * out_width) +
                                oc * (out_height * out_width) +
                                oh * out_width +
                                ow;

                            // Weight index (reversed for backward)
                            int w_idx = oc * (in_channels * kernel_size * kernel_size) +
                                ic * (kernel_size * kernel_size) +
                                kh * kernel_size +
                                kw;

                            sum += grad_output[go_idx] * weights[w_idx];
                        }
                    }
                }
            }

            grad_input[idx] = sum;
        }

        __global__ void conv2d_backward_weights(const float* input, const float* grad_output,
            float* grad_weights, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= out_channels * in_channels * kernel_size * kernel_size) return;

            int oc = idx / (in_channels * kernel_size * kernel_size);
            int ic = (idx / (kernel_size * kernel_size)) % in_channels;
            int kh = (idx / kernel_size) % kernel_size;
            int kw = idx % kernel_size;

            float sum = 0.0f;

            for (int b = 0; b < batch_size; ++b) {
                for (int oh = 0; oh < out_height; ++oh) {
                    for (int ow = 0; ow < out_width; ++ow) {
                        int ih = oh * stride - padding + kh;
                        int iw = ow * stride - padding + kw;

                        if (ih < 0 || ih >= in_height || iw < 0 || iw >= in_width) continue;

                        int in_idx = b * (in_channels * in_height * in_width) +
                            ic * (in_height * in_width) +
                            ih * in_width +
                            iw;

                        int go_idx = b * (out_channels * out_height * out_width) +
                            oc * (out_height * out_width) +
                            oh * out_width +
                            ow;

                        sum += input[in_idx] * grad_output[go_idx];
                    }
                }
            }

            grad_weights[idx] = sum / static_cast<float>(batch_size);
        }

        __global__ void conv2d_backward_bias(const float* grad_output, float* grad_bias,
            int batch_size, int out_channels, int out_height, int out_width) {

            int oc = blockIdx.x * blockDim.x + threadIdx.x;
            if (oc >= out_channels) return;

            float sum = 0.0f;

            for (int b = 0; b < batch_size; ++b) {
                for (int oh = 0; oh < out_height; ++oh) {
                    for (int ow = 0; ow < out_width; ++ow) {
                        int go_idx = b * (out_channels * out_height * out_width) +
                            oc * (out_height * out_width) +
                            oh * out_width +
                            ow;

                        sum += grad_output[go_idx];
                    }
                }
            }

            grad_bias[oc] = sum / static_cast<float>(batch_size);
        }

        void launchConv2DForward(const float* input, const float* weights, const float* bias,
            float* output, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,
            cudaStream_t stream) {
#ifdef THOT_WITH_CUDNN
            launchConv2DForwardCuDNN(input, weights, bias, output,
                batch_size, in_channels, in_height, in_width,
                out_channels, kernel_size, stride, padding,
                out_height, out_width, stream);
            return;
#endif

            // Optimize 1x1 convolutions using GEMM
            if (kernel_size == 1 && stride == 1 && padding == 0) {
                int spatial = in_height * in_width;
                int N = batch_size * spatial;        // columns
                int M = out_channels;                // rows
                int K = in_channels;                 // inner dimension

                bool use_cublas = (M * N * K) > 4096; // heuristic threshold
                if (use_cublas) {
                    cublasHandle_t handle;
                    cublasCreate(&handle);
                    cublasSetStream(handle, stream);
                    const float alpha = 1.0f;
                    const float beta = 0.0f;
                    // Treat row-major matrices as transposed column-major
                    cublasStatus_t stat = cublasSgemm(
                        handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K,
                        &alpha,
                        input, N,
                        weights, K,
                        &beta,
                        output, N);
                    if (stat != CUBLAS_STATUS_SUCCESS) {
                        printf("cuBLAS sgemm failed in launchConv2DForward\n");
                    }
                    cublasDestroy(handle);
                }
                else {
                    dim3 block(16, 16);
                    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
                    small_gemm<<<grid, block, 0, stream>>>(weights, input, output, M, N, K);
                }

                if (bias != nullptr) {
                    int total = M * N;
                    int blockB = 256;
                    int gridB = (total + blockB - 1) / blockB;
                    add_bias<<<gridB, blockB, 0, stream>>>(output, bias, M, N);
                }

                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("Kernel launch error in launchConv2DForward (1x1): %s\n", cudaGetErrorString(err));
                }
                cudaDeviceSynchronize();
                return;
            }

            // General case using im2col + GEMM
            launchConv2DForwardIm2Col(input, weights, bias, output,
                batch_size, in_channels, in_height, in_width,
                out_channels, kernel_size, stride, padding,
                out_height, out_width, stream);
        }

        void launchConv2DBackwardInput(const float* grad_output, const float* weights,
            float* grad_input, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,
            cudaStream_t stream) {

            const int blockSize = 256;
            const int numElements = batch_size * in_channels * in_height * in_width;
            const int numBlocks = (numElements + blockSize - 1) / blockSize;

#ifdef THOT_WITH_CUDNN
            launchConv2DBackwardInputCuDNN(grad_output, weights, grad_input,
                batch_size, in_channels, in_height, in_width,
                out_channels, kernel_size, stride, padding,
                out_height, out_width, stream);
            return;
#endif
            conv2d_backward_input<<<numBlocks, blockSize, 0, stream>>>(
                grad_output, weights, grad_input,
                batch_size, in_channels, in_height, in_width,
                out_channels, kernel_size, stride, padding,
                out_height, out_width);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchConv2DBackwardInput: %s\n", cudaGetErrorString(err));
            }
            cudaDeviceSynchronize();
        }

        void launchConv2DBackwardWeights(const float* input, const float* grad_output,
            float* grad_weights, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,
            cudaStream_t stream) {

            const int blockSize = 256;
            const int numElements = out_channels * in_channels * kernel_size * kernel_size;
            const int numBlocks = (numElements + blockSize - 1) / blockSize;

#ifdef THOT_WITH_CUDNN
            launchConv2DBackwardWeightsCuDNN(input, grad_output, grad_weights,
                batch_size, in_channels, in_height, in_width,
                out_channels, kernel_size, stride, padding,
                out_height, out_width, stream);
            return;
#endif
            conv2d_backward_weights<<<numBlocks, blockSize, 0, stream>>>(
                input, grad_output, grad_weights,
                batch_size, in_channels, in_height, in_width,
                out_channels, kernel_size, stride, padding,
                out_height, out_width
                );
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchConv2DBackwardWeights: %s\n", cudaGetErrorString(err));
            }
            cudaDeviceSynchronize();
        }

        void launchConv2DBackwardBias(const float* grad_output, float* grad_bias,
            int batch_size, int out_channels, int out_height, int out_width,
            cudaStream_t stream) {

            const int blockSize = 256;
            const int numBlocks = (out_channels + blockSize - 1) / blockSize;

#ifdef THOT_WITH_CUDNN
            launchConv2DBackwardBiasCuDNN(grad_output, grad_bias,
                batch_size, out_channels, out_height, out_width, stream);
            return;
#endif
            conv2d_backward_bias<<<numBlocks, blockSize, 0, stream>>>(
                grad_output, grad_bias,
                batch_size, out_channels, out_height, out_width
                );
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch error in launchConv2DBackwardBias: %s\n", cudaGetErrorString(err));
            }
            cudaDeviceSynchronize();
        }
    }
}