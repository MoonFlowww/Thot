#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <complex>
#include <cufft.h>
#include "../../cuh/layers/conv2d.cuh"
#ifdef THOT_WITH_CUDNN
#include <cudnn.h>
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    return; \
    } } while(0)
#endif

#ifndef CUFFT_CHECK
#define CUFFT_CHECK(x) do { cufftResult _e = (x); if (_e != CUFFT_SUCCESS) { \
printf("cuFFT error %s:%d: %d\n", __FILE__, __LINE__, (int)_e); return; } } while(0)
#endif




namespace cuda {
    namespace layers {
        // Winograd F(2x2,3x3) transform matrices. These are stored in constant
        // memory so kernels can access them without additional global loads.
        // G: 4x3, B^T: 4x4, A^T: 2x4
        __device__ __constant__ float WINOGRAD_F2X2_G[12] = {
            1.f, 0.f, 0.f,
            0.5f, 0.5f, 0.5f,
            0.5f,-0.5f, 0.5f,
            0.f, 0.f, 1.f
        };
        __device__ __constant__ float WINOGRAD_F2X2_BT[16] = {
            1.f, 0.f,-1.f, 0.f,
            0.f, 1.f, 1.f, 0.f,
            0.f,-1.f, 1.f, 0.f,
            0.f, 1.f, 0.f,-1.f
        };
        __device__ __constant__ float WINOGRAD_F2X2_AT[8] = {
            1.f, 1.f, 1.f, 0.f,
            0.f, 1.f,-1.f,-1.f
        };

        __forceinline__ __device__ int div_floor(int a, int b) {
            // b > 0
            int q = a / b;
            int r = a % b;
            if ((r != 0) && ((r > 0) != (b > 0))) --q;
            return q;
        }
        __forceinline__ __device__ int div_ceil(int a, int b) {
            // b > 0
            return -div_floor(-a, b);
        }

        template <typename T>
        __forceinline__ __device__ T ro(const T* p) {
        #if __CUDA_ARCH__ >= 350
            return __ldg(p);
        #else
            return *p;
        #endif
        }

        // ---------------- CPU Winograd helpers -----------------

        __device__ inline void winograd_input_transform_device(const float d[16], float U[16]) {
            float temp[16];
            const float* BT = WINOGRAD_F2X2_BT;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    float sum = 0.f;
                    for (int k = 0; k < 4; ++k)
                        sum += BT[i*4 + k] * d[k*4 + j];
                    temp[i*4 + j] = sum;
                }
            }
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    float sum = 0.f;
                    for (int k = 0; k < 4; ++k)
                        sum += temp[i*4 + k] * BT[j*4 + k];
                    U[i*4 + j] = sum;
                }
            }
        }

        __device__ inline void winograd_filter_transform_device(const float g[9], float V[16]) {
            const float* G = WINOGRAD_F2X2_G;
            float temp[12];
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 3; ++j) {
                    float sum = 0.f;
                    for (int k = 0; k < 3; ++k)
                        sum += G[i*3 + k] * g[k*3 + j];
                    temp[i*3 + j] = sum;
                }
            }
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    float sum = 0.f;
                    for (int k = 0; k < 3; ++k)
                        sum += temp[i*3 + k] * G[j*3 + k];
                    V[i*4 + j] = sum;
                }
            }
        }

        __device__ inline void winograd_output_transform_device(const float M[16], float Y[4]) {
            const float* AT = WINOGRAD_F2X2_AT;
            float temp[8];
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 4; ++j) {
                    float sum = 0.f;
                    for (int k = 0; k < 4; ++k)
                        sum += AT[i*4 + k] * M[k*4 + j];
                    temp[i*4 + j] = sum;
                }
            }
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    float sum = 0.f;
                    for (int k = 0; k < 4; ++k)
                        sum += temp[i*4 + k] * AT[j*4 + k];
                    Y[i*2 + j] = sum;
                }
            }
        }





        // ---------------- GPU Winograd implementation -----------------
        __global__ void conv2d_forward_winograd_kernel(
            const float* __restrict__ input,
            const float* __restrict__ weights,
            const float* __restrict__ bias,
            float* __restrict__ output,
            int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int padding, int out_height, int out_width)
        {
            int tiles_w = (out_width + 1) / 2;
            int tiles_h = (out_height + 1) / 2;
            int tile_count = tiles_w * tiles_h;
            int grid_stride = blockDim.x * gridDim.x;
            int oc = blockIdx.y;
            int b = blockIdx.z;
            for (int tile_idx = blockIdx.x * blockDim.x + threadIdx.x; tile_idx < tile_count; tile_idx += grid_stride) {
                int th = tile_idx / tiles_w;
                int tw = tile_idx - th * tiles_w;
                if (th >= tiles_h || tw >= tiles_w) continue;
                float M[16] = {0};
                for (int ic = 0; ic < in_channels; ++ic) {
                    float d[16];
                    for (int ih = 0; ih < 4; ++ih) {
                        int ih_g = th*2 + ih - padding;
                        for (int iw = 0; iw < 4; ++iw) {
                            int iw_g = tw*2 + iw - padding;
                            float val = 0.f;
                            if (ih_g >= 0 && ih_g < in_height && iw_g >= 0 && iw_g < in_width)
                                val = input[(((b*in_channels + ic)*in_height + ih_g)*in_width) + iw_g];
                            d[ih*4 + iw] = val;
                        }
                    }
                    float U[16], V[16];
                    winograd_input_transform_device(d, U);
                    winograd_filter_transform_device(weights + ((oc*in_channels + ic) * 9), V);
                    for (int m = 0; m < 16; ++m) M[m] += U[m] * V[m];
                }
                float Y[4];
                winograd_output_transform_device(M, Y);
                for (int oh = 0; oh < 2; ++oh) {
                    for (int ow = 0; ow < 2; ++ow) {
                        int oh_g = th*2 + oh;
                        int ow_g = tw*2 + ow;
                        if (oh_g < out_height && ow_g < out_width) {
                            float val = Y[oh*2 + ow];
                            if (bias) val += bias[oc];
                            output[(((b*out_channels + oc)*out_height + oh_g)*out_width) + ow_g] = val;
                        }
                    }
                }
            }
        }

        static void conv2d_forward_winograd_gpu(
            const float* input, const float* weights, const float* bias, float* output,
            int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int padding,
            int out_height, int out_width, cudaStream_t stream)
        {
            if (kernel_size != 3) {
                const int64_t N = (int64_t)batch_size * out_channels * out_height * out_width;
                const int blockSize = 256;
                const int gridSize  = (int)std::min<int64_t>((N + blockSize - 1) / blockSize, 65535);
                conv2d_forward<<<gridSize, blockSize, 0, stream>>>(
                    input, weights, bias, output,
                    batch_size, in_channels, in_height, in_width,
                    out_channels, kernel_size, 1, padding,
                    out_height, out_width);
                CUDA_CHECK(cudaGetLastError());
                return;
            }
            int tiles_w = (out_width + 1) / 2;
            int tiles_h = (out_height + 1) / 2;
            dim3 block(64);
            dim3 grid((tiles_w*tiles_h + block.x - 1) / block.x, out_channels, batch_size);
            conv2d_forward_winograd_kernel<<<grid, block, 0, stream>>>(
                input, weights, bias, output,
                batch_size, in_channels, in_height, in_width,
                out_channels, padding, out_height, out_width);
            CUDA_CHECK(cudaGetLastError());
        }

        // ---------------- GPU FFT implementation -----------------
        __global__ void pad_2d(const float* src, float* dst,
                               int src_h, int src_w, int dst_h, int dst_w) {
            int h = blockIdx.y * blockDim.y + threadIdx.y;
            int w = blockIdx.x * blockDim.x + threadIdx.x;
            if (h >= dst_h || w >= dst_w) return;
            float val = 0.f;
            if (h < src_h && w < src_w)
                val = src[h*src_w + w];
            dst[h*dst_w + w] = val;
        }

        __global__ void crop_2d_bias(const float* src, float* dst,
                                      int src_w, int dst_h, int dst_w,
                                      float bias, int fft_h, int fft_w) {
            int h = blockIdx.y * blockDim.y + threadIdx.y;
            int w = blockIdx.x * blockDim.x + threadIdx.x;
            if (h >= dst_h || w >= dst_w) return;
            float val = src[h*src_w + w] / (float)(fft_h * fft_w);
            dst[h*dst_w + w] = val + bias;
        }

        __global__ void complex_mul_accum(const cufftComplex* a,
                                          const cufftComplex* b,
                                          cufftComplex* c, size_t n) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;
            cufftComplex r;
            r.x = a[idx].x * b[idx].x - a[idx].y * b[idx].y;
            r.y = a[idx].x * b[idx].y + a[idx].y * b[idx].x;
            c[idx].x += r.x;
            c[idx].y += r.y;
        }

        static void conv2d_forward_fft_gpu(
            const float* input, const float* weights, const float* bias, float* output,
            int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int padding,
            int out_height, int out_width, cudaStream_t stream)
        {
            int req_h = in_height + kernel_size - 1;
            int req_w = in_width  + kernel_size - 1;
            int fft_h = 1; while (fft_h < req_h) fft_h <<= 1;
            int fft_w = 1; while (fft_w < req_w)  fft_w <<= 1;
            if (fft_h < req_h || fft_w < req_w) {
                printf("Invalid FFT dimensions\n");
                return;
            }
            size_t real_size = (size_t)fft_h * fft_w;
            size_t complex_size = (size_t)fft_h * (fft_w/2 + 1);

            cufftHandle plan_fwd, plan_inv;
            CUFFT_CHECK(cufftPlan2d(&plan_fwd, fft_h, fft_w, CUFFT_R2C));
            CUFFT_CHECK(cufftPlan2d(&plan_inv, fft_h, fft_w, CUFFT_C2R));
            CUFFT_CHECK(cufftSetStream(plan_fwd, stream));
            CUFFT_CHECK(cufftSetStream(plan_inv, stream));

            float *d_in_pad, *d_w_pad;
            cufftComplex *d_in_fft, *d_w_fft, *d_prod;
            CUDA_CHECK(cudaMalloc(&d_in_pad, real_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_w_pad, real_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_in_fft, complex_size * sizeof(cufftComplex)));
            CUDA_CHECK(cudaMalloc(&d_w_fft, complex_size * sizeof(cufftComplex)));
            CUDA_CHECK(cudaMalloc(&d_prod, complex_size * sizeof(cufftComplex)));

            CUDA_CHECK(cudaMemsetAsync(d_in_pad, 0, real_size * sizeof(float), stream));
            CUDA_CHECK(cudaMemsetAsync(d_w_pad, 0, real_size * sizeof(float), stream));
            CUDA_CHECK(cudaMemsetAsync(d_in_fft, 0, complex_size * sizeof(cufftComplex), stream));
            CUDA_CHECK(cudaMemsetAsync(d_w_fft, 0, complex_size * sizeof(cufftComplex), stream));
            CUDA_CHECK(cudaMemsetAsync(d_prod, 0, complex_size * sizeof(cufftComplex), stream));

            dim3 block2d(16,16);
            dim3 gridPad((fft_w + 15)/16, (fft_h + 15)/16);
            dim3 gridCrop((out_width +15)/16, (out_height +15)/16);
            int block1d = 256;
            int grid1d = (int)((complex_size + block1d -1)/block1d);

            for (int b = 0; b < batch_size; ++b) {
                for (int oc = 0; oc < out_channels; ++oc) {
                    CUDA_CHECK(cudaMemsetAsync(d_prod, 0, complex_size*sizeof(cufftComplex), stream));
                    for (int ic = 0; ic < in_channels; ++ic) {
                        const float* in_src = input + ((b*in_channels + ic) * in_height * in_width);
                        const float* w_src  = weights + ((oc*in_channels + ic) * kernel_size * kernel_size);
                        CUDA_CHECK(cudaMemsetAsync(d_in_pad, 0, real_size*sizeof(float), stream));
                        CUDA_CHECK(cudaMemsetAsync(d_w_pad, 0, real_size*sizeof(float), stream));
                        pad_2d<<<gridPad, block2d, 0, stream>>>(in_src, d_in_pad, in_height, in_width, fft_h, fft_w);
                        CUDA_CHECK(cudaGetLastError());
                        pad_2d<<<gridPad, block2d, 0, stream>>>(w_src, d_w_pad, kernel_size, kernel_size, fft_h, fft_w);
                        CUDA_CHECK(cudaGetLastError());
                        CUFFT_CHECK(cufftExecR2C(plan_fwd, d_in_pad, d_in_fft));
                        CUFFT_CHECK(cufftExecR2C(plan_fwd, d_w_pad, d_w_fft));
                        complex_mul_accum<<<grid1d, block1d, 0, stream>>>(d_in_fft, d_w_fft, d_prod, complex_size);
                        CUDA_CHECK(cudaGetLastError());
                    }
                    CUFFT_CHECK(cufftExecC2R(plan_inv, d_prod, d_in_pad));
                    float bias_val = bias ? bias[oc] : 0.f;
                    crop_2d_bias<<<gridCrop, block2d, 0, stream>>>(d_in_pad, output + ((b*out_channels + oc)*out_height*out_width), fft_w, out_height, out_width, bias_val, fft_h, fft_w);
                    CUDA_CHECK(cudaGetLastError());
                }
            }

            CUDA_CHECK(cudaFree(d_in_pad));
            CUDA_CHECK(cudaFree(d_w_pad));
            CUDA_CHECK(cudaFree(d_in_fft));
            CUDA_CHECK(cudaFree(d_w_fft));
            CUDA_CHECK(cudaFree(d_prod));
            CUFFT_CHECK(cufftDestroy(plan_fwd));
            CUFFT_CHECK(cufftDestroy(plan_inv));
        }

        #ifdef THOT_WITH_CUDNN
        static void conv2d_forward_cudnn(
            const float* input, const float* weights, const float* bias, float* output,
            int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding,
            int out_height, int out_width, cudaStream_t stream)
        {
            cudnnHandle_t handle;
            cudnnCreate(&handle);
            cudnnSetStream(handle, stream);

            cudnnTensorDescriptor_t in_desc, out_desc, bias_desc;
            cudnnFilterDescriptor_t w_desc;
            cudnnConvolutionDescriptor_t conv_desc;
            cudnnCreateTensorDescriptor(&in_desc);
            cudnnCreateTensorDescriptor(&out_desc);
            cudnnCreateTensorDescriptor(&bias_desc);
            cudnnCreateFilterDescriptor(&w_desc);
            cudnnCreateConvolutionDescriptor(&conv_desc);

            cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                       batch_size, in_channels, in_height, in_width);
            cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                       batch_size, out_channels, out_height, out_width);
            cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                       out_channels, in_channels, kernel_size, kernel_size);
            cudnnSetConvolution2dDescriptor(conv_desc,
                padding, padding, stride, stride, 1, 1,
                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

            cudnnConvolutionFwdAlgo_t algo;
            cudnnGetConvolutionForwardAlgorithm(handle, in_desc, w_desc, conv_desc, out_desc,
                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
            size_t ws_bytes = 0;
            cudnnGetConvolutionForwardWorkspaceSize(handle, in_desc, w_desc, conv_desc, out_desc,
                algo, &ws_bytes);
            void* workspace = nullptr;
            if (ws_bytes > 0) cudaMalloc(&workspace, ws_bytes);

            const float alpha = 1.f, beta = 0.f;
            cudnnConvolutionForward(handle, &alpha, in_desc, input, w_desc, weights,
                                    conv_desc, algo, workspace, ws_bytes,
                                    &beta, out_desc, output);

            if (bias) {
                cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           1, out_channels, 1, 1);
                cudnnAddTensor(handle, &alpha, bias_desc, bias, &alpha, out_desc, output);
            }

            if (workspace) cudaFree(workspace);
            cudnnDestroyConvolutionDescriptor(conv_desc);
            cudnnDestroyFilterDescriptor(w_desc);
            cudnnDestroyTensorDescriptor(in_desc);
            cudnnDestroyTensorDescriptor(out_desc);
            cudnnDestroyTensorDescriptor(bias_desc);
            cudnnDestroy(handle);
        }
#endif // THOT_WITH_CUDNN
        __global__ void conv2d_forward(
            const float* __restrict__ input,
            const float* __restrict__ weights,
            const float* __restrict__ bias,
            float* __restrict__ output,
            int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding,
            int out_height, int out_width)
        {
            const int64_t outHW = (int64_t)out_height * out_width;
            const int64_t inHW  = (int64_t)in_height * in_width;
            const int64_t perB  = (int64_t)out_channels * outHW;
            const int64_t N     = (int64_t)batch_size * perB;

            for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                 idx < N;
                 idx += (int64_t)blockDim.x * gridDim.x)
            {
                int64_t t = idx;
                const int b  = t / perB;           t -= (int64_t)b * perB;
                const int oc = t / outHW;          t -= (int64_t)oc * outHW;
                const int h  = t / out_width;
                const int w  = t - h * out_width;

                float sum = 0.0f;

                // anchor in input
                const int ih0 = h * stride - padding;
                const int iw0 = w * stride - padding;

                // valid filter ranges for borders (case-independent)
                const int kh_min = max(0, -ih0);
                const int kw_min = max(0, -iw0);
                const int kh_max = min(kernel_size, in_height - ih0);
                const int kw_max = min(kernel_size, in_width  - iw0);

                const int64_t inB  = (int64_t)b  * in_channels * inHW;
                const int64_t wOC  = (int64_t)oc * in_channels * kernel_size * kernel_size;

                for (int ic = 0; ic < in_channels; ++ic) {
                    const int64_t inC = inB + (int64_t)ic * inHW;
                    const int64_t wIC = wOC + (int64_t)ic * kernel_size * kernel_size;

                    for (int kh = kh_min; kh < kh_max; ++kh) {
                        const int ih = ih0 + kh;
                        const int64_t inRow = inC + (int64_t)ih * in_width;

                        const int64_t wRow = wIC + (int64_t)kh * kernel_size;
                        for (int kw = kw_min; kw < kw_max; ++kw) {
                            const int iw = iw0 + kw;
                            sum += ro(input + inRow + iw) * ro(weights + wRow + kw);
                        }
                    }
                }

                if (bias) sum += ro(bias + oc);
                output[idx] = sum;
            }
        }

        __global__ void flip_weights_kernel( const float* __restrict__ src, float* __restrict__ dst, int out_channels, int in_channels, int k)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total = out_channels * in_channels * k * k;
            if (idx >= total) return;
            int kw = idx % k;
            int kh = (idx / k) % k;
            int ic = (idx / (k * k)) % in_channels;
            int oc = idx / (k * k * in_channels);
            int src_idx = ((oc * in_channels + ic) * k + (k - 1 - kh)) * k + (k - 1 - kw);
            dst[idx] = src[src_idx];
        }

        __global__ void conv2d_backward_input(
            const float* __restrict__ grad_output,
            const float* __restrict__ weights,
            float* __restrict__ grad_input,
            int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding,
            int out_height, int out_width)
        {
            const int64_t inHW  = (int64_t)in_height * in_width;
            const int64_t outHW = (int64_t)out_height * out_width;
            const int64_t perB  = (int64_t)in_channels * inHW;
            const int64_t N     = (int64_t)batch_size * perB;

            for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                 idx < N;
                 idx += (int64_t)blockDim.x * gridDim.x)
            {
                int64_t t = idx;
                const int b  = t / perB;          t -= (int64_t)b * perB;
                const int ic = t / inHW;          t -= (int64_t)ic * inHW;
                const int h  = t / in_width;
                const int w  = t - h * in_width;

                float sum = 0.0f;

                // oh, ow ranges that hit this (h,w)
                // ih = oh*stride - padding + kh  in [0, in_h-1] and 0<=kh<K => oh in:
                const int oh_start = max(0, div_ceil(h + padding - (kernel_size - 1), stride));
                const int oh_end   = min(out_height - 1, div_floor(h + padding, stride));
                const int ow_start = max(0, div_ceil(w + padding - (kernel_size - 1), stride));
                const int ow_end   = min(out_width  - 1, div_floor(w + padding, stride));

                const int64_t goB = (int64_t)b * out_channels * outHW;

                for (int oc = 0; oc < out_channels; ++oc) {
                    const int64_t goOC = goB + (int64_t)oc * outHW;
                    const int64_t wOC  = (int64_t)oc * in_channels * kernel_size * kernel_size
                                       + (int64_t)ic * kernel_size * kernel_size;

                    for (int oh = oh_start; oh <= oh_end; ++oh) {
                        const int kh = h + padding - oh * stride; // in [0, K-1]
                        const int64_t goRow = goOC + (int64_t)oh * out_width;

                        const int64_t wRow  = wOC + (int64_t)kh * kernel_size;

                        for (int ow = ow_start; ow <= ow_end; ++ow) {
                            const int kw = w + padding - ow * stride; // in [0, K-1]
                            sum += ro(grad_output + goRow + ow) * ro(weights + wRow + kw);
                        }
                    }
                }

                grad_input[idx] = sum;
            }
        }

        __global__ void conv2d_backward_weights(
            const float* __restrict__ input,
            const float* __restrict__ grad_output,
            float* __restrict__ grad_weights,
            int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding,
            int out_height, int out_width)
        {
            const int64_t K2     = (int64_t)kernel_size * kernel_size;
            const int64_t Nw     = (int64_t)out_channels * in_channels * K2;
            const int64_t inHW   = (int64_t)in_height * in_width;
            const int64_t outHW  = (int64_t)out_height * out_width;

            for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                 idx < Nw;
                 idx += (int64_t)blockDim.x * gridDim.x)
            {
                int64_t t = idx;
                const int oc = t / (in_channels * K2); t -= (int64_t)oc * (in_channels * K2);
                const int ic = t / K2;                 t -= (int64_t)ic * K2;
                const int kh = t / kernel_size;
                const int kw = t - kh * kernel_size;

                float sum = 0.0f;

                // oh, ow ranges that make ih,iw in-bounds
                // ih = oh*stride - padding + kh in [0, in_h-1]
                const int oh_start = max(0, div_ceil(        padding - kh, stride));
                const int oh_end   = min(out_height - 1, div_floor(in_height - 1 + padding - kh, stride));
                // iw = ow*stride - padding + kw in [0, in_w-1]
                const int ow_start = max(0, div_ceil(        padding - kw, stride));
                const int ow_end   = min(out_width  - 1, div_floor(in_width  - 1 + padding - kw, stride));

                for (int b = 0; b < batch_size; ++b) {
                    const int64_t inB = (int64_t)b * in_channels * inHW + (int64_t)ic * inHW;
                    const int64_t goB = (int64_t)b * out_channels * outHW + (int64_t)oc * outHW;

                    for (int oh = oh_start; oh <= oh_end; ++oh) {
                        const int ih = oh * stride - padding + kh;
                        const int64_t inRow = inB + (int64_t)ih * in_width;
                        const int64_t goRow = goB + (int64_t)oh * out_width;

                        for (int ow = ow_start; ow <= ow_end; ++ow) {
                            const int iw = ow * stride - padding + kw;
                            sum += ro(input + inRow + iw) * ro(grad_output + goRow + ow);
                        }
                    }
                }

                grad_weights[idx] = sum / static_cast<float>(batch_size);
            }
        }

        __global__ void conv2d_backward_bias(
            const float* __restrict__ grad_output,
            float* __restrict__ grad_bias,
            int batch_size, int out_channels, int out_height, int out_width)
        {
            // One thread per oc, grid-stride over N= B*H*W
            const int oc = blockIdx.x * blockDim.x + threadIdx.x;
            if (oc >= out_channels) return;

            const int64_t outHW = (int64_t)out_height * out_width;
            const int64_t perB  = (int64_t)out_channels * outHW;
            const int64_t N     = (int64_t)batch_size * outHW;

            float sum = 0.0f;
            for (int64_t n = 0; n < N; ++n) {
                const int b  = n / outHW;
                const int64_t goB = (int64_t)b * perB + (int64_t)oc * outHW;
                const int64_t offs = n - (int64_t)b * outHW;
                sum += ro(grad_output + goB + offs);
            }
            grad_bias[oc] = sum / static_cast<float>(batch_size);
        }

        void launchConv2DForward(
            const float* input, const float* weights, const float* bias,
            float* output, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width, int ConvAlgo,
            cudaStream_t stream)
        {
            const int64_t N = (int64_t)batch_size * out_channels * out_height * out_width;
            const int blockSize = 256;
            const int gridSize  = (int)std::min<int64_t>((N + blockSize - 1) / blockSize, 65535);
            if (ConvAlgo == 1) { // Winograde
                conv2d_forward_winograd_gpu(input, weights, bias, output,
                    batch_size, in_channels, in_height, in_width,
                    out_channels, kernel_size, padding, out_height, out_width, stream);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    // Fallback to direct convolution on failure
                    conv2d_forward<<<gridSize, blockSize, 0, stream>>>(
                        input, weights, bias, output,
                        batch_size, in_channels, in_height, in_width,
                        out_channels, kernel_size, stride, padding,
                        out_height, out_width);
                    CUDA_CHECK(cudaGetLastError());
                }
                return;
            } else if (ConvAlgo == 2) { // FFT
                conv2d_forward_fft_gpu(input, weights, bias, output,
                    batch_size, in_channels, in_height, in_width,
                    out_channels, kernel_size, padding, out_height, out_width, stream);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    // Fallback to direct convolution on failure
                    conv2d_forward<<<gridSize, blockSize, 0, stream>>>(
                        input, weights, bias, output,
                        batch_size, in_channels, in_height, in_width,
                        out_channels, kernel_size, stride, padding,
                        out_height, out_width);
                    CUDA_CHECK(cudaGetLastError());
                }

                return;
            }
#ifdef THOT_WITH_CUDNN
            else if (ConvAlgo == 3) { // cuDNN
                conv2d_forward_cudnn(input, weights, bias, output,
                    batch_size, in_channels, in_height, in_width,
                    out_channels, kernel_size, stride, padding,
                    out_height, out_width, stream);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    // Fallback to direct convolution on failure
                    conv2d_forward<<<gridSize, blockSize, 0, stream>>>(
                        input, weights, bias, output,
                        batch_size, in_channels, in_height, in_width,
                        out_channels, kernel_size, stride, padding,
                        out_height, out_width);
                    CUDA_CHECK(cudaGetLastError());
                }
                return;
            }
#endif
            conv2d_forward<<<gridSize, blockSize, 0, stream>>>(
                input, weights, bias, output,
                batch_size, in_channels, in_height, in_width,
                out_channels, kernel_size, stride, padding,
                out_height, out_width);
            CUDA_CHECK(cudaGetLastError());
        }

        void launchConv2DBackwardInput(
            const float* grad_output, const float* weights,
            float* grad_input, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,  int ConvAlgo,
            cudaStream_t stream)
        {
            if (ConvAlgo == 1 || ConvAlgo == 2) { // winograd or FFT
                int pad = kernel_size - 1 - padding;
                size_t wsize = (size_t)out_channels * in_channels * kernel_size * kernel_size;
                std::vector<float> flipped(wsize);

                float* d_flipped;
                CUDA_CHECK(cudaMalloc(&d_flipped, wsize * sizeof(float)));
                int block = 256;
                int grid = (int)((wsize + block - 1) / block);
                flip_weights_kernel<<<grid, block, 0, stream>>>(weights, d_flipped, out_channels, in_channels, kernel_size);
                CUDA_CHECK(cudaGetLastError());
                if (ConvAlgo == 1) // winograd
                    conv2d_forward_winograd_gpu(grad_output, d_flipped, nullptr, grad_input,
                        batch_size, out_channels, out_height, out_width,
                        in_channels, kernel_size, pad, in_height, in_width, stream);
                else
                    conv2d_forward_fft_gpu(grad_output, d_flipped, nullptr, grad_input,
                        batch_size, out_channels, out_height, out_width,
                        in_channels, kernel_size, pad, in_height, in_width, stream);
                CUDA_CHECK(cudaFree(d_flipped));
                return;
            }
            const int64_t N = (int64_t)batch_size * in_channels * in_height * in_width;
            const int blockSize = 256;
            const int gridSize  = (int)std::min<int64_t>((N + blockSize - 1) / blockSize, 65535);
            conv2d_backward_input<<<gridSize, blockSize, 0, stream>>>(
                grad_output, weights, grad_input,
                batch_size, in_channels, in_height, in_width,
                out_channels, kernel_size, stride, padding,
                out_height, out_width);
            CUDA_CHECK(cudaGetLastError());
        }

        void launchConv2DBackwardWeights(
            const float* input, const float* grad_output,
            float* grad_weights, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width, int ConvAlgo,
            cudaStream_t stream)
        {

            const int64_t N = (int64_t)out_channels * in_channels * kernel_size * kernel_size;
            const int blockSize = 256;
            const int gridSize  = (int)std::min<int64_t>((N + blockSize - 1) / blockSize, 65535);
            (void)ConvAlgo; // placeholder
            conv2d_backward_weights<<<gridSize, blockSize, 0, stream>>>(
                input, grad_output, grad_weights,
                batch_size, in_channels, in_height, in_width,
                out_channels, kernel_size, stride, padding,
                out_height, out_width);
            CUDA_CHECK(cudaGetLastError());
        }

        void launchConv2DBackwardBias(
            const float* grad_output, float* grad_bias,
            int batch_size, int out_channels, int out_height, int out_width,
            cudaStream_t stream)
        {
            const int blockSize = 256;
            const int gridSize  = (out_channels + blockSize - 1) / blockSize;
            conv2d_backward_bias<<<gridSize, blockSize, 0, stream>>>(
                grad_output, grad_bias,
                batch_size, out_channels, out_height, out_width);
            CUDA_CHECK(cudaGetLastError());
        }

    } // namespace layers
} // namespace cuda