#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include "../../cuh/layers/conv2d.cuh"

#ifdef THOT_WITH_CUDNN
#include <cudnn.h>
#endif

namespace cuda {
    namespace layers {

        // Transform input into column matrix
        __global__ void im2col_kernel(const float* input, float* col,
            int batch_size, int channels, int height, int width,
            int kernel_size, int stride, int padding,
            int out_height, int out_width) {
            int K = channels * kernel_size * kernel_size;
            int N = batch_size * out_height * out_width;
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= K * N) return;

            int n = idx / K;
            int k = idx % K;
            int b = n / (out_height * out_width);
            int oh = (n / out_width) % out_height;
            int ow = n % out_width;

            int c = k / (kernel_size * kernel_size);
            int kh = (k / kernel_size) % kernel_size;
            int kw = k % kernel_size;

            int ih = oh * stride - padding + kh;
            int iw = ow * stride - padding + kw;
            float val = 0.0f;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                int in_idx = b * (channels * height * width) +
                    c * (height * width) +
                    ih * width +
                    iw;
                val = input[in_idx];
            }
            col[k * N + n] = val;
        }

        __global__ void add_bias(float* C, const float* bias, int M, int N) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= M * N) return;
            int oc = idx / N;
            C[idx] += bias[oc];
        }

        void launchConv2DForwardIm2Col(const float* input, const float* weights, const float* bias,
            float* output, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,
            cudaStream_t stream) {

            int K = in_channels * kernel_size * kernel_size;
            int N = batch_size * out_height * out_width;
            size_t col_size = static_cast<size_t>(K) * N * sizeof(float);
            float* col;
            cudaMalloc(&col, col_size);

            int threads = 256;
            int blocks = (K * N + threads - 1) / threads;
            im2col_kernel<<<blocks, threads, 0, stream>>>(input, col,
                batch_size, in_channels, in_height, in_width,
                kernel_size, stride, padding,
                out_height, out_width);

            cublasHandle_t handle;
            cublasCreate(&handle);
            cublasSetStream(handle, stream);

            const float alpha = 1.0f;
            const float beta = 0.0f;
            // weights: [out_channels, K], col: [K, N], output: [out_channels, N]
            cublasStatus_t stat = cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, out_channels, K,
                &alpha,
                col, N,
                weights, K,
                &beta,
                output, N);
            if (stat != CUBLAS_STATUS_SUCCESS) {
                printf("cuBLAS sgemm failed in launchConv2DForwardIm2Col\n");
            }

            if (bias != nullptr) {
                int total = out_channels * N;
                int blockB = 256;
                int gridB = (total + blockB - 1) / blockB;
                add_bias<<<gridB, blockB, 0, stream>>>(output, bias, out_channels, N);
            }

            cublasDestroy(handle);
            cudaFree(col);
            cudaDeviceSynchronize();
        }

#ifdef THOT_WITH_CUDNN
        void launchConv2DForwardCuDNN(const float* input, const float* weights, const float* bias,
            float* output, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,
            cudaStream_t stream) {

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
            cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                1, out_channels, 1, 1);
            cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                out_channels, in_channels, kernel_size, kernel_size);
            cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride,
                1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

            cudnnConvolutionFwdAlgo_t algo;
            cudnnGetConvolutionForwardAlgorithm(handle, in_desc, w_desc, conv_desc, out_desc,
                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);

            size_t ws_size = 0;
            cudnnGetConvolutionForwardWorkspaceSize(handle, in_desc, w_desc, conv_desc, out_desc,
                algo, &ws_size);
            void* workspace = nullptr;
            if (ws_size > 0) cudaMalloc(&workspace, ws_size);

            const float alpha = 1.0f;
            const float beta = 0.0f;
            cudnnConvolutionForward(handle, &alpha,
                in_desc, input,
                w_desc, weights,
                conv_desc, algo,
                workspace, ws_size,
                &beta,
                out_desc, output);

            if (bias != nullptr) {
                cudnnAddTensor(handle, &alpha, bias_desc, bias, &alpha, out_desc, output);
            }

            if (workspace) cudaFree(workspace);
            cudnnDestroyTensorDescriptor(in_desc);
            cudnnDestroyTensorDescriptor(out_desc);
            cudnnDestroyTensorDescriptor(bias_desc);
            cudnnDestroyFilterDescriptor(w_desc);
            cudnnDestroyConvolutionDescriptor(conv_desc);
            cudnnDestroy(handle);
            cudaDeviceSynchronize();
        }

        void launchConv2DBackwardInputCuDNN(const float* grad_output, const float* weights,
            float* grad_input, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,
            cudaStream_t stream) {

            cudnnHandle_t handle;
            cudnnCreate(&handle);
            cudnnSetStream(handle, stream);

            cudnnTensorDescriptor_t grad_out_desc, grad_in_desc;
            cudnnFilterDescriptor_t w_desc;
            cudnnConvolutionDescriptor_t conv_desc;

            cudnnCreateTensorDescriptor(&grad_out_desc);
            cudnnCreateTensorDescriptor(&grad_in_desc);
            cudnnCreateFilterDescriptor(&w_desc);
            cudnnCreateConvolutionDescriptor(&conv_desc);

            cudnnSetTensor4dDescriptor(grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                batch_size, out_channels, out_height, out_width);
            cudnnSetTensor4dDescriptor(grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                batch_size, in_channels, in_height, in_width);
            cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                out_channels, in_channels, kernel_size, kernel_size);
            cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride,
                1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

            cudnnConvolutionBwdDataAlgo_t algo;
            cudnnGetConvolutionBackwardDataAlgorithm(handle, w_desc, grad_out_desc, conv_desc, grad_in_desc,
                CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &algo);
            size_t ws_size = 0;
            cudnnGetConvolutionBackwardDataWorkspaceSize(handle, w_desc, grad_out_desc, conv_desc, grad_in_desc,
                algo, &ws_size);
            void* workspace = nullptr;
            if (ws_size > 0) cudaMalloc(&workspace, ws_size);

            const float alpha = 1.0f;
            const float beta = 0.0f;
            cudnnConvolutionBackwardData(handle, &alpha,
                w_desc, weights,
                grad_out_desc, grad_output,
                conv_desc, algo,
                workspace, ws_size,
                &beta,
                grad_in_desc, grad_input);

            if (workspace) cudaFree(workspace);
            cudnnDestroyTensorDescriptor(grad_out_desc);
            cudnnDestroyTensorDescriptor(grad_in_desc);
            cudnnDestroyFilterDescriptor(w_desc);
            cudnnDestroyConvolutionDescriptor(conv_desc);
            cudnnDestroy(handle);
            cudaDeviceSynchronize();
        }

        void launchConv2DBackwardWeightsCuDNN(const float* input, const float* grad_output,
            float* grad_weights, int batch_size, int in_channels, int in_height, int in_width,
            int out_channels, int kernel_size, int stride, int padding, int out_height, int out_width,
            cudaStream_t stream) {

            cudnnHandle_t handle;
            cudnnCreate(&handle);
            cudnnSetStream(handle, stream);

            cudnnTensorDescriptor_t grad_out_desc, in_desc;
            cudnnFilterDescriptor_t grad_w_desc;
            cudnnConvolutionDescriptor_t conv_desc;

            cudnnCreateTensorDescriptor(&grad_out_desc);
            cudnnCreateTensorDescriptor(&in_desc);
            cudnnCreateFilterDescriptor(&grad_w_desc);
            cudnnCreateConvolutionDescriptor(&conv_desc);

            cudnnSetTensor4dDescriptor(grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                batch_size, out_channels, out_height, out_width);
            cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                batch_size, in_channels, in_height, in_width);
            cudnnSetFilter4dDescriptor(grad_w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                out_channels, in_channels, kernel_size, kernel_size);
            cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride,
                1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

            cudnnConvolutionBwdFilterAlgo_t algo;
            cudnnGetConvolutionBackwardFilterAlgorithm(handle, in_desc, grad_out_desc, conv_desc, grad_w_desc,
                CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &algo);
            size_t ws_size = 0;
            cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, in_desc, grad_out_desc, conv_desc, grad_w_desc,
                algo, &ws_size);
            void* workspace = nullptr;
            if (ws_size > 0) cudaMalloc(&workspace, ws_size);

            const float alpha = 1.0f;
            const float beta = 0.0f;
            cudnnConvolutionBackwardFilter(handle, &alpha,
                in_desc, input,
                grad_out_desc, grad_output,
                conv_desc, algo,
                workspace, ws_size,
                &beta,
                grad_w_desc, grad_weights);

            if (workspace) cudaFree(workspace);
            cudnnDestroyTensorDescriptor(grad_out_desc);
            cudnnDestroyTensorDescriptor(in_desc);
            cudnnDestroyFilterDescriptor(grad_w_desc);
            cudnnDestroyConvolutionDescriptor(conv_desc);
            cudnnDestroy(handle);
            cudaDeviceSynchronize();
        }

        void launchConv2DBackwardBiasCuDNN(const float* grad_output, float* grad_bias,
            int batch_size, int out_channels, int out_height, int out_width,
            cudaStream_t stream) {

            cudnnHandle_t handle;
            cudnnCreate(&handle);
            cudnnSetStream(handle, stream);

            cudnnTensorDescriptor_t grad_out_desc, bias_desc;
            cudnnCreateTensorDescriptor(&grad_out_desc);
            cudnnCreateTensorDescriptor(&bias_desc);

            cudnnSetTensor4dDescriptor(grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                batch_size, out_channels, out_height, out_width);
            cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                1, out_channels, 1, 1);

            const float alpha = 1.0f;
            const float beta = 0.0f;
            cudnnConvolutionBackwardBias(handle, &alpha,
                grad_out_desc, grad_output,
                &beta,
                bias_desc, grad_bias);

            cudnnDestroyTensorDescriptor(grad_out_desc);
            cudnnDestroyTensorDescriptor(bias_desc);
            cudnnDestroy(handle);
            cudaDeviceSynchronize();
        }
#endif // THOT_WITH_CUDNN

    }
}
