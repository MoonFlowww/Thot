#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include "../../cuh/layers/rcnn.cuh"

namespace cuda {
    namespace layers {

        __global__ void roi_pool_forward(const float* input, const float* rois, float* output,
            int num_rois, int channels, int height, int width, int pooled_h, int pooled_w) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int total = num_rois * channels * pooled_h * pooled_w;
            if (index >= total) return;
            int pw = index % pooled_w;
            int ph = (index / pooled_w) % pooled_h;
            int c = (index / (pooled_w * pooled_h)) % channels;
            int n = index / (pooled_w * pooled_h * channels);

            const float* roi = rois + n * 5;
            int batch = static_cast<int>(roi[0]);
            float x1 = roi[1];
            float y1 = roi[2];
            float x2 = roi[3];
            float y2 = roi[4];
            float roi_w = fmaxf(x2 - x1, 1.0f);
            float roi_h = fmaxf(y2 - y1, 1.0f);
            float bin_w = roi_w / pooled_w;
            float bin_h = roi_h / pooled_h;

            int hstart = static_cast<int>(floorf(y1 + ph * bin_h));
            int wstart = static_cast<int>(floorf(x1 + pw * bin_w));
            int hend = static_cast<int>(ceilf(y1 + (ph + 1) * bin_h));
            int wend = static_cast<int>(ceilf(x1 + (pw + 1) * bin_w));

            hstart = min(max(hstart, 0), height);
            hend = min(max(hend, 0), height);
            wstart = min(max(wstart, 0), width);
            wend = min(max(wend, 0), width);

            float maxval = -FLT_MAX;
            const float* input_ptr = input + (batch * channels + c) * height * width;
            for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                    int idx = h * width + w;
                    float val = input_ptr[idx];
                    if (val > maxval) maxval = val;
                }
            }
            if (hend <= hstart || wend <= wstart) maxval = 0.0f;
            output[index] = maxval;
        }

        __global__ void roi_pool_backward(const float* grad_output, const float* input, const float* rois,
            float* grad_input, int num_rois, int channels, int height, int width, int pooled_h, int pooled_w) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int total = num_rois * channels * pooled_h * pooled_w;
            if (index >= total) return;
            int pw = index % pooled_w;
            int ph = (index / pooled_w) % pooled_h;
            int c = (index / (pooled_w * pooled_h)) % channels;
            int n = index / (pooled_w * pooled_h * channels);

            const float* roi = rois + n * 5;
            int batch = static_cast<int>(roi[0]);
            float x1 = roi[1];
            float y1 = roi[2];
            float x2 = roi[3];
            float y2 = roi[4];
            float roi_w = fmaxf(x2 - x1, 1.0f);
            float roi_h = fmaxf(y2 - y1, 1.0f);
            float bin_w = roi_w / pooled_w;
            float bin_h = roi_h / pooled_h;

            int hstart = static_cast<int>(floorf(y1 + ph * bin_h));
            int wstart = static_cast<int>(floorf(x1 + pw * bin_w));
            int hend = static_cast<int>(ceilf(y1 + (ph + 1) * bin_h));
            int wend = static_cast<int>(ceilf(x1 + (pw + 1) * bin_w));

            hstart = min(max(hstart, 0), height);
            hend = min(max(hend, 0), height);
            wstart = min(max(wstart, 0), width);
            wend = min(max(wend, 0), width);

            const float* input_ptr = input + (batch * channels + c) * height * width;
            int maxh = hstart;
            int maxw = wstart;
            float maxval = -FLT_MAX;
            for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                    int idx = h * width + w;
                    float val = input_ptr[idx];
                    if (val > maxval) {
                        maxval = val;
                        maxh = h; maxw = w;
                    }
                }
            }
            if (hend <= hstart || wend <= wstart) return;
            float g = grad_output[index];
            float* grad_ptr = grad_input + (batch * channels + c) * height * width;
            atomicAdd(&grad_ptr[maxh * width + maxw], g);
        }

        void launchROIPoolForward(const float* input, const float* rois, float* output,
            int num_rois, int channels, int height, int width, int pooled_h, int pooled_w,
            cudaStream_t stream) {
            int total = num_rois * channels * pooled_h * pooled_w;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            roi_pool_forward<<<blocks, threads, 0, stream>>>(input, rois, output, num_rois, channels, height, width, pooled_h, pooled_w);
        }

        void launchROIPoolBackward(const float* grad_output, const float* input, const float* rois,
            float* grad_input, int num_rois, int channels, int height, int width, int pooled_h, int pooled_w,
            cudaStream_t stream) {
            int total = num_rois * channels * pooled_h * pooled_w;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            roi_pool_backward<<<blocks, threads, 0, stream>>>(grad_output, input, rois, grad_input, num_rois, channels, height, width, pooled_h, pooled_w);
        }
    }
}