#pragma once

#include <cuda_runtime.h>

namespace cuda {
    namespace layers {
        __global__ void roi_pool_forward(const float* input, const float* rois, float* output,
            int num_rois, int channels, int height, int width, int pooled_h, int pooled_w);

        __global__ void roi_pool_backward(const float* grad_output, const float* input, const float* rois,
            float* grad_input, int num_rois, int channels, int height, int width, int pooled_h, int pooled_w);

        void launchROIPoolForward(const float* input, const float* rois, float* output,
            int num_rois, int channels, int height, int width, int pooled_h, int pooled_w,
            cudaStream_t stream = 0);

        void launchROIPoolBackward(const float* grad_output, const float* input, const float* rois,
            float* grad_input, int num_rois, int channels, int height, int width, int pooled_h, int pooled_w,
            cudaStream_t stream = 0);
    }
}