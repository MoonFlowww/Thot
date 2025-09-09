#pragma once

#include <cuda_runtime.h>

namespace cuda::layernorm {

    void launchRMSForward(const float *input, float *output, int rows, int cols,
                          cudaStream_t stream = 0);
    void launchRMSBackward(const float *input, const float *grad_output,
                           float *grad_input, int rows, int cols,
                           cudaStream_t stream = 0);

    void launchDyTForward(const float *input, float *output, int rows, int cols,
                          cudaStream_t stream = 0);
    void launchDyTBackward(const float *input, const float *output,
                           const float *grad_output, float *grad_input, int rows,
                           int cols, cudaStream_t stream = 0);

} // namespace cuda::layernorm