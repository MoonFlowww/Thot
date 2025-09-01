#ifndef THOT_FLATTEN_CUH
#define THOT_FLATTEN_CUH
#pragma once

#include <cuda_runtime.h>

namespace cuda {
    namespace layers {
        void launchFlattenForward(const float* input, float* output, int batch_size, int feature_size, cudaStream_t stream = 0);
        void launchFlattenBackward(const float* grad_output, float* grad_input, int batch_size, int feature_size, cudaStream_t stream = 0);
    }
}
#endif //THOT_FLATTEN_CUH