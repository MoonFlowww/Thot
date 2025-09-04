#pragma once
#include <cuda_runtime.h>

namespace cuda {
    namespace optimizations {
        __global__ void adamuon_update(float* weights, const float* gradients,
                               float learning_rate, float beta1, float beta2,
                               float weight_decay, int size);

        void launchAdaMuonUpdate(float* weights, const float* gradients,
                                 float learning_rate, float beta1, float beta2,
                                 float weight_decay, int size, cudaStream_t stream = 0);
    }
}