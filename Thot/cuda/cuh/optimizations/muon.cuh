#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
namespace cuda {
    namespace optimizations {
        __global__ void muon_update(float* weights, const float* gradients,
                            float learning_rate, float beta, float weight_decay,
                            int size);
        void launchMuonUpdate(float* weights, const float* gradients,
                              float learning_rate, float beta, float weight_decay,
                              int size, cudaStream_t stream = 0);
    }
}