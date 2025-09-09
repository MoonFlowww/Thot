#pragma once

#include <memory>
#include <string>

#include "../../cuda/cuh/layernorm/layernorm.cuh"
#include "../layers/layers.hpp"

namespace Thot {
    class Layer;
    namespace LayerNorm {
        std::shared_ptr<Layer> RMS(int normalized_size,
                                   const std::string &name = "RMSLayerNorm");
        std::shared_ptr<Layer> DyT(int normalized_size,
                                   const std::string &name = "DyTLayerNorm");

        void RMSLayerNorm(const float *input, float *output, int rows, int cols);
        void DyTLayerNorm(const float *input, float *output, int rows, int cols);
    } // namespace LayerNorm
} // namespace Thot

#include "details/dyt.hpp"
#include "details/rms.hpp"

namespace Thot {
    namespace LayerNorm {
        inline std::shared_ptr<Layer> RMS(int normalized_size, const std::string &name) {
            return std::make_shared<RMSLayerNormLayer>(normalized_size, name);
        }
        inline std::shared_ptr<Layer> DyT(int normalized_size, const std::string &name) {
            return std::make_shared<DyTLayerNormLayer>(normalized_size, name);
        }
        inline void RMSLayerNorm(const float *input, float *output, int rows, int cols) {
            cuda::layernorm::launchRMSForward(input, output, rows, cols);
        }
        inline void DyTLayerNorm(const float *input, float *output, int rows, int cols) {
            cuda::layernorm::launchDyTForward(input, output, rows, cols);
        }
    } // namespace LayerNorm
} // namespace Thot
