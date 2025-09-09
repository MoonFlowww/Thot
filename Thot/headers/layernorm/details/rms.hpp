#pragma once

#include "../../../cuda/cuh/layernorm/layernorm.cuh"
#include "../../initializations/initializations.hpp"
#include "../../layers/layers.hpp"
#include "../../tensor.hpp"
#include <string>

namespace Thot {

    class RMSLayerNormLayer : public Layer {
    private:
        int normalized_size_;
        float epsilon_;

    public:
        RMSLayerNormLayer(int normalized_size, const std::string &name = "RMSLayerNorm", float epsilon = 1e-5f) : Layer(name), normalized_size_(normalized_size), epsilon_(epsilon) {}

        Utils::Tensor forward(const Utils::Tensor &input) override {
            this->input_cache_ = Utils::Tensor(input.shape());
            ::cudaMemcpy(this->input_cache_.data(), input.data(), input.bytes(),
                     ::cudaMemcpyDeviceToDevice);

            int rows = static_cast<int>(input.size() / normalized_size_);
            Utils::Tensor output(input.shape());
            cuda::layernorm::launchRMSForward(
                static_cast<const float *>(this->input_cache_.data()),
                static_cast<float *>(output.data()), rows, normalized_size_);
            return output;
        }

        Utils::Tensor backward(const Utils::Tensor &grad_output) override {
            int rows = static_cast<int>(grad_output.size() / normalized_size_);
            Utils::Tensor grad_input(grad_output.shape());
            cuda::layernorm::launchRMSBackward(
                static_cast<const float *>(this->input_cache_.data()),
                static_cast<const float *>(grad_output.data()),
                static_cast<float *>(grad_input.data()), rows, normalized_size_);
            return grad_input;
        }

        size_t get_flops(int batch_size = 1) const override {
            return static_cast<size_t>(batch_size) * normalized_size_ * 4;
        }

        size_t get_parameters() const override { return 0; }

        Activation get_activation() const override { return Activation::Linear; }

        Initialization get_initialization() const override {
            return Initialization::Xavier;
        }

        float get_latency() const override { return -1.0f; }

        int get_input_size() const override { return normalized_size_; }

        int get_output_size() const override { return normalized_size_; }
    };

} // namespace Thot