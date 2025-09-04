#pragma once

#include "../../tensor.hpp"
#include "../../activations/activations.hpp"
#include "../../initializations/initializations.hpp"
#include "../../../cuda/cuh/layers/spike.cuh"

#include <vector>

namespace Thot {

    class Layer;
    class Network;

    // Simple spiking neuron layer maintaining membrane potentials.
    class SpikeLayer : public Layer {
    private:
        friend class Network;
        int size_;
        float threshold_;

        // Membrane potential state per neuron
        Utils::Tensor membrane_; 
        // Store last spike output for backpropagation
        Utils::Tensor last_spikes_;

    public:
        SpikeLayer(int size, float threshold = 1.0f, const std::string& name = "Spike")
            : Layer(name), size_(size), threshold_(threshold) {}

        // Reset membrane potentials (used on mode switches)
        void reset_state() {
            if (membrane_.size() > 0) {
                std::vector<float> zeros(membrane_.size(), 0.0f);
                membrane_.upload(zeros);
            }
        }

        int get_input_size() const override { return size_; }
        int get_output_size() const override { return size_; }
        size_t get_parameters() const override { return 0; }
        size_t get_flops(int batch_size = 1) const override { return batch_size * size_ * 2; }
        Activation get_activation() const override { return Activation::Linear; }
        Initialization get_initialization() const override { return Initialization::Zeros; }

        Utils::Tensor forward(const Utils::Tensor& input) override {
            int batch = input.shape()[0];
            int neurons = input.shape()[1];
            if (membrane_.size() == 0 || membrane_.shape()[0] != batch) {
                membrane_ = Utils::Tensor({batch, neurons}, true); // zero init
            }
            Utils::Tensor output({batch, neurons});

            float* in_ptr = static_cast<float*>(input.data());
            float* mem_ptr = static_cast<float*>(membrane_.data());
            float* out_ptr = static_cast<float*>(output.data());

            ::cuda::layers::launchSpikeForward(in_ptr, mem_ptr, out_ptr, batch, neurons, threshold_);

            // cache spikes for backward
            last_spikes_ = Utils::Tensor({batch, neurons});
            ::cudaMemcpy(last_spikes_.data(), out_ptr, output.size()*sizeof(float), ::cudaMemcpyDeviceToDevice);

            return output;
        }

        Utils::Tensor backward(const Utils::Tensor& grad_output) override {
            int batch = grad_output.shape()[0];
            int neurons = grad_output.shape()[1];
            Utils::Tensor grad_input({batch, neurons});

            float* spike_ptr = static_cast<float*>(last_spikes_.data());
            float* grad_out_ptr = static_cast<float*>(grad_output.data());
            float* grad_in_ptr = static_cast<float*>(grad_input.data());

            ::cuda::layers::launchSpikeBackward(spike_ptr, grad_out_ptr, grad_in_ptr, batch, neurons);
            return grad_input;
        }
    };

} // namespace Thot