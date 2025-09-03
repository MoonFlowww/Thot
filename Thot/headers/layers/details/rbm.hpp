#pragma once

#include "../../tensor.hpp"
#include "../../initializations/initializations.hpp"
#include "../../../cuda/cuh/layers/rbm.cuh"
#include "../../activations/activations.hpp"

#include <random>
#include <cmath>
#include <iostream>
#include <utility> 

namespace Thot {

    class Layer;
    class Network;

    class RBMLayer : public Layer {
    private:
        friend class Network;
        int visible_size_;
        int hidden_size_;
        Activation activation_type_;
        Initialization initialization_type_;
        int cd_steps_; // Number of contrastive divergence steps

        Utils::Tensor weights_;
        Utils::Tensor visible_bias_;
        Utils::Tensor hidden_bias_;
        Utils::Tensor grad_weights_;
        Utils::Tensor grad_visible_bias_;
        Utils::Tensor grad_hidden_bias_;

        Utils::Tensor hidden_probs_;     // Probabilities of hidden units being active
        Utils::Tensor hidden_states_;    // Binary states of hidden units
        Utils::Tensor visible_recon_;    // Reconstruction of visible units
        Utils::Tensor hidden_recon_probs_; // Probabilities for the reconstruction

    public:
        RBMLayer(int visible_size, int hidden_size, int cd_steps = 1,
            Activation activation_type = Activation::Sigmoid,
            Initialization weight_init = Initialization::Xavier,
            const std::string& name = "RBM")
            : Layer(name),
            visible_size_(visible_size),
            hidden_size_(hidden_size),
            cd_steps_(cd_steps),
            activation_type_(activation_type),
            initialization_type_(weight_init) {

            // Initialize weights
            weights_ = Utils::Tensor({ hidden_size, visible_size });
            Initializers::initialize_tensor(weights_, weight_init, visible_size, hidden_size);
            grad_weights_ = Utils::Tensor({ hidden_size, visible_size }, true);

            // Initialize visible bias
            visible_bias_ = Utils::Tensor({ visible_size });
            Initializers::zeros(visible_bias_);
            grad_visible_bias_ = Utils::Tensor({ visible_size }, true);

            // Initialize hidden bias
            hidden_bias_ = Utils::Tensor({ hidden_size });
            Initializers::zeros(hidden_bias_);
            grad_hidden_bias_ = Utils::Tensor({ hidden_size }, true);

            // Initialize temporary storage
            hidden_probs_ = Utils::Tensor({ 1, hidden_size }, true);
            hidden_states_ = Utils::Tensor({ 1, hidden_size }, true);
            visible_recon_ = Utils::Tensor({ 1, visible_size }, true);
            hidden_recon_probs_ = Utils::Tensor({ 1, hidden_size }, true);
        }

        size_t get_flops(int batch_size = 1) const override {
            // For each sample, one CD step involves:
            // 1. Visible to hidden: 2 * visible_size * hidden_size + hidden_size FLOPs
            // 2. Hidden to visible: 2 * visible_size * hidden_size + visible_size FLOPs
            // Multiply by CD steps and batch size
            return batch_size * cd_steps_ *
                (2 * visible_size_ * hidden_size_ + hidden_size_ +
                    2 * visible_size_ * hidden_size_ + visible_size_);
        }

        Activation get_activation() const override {
            return activation_type_;
        }

        Initialization get_initialization() const override {
            return initialization_type_;
        }

        Utils::Tensor forward(const Utils::Tensor& input) override {
            int batch_size = input.shape()[0];

            this->input_cache_ = Utils::Tensor(input.shape());
            float* src_ptr = static_cast<float*>(input.data());
            float* dst_ptr = static_cast<float*>(this->input_cache_.data());
            size_t size = input.size() * sizeof(float);
            ::cudaMemcpy(dst_ptr, src_ptr, size, ::cudaMemcpyDeviceToDevice);

            if (hidden_probs_.shape()[0] != batch_size) {
                hidden_probs_ = Utils::Tensor({ batch_size, hidden_size_ }, true);
                hidden_states_ = Utils::Tensor({ batch_size, hidden_size_ }, true);
                visible_recon_ = Utils::Tensor({ batch_size, visible_size_ }, true);
                hidden_recon_probs_ = Utils::Tensor({ batch_size, hidden_size_ }, true);
            }

            Utils::Tensor output({ batch_size, hidden_size_ });

            float* visible_ptr = static_cast<float*>(this->input_cache_.data());
            float* weights_ptr = static_cast<float*>(weights_.data());
            float* hidden_bias_ptr = static_cast<float*>(hidden_bias_.data());
            float* hidden_probs_ptr = static_cast<float*>(hidden_probs_.data());
            float* hidden_states_ptr = static_cast<float*>(hidden_states_.data());

            ::cuda::layers::launchRBMVisibleToHidden(
                visible_ptr, weights_ptr, hidden_bias_ptr,
                hidden_probs_ptr, hidden_states_ptr,
                batch_size, visible_size_, hidden_size_
            );

            // Copy hidden states to output
            float* output_ptr = static_cast<float*>(output.data());
            size_t hidden_size = batch_size * hidden_size_ * sizeof(float);
            ::cudaMemcpy(output_ptr, hidden_states_ptr, hidden_size, ::cudaMemcpyDeviceToDevice);

            return std::move(output);
        }

        Utils::Tensor backward(const Utils::Tensor& grad_output) override {
            int batch_size = this->input_cache_.shape()[0];

            // Note: For RBMs, we typically don't use the grad_output for training
            // Instead, we perform contrastive divergence (CD) algorithm

            Utils::Tensor grad_visible({ batch_size, visible_size_ });

            float* visible_ptr = static_cast<float*>(this->input_cache_.data());
            float* weights_ptr = static_cast<float*>(weights_.data());
            float* visible_bias_ptr = static_cast<float*>(visible_bias_.data());
            float* hidden_bias_ptr = static_cast<float*>(hidden_bias_.data());
            float* hidden_probs_ptr = static_cast<float*>(hidden_probs_.data());
            float* hidden_states_ptr = static_cast<float*>(hidden_states_.data());
            float* visible_recon_ptr = static_cast<float*>(visible_recon_.data());
            float* hidden_recon_probs_ptr = static_cast<float*>(hidden_recon_probs_.data());
            float* grad_weights_ptr = static_cast<float*>(grad_weights_.data());
            float* grad_visible_bias_ptr = static_cast<float*>(grad_visible_bias_.data());
            float* grad_hidden_bias_ptr = static_cast<float*>(grad_hidden_bias_.data());

            // Contrastive Divergence step - reconstruct visible units from hidden states
            ::cuda::layers::launchRBMHiddenToVisible(
                hidden_states_ptr, weights_ptr, visible_bias_ptr,
                visible_recon_ptr, visible_recon_ptr, // Use same buffer for probs and states
                batch_size, visible_size_, hidden_size_
            );

            // Calculate activation of hidden units from the reconstruction
            ::cuda::layers::launchRBMVisibleToHidden(
                visible_recon_ptr, weights_ptr, hidden_bias_ptr,
                hidden_recon_probs_ptr, hidden_recon_probs_ptr, // Use same buffer for probs and states
                batch_size, visible_size_, hidden_size_
            );

            ::cuda::layers::launchRBMComputeGradients(
                visible_ptr, visible_recon_ptr,
                hidden_probs_ptr, hidden_recon_probs_ptr,
                grad_weights_ptr, grad_visible_bias_ptr, grad_hidden_bias_ptr,
                batch_size, visible_size_, hidden_size_
            );

            if (this->optimizer_) {
                this->optimizer_->update(weights_, grad_weights_);
                this->optimizer_->update(visible_bias_, grad_visible_bias_);
                this->optimizer_->update(hidden_bias_, grad_hidden_bias_);
            }
            else { // SGD by default
                // Update w
                const int weights_size = weights_.size();
                for (int i = 0; i < weights_size; ++i) {
                    float* w_ptr = static_cast<float*>(weights_.data());
                    float* gw_ptr = static_cast<float*>(grad_weights_.data());
                    w_ptr[i] -= this->optimizer_->get_learning_rate() * gw_ptr[i];
                }

                // Update visible b
                const int visible_bias_size = visible_bias_.size();
                for (int i = 0; i < visible_bias_size; ++i) {
                    float* b_ptr = static_cast<float*>(visible_bias_.data());
                    float* gb_ptr = static_cast<float*>(grad_visible_bias_.data());
                    b_ptr[i] -= this->optimizer_->get_learning_rate() * gb_ptr[i];
                }

                // Update hidden b
                const int hidden_bias_size = hidden_bias_.size();
                for (int i = 0; i < hidden_bias_size; ++i) {
                    float* b_ptr = static_cast<float*>(hidden_bias_.data());
                    float* gb_ptr = static_cast<float*>(grad_hidden_bias_.data());
                    b_ptr[i] -= this->optimizer_->get_learning_rate() * gb_ptr[i];
                }
            }

            return std::move(grad_visible);
        }

        int get_input_size() const override { return visible_size_; }
        int get_output_size() const override { return hidden_size_; }

        int get_cd_steps() const { return cd_steps_; }
        Utils::Tensor& weights() { return weights_; }
        const Utils::Tensor& weights() const { return weights_; }
        Utils::Tensor& visible_bias() { return visible_bias_; }
        const Utils::Tensor& visible_bias() const { return visible_bias_; }
        Utils::Tensor& hidden_bias() { return hidden_bias_; }
        const Utils::Tensor& hidden_bias() const { return hidden_bias_; }
    };
} // namespace Thot 