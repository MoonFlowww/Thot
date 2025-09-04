#pragma once

#include "../../tensor.hpp"
#include "../../initializations/initializations.hpp"
#include "../../activations/activations.hpp"
#include "../../optimizations/optimizations.hpp"

#include "../../../cuda/cuh/layers/fc.cuh"
#include "../../../cuda/cuh/layers/sparse_contractive_ae.cuh"

#include <string>

namespace Thot {

class SparseContractiveAELayer : public Layer {
private:
    friend class Network;
    int input_size_;
    int latent_size_;
    Activation activation_type_;
    Initialization initialization_type_;
    bool use_sparsity_;
    bool use_contractive_;
    float sparsity_rho_;
    float sparsity_beta_;
    float contractive_lambda_;

    Utils::Tensor enc_weights_;
    Utils::Tensor dec_weights_;
    Utils::Tensor enc_bias_;
    Utils::Tensor dec_bias_;

    Utils::Tensor grad_enc_weights_;
    Utils::Tensor grad_dec_weights_;
    Utils::Tensor grad_enc_bias_;
    Utils::Tensor grad_dec_bias_;

    Utils::Tensor pre_latent_;
    Utils::Tensor latent_;

    float regularization_loss_;

public:
    SparseContractiveAELayer(int input_size, int latent_size,
                             Activation activation = Activation::Sigmoid,
                             Initialization weight_init = Initialization::Xavier,
                             bool use_sparsity = false,
                             bool use_contractive = false,
                             float sparsity_rho = 0.05f,
                             float sparsity_beta = 1e-3f,
                             float contractive_lambda = 1e-3f,
                             const std::string &name = "SparseAE")
        : Layer(name),
          input_size_(input_size),
          latent_size_(latent_size),
          activation_type_(activation),
          initialization_type_(weight_init),
          use_sparsity_(use_sparsity),
          use_contractive_(use_contractive),
          sparsity_rho_(sparsity_rho),
          sparsity_beta_(sparsity_beta),
          contractive_lambda_(contractive_lambda),
          enc_weights_({latent_size, input_size}),
          dec_weights_({input_size, latent_size}),
          enc_bias_({latent_size}),
          dec_bias_({input_size}),
          grad_enc_weights_({latent_size, input_size}, true),
          grad_dec_weights_({input_size, latent_size}, true),
          grad_enc_bias_({latent_size}, true),
          grad_dec_bias_({input_size}, true),
          regularization_loss_(0.0f) {
        Initializations::initialize_tensor(enc_weights_, weight_init, input_size, latent_size);
        Initializations::initialize_tensor(dec_weights_, weight_init, latent_size, input_size);
        Initializations::zeros(enc_bias_);
        Initializations::zeros(dec_bias_);
    }

    Utils::Tensor forward(const Utils::Tensor &input) override {
        this->input_cache_ = Utils::Tensor(input.shape());
        ::cudaMemcpy(this->input_cache_.data(), input.data(), input.size() * sizeof(float), ::cudaMemcpyDeviceToDevice);
        int batch_size = input.shape()[0];

        Utils::Tensor enc_pre({batch_size, latent_size_});
        ::cuda::layers::launchFCForward(static_cast<float *>(this->input_cache_.data()),
                                        static_cast<float *>(enc_weights_.data()),
                                        static_cast<float *>(enc_bias_.data()),
                                        static_cast<float *>(enc_pre.data()), batch_size, input_size_, latent_size_);

        pre_latent_ = Utils::Tensor(enc_pre.shape());
        ::cudaMemcpy(pre_latent_.data(), enc_pre.data(), enc_pre.size() * sizeof(float), ::cudaMemcpyDeviceToDevice);

        Utils::Tensor enc_act({batch_size, latent_size_});
        Activations::apply_activation(enc_pre, enc_act, activation_type_);

        latent_ = Utils::Tensor(enc_act.shape());
        ::cudaMemcpy(latent_.data(), enc_act.data(), enc_act.size() * sizeof(float), ::cudaMemcpyDeviceToDevice);

        Utils::Tensor recon({batch_size, input_size_});
        ::cuda::layers::launchFCForward(static_cast<float *>(enc_act.data()),
                                        static_cast<float *>(dec_weights_.data()),
                                        static_cast<float *>(dec_bias_.data()),
                                        static_cast<float *>(recon.data()), batch_size, latent_size_, input_size_);

        regularization_loss_ = 0.0f;
        if (use_sparsity_) {
            float kl = ::cuda::layers::computeKLSparsity(static_cast<float *>(latent_.data()),
                                                         batch_size, latent_size_, sparsity_rho_);
            regularization_loss_ += sparsity_beta_ * kl;
        }
        if (use_contractive_) {
            float contr = ::cuda::layers::computeContractiveLoss(static_cast<float *>(latent_.data()),
                                                                 static_cast<float *>(enc_weights_.data()),
                                                                 batch_size, input_size_, latent_size_);
            regularization_loss_ += contractive_lambda_ * contr;
        }

        return recon;
    }

    Utils::Tensor backward(const Utils::Tensor &grad_output) override {
        int batch_size = grad_output.shape()[0];

        Utils::Tensor grad_latent({batch_size, latent_size_});
        Utils::Tensor grad_input({batch_size, input_size_});

        ::cuda::layers::launchFCBackwardInput(static_cast<float *>(grad_output.data()),
                                              static_cast<float *>(dec_weights_.data()),
                                              static_cast<float *>(grad_latent.data()),
                                              batch_size, latent_size_, input_size_);

        ::cuda::layers::launchFCBackwardWeights(static_cast<float *>(latent_.data()),
                                                static_cast<float *>(grad_output.data()),
                                                static_cast<float *>(grad_dec_weights_.data()),
                                                batch_size, latent_size_, input_size_);

        ::cuda::layers::launchFCBackwardBias(static_cast<float *>(grad_output.data()),
                                             static_cast<float *>(grad_dec_bias_.data()),
                                             batch_size, input_size_);

        if (activation_type_ != Activation::Linear) {
            Utils::Tensor temp_input({batch_size, latent_size_});
            Utils::Tensor temp_output({batch_size, latent_size_});
            Utils::Tensor temp_grad({batch_size, latent_size_});
            Utils::Tensor grad_pre({batch_size, latent_size_});
            ::cudaMemcpy(temp_input.data(), pre_latent_.data(), pre_latent_.size() * sizeof(float), ::cudaMemcpyDeviceToDevice);
            ::cudaMemcpy(temp_output.data(), latent_.data(), latent_.size() * sizeof(float), ::cudaMemcpyDeviceToDevice);
            ::cudaMemcpy(temp_grad.data(), grad_latent.data(), grad_latent.size() * sizeof(float), ::cudaMemcpyDeviceToDevice);
            Activations::apply_activation_gradient(temp_input, temp_output, temp_grad, grad_pre, activation_type_);
            grad_latent = std::move(grad_pre);
        }

        ::cuda::layers::launchFCBackwardInput(static_cast<float *>(grad_latent.data()),
                                              static_cast<float *>(enc_weights_.data()),
                                              static_cast<float *>(grad_input.data()),
                                              batch_size, input_size_, latent_size_);

        ::cuda::layers::launchFCBackwardWeights(static_cast<float *>(this->input_cache_.data()),
                                                static_cast<float *>(grad_latent.data()),
                                                static_cast<float *>(grad_enc_weights_.data()),
                                                batch_size, input_size_, latent_size_);

        ::cuda::layers::launchFCBackwardBias(static_cast<float *>(grad_latent.data()),
                                             static_cast<float *>(grad_enc_bias_.data()),
                                             batch_size, latent_size_);

        if (this->optimizer_) {
            this->optimizer_->update(enc_weights_, grad_enc_weights_);
            this->optimizer_->update(dec_weights_, grad_dec_weights_);
            this->optimizer_->update(enc_bias_, grad_enc_bias_);
            this->optimizer_->update(dec_bias_, grad_dec_bias_);
        }

        return grad_input;
    }

    float regularization_loss() const override { return regularization_loss_; }

    size_t get_flops(int batch_size = 1) const override {
        return batch_size * (4 * input_size_ * latent_size_);
    }

    size_t get_parameters() const override {
        return enc_weights_.size() + dec_weights_.size() + enc_bias_.size() + dec_bias_.size();
    }

    int get_input_size() const override { return input_size_; }
    int get_output_size() const override { return input_size_; }

    Utils::Tensor &encoder_weights() { return enc_weights_; }
    Utils::Tensor &decoder_weights() { return dec_weights_; }
    Utils::Tensor &encoder_bias() { return enc_bias_; }
    Utils::Tensor &decoder_bias() { return dec_bias_; }
};

} // namespace Thot