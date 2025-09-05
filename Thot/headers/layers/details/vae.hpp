#pragma once

#include "../../tensor.hpp"
#include "../../../cuda/cuh/layers/vae.cuh"
#include "../../../cuda/cuh/LowRankCuda/lowrank.cuh"
#include "fc.hpp"

#include <memory>

namespace Thot {

    class VAELayer : public Layer {
    private:
        friend class Network;
        int input_size_;
        int latent_size_;
        std::shared_ptr<FCLayer> enc_mean_;
        std::shared_ptr<FCLayer> enc_logvar_;
        std::shared_ptr<FCLayer> dec_;

        Utils::Tensor mu_;
        Utils::Tensor logvar_;
        Utils::Tensor z_;
        Utils::Tensor eps_;
    public:
        VAELayer(int input_size, int latent_size,
                 Activation activation_type = Activation::ReLU,
                 Initialization weight_init = Initialization::Xavier,
                 const std::string& name = "VAE")
            : Layer(name), input_size_(input_size), latent_size_(latent_size) {
            enc_mean_ = std::make_shared<FCLayer>(input_size, latent_size,
                                                  Activation::Linear, weight_init, name+"_enc_mean");
            enc_logvar_ = std::make_shared<FCLayer>(input_size, latent_size,
                                                    Activation::Linear, weight_init, name+"_enc_logvar");
            dec_ = std::make_shared<FCLayer>(latent_size, input_size,
                                             activation_type, weight_init, name+"_dec");
        }

        void set_optimizer(std::shared_ptr<Optimizer> optimizer) {
            optimizer_ = optimizer;
            enc_mean_->set_optimizer(optimizer);
            enc_logvar_->set_optimizer(optimizer);
            dec_->set_optimizer(optimizer);
        }

        Utils::Tensor forward(const Utils::Tensor& input) override {
            mu_ = enc_mean_->forward(input);
            logvar_ = enc_logvar_->forward(input);
            z_ = Utils::Tensor(mu_.shape());
            eps_ = Utils::Tensor(mu_.shape());
            cuda::layers::launchVAESample(
                static_cast<float*>(mu_.data()),
                static_cast<float*>(logvar_.data()),
                static_cast<float*>(z_.data()),
                static_cast<float*>(eps_.data()),
                z_.size());
            return dec_->forward(z_);
        }

        Utils::Tensor backward(const Utils::Tensor& grad_output) override {
            Utils::Tensor grad_z = dec_->backward(grad_output);
            Utils::Tensor grad_mu(mu_.shape(), true);
            Utils::Tensor grad_logvar(mu_.shape(), true);
            cuda::layers::launchVAESampleBackward(
                static_cast<float*>(grad_z.data()),
                static_cast<float*>(eps_.data()),
                static_cast<float*>(logvar_.data()),
                static_cast<float*>(grad_mu.data()),
                static_cast<float*>(grad_logvar.data()),
                grad_z.size());
            Utils::Tensor grad_input_mu = enc_mean_->backward(grad_mu);
            Utils::Tensor grad_input_logvar = enc_logvar_->backward(grad_logvar);
            Utils::Tensor grad_input(grad_input_mu.shape(), true);
            launchAdd(static_cast<float*>(grad_input_mu.data()),
                      static_cast<float*>(grad_input_logvar.data()),
                      static_cast<float*>(grad_input.data()),
                      grad_input.size());
            return grad_input;
        }

        size_t get_flops(int batch_size = 1) const override {
            return enc_mean_->get_flops(batch_size) +
                   enc_logvar_->get_flops(batch_size) +
                   dec_->get_flops(batch_size);
        }

        size_t get_parameters() const override {
            return enc_mean_->get_parameters() +
                   enc_logvar_->get_parameters() +
                   dec_->get_parameters();
        }

        Activation get_activation() const override { return dec_->get_activation(); }

        int get_input_size() const override { return input_size_; }
        int get_output_size() const override { return input_size_; }

        const Utils::Tensor& mean() const { return mu_; }
        const Utils::Tensor& logvar() const { return logvar_; }
    };

} // namespace Thot