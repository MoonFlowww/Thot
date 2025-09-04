#pragma once
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <chrono>

#include "../../tensor.hpp"
#include "../../initializations/initializations.hpp"
#include "../../../cuda/cuh/attentions/mla.cuh"
#include "../../layers/layers.hpp"

namespace Thot {
    class MLAAttentionLayer : public Layer {
    private:
        int embed_dim_;
        int num_heads_;
        int head_dim_;
        int latent_dim_;
        Initialization initialization_;

        // Cached shapes from last forward
        int last_batch_ = 0;
        int last_seq_ = 0;

        // Params
        Utils::Tensor W_DKV_, b_DKV_;
        Utils::Tensor W_UK_,  b_UK_;
        Utils::Tensor W_UV_,  b_UV_;
        Utils::Tensor W_Q_,   b_Q_;
        Utils::Tensor W_O_,   b_O_;

        // Grads
        Utils::Tensor gW_DKV_, gB_DKV_;
        Utils::Tensor gW_UK_,  gB_UK_;
        Utils::Tensor gW_UV_,  gB_UV_;
        Utils::Tensor gW_Q_,   gB_Q_;
        Utils::Tensor gW_O_,   gB_O_;

        // Caches
        Utils::Tensor input_cache_;
        Utils::Tensor q_, k_, v_, c_kv_, attn_probs_, concat_;

        std::chrono::nanoseconds total_init_;



    public:
    MLAAttentionLayer(int embed_dim, int num_heads, int latent_dim, Initialization init = Initialization::He, const std::string &name = "MLA")
        : Layer(name), embed_dim_(embed_dim), num_heads_(num_heads), latent_dim_(latent_dim), initialization_(init) {
        auto t1 = std::chrono::high_resolution_clock::now();
        if (embed_dim_ % num_heads_ != 0) {
            throw std::invalid_argument("embed_dim must be divisible by num_heads");
        }
        head_dim_ = embed_dim_ / num_heads_;
        if (latent_dim_ <= 0 || latent_dim_ >= embed_dim_) {
            throw std::invalid_argument("latent_dim must be in (0, embed_dim)");
        }

        W_DKV_ = Utils::Tensor({latent_dim_, embed_dim_});
        b_DKV_ = Utils::Tensor({latent_dim_});

        W_UK_  = Utils::Tensor({embed_dim_, latent_dim_});
        b_UK_  = Utils::Tensor({embed_dim_});
        W_UV_  = Utils::Tensor({embed_dim_, latent_dim_});
        b_UV_  = Utils::Tensor({embed_dim_});

        W_Q_   = Utils::Tensor({embed_dim_, embed_dim_});
        b_Q_   = Utils::Tensor({embed_dim_});

        W_O_   = Utils::Tensor({embed_dim_, embed_dim_});
        b_O_   = Utils::Tensor({embed_dim_});

        gW_DKV_ = Utils::Tensor({latent_dim_, embed_dim_}, true);
        gB_DKV_ = Utils::Tensor({latent_dim_}, true);
        gW_UK_  = Utils::Tensor({embed_dim_, latent_dim_}, true);
        gB_UK_  = Utils::Tensor({embed_dim_}, true);
        gW_UV_  = Utils::Tensor({embed_dim_, latent_dim_}, true);
        gB_UV_  = Utils::Tensor({embed_dim_}, true);
        gW_Q_   = Utils::Tensor({embed_dim_, embed_dim_}, true);
        gB_Q_   = Utils::Tensor({embed_dim_}, true);
        gW_O_   = Utils::Tensor({embed_dim_, embed_dim_}, true);
        gB_O_   = Utils::Tensor({embed_dim_}, true);

        Initializations::initialize_tensor(W_DKV_, init, latent_dim_, embed_dim_);
        Initializations::initialize_tensor(W_UK_,  init, embed_dim_, latent_dim_);
        Initializations::initialize_tensor(W_UV_,  init, embed_dim_, latent_dim_);
        Initializations::initialize_tensor(W_Q_,   init, embed_dim_, embed_dim_);
        Initializations::initialize_tensor(W_O_,   init, embed_dim_, embed_dim_);
        Initializations::initialize_tensor(b_DKV_, init, latent_dim_, 1);
        Initializations::initialize_tensor(b_UK_,  init, embed_dim_, 1);
        Initializations::initialize_tensor(b_UV_,  init, embed_dim_, 1);
        Initializations::initialize_tensor(b_Q_,   init, embed_dim_, 1);
        Initializations::initialize_tensor(b_O_,   init, embed_dim_, 1);

        auto t2 = std::chrono::high_resolution_clock::now();
        total_init_ = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1);
    }

    Utils::Tensor forward(const Utils::Tensor &input) override {
        int dims = static_cast<int>(input.shape().size());
        int batch, seq_len, embed_dim;

        if (dims == 2) {
            batch = input.shape()[0];
            seq_len = 1;
            embed_dim = input.shape()[1];
            input_cache_ = Utils::Tensor({batch, seq_len, embed_dim});
        } else if (dims == 3) {
            batch = input.shape()[0];
            seq_len = input.shape()[1];
            embed_dim = input.shape()[2];
            input_cache_ = Utils::Tensor({batch, seq_len, embed_dim});
        } else if (dims == 4) {
            batch = input.shape()[0];
            seq_len = 1;
            embed_dim = input.shape()[1] * input.shape()[2] * input.shape()[3];
            input_cache_ = Utils::Tensor({batch, seq_len, embed_dim});
        } else {
            throw std::invalid_argument("Input tensor must be 2D, 3D or 4D");
        }

        ::cudaMemcpy(input_cache_.data(), input.data(),
                     input.size() * sizeof(float), ::cudaMemcpyDeviceToDevice);


        if (embed_dim != embed_dim_) {
            throw std::invalid_argument(
                "Input embed_dim " + std::to_string(embed_dim) +
                " differs from layer embed_dim " + std::to_string(embed_dim_));
        }

        last_batch_ = batch;
        last_seq_ = seq_len;

        Utils::Tensor output({batch, seq_len, embed_dim});
        q_      = Utils::Tensor({batch, seq_len, embed_dim});
        k_      = Utils::Tensor({batch, seq_len, embed_dim});
        v_      = Utils::Tensor({batch, seq_len, embed_dim});
        c_kv_   = Utils::Tensor({batch, seq_len, latent_dim_});
        attn_probs_ = Utils::Tensor({batch, num_heads_, seq_len, seq_len});
        concat_ = Utils::Tensor({batch, seq_len, embed_dim});

        cuda::attentions::launchMLAForward(
            static_cast<const float *>(input_cache_.data()),
            static_cast<const float *>(W_DKV_.data()), static_cast<const float *>(b_DKV_.data()),
            static_cast<const float *>(W_UK_.data()),  static_cast<const float *>(b_UK_.data()),
            static_cast<const float *>(W_UV_.data()),  static_cast<const float *>(b_UV_.data()),
            static_cast<const float *>(W_Q_.data()),   static_cast<const float *>(b_Q_.data()),
            static_cast<const float *>(W_O_.data()),   static_cast<const float *>(b_O_.data()),
            static_cast<float *>(output.data()),
            static_cast<float *>(q_.data()), static_cast<float *>(k_.data()),
            static_cast<float *>(v_.data()), static_cast<float *>(c_kv_.data()),
            static_cast<float *>(attn_probs_.data()),
            static_cast<float *>(concat_.data()),
            batch, seq_len, embed_dim, num_heads_, latent_dim_);

        return output;
    }

    Utils::Tensor backward(const Utils::Tensor &grad_output) override {
        int batch = input_cache_.shape()[0];
        int seq   = input_cache_.shape().size() > 1 ? input_cache_.shape()[1] : 1;

        Utils::Tensor grad_input(input_cache_.shape());

        cuda::attentions::launchMLABackward(
            static_cast<const float *>(input_cache_.data()),
            static_cast<const float *>(W_DKV_.data()), static_cast<const float *>(b_DKV_.data()),
            static_cast<const float *>(W_UK_.data()),  static_cast<const float *>(b_UK_.data()),
            static_cast<const float *>(W_UV_.data()),  static_cast<const float *>(b_UV_.data()),
            static_cast<const float *>(W_Q_.data()),   static_cast<const float *>(b_Q_.data()),
            static_cast<const float *>(W_O_.data()),   static_cast<const float *>(b_O_.data()),
            static_cast<const float *>(q_.data()), static_cast<const float *>(k_.data()),
            static_cast<const float *>(v_.data()), static_cast<const float *>(c_kv_.data()),
            static_cast<const float *>(attn_probs_.data()),
            static_cast<const float *>(concat_.data()),
            static_cast<const float *>(grad_output.data()),
            static_cast<float *>(grad_input.data()),
            static_cast<float *>(gW_DKV_.data()), static_cast<float *>(gB_DKV_.data()),
            static_cast<float *>(gW_UK_.data()),  static_cast<float *>(gB_UK_.data()),
            static_cast<float *>(gW_UV_.data()),  static_cast<float *>(gB_UV_.data()),
            static_cast<float *>(gW_Q_.data()),   static_cast<float *>(gB_Q_.data()),
            static_cast<float *>(gW_O_.data()),   static_cast<float *>(gB_O_.data()),
            batch, seq, embed_dim_, num_heads_, latent_dim_);

        return grad_input;
    }

    Activation get_activation() const override { return Activation::Linear; }
    Initialization get_initialization() const override { return initialization_; }
    float get_latency() const override { return total_init_.count(); }
    int get_input_size()  const override { return embed_dim_; }
    int get_output_size() const override { return embed_dim_; }


        //last_seq define how many flops, without ->0
    size_t get_flops(int batch_size = 1) const override {
        const int B = (last_batch_ > 0) ? last_batch_ : batch_size;
        const int S = (last_seq_ > 0) ? last_seq_ : 0;
        if (S == 0) return 0;

        const double D  = static_cast<double>(embed_dim_);
        const double Dc = static_cast<double>(latent_dim_);
        const double H  = static_cast<double>(num_heads_);
        const double Dh = static_cast<double>(head_dim_);

        double flops = 0.0;
        flops += B * S * (2.0 * Dc * D);      // W_DKV * X
        flops += B * S * (2.0 * D * D);       // W_Q  * X
        flops += B * S * (2.0 * D * Dc);      // W_UK * C
        flops += B * S * (2.0 * D * Dc);      // W_UV * C

        flops += B * H * S * S * (2.0 * Dh);  // QK^T
        flops += B * H * S * S;               // softmax approx
        flops += B * H * S * S * (2.0 * Dh);  // P * V

        flops += B * S * (2.0 * D * D);       // W_O * concat

        return static_cast<size_t>(flops);
    }

    size_t get_parameters() const override {
        size_t params = 0;
        params += static_cast<size_t>(latent_dim_) * embed_dim_; // W_DKV
        params += static_cast<size_t>(latent_dim_);               // b_DKV
        params += static_cast<size_t>(embed_dim_) * latent_dim_;  // W_UK
        params += static_cast<size_t>(embed_dim_);                // b_UK
        params += static_cast<size_t>(embed_dim_) * latent_dim_;  // W_UV
        params += static_cast<size_t>(embed_dim_);                // b_UV
        params += static_cast<size_t>(embed_dim_) * embed_dim_;   // W_Q
        params += static_cast<size_t>(embed_dim_);                // b_Q
        params += static_cast<size_t>(embed_dim_) * embed_dim_;   // W_O
        params += static_cast<size_t>(embed_dim_);                // b_O
        return params;
    }
};

} // namespace Thot
