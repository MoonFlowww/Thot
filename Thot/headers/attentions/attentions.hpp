#pragma once

#include "../tensor.hpp"
#include "../activations/activations.hpp"
#include "../initializations/initializations.hpp"
#include "../optimizations/optimizations.hpp"
#include "../layers/layers.hpp"


namespace Thot {

    class Optimizer;
    class MHAAtt;

    class Attention : public Layer {


    public:
        Attention(const std::string& name = "attention") : Layer(name) {}
        ~Attention() override = default;

        virtual Utils::Tensor forward(const Utils::Tensor& input) = 0;
        virtual Utils::Tensor backward(const Utils::Tensor& gradient_output) = 0;

        virtual size_t get_flops(int batch_size = 1) const = 0;
        virtual Initialization get_initialization() const { return static_cast<const Attention*>(this)->get_initialization(); }

        Activation get_activation() const override { return Activation::Linear; }

        static std::shared_ptr<Attention> MHA(int embed_dim, int num_heads, Initialization weight_init = Initialization::Xavier, const std::string& name = "MHA");
    };

} // namespace Thot

#include "../attentions/details/mha.hpp"

namespace Thot {
    inline std::shared_ptr<Attention> Attention::MHA(int embed_dim, int num_heads, Initialization weight_init, const std::string& name) {
        return std::make_shared<MHAAtt>(embed_dim, num_heads, weight_init, name);
    }
}