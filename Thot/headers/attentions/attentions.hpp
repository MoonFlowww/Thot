#pragma once

#include "../tensor.hpp"
#include "../initializations/initializations.hpp"
#include "../optimizations/optimizations.hpp"

namespace Thot {

    class Optimizer;
    class MHAAtt;

    class Attention {
    protected:
        std::string name_;
        bool IsTraining;
        std::shared_ptr<Optimizer> optimizer_;
        Utils::Tensor input_cache_;
    public:
        Attention(const std::string& name = "attention") : name_(name), IsTraining(true) {}
        virtual ~Attention() = default;

        virtual Utils::Tensor forward(const Utils::Tensor& input) = 0;
        virtual Utils::Tensor backward(const Utils::Tensor& gradient_output) = 0;

        std::string get_name() const { return name_; }
        virtual size_t get_flops(int batch_size = 1) const = 0;
        virtual Initialization get_initialization() const { return static_cast<const Attention*>(this)->get_initialization(); }

        void set_training(bool training) { IsTraining = training; }
        bool is_training() const { return IsTraining; }

        void set_optimizer(std::shared_ptr<Optimizer> optimizer) { optimizer_ = optimizer; }

        virtual int get_input_size() const { return -1; }
        virtual int get_output_size() const { return -1; }

        static std::shared_ptr<Attention> MHA(int embed_dim, int num_heads, Initialization weight_init = Initialization::Xavier, const std::string& name = "MHA");
    };

} // namespace Thot

#include "../attentions/details/mha.hpp"

namespace Thot {
    inline std::shared_ptr<Attention> Attention::MHA(int embed_dim, int num_heads, Initialization weight_init, const std::string& name) {
        return std::make_shared<MHAAtt>(embed_dim, num_heads, weight_init, name);
    }
}