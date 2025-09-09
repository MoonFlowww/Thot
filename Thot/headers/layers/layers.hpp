#pragma once

#include "../tensor.hpp"
#include "../activations/activations.hpp"
#include "../initializations/initializations.hpp"
#include "../optimizations/optimizations.hpp"

namespace Thot {

	class Optimizer;
    //enum class ConvAlgo { Auto = -1, Direct = 0, Winograd = 1, FFT = 2 };



    class FCLayer;
    class RNNLayer;
    class Conv2DLayer;
    class RBMLayer;
    class FlattenLayer;
    class MaxPool2DLayer;

    class VAELayer;
    class RCNNLayer;
    class SpikeLayer;
    class SparseContractiveAELayer;


	class Layer {
    protected:
        std::string name_;
        bool IsTraining;
        std::shared_ptr<Optimizer> optimizer_;
        Utils::Tensor input_cache_;

    public:
        Layer(const std::string& name = "layer") : name_(name), IsTraining(true) {};
        virtual ~Layer() = default;

        virtual Utils::Tensor forward(const Utils::Tensor& input) = 0;
        virtual Utils::Tensor backward(const Utils::Tensor& gradient_output) = 0;

        std::string get_name() const { return name_; }

        virtual size_t get_flops(int batch_size = 1) const = 0;
        virtual size_t get_parameters() const = 0;
        virtual Activation get_activation() const { return static_cast<const Layer*>(this)->get_activation(); }
        virtual Initialization get_initialization() const { return static_cast<const Layer*>(this)->get_initialization(); }
        virtual float get_latency() const { return -1.0f; }

        void set_training(bool training) { IsTraining = training; }
        bool is_training() const { return IsTraining; }
        void set_optimizer(std::shared_ptr<Optimizer> optimizer) { optimizer_ = optimizer; }

        virtual int get_input_size() const { return -1; }
        virtual int get_output_size() const { return -1; }
        virtual float regularization_loss() const { return 0.0f; }

        static std::shared_ptr<Layer> FC(int input_size, int output_size,
                                         Activation activation_type = Activation::ReLU,
                                         Initialization weight_init = Initialization::Xavier,
                                         const std::string& name = "FeedForward");

        static std::shared_ptr<Layer> RNN(int input_size, int hidden_size, int seq_length,
                                          Activation activation_type = Activation::ReLU,
                                          Initialization weight_init = Initialization::Xavier,
                                          const std::string& name = "Recurrent Layer");

        static std::shared_ptr<Layer> Conv2D(int in_channels, int in_height, int in_width,
                                             int out_channels, int kernel_size, int stride, int padding,
                                             Activation activation_type = Activation::ReLU,
                                             Initialization weight_init = Initialization::Xavier,
                                             /*ConvAlgo conv_algo = ConvAlgo::Auto,*/
                                             const std::string& name = "Conv2D");

        static std::shared_ptr<Layer> RBM(int visible_size, int hidden_size, int cd_steps,
                                          Activation activation_type = Activation::ReLU,
                                          Initialization weight_init = Initialization::Xavier,
                                          const std::string& name = "Restriced Boltzman Layer");

        static std::shared_ptr<Layer> Flatten(int in_channels, int in_height, int in_width,
                                              const std::string& name = "Flatten");

        static std::shared_ptr<Layer> MaxPool2D(int in_channels, int in_height, int in_width,
                                               int kernel_size, int stride = 2,
                                               const std::string& name = "MaxPool2D");

        static std::shared_ptr<Layer> VAE(int input_size, int latent_size,
                                          Activation activation_type = Activation::ReLU,
                                          Initialization weight_init = Initialization::Xavier,
                                          const std::string& name = "VAE");

        static std::shared_ptr<Layer> RCNN(int in_channels, int in_height, int in_width,
                                           int out_channels, int kernel_size, int stride, int padding,
                                           int pooled_h, int pooled_w,
                                           Activation activation_type = Activation::ReLU,
                                           Initialization weight_init = Initialization::Xavier,
                                           /*ConvAlgo conv_algo = ConvAlgo::Auto,*/
                                           const std::string& name = "RCNN");

        static std::shared_ptr<Layer> Spike(int size, float threshold = 1.0f,
                                            const std::string& name = "Spike");

        static std::shared_ptr<Layer> SparseAE(int input_size, int latent_size,
                                              Activation activation_type = Activation::Sigmoid,
                                              Initialization weight_init = Initialization::Xavier,
                                              bool use_sparsity = false,
                                              bool use_contractive = false,
                                              float sparsity_rho = 0.05f,
                                              float sparsity_beta = 1e-3f,
                                              float contractive_lambda = 1e-3f,
                                              const std::string& name = "SparseAE");
	};

};


#include "details/fc.hpp"
#include "details/rnn.hpp"
#include "details/conv2d.hpp"
#include "details/rbm.hpp"
#include "details/flatten.hpp"
#include "details/maxpool2d.hpp"
#include "details/vae.hpp"
#include "details/rcnn.hpp"
#include "details/spike.hpp"
#include "details/sparse_contractive_ae.hpp"


namespace Thot {
	inline std::shared_ptr<Layer> Layer::FC(int input_size, int output_size, Activation activation_type, Initialization weight_init, const std::string& name) {
		return std::make_shared<FCLayer>(input_size, output_size, activation_type, weight_init, name);
	}

	inline std::shared_ptr<Layer> Layer::RNN(int input_size, int hidden_size, int seq_length, Activation activation_type, Initialization weight_init, const std::string& name) {
		return std::make_shared<RNNLayer>(input_size, hidden_size, seq_length, activation_type, weight_init, name);
	}

    inline std::shared_ptr<Layer> Layer::Conv2D(int in_channels, int in_height, int in_width, int out_channels, int kernel_size, int stride, int padding, Activation activation_type, Initialization weight_init, /*ConvAlgo conv_algo,*/ const std::string& name) {
	    return std::make_shared<Conv2DLayer>(in_channels, in_height, in_width, out_channels, kernel_size, stride, padding, activation_type, weight_init, /*conv_algo,*/ name);
	}

    inline std::shared_ptr<Layer> Layer::RBM(int visible_size, int hidden_size, int cd_steps, Activation activation_type, Initialization weight_init, const std::string& name) {
	    return std::make_shared<RBMLayer>(visible_size, hidden_size, cd_steps, activation_type, weight_init, name);
	}

    inline std::shared_ptr<Layer> Layer::Flatten(int in_channels, int in_height, int in_width, const std::string& name) {
	    return std::make_shared<FlattenLayer>(in_channels, in_height, in_width, name);
	}

    inline std::shared_ptr<Layer> Layer::MaxPool2D(int in_channels, int in_height, int in_width, int kernel_size, int stride, const std::string& name) {
	    return std::make_shared<MaxPool2DLayer>(in_channels, in_height, in_width, kernel_size, stride, name);
	}

    inline std::shared_ptr<Layer> Layer::VAE(int input_size, int latent_size, Activation activation_type, Initialization weight_init, const std::string& name) {
	    return std::make_shared<VAELayer>(input_size, latent_size, activation_type, weight_init, name);
	}

    inline std::shared_ptr<Layer> Layer::RCNN(int in_channels, int in_height, int in_width, int out_channels, int kernel_size, int stride, int padding, int pooled_h, int pooled_w, Activation activation_type, Initialization weight_init, /*ConvAlgo conv_algo,*/ const std::string& name) {
	    return std::make_shared<RCNNLayer>(in_channels, in_height, in_width, out_channels, kernel_size, stride, padding, pooled_h, pooled_w, activation_type, weight_init, /*conv_algo,*/ name);
	}

    inline std::shared_ptr<Layer> Layer::Spike(int size, float threshold, const std::string& name) {
	    return std::make_shared<SpikeLayer>(size, threshold, name);
	}

    inline std::shared_ptr<Layer> Layer::SparseAE(int input_size, int latent_size, Activation activation_type, Initialization weight_init,  bool use_sparsity, bool use_contractive,
                                                float sparsity_rho, float sparsity_beta, float contractive_lambda, const std::string& name) {
	    return std::make_shared<SparseContractiveAELayer>(input_size, latent_size, activation_type, weight_init, use_sparsity, use_contractive, sparsity_rho, sparsity_beta, contractive_lambda, name);
	}
}