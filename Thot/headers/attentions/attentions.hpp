#pragma once

#include "../tensor.hpp"
#include "../activations/activations.hpp"
#include "../initializations/initializations.hpp"
#include "../optimizations/optimizations.hpp"
#include "layers/layers.hpp"

namespace Thot {

	class Optimizer;

    class MHAAtt;

	class Attention : public Layer {
	protected:
		std::string name_;
		bool IsTraining;
		std::shared_ptr<Optimizer> optimizer_;
		Utils::Tensor input_cache_;
	public:
		Attention(const std::string& name = "Attention") : name_(name), IsTraining(true) {};
		virtual ~Attention() = default;

		virtual Utils::Tensor forward(const Utils::Tensor& input) = 0;

		virtual Utils::Tensor backward(const Utils::Tensor& gradient_output) = 0;

		std::string get_name() const { return name_; }
		virtual size_t get_flops(int batch_size = 1) const = 0;
		virtual Activation get_activation() const { return static_cast<const Layer*>(this)->get_activation(); }

		virtual Initialization get_initialization() const { return static_cast<const Layer*>(this)->get_initialization(); }


		void set_training(bool training) { IsTraining = training; }

		bool is_training() const { return IsTraining; }

		void set_optimizer(std::shared_ptr<Optimizer> optimizer) { optimizer_ = optimizer; }

	    virtual int get_input_size() const { return -1; }
	    virtual int get_output_size() const { return -1; }


		static std::shared_ptr<Layer> FC(int input_size, int output_size, Activation activation_type = Activation::ReLU, Initialization weight_init = Initialization::Xavier, const std::string& name = "FeedForward");

		static std::shared_ptr<Layer> RNN(int input_size, int hidden_size, int seq_length, Activation activation_type = Activation::ReLU, Initialization weight_init = Initialization::Xavier, const std::string& name = "Recurrent Layer");

	    static std::shared_ptr<Layer> Conv2D(int in_channels, int in_height, int in_width, int out_channels, int kernel_size, int stride, int padding, Activation activation_type = Activation::ReLU, Initialization weight_init = Initialization::Xavier, const std::string& name = "Conv2D");

	    static std::shared_ptr<Layer> RBM(int visible_size, int hidden_size, int cd_steps, Activation activation_type = Activation::ReLU, Initialization weight_init = Initialization::Xavier, const std::string& name = "Restriced Boltzman Layer");

	    static std::shared_ptr<Layer> Flatten(int in_channels, int in_height, int in_width, const std::string& name = "Flatten");

	    static std::shared_ptr<Layer> MaxPool2D(int in_channels, int in_height, int in_width, int kernel_size, int stride = 2, const std::string& name = "MaxPool2D");
	};

}

#include "attentions/details/mha.hpp"
namespace Thot {

}