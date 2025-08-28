#pragma once

#include "../tensor.hpp"
#include "../activations/activations.hpp"
#include "../initializations/initializations.hpp"
#include "../optimizations/optimizations.hpp"

namespace Thot {

	class Optimizer;
	class FCLayer;
	class RNNLayer;
	class Conv2DLayer;
	class RBMLayer;

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
		virtual Activation get_activation() const { return static_cast<const Layer*>(this)->get_activation(); }

		virtual Initialization get_initialization() const { return static_cast<const Layer*>(this)->get_initialization(); }


		void set_training(bool training) { IsTraining = training; }

		bool is_training() const { return IsTraining; }

		void set_optimizer(std::shared_ptr<Optimizer> optimizer) { optimizer_ = optimizer; }

		static std::shared_ptr<Layer> FC(int input_size, int output_size, Activation activation_type = Activation::ReLU, Initialization weight_init = Initialization::Xavier, const std::string& name = "FeedForward");

		static std::shared_ptr<Layer> RNN(int input_size, int hidden_size, int seq_length, Activation activation_type = Activation::ReLU, Initialization weight_init = Initialization::Xavier, const std::string& name = "Recurrent Layer");

		static std::shared_ptr<Layer> Conv2D(int in_channels, int in_height, int in_width, int out_channels, int kernel_size, int stride, int padding, Activation activation_type = Activation::ReLU, Initialization weight_init = Initialization::Xavier, const std::string& name = "Conv2D");

		static std::shared_ptr<Layer> RBM(int visible_size, int hidden_size, int cd_steps, Activation activation_type = Activation::ReLU, Initialization weight_init = Initialization::Xavier, const std::string& name = "Restriced Boltzman Layer");
	};

}
#include "../layers/details/fc.hpp"
#include "../layers/details/rnn.hpp"
#include "../layers/details/conv2d.hpp"
#include "../layers/details/rbm.hpp"

namespace Thot {
	inline std::shared_ptr<Layer> Layer::FC(int input_size, int output_size, Activation activation_type, Initialization weight_init, const std::string& name) {
		return std::make_shared<FCLayer>(input_size, output_size, activation_type, weight_init, name);
	}

	inline std::shared_ptr<Layer> Layer::RNN(int input_size, int hidden_size, int seq_length, Activation activation_type, Initialization weight_init, const std::string& name) {
		return std::make_shared<RNNLayer>(input_size, hidden_size, seq_length, activation_type, weight_init, name);
	}

	inline std::shared_ptr<Layer> Layer::Conv2D(int in_channels, int in_height, int in_width, int out_channels, int kernel_size, int stride, int padding, Activation activation_type, Initialization weight_init, const std::string& name) {
		return std::make_shared<Conv2DLayer>(in_channels, in_height, in_width, out_channels, kernel_size, stride, padding, activation_type, weight_init, name);
	}

	inline std::shared_ptr<Layer> Layer::RBM(int visible_size, int hidden_size, int cd_steps, Activation activation_type, Initialization weight_init, const std::string& name) {
		return std::make_shared<RBMLayer>(visible_size, hidden_size, cd_steps, activation_type, weight_init, name);
	}
}