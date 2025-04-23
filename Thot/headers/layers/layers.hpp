#pragma once

#include "../tensor.hpp"
#include "../activations/activations.hpp"
#include "../initializations/initializations.hpp"
#include "../optimizations/optimizations.hpp"

namespace Thot {

	class Optimizer;
	class FCLayer;

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

		virtual Utils::Tensor backward(const Utils::Tensor& gradient_output, float learning_rate) = 0;

		std::string get_name() const { return name_; }
		virtual size_t get_flops(int batch_size = 1) const = 0;
		virtual Activation get_activation() const { return static_cast<const Layer*>(this)->get_activation(); }

		virtual Initialization get_initialization() const { return static_cast<const Layer*>(this)->get_initialization(); }


		void set_training(bool training) { IsTraining = training; }

		bool is_training() const { return IsTraining; }

		void set_optimizer(std::shared_ptr<Optimizer> optimizer) { optimizer_ = optimizer; }

		static std::shared_ptr<Layer> FC(int input_size, int output_size, Activation activation_type = Activation::ReLU, Initialization weight_init = Initialization::Xavier, const std::string& name = "FC");
	};

}

#include "../layers/details/fc.hpp"

// Define the FC static method after including FCLayer definition
namespace Thot {
	inline std::shared_ptr<Layer> Layer::FC(int input_size, int output_size, Activation activation_type, Initialization weight_init, const std::string& name) {
		return std::make_shared<FCLayer>(input_size, output_size, activation_type, weight_init, name);
	}
}