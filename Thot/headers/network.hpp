#pragma once
#include <string>
#include <vector>
#include <memory>
#include <iostream>

#include "tensor.hpp"
#include "layers/layers.hpp"

#include "activations/activations.hpp"
#include "optimizations/optimizations.hpp"


namespace Thot {
	class Layer;
	class Optimizer;

	class Network {
	private:
		std::string name_;
		std::vector<std::shared_ptr<Layer>> layers_;
		bool Istraining_;
		std::shared_ptr<Optimizer> optimizer_;

	public:
		Network(const std::string& name = "Thot_Network") : name_(name), Istraining_(true) {};

		inline void add(std::shared_ptr<Layer> layer) {
			layers_.push_back(layer);
		}

		inline void set_optimizer(std::shared_ptr<Optimizer> optimizer) {
			optimizer_ = optimizer;
			for (auto& L : layers_) {
				L->set_optimizer(optimizer);
			}
		}

		inline Utils::Tensor forward_gpu(const Utils::Tensor& input) {
			Utils::Tensor output(input.shape());

			float* src_ptr = static_cast<float*>(input.data());
			float* dst_ptr = static_cast<float*>(output.data());
			size_t size = input.size() * sizeof(float);
			::cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToDevice);

			for (auto& L : layers_) {
				output = L->forward(output);
			}
			return output;
		}

		inline std::vector<float> forward(const std::vector<float>& input, const std::vector<int>& input_shape) {
			Utils::Tensor input_tensor(input_shape);
			input_tensor.upload(input); // CPU -> GPU

			Utils::Tensor output_tensor = forward_gpu(input_tensor);

			return output_tensor.download(); // GPU -> CPU
		}

		inline void backward(const Utils::Tensor& grad_output, float learning_rate) {
			Utils::Tensor current_gradient(grad_output.shape());

			float* src_ptr = static_cast<float*>(grad_output.data());
			float* dst_ptr = static_cast<float*>(current_gradient.data());
			size_t size = grad_output.size() * sizeof(float);
			::cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToDevice);

			for (int i = layers_.size() - 1; i >= 0; --i) {
				current_gradient = layers_[i]->backward(current_gradient, learning_rate);
			}
		}

		inline void train() {
			Istraining_ = true;
			for (auto& layer : layers_) {
				layer->set_training(true);
			}
		}

		inline void eval() {
			Istraining_ = false;
			for (auto& layer : layers_) {
				layer->set_training(false);
			}
		}

		inline void summary() {
			std::cout << "Network: " << name_ << std::endl;
			std::cout << "Layers:" << std::endl;

			for (size_t i = 0; i < layers_.size(); ++i) {
				std::cout << "  " << i << ": " << layers_[i]->get_name() << std::endl;
			}
		}
	};
}