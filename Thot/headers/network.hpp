#pragma once
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>
#include <chrono>

#include "tensor.hpp"
#include "layers/layers.hpp"

#include "activations/activations.hpp"
#include "optimizations/optimizations.hpp"
#include "losses\losses.hpp"
#include "metrics\metrics.hpp"


namespace Thot {
	class Layer;
	class Optimizer;

	class Network {
	private:
		std::string name_;
		std::vector<std::shared_ptr<Layer>> layers_;
		bool Istraining_;
		std::shared_ptr<Optimizer> optimizer_;
		std::shared_ptr<Losses> loss_function_;

		std::vector<float> latencies_;
		std::vector<std::vector<float>> model_parameters_;



		void print_vector(const std::vector<float>& vec) {
			std::cout << "[";
			for (size_t i = 0; i < vec.size(); ++i) {
				std::cout << vec[i];
				if (i < vec.size() - 1) std::cout << ", ";
			}
			std::cout << "]";
		}


		float train_batch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, float learning_rate ) {
			float total_loss = 0.0f;

			for (size_t i = 0; i < inputs.size(); ++i) {
				std::vector<int> input_shape = { 1, static_cast<int>(inputs[i].size()) };
				std::vector<float> output = forward(inputs[i], input_shape);

				float loss = 0.0f;
				std::vector<float> grad_output(output.size(), 0.0f);

				for (size_t j = 0; j < output.size(); ++j) {
					float error = output[j] - targets[i][j];
					loss += error * error;
					grad_output[j] = 2.0f * error;
				}
				loss *= 0.5f;
				total_loss += loss;

				Utils::Tensor grad_tensor({ 1, static_cast<int>(output.size()) });
				grad_tensor.upload(grad_output);

				backward(grad_tensor, learning_rate);
			}

			return total_loss / inputs.size();
		}

		std::string format_time(float seconds) {
			std::ostringstream oss;
			oss << std::fixed << std::setprecision(2);

			if (seconds < 1e-6) {
				oss << seconds * 1e9 << " ns";
			}
			else if (seconds < 1e-3) {
				oss << seconds * 1e6 << " us";
			}
			else if (seconds < 1.0) {
				oss << seconds * 1e3 << " ms";
			}
			else if (seconds < 60.0) {
				oss << seconds << " s";
			}
			else if (seconds < 3600.0) {
				int minutes = static_cast<int>(seconds / 60);
				float remaining_seconds = seconds - (minutes * 60);
				oss << minutes << " m " << remaining_seconds << " s";
			}
			else {
				int hours = static_cast<int>(seconds / 3600);
				int minutes = static_cast<int>((seconds - (hours * 3600)) / 60);
				float remaining_seconds = seconds - (hours * 3600) - (minutes * 60);
				oss << hours << " h " << minutes << " m " << remaining_seconds << " s";
			}
			std::cout.unsetf(std::ios_base::floatfield);
			std::cout << std::setprecision(6);
			return oss.str();
		}

		std::string format_samples_per_second(float samples_per_second) {
			std::ostringstream oss;
			oss << std::fixed << std::setprecision(2);

			if (samples_per_second < 1e3) {
				oss << samples_per_second << " samples/s";
			}
			else if (samples_per_second < 1e6) {
				oss << samples_per_second / 1e3 << "K samples/s";
			}
			else if (samples_per_second < 1e9) {
				oss << samples_per_second / 1e6 << "M samples/s";
			}
			else {
				oss << samples_per_second / 1e9 << "B samples/s";
			}
			std::cout.unsetf(std::ios_base::floatfield);
			std::cout << std::setprecision(6);
			return oss.str();
		}


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
		size_t get_flops(int batch_size = 1) const {
			size_t total_flops = 0;
			for (const auto& layer : layers_) {
				total_flops += layer->get_flops(batch_size);
			}
			return total_flops;
		}

		void evaluate( const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, bool verbose = true ) {
			std::vector<std::vector<float>> predictions;
			std::vector<float> latencies;


			for (size_t i = 0; i < inputs.size(); ++i) {
				auto start = std::chrono::high_resolution_clock::now();
				std::vector<float> output = forward(inputs[i], { 1, static_cast<int>(inputs[i].size()) });
				auto end = std::chrono::high_resolution_clock::now();

				float latency = std::chrono::duration<float>(end - start).count();
				latencies.push_back(latency);
				predictions.push_back(output);

				if (verbose) {
					std::cout << "Input: [";
					for (float x : inputs[i]) std::cout << x << " ";
					std::cout << "] -> Output: [";
					for (float y : output) std::cout << y << " ";
					std::cout << "] -> Expected: [";
					for (float t : targets[i]) std::cout << t << " ";
					std::cout << "]\n";
				}
			}

			auto metrics = Metrics::compute_metrics(predictions, targets, latencies, get_flops());
			auto frontier = Metrics::compute_pareto_frontier(predictions, targets, latencies, get_flops());
			Metrics::print_metrics(metrics, frontier);
		}

		void set_loss(Loss type, float epsilon = 1e-8f, float delta = 1.0f) {
			loss_function_ = std::make_shared<Losses>(type, epsilon, delta);
		}

		float compute_loss(const Utils::Tensor& predictions, const Utils::Tensor& targets) {
			return loss_function_->compute(predictions, targets);
		}

		Utils::Tensor compute_gradients(const Utils::Tensor& predictions, const Utils::Tensor& targets) {
			return loss_function_->compute_gradients(predictions, targets);
		}

		inline void summary() {
			std::cout << "Network: " << name_ << std::endl;
			std::cout << "Layers:" << std::endl;

			size_t total_flops = 0;
			size_t batch_size = 1;

			std::cout << "+---------------+----------------------+----------------------+----------------------+---------------+" << std::endl;
			std::cout << "| Layer         | Type                 | Activation           | Initialization       | FLOPs         |" << std::endl;
			std::cout << "+---------------+----------------------+----------------------+----------------------+---------------+" << std::endl;

			for (size_t i = 0; i < layers_.size(); ++i) {
				auto& layer = layers_[i];
				std::string layer_name = layer->get_name();
				std::string activation_name = Thot::Activations::to_string(layer->get_activation());
				std::string init_name = Thot::Initializers::to_string(layer->get_initialization());

				size_t layer_flops = layer->get_flops(batch_size);
				total_flops += layer_flops;

				if (layer_name.length() > 20) layer_name = layer_name.substr(0, 17) + "...";
				if (activation_name.length() > 20) activation_name = activation_name.substr(0, 17) + "...";
				if (init_name.length() > 20) init_name = init_name.substr(0, 17) + "...";

				std::cout << "| " << std::left << std::setw(13) << i + 1
					<< " | " << std::left << std::setw(20) << layer_name
					<< " | " << std::left << std::setw(20) << activation_name
					<< " | " << std::left << std::setw(20) << init_name
					<< " | " << std::right << std::setw(13) << layer_flops << " |" << std::endl;
			}

			std::cout << "+---------------+----------------------+----------------------+----------------------+---------------+" << std::endl;
			std::cout << "| Thot Model    |                                                                            " << std::right << std::setw(7) << total_flops << " |" << std::endl;
			std::cout << "+---------------+------------------------------------------------------------------------------------+" << std::endl;
		}


		

		void fit( const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, int epochs, int batch_size = 1, float learning_rate = 0.01f, int log_interval = 100 ) {
			auto total_start = std::chrono::high_resolution_clock::now();
			std::vector<float> epoch_times;
			bool zero_hit = false;
			for (int epoch = 0; epoch < epochs; ++epoch) {
				auto epoch_start = std::chrono::high_resolution_clock::now();

				double epoch_loss = train_batch(inputs, targets, learning_rate);

				auto epoch_end = std::chrono::high_resolution_clock::now();
				float epoch_time = std::chrono::duration<float>(epoch_end - epoch_start).count();
				epoch_times.push_back(epoch_time);

				if (epoch % log_interval == 0 || epoch == epochs - 1) {
					std::cout << "Epoch " << epoch << " - Average Loss: " << epoch_loss << std::endl;
				} if (epoch_loss < 1e-15) {
					zero_hit = true;
					break;
				}
			}
			if (zero_hit) std::cout << " -> [ADMIN]" << name_ << " loss <1e-15, not necessary to continue" << std::endl;
			auto total_end = std::chrono::high_resolution_clock::now();
			float total_time = std::chrono::duration<float>(total_end - total_start).count();

			float avg_epoch_time = std::accumulate(epoch_times.begin(), epoch_times.end(), 0.0f) / epochs;
			float min_epoch_time = *std::min_element(epoch_times.begin(), epoch_times.end());
			float max_epoch_time = *std::max_element(epoch_times.begin(), epoch_times.end());
			float samples_per_second = (inputs.size() * epochs) / total_time;

			std::cout << std::fixed << std::setprecision(2);

			std::cout << "\nTraining Summary:\n";
			std::cout << "----------------\n";
			std::cout << "Total Epochs: " << epochs << "\n";
			std::cout << "Total Training Time: " << format_time(total_time) << "\n";
			std::cout << "Average Epoch Time: " << format_time(avg_epoch_time) << "\n";
			std::cout << "Min Epoch Time: " << format_time(min_epoch_time) << "\n";
			std::cout << "Max Epoch Time: " << format_time(max_epoch_time) << "\n";
			std::cout << "Throughput: " << format_samples_per_second(samples_per_second) << "\n";

			std::cout.unsetf(std::ios_base::floatfield);
			std::cout << std::setprecision(6);

		}
	};
}