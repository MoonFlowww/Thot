#pragma once
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <mutex>
#include <future>
#include <algorithm>

#include "tensor.hpp"
#include "layers/layers.hpp"

#include "activations/activations.hpp"
#include "optimizations/optimizations.hpp"
#include "losses/losses.hpp"
#include "metrics/metrics.hpp"
#include "evaluations/evaluation.hpp"

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
		std::mutex mutex_;

		std::vector<float> latencies_;
		std::vector<std::vector<float>> model_parameters_;

		size_t max_gpu_batches_;
		std::vector<cudaStream_t> cuda_streams_;

		void print_vector(const std::vector<float>& vec) {
			std::cout << "[";
			for (size_t i = 0; i < vec.size(); ++i) {
				std::cout << vec[i];
				if (i < vec.size() - 1) std::cout << ", ";
			}
			std::cout << "]";
		}

		void k_fold_split(const std::vector<std::vector<float>>& inputs,
			const std::vector<std::vector<float>>& targets,
			int k, int fold,
			std::vector<std::vector<float>>& train_inputs,
			std::vector<std::vector<float>>& train_targets,
			std::vector<std::vector<float>>& val_inputs,
			std::vector<std::vector<float>>& val_targets) {
			size_t fold_size = inputs.size() / k;
			size_t start_idx = fold * fold_size;
			size_t end_idx = (fold == k - 1) ? inputs.size() : (fold + 1) * fold_size;

			train_inputs.clear();
			train_targets.clear();
			val_inputs.clear();
			val_targets.clear();

			for (size_t i = 0; i < inputs.size(); ++i) {
				if (i >= start_idx && i < end_idx) {
					val_inputs.push_back(inputs[i]);
					val_targets.push_back(targets[i]);
				}
				else {
					train_inputs.push_back(inputs[i]);
					train_targets.push_back(targets[i]);
				}
			}
		}

		float train_batch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets) {
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

				backward(grad_tensor);
			}

			return total_loss / inputs.size();
		}

		float train_batch_threaded(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, size_t start_idx, size_t end_idx) {
			float total_loss = 0.0f;

			for (size_t i = start_idx; i < end_idx; ++i) {
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

				std::lock_guard<std::mutex> lock(mutex_);
				backward(grad_tensor);
			}

			return total_loss / (end_idx - start_idx);
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

		int get_safe_thread_count(int requested_threads) {
			int max_threads = std::thread::hardware_concurrency();
			int safe_threads = static_cast<int>(max_threads * 0.7);
			return std::min(requested_threads, safe_threads);
		}

		size_t calculate_max_gpu_batches(size_t single_batch_size) {
			// Get available GPU memory
			size_t free_memory, total_memory;
			cudaMemGetInfo(&free_memory, &total_memory);

			// Calculate memory needed per batch
			size_t memory_per_batch = 0;
			for (const auto& layer : layers_) {
				memory_per_batch += layer->get_memory_requirements(single_batch_size);
			}

			// Use 70% of available memory as safety margin
			size_t safe_memory = static_cast<size_t>(free_memory * 0.7);
			return std::max(size_t(1), safe_memory / memory_per_batch);
		}

		void initialize_cuda_streams(size_t num_streams) {
			cuda_streams_.resize(num_streams);
			for (auto& stream : cuda_streams_) {
				cudaStreamCreate(&stream);
			}
		}

		void cleanup_cuda_streams() {
			for (auto& stream : cuda_streams_) {
				cudaStreamSynchronize(stream);
				cudaStreamDestroy(stream);
			}
			cuda_streams_.clear();
		}

		float train_epoch_parallel(const std::vector<std::vector<float>>& inputs,
			const std::vector<std::vector<float>>& targets,
			size_t batch_size) {
			float total_loss = 0.0f;
			size_t num_batches = (inputs.size() + batch_size - 1) / batch_size;
			size_t max_parallel_batches = calculate_max_gpu_batches(batch_size);

			// Initialize CUDA streams for parallel processing
			initialize_cuda_streams(max_parallel_batches);

			// Process batches in parallel groups
			for (size_t batch_group = 0; batch_group < num_batches; batch_group += max_parallel_batches) {
				size_t current_group_size = std::min(max_parallel_batches,
					num_batches - batch_group);

				std::vector<Utils::Tensor> batch_gradients(current_group_size);

				// Launch parallel batch processing
				for (size_t i = 0; i < current_group_size; i++) {
					size_t batch_idx = batch_group + i;
					size_t start_idx = batch_idx * batch_size;
					size_t end_idx = std::min(start_idx + batch_size, inputs.size());

					// Process batch using CUDA stream
					cudaStream_t& stream = cuda_streams_[i];

					// Forward pass with stream
					std::vector<float> batch_loss = forward_batch_async(
						inputs, targets, start_idx, end_idx, stream);

					// Backward pass with stream
					batch_gradients[i] = backward_batch_async(stream);

					total_loss += std::accumulate(batch_loss.begin(), batch_loss.end(), 0.0f);
				}

				// Synchronize all streams in this group
				for (size_t i = 0; i < current_group_size; i++) {
					cudaStreamSynchronize(cuda_streams_[i]);
				}

				// Accumulate gradients from all batches in this group
				accumulate_gradients(batch_gradients);
			}

			// Apply optimizer updates once for the entire epoch
			if (optimizer_) {
				for (auto& layer : layers_) {
					layer->apply_accumulated_gradients(optimizer_);
				}
			}

			cleanup_cuda_streams();
			return total_loss / inputs.size();
		}

		Utils::Tensor forward_batch_async(const std::vector<std::vector<float>>& inputs,
			const std::vector<std::vector<float>>& targets,
			size_t start_idx, size_t end_idx,
			cudaStream_t& stream) {
			// Implement async forward pass using the specified CUDA stream
			Utils::Tensor batch_input = prepare_batch(inputs, start_idx, end_idx);
			Utils::Tensor output = forward_gpu_async(batch_input, stream);
			return output;
		}

		Utils::Tensor backward_batch_async(cudaStream_t& stream) {
			// Implement async backward pass using the specified CUDA stream
			// This should store gradients but not apply them
			return compute_gradients_async(stream);
		}

		void accumulate_gradients(const std::vector<Utils::Tensor>& batch_gradients) {
			// Sum up gradients from all batches
			for (auto& layer : layers_) {
				layer->accumulate_gradients(batch_gradients);
			}
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
			input_tensor.upload(input);

			Utils::Tensor output_tensor = forward_gpu(input_tensor);

			return output_tensor.download();
		}

		inline void backward(const Utils::Tensor& grad_output) {
			Utils::Tensor current_gradient(grad_output.shape());

			float* src_ptr = static_cast<float*>(grad_output.data());
			float* dst_ptr = static_cast<float*>(current_gradient.data());
			size_t size = grad_output.size() * sizeof(float);
			::cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToDevice);

			for (int i = layers_.size() - 1; i >= 0; --i) {
				current_gradient = layers_[i]->backward(current_gradient);
			}
		}

		size_t get_flops(int batch_size = 1) const {
			size_t total_flops = 0;
			for (const auto& layer : layers_) {
				total_flops += layer->get_flops(batch_size);
			}
			return total_flops;
		}

		void evaluate(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, Evaluation type = Evaluation::Regression, bool verbose = true) {
			std::vector<std::vector<float>> predictions;
			std::vector<float> latencies;

			for (size_t i = 0; i < inputs.size(); ++i) {
				auto start = std::chrono::high_resolution_clock::now();
				std::vector<float> output = forward(inputs[i], { 1, static_cast<int>(inputs[i].size()) });
				auto end = std::chrono::high_resolution_clock::now();

				float latency = std::chrono::duration<float>(end - start).count();
				latencies.push_back(latency);
				predictions.push_back(output);
			}

			Evaluations::evaluate(predictions, targets, latencies, get_flops(), type, verbose);
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

			std::cout << "\nTraining Configuration:" << std::endl;
			std::cout << "+----------------------+----------------------+----------------------+" << std::endl;
			std::cout << "| Optimizer           | Parameters           | Loss Function        |" << std::endl;
			std::cout << "+----------------------+----------------------+----------------------+" << std::endl;

			std::string optimizer_name = optimizer_ ? optimizer_->get_name() : "None";
			std::string optimizer_params = optimizer_ ? optimizer_->get_params() : "None";
			std::string loss_name = loss_function_ ? Thot::Losses::to_string(loss_function_->get_type()) : "None";
			std::string loss_params = loss_function_ ? loss_function_->get_params() : "None";

			if (optimizer_name.length() > 20) optimizer_name = optimizer_name.substr(0, 17) + "...";
			if (optimizer_params.length() > 20) optimizer_params = optimizer_params.substr(0, 17) + "...";
			if (loss_name.length() > 20) loss_name = loss_name.substr(0, 17) + "...";
			if (loss_params.length() > 20) loss_params = loss_params.substr(0, 17) + "...";

			std::cout << "| " << std::left << std::setw(20) << optimizer_name
				<< " | " << std::left << std::setw(20) << optimizer_params
				<< " | " << std::left << std::setw(20) << loss_name << " |" << std::endl;
			std::cout << "| " << std::left << std::setw(20) << ""
				<< " | " << std::left << std::setw(20) << ""
				<< " | " << std::left << std::setw(20) << loss_params << " |" << std::endl;
			std::cout << "+----------------------+----------------------+----------------------+" << std::endl;
		}

		void train(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, int epochs, int batch_size = 1, int log_interval = 100, int folds = 1) {
			if (!optimizer_) {
				optimizer_ = Thot::Optimizer::SGD(0.01f);
				for (auto& L : layers_) {
					L->set_optimizer(optimizer_);
				}
			}

			auto total_start = std::chrono::high_resolution_clock::now();
			std::vector<float> epoch_times;
			std::vector<float> fold_losses;

			// K-fold cross-validation
			for (int fold = 0; fold < folds; ++fold) {
				if (folds > 1) {
					std::cout << "\nTraining Fold " << fold + 1 << "/" << folds << std::endl;
				}

				std::vector<std::vector<float>> train_inputs, train_targets, val_inputs, val_targets;
				k_fold_split(inputs, targets, folds, fold, train_inputs, train_targets, val_inputs, val_targets);

				for (int epoch = 0; epoch < epochs; ++epoch) {
					auto epoch_start = std::chrono::high_resolution_clock::now();

					double epoch_loss = train_batch(train_inputs, train_targets);

					auto epoch_end = std::chrono::high_resolution_clock::now();
					float epoch_time = std::chrono::duration<float>(epoch_end - epoch_start).count();
					epoch_times.push_back(epoch_time);

					if (epoch % log_interval == 0 || epoch == epochs - 1) {
						std::cout << "Epoch " << epoch << " - Average Loss: " << epoch_loss;

						if (folds > 1) {
							double val_loss = 0.0;
							for (size_t i = 0; i < val_inputs.size(); ++i) {
								std::vector<float> output = forward(val_inputs[i], { 1, static_cast<int>(val_inputs[i].size()) });
								for (size_t j = 0; j < output.size(); ++j) {
									float error = output[j] - val_targets[i][j];
									val_loss += error * error;
								}
							}
							val_loss = val_loss / (2.0 * val_inputs.size());
							std::cout << " - Validation Loss: " << val_loss;
							fold_losses.push_back(val_loss);
						}
						std::cout << std::endl;
					}
				}
			}

			auto total_end = std::chrono::high_resolution_clock::now();
			float total_time = std::chrono::duration<float>(total_end - total_start).count();

			float avg_epoch_time = std::accumulate(epoch_times.begin(), epoch_times.end(), 0.0f) / (epochs * folds);
			float min_epoch_time = *std::min_element(epoch_times.begin(), epoch_times.end());
			float max_epoch_time = *std::max_element(epoch_times.begin(), epoch_times.end());
			float samples_per_second = (inputs.size() * epochs * folds) / total_time;

			std::cout << std::fixed << std::setprecision(2);

			std::cout << "\nTraining Summary:\n";
			std::cout << "----------------\n";
			std::cout << "Total Epochs: " << epochs * folds << "\n";
			if (folds > 1) {
				float avg_fold_loss = std::accumulate(fold_losses.begin(), fold_losses.end(), 0.0f) / fold_losses.size();
				float min_fold_loss = *std::min_element(fold_losses.begin(), fold_losses.end());
				float max_fold_loss = *std::max_element(fold_losses.begin(), fold_losses.end());
				std::cout << "Average Validation Loss: " << avg_fold_loss << "\n";
				std::cout << "Min Validation Loss: " << min_fold_loss << "\n";
				std::cout << "Max Validation Loss: " << max_fold_loss << "\n";
			}
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
