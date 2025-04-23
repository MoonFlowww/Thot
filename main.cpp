#include <iostream>
#include <vector>
#include "Thot/Thot.hpp"

void print_vector(const std::vector<float>& vec) {
	std::cout << "[";
	for (size_t i = 0; i < vec.size(); ++i) {
		std::cout << vec[i];
		if (i < vec.size() - 1) std::cout << ", ";
	}
	std::cout << "]";
}

int main() {
	Thot::Network model("Thot Network for XOR testing");

	model.add(Thot::Layer::FC(2, 6, Thot::Activation::ReLU, Thot::Initialization::Xavier));
	model.add(Thot::Layer::FC(6, 1, Thot::Activation::ReLU, Thot::Initialization::Xavier));



	model.set_optimizer(Thot::Optimizer::SGD(0.01f));

	model.summary();

	std::vector<std::vector<float>> x_train = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1}
	};

	// Output format: {batch_size, output}
	std::vector<std::vector<float>> y_train = {
		{0},
		{1},
		{1},
		{0}
	};

	int epochs = 1000;

	std::cout << "\nTraining XOR function...\n" << std::endl;

	for (int epoch = 0; epoch < epochs; ++epoch) {
		float total_loss = 0.0f;

		for (size_t i = 0; i < x_train.size(); ++i) {
			std::vector<float> input = x_train[i];
			std::vector<float> target = y_train[i];

			std::vector<int> input_shape = { 1, 2 }; 
			std::vector<float> output = model.forward(input, input_shape);

			float loss = 0.0f;
			std::vector<float> grad_output(output.size(), 0.0f);

			for (size_t j = 0; j < output.size(); ++j) {
				float error = output[j] - target[j];
				loss += error * error;
				// Gradient of MSE is 2 * (output - target)
				grad_output[j] = 2.0f * error;
			}
			loss *= 0.5f; 
			total_loss += loss;

			Thot::Utils::Tensor grad_tensor({ 1, 1 });
			grad_tensor.upload(grad_output);

			model.backward(grad_tensor, 0.01f);
		}

		if (epoch % 50 == 0 || epoch == epochs - 1) {
			std::cout << "Epoch " << epoch << " - Average Loss: " << (total_loss / x_train.size()) << std::endl;
		}
	}

	std::cout << "\nTesting XOR function:\n" << std::endl;

	for (size_t i = 0; i < x_train.size(); ++i) {
		std::vector<int> input_shape = { 1, 2 }; 
		std::vector<float> output = model.forward(x_train[i], input_shape);

		std::cout << "Input: ";
		print_vector(x_train[i]);
		std::cout << " -> Output: ";
		print_vector(output);
		std::cout << " (Expected: " << y_train[i][0] << ")" << std::endl;
	}

	return 0;
}
