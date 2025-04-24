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
	Thot::Network model("Thot Network [Squared]");

	model.add(Thot::Layer::FC(1, 16, Thot::Activation::Sigmoid, Thot::Initialization::He));
	model.add(Thot::Layer::FC(16, 16, Thot::Activation::ReLU, Thot::Initialization::He));
	model.add(Thot::Layer::FC(16, 16, Thot::Activation::ReLU, Thot::Initialization::He));
	model.add(Thot::Layer::FC(16, 8, Thot::Activation::ReLU, Thot::Initialization::He));
	model.add(Thot::Layer::FC(8, 1, Thot::Activation::Linear, Thot::Initialization::He));



	model.set_optimizer(Thot::Optimizer::Adam(0.001f));

	model.summary();

	std::vector<std::vector<float>> x_train = { {0.1}, {0.2}, {0.3}, {0.4}, {0.5}, {0.6}, {0.7}, {0.8}, {0.9} };

	std::vector<std::vector<float>> y_train = { {0.01}, {0.04}, {0.09}, {0.16}, {0.25}, {0.36}, {0.49}, {0.64}, {0.81} };


	std::cout << "\nTraining XOR function...\n" << std::endl;

	model.train(x_train, y_train, 1000, 5, 0.01f, 50);

	std::cout << "\nTesting XOR function:\n" << std::endl;

	model.evaluate(x_train, y_train);

	return 0;
}
