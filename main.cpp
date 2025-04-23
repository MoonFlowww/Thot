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

	model.add(Thot::Layer::FC(2, 6, Thot::Activation::ReLU, Thot::Initialization::He));
	model.add(Thot::Layer::FC(6, 6, Thot::Activation::ReLU, Thot::Initialization::Xavier));
	model.add(Thot::Layer::FC(6, 1, Thot::Activation::ReLU, Thot::Initialization::LeCun));



	model.set_optimizer(Thot::Optimizer::SGDM(0.01f));
	model.summary();

	std::vector<std::vector<float>> x_train = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1}
	};

	std::vector<std::vector<float>> y_train = {
		{0},
		{1},
		{1},
		{0}
	};

	int epochs = 5000;

	std::cout << "\nTraining XOR function...\n" << std::endl;

	model.fit(x_train, y_train, 1000, 1, 0.01f, 50);

	std::cout << "\nTesting XOR function:\n" << std::endl;

	model.evaluate(x_train, y_train);

	return 0;
}
