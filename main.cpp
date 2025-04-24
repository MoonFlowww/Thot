#include <iostream>
#include <vector>
#include "Thot/Thot.hpp"

int main() {
	Thot::Network model("Thot Network [Squared]");

	model.add(Thot::Layer::FC(2, 6, Thot::Activation::Linear, Thot::Initialization::Xavier));
	model.add(Thot::Layer::FC(6, 6, Thot::Activation::LeakyReLU, Thot::Initialization::He));
	model.add(Thot::Layer::FC(6, 1, Thot::Activation::LeakyReLU, Thot::Initialization::He));



	model.set_optimizer(Thot::Optimizer::SGDM(0.01f));

	model.summary();

	std::vector<std::vector<float>> x_train = {
		{0.0f, 0.0f},
		{0.0f, 1.0f},
		{1.0f, 0.0f},
		{1.0f, 1.0f}
	};

	std::vector<std::vector<float>> y_train = {
		{0.0f},
		{1.0f},
		{1.0f},
		{0.0f}
	};


	std::cout << "\nTraining:\n" << std::endl;

	model.train(x_train, y_train, 5000, 2, 100);

	model.evaluate(x_train, y_train, Thot::Evaluation::Binary);



	return 0;
}
