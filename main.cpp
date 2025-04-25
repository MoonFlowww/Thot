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


	std::string mnist_path = "C:\\Users\\PC\\Downloads\\MNIST";
	auto [train_images, train_labels] = Thot::Data::load_mnist_train(mnist_path, 0.1f); // 10%

	model.train(train_images, train_labels, 100, 5, 5);
	//auto [test_images, test_labels] = Thot::Data::load_mnist_test(mnist_path, 0.5f);
	model.evaluate(train_images, train_labels, Thot::Evaluation::Binary);



	return 0;
}
