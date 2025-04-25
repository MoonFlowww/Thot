#include <iostream>
#include <vector>
#include "Thot/Thot.hpp"

int main() {
	Thot::Network model("Thot Network [Squared]");

	model.add(Thot::Layer::Conv2D(1, 28, 28, 32, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::LeCun));
	model.add(Thot::Layer::Conv2D(32, 28, 28, 64, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::LeCun));
	model.add(Thot::Layer::Conv2D(64, 28, 28, 126, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::LeCun));
	model.add(Thot::Layer::FC(128 * 28 * 28, 10, Thot::Activation::Softmax, Thot::Initialization::LeCun));
	model.set_optimizer(Thot::Optimizer::Adam(0.00001f));
	model.set_loss(Thot::Loss::BinaryCrossEntropy);
	model.summary();


	std::string mnist_path = "C:\\Users\\PC\\Downloads\\MNIST";
	auto [train_images, train_labels] = Thot::Data::load_mnist_train(mnist_path, 0.001f); // 10%

	model.train(train_images, train_labels, 10, 64, 100, 5);
	model.evaluate(train_images, train_labels, Thot::Evaluation::Classification);


	return 0;
}
