#include <iostream>
#include <vector>
#include "Thot/Thot.hpp"

int main() {
	Thot::Network model("Thot Network [Sine Wave]");

	/*
		model.add(Thot::Layer::Conv2D(1, 28, 28, 32, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::LeCun));
	model.add(Thot::Layer::Conv2D(32, 28, 28, 64, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::LeCun));
	model.add(Thot::Layer::Conv2D(64, 28, 28, 126, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::LeCun));
	model.add(Thot::Layer::FC(128 * 28 * 28, 10, Thot::Activation::Softmax, Thot::Initialization::LeCun));

	*/
	model.add(Thot::Layer::RNN(5, 8, 4, Thot::Activation::Tanh));
	model.add(Thot::Layer::RNN(8, 16, 4, Thot::Activation::Tanh));
	model.add(Thot::Layer::RNN(16, 32, 4, Thot::Activation::Tanh));
	model.add(Thot::Layer::RNN(32, 16, 4, Thot::Activation::Tanh));
	model.add(Thot::Layer::RNN(16, 1, 4, Thot::Activation::Tanh));
	model.set_optimizer(Thot::Optimizer::Adam(0.001f));
	model.set_loss(Thot::Loss::MSE);
	model.summary();


	std::string mnist_path = "C:\\Users\\PC\\Downloads\\MNIST";
	//auto [train_images, train_labels] = Thot::Data::load_mnist_train(mnist_path, 0.001f); // 10%
	auto [x, y] = Thot::Data::generate_data(Thot::DataType::Sine, 2000, 0.1, true);
    model.train(x, y, Thot::Batch::Classic(16, 200), Thot::KFold::Classic(1), 20, true);
    model.evaluate(x, y, Thot::Evaluation::Regression, true);


	return 0;
}
