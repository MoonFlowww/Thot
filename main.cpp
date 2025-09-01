#include <iostream>
#include <vector>
#include "Thot/Thot.hpp"

int main() {
	Thot::Network model("Thot Network [Sine Wave]");

    /*
    model.add(Thot::Layer::Conv2D(1, 28, 28, 32, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::Uniform));
	model.add(Thot::Layer::Conv2D(32, 28, 28, 64, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::Uniform));
	model.add(Thot::Layer::Conv2D(64, 28, 28, 64, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::Uniform));
	model.add(Thot::Layer::Conv2D(64, 28, 28, 64, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::Uniform));
	model.add(Thot::Layer::Conv2D(64, 28, 28, 64, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::Uniform));
	model.add(Thot::Layer::Conv2D(64, 28, 28, 168, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::Uniform));
	model.add(Thot::Layer::FC(10, 10, Thot::Activation::Softmax, Thot::Initialization::LeCun));
    */

    //model.add(Thot::Layer::FC(1, 1, Thot::Activation::ReLU, Thot::Initialization::Uniform));


    model.add(Thot::Layer::FC(2, 4, Thot::Activation::Tanh, Thot::Initialization::Xavier));
    model.add(Thot::Layer::FC(4, 1, Thot::Activation::Sigmoid, Thot::Initialization::Xavier));

    model.set_optimizer(Thot::Optimizer::Adam(0.1f));
    model.set_loss(Thot::Loss::MSE);


	model.summary();


	//std::string mnist_path = "/home/moonfloww/Projects/DATASETS/MNIST/train";
	//auto [x, y] = Thot::Data::load_mnist_train(mnist_path, 0.005f);
	auto [x, y] = Thot::Data::generate_data(Thot::DataType::XOR, 200, 0.0001, true);
    model.train(x, y, Thot::Batch::Classic(32, 10), Thot::KFold::Classic(1), 1, true);
    model.evaluate(x, y, Thot::Evaluation::Binary, true);


	return 0;
}
