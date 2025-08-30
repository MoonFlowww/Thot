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


	model.add(Thot::Layer::RNN(2, 64, 4, Thot::Activation::Tanh, Thot::Initialization::Uniform));
	model.add(Thot::Layer::RNN(64, 64, 4, Thot::Activation::Tanh, Thot::Initialization::Uniform));
	model.add(Thot::Layer::RNN(64, 64, 4, Thot::Activation::Tanh, Thot::Initialization::Uniform));
	model.add(Thot::Layer::RNN(64, 64, 4, Thot::Activation::Tanh, Thot::Initialization::Uniform));
	model.add(Thot::Layer::RNN(64, 64, 4, Thot::Activation::Tanh, Thot::Initialization::Uniform));
	model.add(Thot::Layer::RNN(64, 64, 4, Thot::Activation::Tanh, Thot::Initialization::Uniform));
	model.add(Thot::Layer::RNN(64, 64, 4, Thot::Activation::Tanh, Thot::Initialization::Uniform));
	model.add(Thot::Layer::RNN(64, 64, 4, Thot::Activation::Tanh, Thot::Initialization::Uniform));
	model.add(Thot::Layer::RNN(64, 64, 4, Thot::Activation::Tanh, Thot::Initialization::Uniform));
	model.add(Thot::Layer::RNN(64, 64, 4, Thot::Activation::Tanh, Thot::Initialization::Uniform));
	model.add(Thot::Layer::RNN(64, 64, 4, Thot::Activation::Tanh, Thot::Initialization::Uniform));
	model.add(Thot::Layer::RNN(64, 64, 4, Thot::Activation::Tanh, Thot::Initialization::Uniform));
	model.add(Thot::Layer::RNN(64, 32, 4, Thot::Activation::Tanh, Thot::Initialization::Uniform));
	model.add(Thot::Layer::RNN(32, 16, 4, Thot::Activation::Tanh, Thot::Initialization::Uniform));
	model.add(Thot::Layer::RNN(8, 1, 4, Thot::Activation::Sigmoid, Thot::Initialization::Uniform));


    model.set_optimizer(Thot::Optimizer::Adam(0.001f));
    model.set_loss(Thot::Loss::MSE);


	model.summary();


	//std::string mnist_path = "/home/moonfloww/Projects/DATASETS/MNIST/train";
	//auto [x, y] = Thot::Data::load_mnist_train(mnist_path, 0.005f);
	auto [x, y] = Thot::Data::generate_data(Thot::DataType::XOR, 20000, 0.0001, true);
    model.train(x, y, Thot::Batch::Classic(64, 50), Thot::KFold::Classic(5), 1, true);
    model.evaluate(x, y, Thot::Evaluation::Classification, true);


	return 0;
}
