#include <iostream>
#include <vector>
#include "Thot/Thot.hpp"

int main() {
	Thot::Network model("Thot Network");

    model.add(Thot::Layer::Conv2D(1, 28, 28, 32, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He));
    model.add(Thot::Layer::Conv2D(32, 28, 28, 32, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He));
    model.add(Thot::Layer::MaxPool2D(32, 28, 28, 2, 2));

    model.add(Thot::Layer::Flatten(32, 14, 14));

    model.add(Thot::Layer::FC(32 * 14 * 14, 128, Thot::Activation::ReLU, Thot::Initialization::He));
    model.add(Thot::Layer::FC(128, 10, Thot::Activation::Softmax, Thot::Initialization::Xavier));


    model.set_optimizer(Thot::Optimizer::Adam(0.001f));
    model.set_loss(Thot::Loss::CrossEntropy);


	model.summary();


	std::string mnist_path = "/home/moonfloww/Projects/DATASETS/MNIST/train";
	auto [x, y] = Thot::Data::load_mnist_train(mnist_path, 0.05f);
	//auto [x, y] = Thot::Data::generate_data(Thot::DataType::XOR, 200, 0.0001, true);
    model.train(x, y, Thot::Batch::Classic(32, 10), Thot::KFold::Classic(5), 1, true);
    model.evaluate(x, y, Thot::Evaluation::Classification, true);


	return 0;
}
