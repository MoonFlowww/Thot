#include "data/details/cifar.hpp"
#include "Thot/Thot.hpp"

int main() {
	Thot::Network model("Thot Network");

    // input: 3x32x32
    model.add(Thot::Layer::Conv2D(3, 32, 32, 32, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He));
    model.add(Thot::Layer::Conv2D(32, 28, 28, 32, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He));
    model.add(Thot::Layer::MaxPool2D(32, 28, 28, 2, 2));   // → 32x14x14

    model.add(Thot::Layer::Conv2D(32, 14, 14, 64, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He));
    model.add(Thot::Layer::Conv2D(64, 14, 14, 64, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He));
    model.add(Thot::Layer::MaxPool2D(64, 14, 14, 2, 2));   // → 64x7x7

    model.add(Thot::Layer::Conv2D(64, 7, 7, 128, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He));
    model.add(Thot::Layer::Conv2D(128, 7, 7, 128, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He));
    model.add(Thot::Layer::MaxPool2D(128, 7, 7, 2, 2));    // → 128x3x3

    model.add(Thot::Layer::Flatten(128, 3, 3));            // → 1152

    model.add(Thot::Layer::FC(1152, 256, Thot::Activation::ReLU, Thot::Initialization::He));
    model.add(Thot::Layer::FC(256, 10, Thot::Activation::Softmax, Thot::Initialization::Xavier));

    model.set_loss(Thot::Loss::CrossEntropy);
    model.set_optimizer(Thot::Optimizer::Adam(0.001f));



	model.summary();


	//std::string mnist_train = "/home/moonfloww/Projects/DATASETS/MNIST/train";
	//std::string mnist_test = "/home/moonfloww/Projects/DATASETS/MNIST/test";
	std::string cifar = "/home/moonfloww/Projects/DATASETS/CIFAR10";
    auto [x, y, x_test, y_test] = Thot::Data::Load_CIFAR10(cifar, 0.05f, 0.05f);

    model.train(x, y, Thot::Batch::Classic(32, 10), Thot::KFold::Classic(5), 1, true);

    model.evaluate(x_test, y_test, Thot::Evaluation::Classification, true);

	return 0;
}
