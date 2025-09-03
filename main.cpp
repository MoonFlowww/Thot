#include "attentions/attentions.hpp"
#include "data/details/cifar.hpp"
#include "Thot/Thot.hpp"

int main() {
	Thot::Network model("Thot Network");

    // input: 3x32x32
    model.add(Thot::Attention::MHA(256,8,Thot::Initialization::He));
    model.add(Thot::Layer::Conv2D(3, 32, 32, 32, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He));
    model.add(Thot::Layer::MaxPool2D(32, 32, 32, 2, 2));   // → 32x16x16

    model.add(Thot::Layer::Conv2D(32, 16, 16, 64, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He));
    model.add(Thot::Layer::Conv2D(64, 16, 16, 64, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He));
    model.add(Thot::Layer::MaxPool2D(64, 16, 16, 2, 2));   // → 64x8x8

    model.add(Thot::Layer::Conv2D(64, 8, 8, 128, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He));
    model.add(Thot::Layer::Conv2D(128, 8, 8, 128, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He));
    model.add(Thot::Layer::MaxPool2D(128, 8, 8, 2, 2));    // → 128x4x4

    model.add(Thot::Layer::Flatten(128, 4, 4));            // → 2048

    model.add(Thot::Layer::FC(2048, 256, Thot::Activation::ReLU, Thot::Initialization::He));
    model.add(Thot::Layer::FC(256, 10, Thot::Activation::Softmax, Thot::Initialization::Xavier));

    model.set_loss(Thot::Loss::CategoricalCrossEntropy);
    model.set_optimizer(Thot::Optimizer::Adam(0.0001f));



	model.summary();


	//std::string mnist_train = "/home/moonfloww/Projects/DATASETS/MNIST/train";
	//std::string mnist_test = "/home/moonfloww/Projects/DATASETS/MNIST/test";
	std::string cifar = "/home/moonfloww/Projects/DATASETS/CIFAR10";
    auto [x, y, x_test, y_test] = Thot::Data::Load_CIFAR10(cifar, 0.005f, 0.1f);

    model.train(x, y, Thot::Batch::Classic(512, 15), Thot::KFold::Classic(5), 5, true);

    model.evaluate(x_test, y_test, Thot::Evaluation::Classification, true);

	return 0;
}
