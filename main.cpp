#include "Thot/Thot.hpp"

int main() {
	Thot::Network model("Thot Network");

    //Core model
    model.add(Thot::Layer::Conv2D(1, 28, 28, 32, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He));
    model.add(Thot::Layer::Conv2D(32, 28, 28, 32, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He));
    model.add(Thot::Layer::MaxPool2D(32, 28, 28, 2, 2));
    model.add(Thot::Layer::Flatten(32, 14, 14));

    //fine-tune
    model.add(Thot::Layer::FC(32 * 14 * 14, 128, Thot::Activation::ReLU, Thot::Initialization::He));
    model.add(Thot::Layer::FC(128, 10, Thot::Activation::Softmax, Thot::Initialization::Xavier));

    //Loss & Optimizer
    model.set_loss(Thot::Loss::CrossEntropy);
    model.set_optimizer(Thot::Optimizer::Adam(0.001f));


    // Model Summary
	model.summary();


    //Loading train and test MNIST data
	std::string mnist_train = "/home/moonfloww/Projects/DATASETS/MNIST/train";
	std::string mnist_test = "/home/moonfloww/Projects/DATASETS/MNIST/test";
    auto [x, y, x_test, y_test] = Thot::Data::Load_MNIST(mnist_train, mnist_test, 0.05f, 0.15f); // 5% of total mnist train and 15% of total mnist test

    //train
    model.train(x, y, Thot::Batch::Classic(32, 10), Thot::KFold::Classic(5), 1, true);

    //test
    model.evaluate(x_test, y_test, Thot::Evaluation::Classification, true);

	return 0;
}
