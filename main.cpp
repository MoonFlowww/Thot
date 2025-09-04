#include "attentions/attentions.hpp"
#include "data/details/cifar.hpp"
#include "Thot/Thot.hpp"


int main() {
    bool IsLoading = false;
    std::string name ="Thot_Network_CIFAR";

	Thot::Network model(name);
    if (IsLoading)model.load("/home/moonfloww/Projects/NNs/Thot");

    else {
        //model.add(Thot::Attention::MLA(3*32*32, 8, 256, Thot::Initialization::He));

        model.add(Thot::Layer::Conv2D(3, 32, 32, 32, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He));
        model.add(Thot::Layer::Conv2D(32, 32, 32, 32, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He));
        model.add(Thot::Layer::MaxPool2D(32, 32, 32, 2, 2));   // → 32x16x16

        model.add(Thot::Layer::Conv2D(32, 16, 16, 64, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He));
        model.add(Thot::Layer::Conv2D(64, 16, 16, 64, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He));
        model.add(Thot::Layer::MaxPool2D(64, 16, 16, 2, 2));   // → 64x8x8

        model.add(Thot::Layer::Conv2D(64, 8, 8, 128, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He));
        model.add(Thot::Layer::Conv2D(128, 8, 8, 128, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He));
        model.add(Thot::Layer::MaxPool2D(128, 8, 8, 2, 2));    // → 128x4x4

        model.add(Thot::Layer::Flatten(128, 4, 4));            // → 2048

        model.add(Thot::Layer::RBM(2048, 1024, 4, Thot::Activation::Tanh, Thot::Initialization::Xavier));
        model.add(Thot::Layer::RBM(1024, 512, 4, Thot::Activation::Tanh, Thot::Initialization::Xavier));
        model.add(Thot::Layer::RBM(512, 256, 4, Thot::Activation::Tanh, Thot::Initialization::Xavier));

        model.add(Thot::Layer::FC(256, 128, Thot::Activation::ReLU, Thot::Initialization::He));
        model.add(Thot::Layer::FC(128, 64, Thot::Activation::ReLU, Thot::Initialization::He));
        model.add(Thot::Layer::FC(64, 32, Thot::Activation::ReLU, Thot::Initialization::He));
        model.add(Thot::Layer::FC(32, 10, Thot::Activation::Softmax, Thot::Initialization::He));

        model.set_loss(Thot::Loss::CategoricalCrossEntropy);
        model.set_optimizer(Thot::Optimizer::AdaMuon(1e-4f));

    }




	model.summary();


	//std::string mnist_train = "/home/moonfloww/Projects/DATASETS/MNIST/train";
	//std::string mnist_test = "/home/moonfloww/Projects/DATASETS/MNIST/test";
	std::string cifar = "/home/moonfloww/Projects/DATASETS/CIFAR10";


    if (!IsLoading) {
        auto [x, y] = Thot::Data::Load_CIFAR10_Train(cifar, 0.5f);
        model.train(x, y, Thot::Batch::Classic(512, 10), Thot::KFold::Classic(10), 5, true);
    }

    auto [x_test, y_test] = Thot::Data::Load_CIFAR10_Test(cifar, 1.f);


    model.evaluate(x_test, y_test, Thot::Evaluation::Classification, true);
    if (!IsLoading) {
        char rep;
        std::cout << "Would you like to save it [y/n]" << std::endl;
        std::cin >> rep;
        if (rep == std::tolower(rep)) {
            model.save("/home/moonfloww/Projects/NNs/Thot");
        }
    }

	return 0;
}
