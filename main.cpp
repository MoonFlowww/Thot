#include "attentions/attentions.hpp"
#include "data/details/cifar.hpp"
#include "Thot/Thot.hpp"


int main() {
    bool IsLoading = false;
    std::string name ="Thot_Network_CIFAR";

	Thot::Network model(name);
    if (IsLoading)model.load("/home/moonfloww/Projects/NNs/Thot");

    else {
        Thot::ConvAlgo ConvAlgo = Thot::ConvAlgo::Direct; // Optim Method for Convulational Layers

        model.add(Thot::Layer::Conv2D(1, 28, 28, 32, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He, ConvAlgo)); // 32×28×28
        model.add(Thot::Layer::Conv2D(32, 28, 28, 32, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He, ConvAlgo)); // 32×28×28
        model.add(Thot::Layer::MaxPool2D(32, 28, 28, 2, 2)); // 32×14×14

        model.add(Thot::Layer::Conv2D(32, 14, 14, 64, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He, ConvAlgo)); // 64×14×14
        model.add(Thot::Layer::Conv2D(64, 14, 14, 64, 3, 1, 1, Thot::Activation::ReLU, Thot::Initialization::He, ConvAlgo)); // 64×14×14
        model.add(Thot::Layer::MaxPool2D(64, 14, 14, 2, 2)); // 64×7×7

        model.add(Thot::Layer::Flatten(64, 7, 7)); // 3136
        model.add(Thot::Layer::FC(3136, 128, Thot::Activation::ReLU, Thot::Initialization::He));
        model.add(Thot::Layer::FC(128, 10, Thot::Activation::Softmax, Thot::Initialization::Xavier)); // or no Softmax if CE expects logits

        model.set_loss(Thot::Loss::CategoricalCrossEntropy);
        model.set_optimizer(Thot::Optimizer::AdaMuon(3e-5f, 0.9f, 0.999f, 0.0f, Thot::LrScheduler::OneCycleDecay(0.01f, (100*(2500/512), 0.3, 25, 1e4))));

    }

	model.summary();


	std::string mnist_train = "/home/moonfloww/Projects/DATASETS/MNIST/train";
	std::string mnist_test = "/home/moonfloww/Projects/DATASETS/MNIST/test";
	//std::string cifar = "/home/moonfloww/Projects/DATASETS/CIFAR10";


    if (!IsLoading) {
        auto [x, y] = Thot::Data::Load_MNIST_Train(mnist_train, 0.1f);
        model.train(x, y, Thot::Batch::Classic(2048, 10), Thot::KFold::Classic(3), 5, true);
    }

    auto [x_test, y_test] = Thot::Data::Load_MNIST_Test(mnist_test, 1.f);


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