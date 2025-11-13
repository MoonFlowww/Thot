#include <torch/torch.h>
#include <../include/Thot.h>
#include <chrono>


struct NetImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr}, conv5{nullptr};
    torch::nn::MaxPool2d pool{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    NetImpl() {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).stride(1).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 3).stride(1).padding(1)));
        pool = register_module("pool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2, 2}).stride({2, 2})));

        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)));
        conv4 = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)));

        conv5 = register_module("conv5", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)));

        fc1 = register_module("fc1", torch::nn::Linear(1152, 524));
        fc2 = register_module("fc2", torch::nn::Linear(524, 126));
        fc3 = register_module("fc3", torch::nn::Linear(126, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1->forward(x));
        x = torch::relu(conv2->forward(x));
        x = pool->forward(x);

        x = torch::relu(conv3->forward(x));
        x = torch::relu(conv4->forward(x));
        x = pool->forward(x);

        x = torch::relu(conv5->forward(x));
        x = pool->forward(x);

        x = x.view({x.size(0), -1});
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }
};
TORCH_MODULE(Net);


int main() {
    Thot::Model model("SpeedTestMNIST");
    model.use_cuda(torch::cuda::is_available());

    model.add(Thot::Layer::Conv2d({.in_channels = 1,  .out_channels = 32,  .kernel_size = {3,3}, .stride={1,1}, .padding={1,1}, .dilation={1,1}}, Thot::Activation::ReLU, Thot::Initialization::HeUniform));
    model.add(Thot::Layer::Conv2d({.in_channels = 32, .out_channels = 32,  .kernel_size = {3,3}, .stride={1,1}, .padding={1,1}, .dilation={1,1}}, Thot::Activation::ReLU, Thot::Initialization::HeUniform));
    model.add(Thot::Layer::MaxPool2d({.kernel_size={2,2}, .stride={2,2}, .padding={0,0}}));

    model.add(Thot::Layer::Conv2d({.in_channels = 32, .out_channels = 64,  .kernel_size = {3,3}, .stride={1,1}, .padding={1,1}, .dilation={1,1}}, Thot::Activation::ReLU, Thot::Initialization::HeUniform));
    model.add(Thot::Layer::Conv2d({.in_channels = 64, .out_channels = 64,  .kernel_size = {3,3}, .stride={1,1}, .padding={1,1}, .dilation={1,1}}, Thot::Activation::ReLU, Thot::Initialization::HeUniform));
    model.add(Thot::Layer::MaxPool2d({.kernel_size={2,2}, .stride={2,2}, .padding={0,0}}));

    model.add(Thot::Layer::Conv2d({.in_channels = 64, .out_channels = 128, .kernel_size = {3,3}, .stride={1,1}, .padding={1,1}, .dilation={1,1}}, Thot::Activation::ReLU, Thot::Initialization::HeUniform));
    model.add(Thot::Layer::MaxPool2d({.kernel_size={2,2}, .stride={2,2}, .padding={0,0}}));

    model.add(Thot::Layer::Flatten());
    model.add(Thot::Layer::FC({1152, 524}, Thot::Activation::ReLU,  Thot::Initialization::HeUniform));
    model.add(Thot::Layer::FC({524, 126},  Thot::Activation::ReLU,  Thot::Initialization::HeUniform));
    model.add(Thot::Layer::FC({126, 10},   Thot::Activation::Identity, Thot::Initialization::HeUniform));

    model.set_optimizer(Thot::Optimizer::SGD({.learning_rate = 1e-3}));
    const auto ce = Thot::Loss::CrossEntropy({.label_smoothing = 0.02f});
    model.set_loss(ce);
    auto [x1, y1, x2, y2] = Thot::Data::Load::MNIST("/home/moonfloww/Projects/DATASETS/MNIST", 1.f, 1.f, true);

    auto [xvalid, yvalid] = Thot::Data::Manipulation::Fraction(x2, y2, 0.1f); // used as test set during training

    const int64_t epochs=100;
    const int64_t B=128;
    const int64_t N = x1.size(0);
    Thot::Data::Check::Size(x1); // (60000, 1, 28, 28)



    //100% Thot
    std::cout << "training 100% Thot" << std::endl;
    std::vector<std::chrono::high_resolution_clock::time_point> timepoints = { std::chrono::high_resolution_clock::now() };
    model.train(x1, y1, {.epoch=epochs, .batch_size = B,  .monitor = false, .enable_amp = true}); // .test = std::vector<at::Tensor>{xvalid, yvalid},
    timepoints.push_back(std::chrono::high_resolution_clock::now());



    //50% Thot
    std::cout << "training 50% Thot" << std::endl;
    for (auto e = 0; e < epochs; ++e) {
        for (int64_t i = 0; i < N; i += B) {
            const int64_t end = std::min(i + B, N);

            auto inputs  = x1.index({torch::indexing::Slice(i, end)}).to(model.device());
            auto targets = y1.index({torch::indexing::Slice(i, end)}).to(model.device());

            model.zero_grad();
            auto logits = model.forward(inputs);
            auto loss = Thot::Loss::Details::compute(ce, logits, targets);
            loss.backward();
            model.step();
        }
    }
    timepoints.push_back(std::chrono::high_resolution_clock::now());

    //100% libtorch
    std::cout << "training 100% LibTorch" << std::endl;

    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    Net net;
    net->to(device);
    torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(1e-3));
    torch::nn::CrossEntropyLoss criterion(torch::nn::CrossEntropyLossOptions().label_smoothing(0.02));

    net->train();
    for (int64_t e = 0; e < epochs; ++e) {
        for (int64_t i = 0; i < N; i += B) {
            const int64_t end = std::min(i + B, N);

            auto inputs  = x1.index({at::indexing::Slice(i, end)}).to(device);
            auto targets = y1.index({at::indexing::Slice(i, end)}).to(device);

            optimizer.zero_grad();
            auto logits = net->forward(inputs);
            auto loss   = criterion(logits, targets);
            loss.backward();
            optimizer.step();
        }
    }
    timepoints.push_back(std::chrono::high_resolution_clock::now());

    std::vector<std::string> titles = {"Thot + PreBuild Train()", "Thot + homemade Train()", "Libtorch Raw"};
    for(int i=1; i<timepoints.size(); ++i) {
        std::cout << titles[i-1] << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(timepoints[i]-timepoints[i-1]).count()/1000.f << "s" << std::endl;
    } std::cout << std::endl;

    const auto ThotPreBuild = std::chrono::duration_cast<std::chrono::milliseconds>(timepoints[1]-timepoints[0]).count();
    const auto ThotHomeMade = std::chrono::duration_cast<std::chrono::milliseconds>(timepoints[2]-timepoints[1]).count();
    const auto LibTorchRaw = std::chrono::duration_cast<std::chrono::milliseconds>(timepoints[3]-timepoints[2]).count();
    std::cout << "Thot vs Libtorch Overhead: " << (ThotPreBuild-LibTorchRaw)/LibTorchRaw<< "%" << std::endl;
    std::cout << "Thot PreBuild Train() Overhead: " << (ThotPreBuild-ThotHomeMade)/ThotHomeMade<< "%" << std::endl;
    std::cout << "Thot HomeMade Train() vs Libtorch Overhead: " << (ThotHomeMade-ThotHomeMade)/ThotHomeMade<< "%" << std::endl;
    return 0;
}


