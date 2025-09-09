# Thot - Neural Network Library


## Features

- GPU-accelerated neural network operations
- Multiple layer types
- Various activation functions
- Different optimization algorithms
- Flexible initialization methods
- Comprehensive evaluation metrics
- Easy-to-use API

## Installation

### Prerequisites
- CUDA Toolkit (version 12.5 or higher)
- C++17 compatible compiler
- GPU architectures (`61;75;86;89`).

To target a specific GPU, pass
`-DCMAKE_CUDA_ARCHITECTURES=<arch>` when invoking CMake.


## Usage

### Creating a Network

```cpp
Thot::Network model("Neural Network");
```

### Layers
#### Adding Layers
```cpp
model.add(Thot::Layer::FC(1, 1, Thot::Activation::ReLU, Thot::Initialization::Xavier));
// Fully connected, 1 input, 1 neuron, ReLU activation and Xavier initialization
```

#### Layers
Available Layers:
```yaml
Thot::Layer::FC(input_size, output_size, __activation_type__, __initialization_type__)
Thot::Layer::RNN(input_size, hidden_size, seq_length, __activation_type__, __initialization_type__)
Thot::Layer::Conv2D(in_channels, in_height, in_width, out_channels, kernel_size, stride, padding, __activation_type__, __initialization_type__, __conv_algo__, "Layer Name")
Thot::Layer::RCNN(in_channels, in_height, in_width, out_channels, kernel_size, stride, padding, pooled_h, pooled_w, __activation_type__, __initialization_type__, "Layer Name")
Thot::Layer::VAE(input_size, latent_size, __activation_type__, __initialization_type__, "Layer Name")
Thot::Layer::RBM(visible_size, hidden_size, cd_steps, __activation_type__, __initialization_type__, "Layer Name")
Thot::Layer::Spike(size, threshold)
Thot::Layer::SparseAE(input_size, latent_size, __activation_type__, __initialization_type__, use_sparsity, use_contractive, sparsity_rho, sparsity_beta, contractive_lambda, "Layer Name")
Thot::Layer::MaxPool2D(in_channels, in_height, in_width, kernel_size, stride)
Thot::Layer::Flatten(in_channels, in_height, in_width)
```
The optional `__conv_algo__` argument for `Conv2D` chooses the convolution
backend. Supported values are `auto`, `direct`, `winograd` and `fft`. The
default `auto` selects a heuristic based on kernel size and stride. `winograd`
implements F(2x2,3x3) transforms with precomputed filter tiles, while `fft`
performs convolution via a frequency-domain multiply using a naïve 2‑D DFT.

#### Attention Layers
Available Attention Mechanisms:
```yaml
Thot::Attention::MHA(embed_dim, num_heads, __initialization_type__, "Layer Name")
Thot::Attention::MLA(embed_dim, num_heads, latent_dim, __initialization_type__, "Layer Name")
```


### Activation Functions
Available activation functions:
```yaml
Thot::Activation::Linear
Thot::Activation::ReLU
Thot::Activation::Sigmoid
Thot::Activation::Tanh
Thot::Activation::LeakyReLU
Thot::Activation::ELU
Thot::Activation::GELU
Thot::Activation::Softmax
```

### Initialization Methods
Available initialization methods:
```yaml
Thot::Initialization::Zeros
Thot::Initialization::Ones
Thot::Initialization::Normal
Thot::Initialization::Uniform
Thot::Initialization::Xavier
Thot::Initialization::He
Thot::Initialization::LeCun
```

### Optimizers
#### Choosing an Optimizer
```cpp
model.set_optimizer(Thot::Optimizer::SGD(learning_rate));
```
#### Available Optimizers:
```yaml
Thot::Optimizer::SGD(learning_rate)
Thot::Optimizer::SGDM(learning_rate, momentum)
Thot::Optimizer::Adam(learning_rate, beta1, beta2, epsilon)
Thot::Optimizer::Muon(learning_rate, beta, weight_decay)
Thot::Optimizer::AdaMuon(learning_rate, beta1, beta2, weight_decay)
```


### Loss
#### Choosing a Loss function
```cpp
model.set_loss(Thot::Loss::MSE);
```
#### Available Losses:
```yaml
Thot::Loss::MSE
Thot::Loss::MAE
Thot::Loss::BinaryCrossEntropy
Thot::Loss::CrossEntropy
Thot::Loss::CategoricalCrossEntropy
Thot::Loss::SparseCategoricalCrossEntropy
Thot::Loss::Hinge
Thot::Loss::Huber
Thot::Loss::KLDivergenc
Thot::Loss::VAE
```



### Train
```cpp
model.train(inputs, targets, Thot::Batch::Classic(batch_size, epochs_per_fold), Thot::KFold::Classic(folds), verbose_every_n_epoch, bool_verbose);
```


### Evaluation
```cpp
model.evaluate(test_inputs, test_targets, __evaluation_mode__, bool_verbose);
```
#### Evaluation Mode
```yaml
Thot::Evaluation::Binary
Thot::Evaluation::Timeseries
Thot::Evaluation::Regression
Thot::Evaluation::Classification
```

### Example: Building a basic CNN for MNIST classification

```cpp
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


    // Print Model Summary
	model.summary();


    // Loading Train and Test MNIST data
	std::string mnist_train = "MNIST/Train"; // path -> folder in which files are
	std::string mnist_test = "MNIST/Test";
    auto [x, y, x_test, y_test] = Thot::Data::Load_MNIST(mnist_train, mnist_test, 0.05f, 0.15f);
								  // Train: 5% of total mnist train
								  // Test: 15% of total mnist test

    // Training via Train Samples
    model.train(x, y, Thot::Batch::Classic(32, 10), Thot::KFold::Classic(5), 1, true);

    // Evaluation via Test Samples
    model.evaluate(x_test, y_test, Thot::Evaluation::Classification, true);
    
	return 0;
}

```




## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 