# Thot - Neural Network Library

Thot is a high-performance neural network library built with CUDA support for efficient deep learning computations.

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
- CUDA Toolkit (version 11.0 or higher)
- C++17 compatible compiler
- CMake (version 3.10 or higher)

### Building
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## Usage

### Creating a Network

```cpp
Thot::Network model("My Neural Network");
```

### Adding Layers

#### Fully Connected Layer
```cpp
model.add(Thot::Layer::FC(
    input_size,          // Number of input neurons
    output_size,         // Number of output neurons
    activation_type,     // Activation function
    weight_init,         // Weight initialization method
    "Layer Name"         // Optional layer name
));
```

#### Recurrent Layer
```cpp
model.add(Thot::Layer::RNN(
    input_size,          // Input dimension
    hidden_size,         // Hidden state dimension
    seq_length,          // Sequence length
    activation_type,     // Activation function
    weight_init,         // Weight initialization method
    "Layer Name"         // Optional layer name
));
```

#### Convolutional Layer
```cpp
model.add(Thot::Layer::Conv2D(
    in_channels,         // Number of input channels
    in_height,          // Input height
    in_width,           // Input width
    out_channels,       // Number of output channels
    kernel_size,        // Kernel size
    stride,             // Stride
    padding,            // Padding
    activation_type,    // Activation function
    weight_init,        // Weight initialization method
    "Layer Name"        // Optional layer name
));
```

#### Restricted Boltzmann Machine
```cpp
model.add(Thot::Layer::RBM(
    visible_size,        // Number of visible units
    hidden_size,         // Number of hidden units
    cd_steps,           // Number of contrastive divergence steps
    activation_type,    // Activation function
    weight_init,        // Weight initialization method
    "Layer Name"        // Optional layer name
));
```

### Activation Functions

Available activation functions:
```cpp
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
```cpp
Thot::Initialization::Xavier
Thot::Initialization::He
Thot::Initialization::Normal
Thot::Initialization::Uniform
```

### Optimizers

#### Stochastic Gradient Descent (SGD)
```cpp
model.set_optimizer(Thot::Optimizer::SGD(learning_rate));
```

#### SGD with Momentum
```cpp
model.set_optimizer(Thot::Optimizer::SGDM(learning_rate, momentum));
```

#### Adam Optimizer
```cpp
model.set_optimizer(Thot::Optimizer::Adam(
    learning_rate,
    beta1,              // Default: 0.9
    beta2,              // Default: 0.999
    epsilon             // Default: 1e-8
));
```

### Training

```cpp
model.train(
    inputs,             // Training inputs
    targets,            // Training targets
    epochs,             // Number of epochs
    batch_size,         // Batch size (default: 1)
    log_interval        // Log interval (default: 100)
);
```

### Evaluation

#### Binary Classification
```cpp
model.evaluate(
    inputs,
    targets,
    Thot::Evaluation::Binary
);
```

#### Time Series
```cpp
model.evaluate(
    inputs,
    targets,
    Thot::Evaluation::Timeseries
);
```

#### Regression
```cpp
model.evaluate(
    inputs,
    targets,
    Thot::Evaluation::Regression
);
```

#### Multi-class Classification
```cpp
model.evaluate(
    inputs,
    targets,
    Thot::Evaluation::Classification
);
```

### Example: Building a Simple Network

```cpp
Thot::Network model("XOR Network");

// Add layers
model.add(Thot::Layer::FC(2, 4, Thot::Activation::Sigmoid, Thot::Initialization::He));
model.add(Thot::Layer::FC(4, 1, Thot::Activation::Sigmoid, Thot::Initialization::He));

// Set optimizer
model.set_optimizer(Thot::Optimizer::Adam(0.1f));

// Print model summary
model.summary();

// XOR training data
std::vector<std::vector<float>> x_train = {
    {0.0f, 0.0f},
    {0.0f, 1.0f},
    {1.0f, 0.0f},
    {1.0f, 1.0f}
};

std::vector<std::vector<float>> y_train = {
    {0.0f},
    {1.0f},
    {1.0f},
    {0.0f}
};

// Train the model
model.train(x_train, y_train, 1000, 4, 100);

// Evaluate
model.evaluate(x_train, y_train, Thot::Evaluation::Binary);

// Test predictions
std::cout << "\nTesting XOR function:\n";
for (const auto& input : x_train) {
    std::vector<float> output = model.forward(input, {1, 2});
    std::cout << "Input: [" << input[0] << ", " << input[1] << "] -> Output: " << output[0] << "\n";
}
```




## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 