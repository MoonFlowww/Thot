# Thot Framework Overview

Thot is a modular C++ deep learning framework built on top of LibTorch. It provides
strongly-typed building blocks for prototyping neural networks, composing data
pipelines, training models, and evaluating their performance. This guide walks you
through the core primitives and shows how to assemble end-to-end experiments

## Creating a Model

Create a model by giving it a descriptive name. The model can be configured to run
on CPU or GPU depending on CUDA availability.

```cpp
Thot::Model model("_Network_Name_");
model.to_device(torch::cuda::is_available()); // accept true || false
```

The model holds a directed acyclic graph of computational blocks and layers. You can
construct it incrementally with `.add()`.

## Adding Blocks and Layers

Each `.add()` call can be tagged with an identifier to make graph connections clearer.

```cpp
model.add(Thot::Layer::FC({258, 10, /*bias*/true}, Thot::Activation::GeLU, Thot::Initialization::HeNormal))
```
or blocks:
```cpp
model.add(Thot::Block::Sequential({ /*vector field*/
    Thot::Layer::Conv2d({3, 64, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false},
        Thot::Activation::Identity, Thot::Initialization::HeNormal),
    Thot::Layer::BatchNorm2d({64, 1e-5, 0.1, true, true},
        Thot::Activation::SiLU),
    Thot::Layer::MaxPool2d({{2, 2}, {2, 2}})
}));
```
The framework ships with a rich catalog of layers (see in [Layers](sub/layer/README.md) or [Blocks](sub/block/README.md)). It will automatically link linearly every item's called via `.add()`. To rewire the network use `.links()` (see in [Links](sub/links/README.md))

## Configuring Optimization

Optimizer and scheduler choices are set once per model by default. 

The example below pairs
AdamW with cosine annealing warm restarts.

```cpp
model.set_optimizer(
    Thot::Optimizer::AdamW({.learning_rate = 1e-4, .weight_decay = 5e-4}),
    Thot::LrScheduler::CosineAnnealing({
        .T_max = steps_per_epoch * epochs,
        .eta_min = 3e-7,
        .warmup_steps = 5 * steps_per_epoch,
        .warmup_start_factor = 0.1
    })
);
```

Losses and regularization follow the same pattern:

```cpp
model.set_loss(Thot::Loss::CrossEntropy({.label_smoothing = 0.02f}));
model.set_regularization({ /*vector field*/
    Thot::Regularization::SWAG({
        .coefficient = 1e-3,
        .variance_epsilon = 1e-6,
        .start_step = static_cast<size_t>(0.85 * steps_per_epoch * epochs),
        .accumulation_stride = static_cast<size_t>(steps_per_epoch),
        .max_snapshots = 20,
    })
});
```
To see the complete list of Optimizers, Losses or Regularizations and their parameters check [Optimizer](sub/optimizer/README.md), [Loss](sub/loss/README.md) and [Regularization](sub/regularization/README.md)

It is also possible to use multiples configurations over the network, check [Local](sub/local/README.md)
## Working with Data

The `Thot::Data::Load` namespace includes ready-made loaders for popular datasets
such as MNIST, CIFAR-10, ETTH, PTBXL. Data manipulations (augmentation, shuffling, and splitting) are
exposed through `Thot::Data::Manipulation` utilities, while consistency checks live
under `Thot::Data::Check`.

```cpp
at::Tensor [train_images, train_labels, test_images, test_labels] = Thot::Data::Load::CIFAR10(dataset_root, 1.f, 1.f, true);

at::Tensor [validation_images, validation_labels] = Thot::Data::Manipulation::Fraction(test_images, test_labels, 0.1f);

std::tie(train_images, train_labels) = Thot::Data::Manipulation::Cutout(train_images, train_labels,{-1, -1}, {12, 12}, -1, 1.f, true, false);
```
More information in [Thot::Data](sub/data/README.md)

## Training

Training is initiated with `model.train`, which accepts tensors and a
`Thot::TrainOptions` struct describing epochs, batch size, graph mode, validation
splits, AMP, and other runtime settings.

```cpp
model.train(train_images, train_labels, {.epoch=120, .batch_size=128, .test={x_val,y_val}});
```
Full details of parameters and process in [Train](sub/training/README.md)
## Evaluation and Metrics

Post-training evaluation is performed with `model.evaluate`, which accepts the test
split, a task type, and a list of metrics. The evaluation API streams batches and
accumulates metrics such as accuracy, precision, recall, calibration errors, and
more.

```cpp
model.evaluate(test_images, test_labels, Thot::Evaluation::Classification, { /*vector field*/
    Thot::Metric::Classification::Precision,
    Thot::Metric::Classification::Recall,
    Thot::Metric::Classification::F1,
    Thot::Metric::Classification::TruePositiveRate,
    Thot::Metric::Classification::LogLoss,
}, {.batch_size = 64});
```
More details inside [Evaluation](sub/evaluation/README.md)

## Saving and Loading

To keep save your Network use `model.save()` and `model.load()`;
`model.save()` will create a folder of `_Network_Name_` name, and save inside `architecture.json` which correspond to Network layers, dimensions parameters, optimizer used, etc. As well as a `parameter.binary` which store learnable parameters of layers. 

NB: Since `model.load()` read `architecture.json`, you don't need to re-code your network via `model.add()`


```cpp
model.save("PATH");
model.load("PATH"+"/_Network_Name_");
```
Details in: [Save&Load](sub/saveload/README.md)
## Next Steps

