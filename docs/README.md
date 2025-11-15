# Welcome to Thot
Thot is a modern C++ deep-learning framework that layers a strongly-typed API over LibTorch. It is built for practitioners who enjoy the ergonomics of high-level frameworks but need deterministic control over kernels, data layout, and optimization steps. Thot lets you compose reusable blocks, stream large datasets, and run telemetry-grade training loops from a single, cohesive interface.
## Why Thot?
* **First-class graph authoring.** Layers and higher-order blocks can be connected as a DAG, letting you express anything from small CNNs to large transformer stacks without wrestling with manual tensor plumbing.
* **Consistent systems model.** Data loaders, augmentations, optimizers, regularizers, and metrics share the same descriptor-driven style so you can mix-and-match building blocks safely.
* **Native performance.** Thot keeps you close to the metal through LibTorch while still providing ergonomic abstractions. Benchmarks at the end of this document detail the runtime overhead compared to pure LibTorch.

## Quick Start
```cpp
#include <thot/model.h>

int main() {
    Thot::Model model("demo");
    model.use_cuda(torch::cuda::is_available());

    model.add(Thot::Layer::FC({784, 256, true}, Thot::Activation::GeLU));
    model.add(Thot::Layer::Dropout({0.1}));
    model.add(Thot::Layer::FC({256, 10, true}, Thot::Activation::Softmax));

    model.set_optimizer(
        Thot::Optimizer::AdamW({.learning_rate = 1e-3}),
        Thot::LrScheduler::CosineAnnealing({.T_max = 50})
    );
    auto [train_images, train_labels, test_images, test_labels] =
        Thot::Data::Load::MNIST("./datasets", 1.f, 1.f, true);
        
    model.train(train_images, train_labels, {.epoch = 10, .batch_size = 64});
    model.evaluate(test_images, test_labels, Thot::Evaluation::Classification, {
        Thot::Metric::Classification::Precision,
        Thot::Metric::Classification::F1,
        Thot::Metric::Classification::Informedness});
}
```



## Adding Layers and Blocks

The model holds a directed acyclic graph of computational blocks and layers. You can
construct it incrementally with `.add()`.

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
The framework ships with a rich catalog of layers (see in [Docs/Layers](sub/layer/README.md) or [Docs/Blocks](sub/block/README.md)). It will automatically link linearly every item's called via `.add()`. To rewire the network use `.links()` (see in [Docs/Links](sub/links/README.md)). Multi-head attention descriptors that power the transformer blocks are documented in [Docs/Attention](sub/attention/README.md).


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
To see the complete list of Optimizers, Losses or Regularizations and their parameters check [Docs/Optimizer](sub/optimizer/README.md), [Docs/Loss](sub/loss/README.md) and [Docs/Regularization](sub/regularization/README.md)

It is also possible to use multiples configurations over the network, check [Docs/Local](sub/local/README.md)
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
More information inside [Docs/Data](sub/data/README.md)

## Training

Training is initiated with `model.train`, which accepts tensors and a
`Thot::TrainOptions` struct describing epochs, batch size, graph mode, validation
splits, AMP, and other runtime settings.

```cpp
model.train(train_images, train_labels, {.epoch=120, .batch_size=128, .test={x_val,y_val}});
```
More information in [Docs/Train](sub/training/README.md)
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
More details inside [Docs/Evaluation](sub/evaluation/README.md)

## Saving and Loading

To keep save your Network use `model.save()` and `model.load()`;
`model.save()` will create a folder of `_Network_Name_` name, and save inside `architecture.json` which correspond to Network layers, dimensions parameters, optimizer used, etc. As well as a `parameter.binary` which store learnable parameters of layers. 

NB: Since `model.load()` read `architecture.json`, you don't need to re-code your network via `model.add()`


```cpp
model.save("PATH");
model.load("PATH"+"/_Network_Name_");
```
`model.save` generates a folder named after the model containing the `architecture.json` (graph, dimensions, optimizer metadata) and `parameters.binary` (learnable weights). Because `model.load` reads the JSON specification, you do not need to recreate the graph via `model.add`. Details live in [Save & Load](sub/saveload/README.md).

## Latency Benchmarks

Results below represent warm runs filtered with a Tukey 0.98 fence on the MNIST workload  
(60k samples, 28×28 | epochs = 100, batch = 64).

Two configurations are reported:

1. **Mixed I/O:** async pinned memory enabled only in **Thot::Train()**.
2. **Unified I/O:** async pinned memory enabled in **all** runners (Thot prebuilt, Thot custom, LibTorch).

---

### 1) Mixed I/O — Async pinned memory only in `Thot::Train()`

| Runner                        | Steps (filtered) | Mean (ms) |    Std |      CV |    P10 |    P50 |    P90 |    P98 |   Mode | Throughput (steps/s) |
|------------------------------|-----------------:|----------:|-------:|--------:|-------:|-------:|-------:|-------:|-------:|----------------------:|
| **Thot — Prebuilt Train()**  |          76 916  | **1.20268** | 0.00157 | **0.00131** | 1.20049 | 1.20302 | 1.20451 | 1.20537 | 1.20398 | **831.47** |
| **Thot — Custom Train()**    |          91 027  | 1.33688 | 0.18792 | 0.14057 | 1.17145 | 1.23006 | 1.65251 | 1.72896 | 1.19031 | 748.01 |
| **LibTorch Raw**             |          90 837  | 1.27572 | 0.18145 | 0.14224 | 1.12161 | 1.16910 | 1.59117 | 1.66006 | 1.13251 | 783.87 |

- **CV** (coefficient of variation) = `Std / Mean`. Lower = less jitter.

#### Overhead (relative to mean latency, positive = slower than reference)

| Comparison                                      |   Value |
|-------------------------------------------------|--------:|
| **Thot** vs **LibTorch** Overhead               |  -5.73% |
| **Thot** Prebuilt vs **Thot** Custom Overhead   | -10.04% |
| **Thot** Custom vs **LibTorch** Overhead        |  +4.79% |

In this configuration, Thot’s prebuilt `Train()` benefits from async pinned memory while the other runners do not, so this setup is *favorable* to the prebuilt runner and mainly illustrates the impact of I/O configuration.

---

### 2) Unified I/O — Async pinned memory in all runners

| Runner                        | Steps (filtered) | Mean (ms) |     Std |      CV |    P10 |    P50 |    P90 |    P98 |   Mode | Throughput (steps/s) |
|------------------------------|-----------------:|----------:|--------:|--------:|-------:|-------:|-------:|-------:|-------:|----------------------:|
| **Thot — Prebuilt Train()**  |          71 288  | 1.06486 | 0.00184 | 0.00172 | 1.06275 | 1.06475 | 1.06702 | 1.06889 | 1.06298 | 939.09 |
| **Thot — Custom Train()**    |          75 622  | 1.06443 | 0.01764 | 0.01657 | 1.04117 | 1.06435 | 1.08850 | 1.10319 | 1.06208 | 939.47 |
| **LibTorch Raw**             |          80 820  | **1.02841** | 0.00512 | 0.00498 | 1.02150 | 1.02813 | 1.03556 | 1.03934 | 1.02704 | **972.37** |

- **CV** (coefficient of variation) = `Std / Mean`. Lower = less jitter.

#### Overhead (relative to mean latency, positive = slower than reference)

| Comparison                                      |   Value |
|-------------------------------------------------|--------:|
| **Thot** vs **LibTorch** Overhead               |  +3.54% |
| **Thot** Prebuilt vs **Thot** Custom Overhead   |  +0.04% |
| **Thot** Custom vs **LibTorch** Overhead        |  +3.50% |

With identical pinned-memory settings, Thot’s prebuilt `Train()` stays within a few percent of raw LibTorch in mean latency and throughput, while keeping jitter extremely low.

---

> **Note on variability.** These numbers are **relative**, not absolute. Modern hardware is noisy: clocks, power limits, thermals and OS scheduling all drift, so repeated runs with identical settings produce slightly different latency distributions. The robust takeaway across both tables is that Thot’s wrapper adds at most a few percent overhead compared to raw LibTorch, and can even be faster under certain I/O configurations, while preserving very low jitter.

Source: [`test/speedtest.cpp`](../test/speedtest.cpp)

