# Layers Reference

The `Thot::Layer` namespace exposes typed factory helpers that wrap LibTorch
modules with the framework's activation, initialization, and locality wiring.  
Each helper follows the `Layer(options, activation, initialization, local)`
signature (activation defaults to `Identity`, initialization to
`Initialization::Default`, and the `LocalConfig` is optional). This document
summarises every layer available in `src/layer/layer.hpp` and details the fields
of their option structures.

## Usage

```cpp
Thot::Model model("Classifier");
model.add("fc1", Thot::Layer::FC(
    {.in_features = 128, .out_features = 64, .bias = true},               // options struct
    Thot::Activation::ReLU,                                               // activation override
    Thot::Initialization::HeNormal,                                       // w&B initialisation
    {.optimizer = Thot::Optimizer::AdamW({.learning_rate = 1e-3})}        // optional local config
));
```

The descriptor tuple mirrors the common calling pattern used throughout the
framework: an options aggregate for the underlying LibTorch module, followed by
an activation decorator, an initialization policy, and (optionally) a
`Thot::LocalConfig` carrying per-layer/per-block overrides.


## Linear

### `Thot::Layer::FC(...)`
*Options type:* `FCOptions`
- `in_features` *(int64)* – number of input features. Must be positive.
- `out_features` *(int64)* – number of output features. Must be positive.
- `bias` *(bool, default: `true`)* – include an additive bias term.

## Convolution

### `Thot::Layer::Conv1d(...)`
*Options type:* `Conv1dOptions`
- `in_channels` *(int64)* – number of input channels. Must be positive.
- `out_channels` *(int64)* – number of output channels. Must be positive.
- `kernel_size` *(vector<int64>, default: `{3}`)* – spatial kernel along the
  temporal axis.
- `stride` *(vector<int64>)* – optional stride per dimension.
- `padding` *(vector<int64>, default: `{0}`)* – padding size. Negative values are
  invalid.
- `dilation` *(vector<int64>, default: `{1}`)* – dilation spacing for the
  kernel.
- `groups` *(int64, default: `1`)* – grouped convolution factor.
- `bias` *(bool, default: `true`)* – include learnable bias terms.
- `padding_mode` *(string, default: `"zeros"`)* – one of `zeros`, `reflect`,
  `replicate`, or `circular`.

### `Thot::Layer::Conv2d(...)`
*Options type:* `Conv2dOptions`
- `in_channels` *(int64)* – number of input channels. Must be positive.
- `out_channels` *(int64)* – number of output channels. Must be positive.
- `kernel_size` *(vector<int64>, default: `{3, 3}`)* – kernel height/width.
- `stride` *(vector<int64>, default: `{1, 1}`)* – spatial stride.
- `padding` *(vector<int64>, default: `{0, 0}`)* – spatial padding.
- `dilation` *(vector<int64>, default: `{1, 1}`)* – dilation factors.
- `groups` *(int64, default: `1`)* – channel grouping.
- `bias` *(bool, default: `true`)* – include learnable bias terms.
- `padding_mode` *(string, default: `"zeros"`)* – `zeros`, `reflect`,
  `replicate`, or `circular`.

## Normalization

### `Thot::Layer::BatchNorm2d(...)`
*Options type:* `BatchNorm2dOptions`
- `num_features` *(int64)* – channel dimension to normalize. Must be positive.
- `eps` *(double, default: `1e-5`)* – numerical stability term added to the
  variance.
- `momentum` *(double, default: `0.1`)* – running-statistics momentum.
- `affine` *(bool, default: `true`)* – whether to learn scale and shift
  parameters.
- `track_running_stats` *(bool, default: `true`)* – keep moving mean/variance.

## Pooling

Pooling layers share the `PoolingDescriptor` wrapper and therefore the same
optional activation/local hooks. Each option structure maps directly to a
LibTorch pooling module.

### `Thot::Layer::MaxPool1d(...)`
*Options type:* `MaxPool1dOptions`
- `kernel_size` *(vector<int64>, default: `{2}`)* – window size.
- `stride` *(vector<int64>)* – optional stride; defaults to kernel size when
  left empty.
- `padding` *(vector<int64>, default: `{0}`)* – implicit zero padding.
- `dilation` *(vector<int64>, default: `{1}`)* – dilation factor.
- `ceil_mode` *(bool, default: `false`)* – use `ceil` instead of `floor` when
  computing the output length.

### `Thot::Layer::AvgPool1d(...)`
*Options type:* `AvgPool1dOptions`
- `kernel_size` *(vector<int64>, default: `{2}`)* – window size.
- `stride` *(vector<int64>)* – optional stride.
- `padding` *(vector<int64>, default: `{0}`)* – implicit zero padding.
- `ceil_mode` *(bool, default: `false`)* – use `ceil` output length.
- `count_include_pad` *(bool, default: `false`)* – include padded zeros in the
  average.

### `Thot::Layer::AdaptiveAvgPool1d`
*Options type:* `AdaptiveAvgPool1dOptions`
- `output_size` *(vector<int64>, default: `{1}`)* – requested output length.

### `Thot::Layer::AdaptiveMaxPool1d`
*Options type:* `AdaptiveMaxPool1dOptions`
- `output_size` *(vector<int64>, default: `{1}`)* – requested output length.

### `Thot::Layer::MaxPool2d(...)`
*Options type:* `MaxPool2dOptions`
- `kernel_size` *(vector<int64>, default: `{2, 2}`)* – spatial window.
- `stride` *(vector<int64>)* – optional stride; defaults to kernel size when
  empty.
- `padding` *(vector<int64>, default: `{0, 0}`)* – implicit zero padding.
- `dilation` *(vector<int64>, default: `{1, 1}`)* – dilation factor.
- `ceil_mode` *(bool, default: `false`)* – use `ceil` output shape.

### `Thot::Layer::AvgPool2d(...)`
*Options type:* `AvgPool2dOptions`
- `kernel_size` *(vector<int64>, default: `{2, 2}`)* – spatial window.
- `stride` *(vector<int64>)* – optional stride.
- `padding` *(vector<int64>, default: `{0, 0}`)* – implicit zero padding.
- `ceil_mode` *(bool, default: `false`)* – use `ceil` output shape.
- `count_include_pad` *(bool, default: `false`)* – include padded zeros in the
  average.

### `Thot::Layer::AdaptiveAvgPool2d(...)`
*Options type:* `AdaptiveAvgPool2dOptions`
- `output_size` *(vector<int64>, default: `{1, 1}`)* – requested output height
  and width.

### `Thot::Layer::AdaptiveMaxPool2d(...)`
*Options type:* `AdaptiveMaxPool2dOptions`
- `output_size` *(vector<int64>, default: `{1, 1}`)* – requested output height
  and width.

## Dropout

### `Thot::Layer::HardDropout(...)`
*Options type:* `HardDropoutOptions`
- `probability` *(double, default: `0.5`)* – probability of masking a unit.
  Must lie in `[0, 1)`.
- `inplace` *(bool, default: `false`)* – modify the input tensor in-place when
  gradients are not required.

### `Thot::Layer::SoftDropout(...)`
*Options type:* `SoftDropoutOptions`
- `probability` *(double, default: `0.5`)* – probability of sampling the noisy
  branch. Must lie in `[0, 1)` and strictly less than `1`.
- `noise_mean` *(double, default: `0.0`)* – mean of the injected noise.
- `noise_std` *(double, default: `1.0`)* – standard deviation of the noise; must
  be non-negative.
- `noise_type` *(enum, default: `Gaussian`)* – distribution used for the noise.
  Options: `Gaussian`, `Poisson`, `Dithering`, `InterleavedGradientNoise`,
  `BlueNoise`, `Bayer`.
- `inplace` *(bool, default: `false`)* – modify inputs in-place when possible.

## Shape

### `Thot::Layer::Flatten(...)`
*Options type:* `FlattenOptions`
- `start_dim` *(int64, default: `1`)* – first dimension to flatten.
- `end_dim` *(int64, default: `-1`)* – last dimension to flatten. Accepts
  negative indexing relative to the tensor rank.

## Sequence Models

All recurrent descriptors accept the usual activation/initialization overrides
and honor the `local` configuration flags in addition to the options listed.

### `Thot::Layer::RNN(...)`
*Options type:* `RNNOptions`
- `input_size` *(int64)* – features per input step. Must be positive.
- `hidden_size` *(int64)* – hidden state size. Must be positive.
- `num_layers` *(int64, default: `1`)* – number of stacked recurrent layers.
- `dropout` *(double, default: `0.0`)* – dropout probability between recurrent
  layers.
- `batch_first` *(bool, default: `false`)* – use `(batch, seq, feature)` layout
  when `true`.
- `bidirectional` *(bool, default: `false`)* – build a bidirectional RNN.
- `nonlinearity` *(string, default: `"tanh"`)* – activation for the recurrent
  cell (`"tanh"` or `"relu"`).

### `Thot::Layer::LSTM(...)`
*Options type:* `LSTMOptions`
- `input_size` *(int64)* – features per time-step. Must be positive.
- `hidden_size` *(int64)* – hidden state size. Must be positive.
- `num_layers` *(int64, default: `1`)* – number of stacked LSTM layers.
- `dropout` *(double, default: `0.0`)* – dropout between layers.
- `batch_first` *(bool, default: `false`)* – `(batch, seq, feature)` layout.
- `bidirectional` *(bool, default: `false`)* – build a bidirectional LSTM.
- `bias` *(bool, default: `true`)* – include bias parameters.
- `forget_gate_bias` *(double, default: `1.0`)* – additive bias applied to the
  forget gate when biases are enabled.
- `param_dtype` *(c10::ScalarType, default: `at::kFloat`)* – dtype for recurrent
  parameters.
- `allow_tf32` *(bool, default: `true`)* – allow TF32 kernels when available on
  CUDA.
- `benchmark_cudnn` *(bool, default: `true`)* – enable cuDNN benchmarking.

### `Thot::Layer::xLSTM(...)`
*Options type:* `xLSTMOptions` (alias of `LSTMOptions`)

In addition to the `LSTMOptions` fields above, the implementation honours the
`forget_gate_bias`, `allow_tf32`, and `benchmark_cudnn` knobs while keeping LibTorch
compatibility.

### `Thot::Layer::GRU(...)`
*Options type:* `GRUOptions`
- `input_size` *(int64)* – features per time-step. Must be positive.
- `hidden_size` *(int64)* – hidden state size. Must be positive.
- `num_layers` *(int64, default: `1`)* – number of stacked GRU layers.
- `dropout` *(double, default: `0.0`)* – dropout between layers.
- `batch_first` *(bool, default: `false`)* – `(batch, seq, feature)` layout.
- `bidirectional` *(bool, default: `false`)* – build a bidirectional GRU.
- `bias` *(bool, default: `true`)* – include bias parameters.
- `param_dtype` *(c10::ScalarType, default: `at::kFloat`)* – dtype for recurrent
  parameters.
- `allow_tf32` *(bool, default: `true`)* – allow TF32 kernels when available on
  CUDA.
- `benchmark_cudnn` *(bool, default: `true`)* – enable cuDNN benchmarking.

## Sequence State-Space

### `Thot::Layer::S4(...)`
*Options type:* `S4Options`
- `input_size` *(int64)* – features per input step. Must be positive.
- `state_size` *(int64)* – size of the internal state vector. Must be positive.
- `rank` *(int64, default: `1`)* – low-rank factorization rank. Must be
  positive.
- `output_size` *(int64, default: `0` → falls back to `input_size`)* – number of
  output channels.
- `batch_first` *(bool, default: `true`)* – `(batch, seq, feature)` layout.
- `bidirectional` *(bool, default: `false`)* – include a backward pass.
- `dropout` *(double, default: `0.0`)* – dropout probability applied to the
  output projection.
- `initialization` *(enum, default: `HiPPO`)* – spectral init strategy
  (`HiPPO` or `S4D`).
- `maximum_length` *(int64, default: `0`)* – precomputes cached kernels when
  greater than zero.

## Tensor Reduction

### `Thot::Layer::Reduce(...)`
*Options type:* `ReduceOptions`
- `op` *(enum, default: `Mean`)* – reduction op (`Sum`, `Mean`, `Max`, `Min`).
- `dims` *(vector<int64>)* – axes to reduce. Empty list reduces across all
  dimensions.
- `keep_dim` *(bool, default: `false`)* – keep reduced axes with size one when
  set to `true`.

---
Every layer descriptor returned by these helpers can be passed to
`Model::add()` or any block container. Combine them with activations,
initializers, and local overrides to tailor the computation graph to your
experiment.


#### HyperLinks related:
- [Docs/Introduction](../../README.md) (`model.add()`)
- [Docs/Activations](../activation/README.md) (`Thot::Activation::*`)
- [Docs/Initialization](../initialization/README.md) (`Thot::Initialization::*`)
- [Docs/Local](../local/README.md) (`model.links()`)


