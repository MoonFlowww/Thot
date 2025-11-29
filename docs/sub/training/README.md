# Training Loop

Omni's `Model::train` orchestrates dataset streaming, optimisation, and telemetry
capture. You can either pass a packed dataset (`std::vector` of `{inputs,
targets}` pairs) or raw tensors. Training requires an optimizer and a loss
descriptor to be set beforehand (see [Docs/Optimizer](../optimizer/README.md) and
[Docs/Loss](../loss/README.md)).

## TrainOptions

`TrainOptions` controls runtime behaviour:

| Field | Purpose |
| --- | --- |
| `epoch` | Number of epochs to run; zero short-circuits the loop. |
| `batch_size` | Mini-batch size. Must be non-zero. |
| `shuffle` | Shuffle dataset between epochs. |
| `buffer_vram` | When > 0, stage batches in pinned memory and stream them to the GPU asynchronously. Requires the model to live on CUDA. |
| `monitor` / `stream` | Enable console logging through `Utils::Terminal` when `stream` is non-null. |
| `restore_best_state` | Keep a shadow copy of parameters and restore the epoch with the lowest validation/test loss. |
| `validation` / `test` | Optional `{inputs, targets}` tensors evaluated at the end of each epoch. Validation is used if test is absent. |
| `graph_mode` | Choose between `GraphMode::Disabled`, `GraphMode::Capture`, and `GraphMode::Replay` (see below). |
| `enable_amp` | Turn on automatic mixed precision (AMP) when CUDA is available. |
| `memory_format` | Request `torch::MemoryFormat::ChannelsLast`; Omni only applies it when convolutional layers and CUDA are present. |

Validation/test splits are supplied as `std::vector<torch::Tensor>{inputs,
targets}` to preserve ownership and allow Omni to reuse contiguous host buffers.

## Graph capture and streaming

CUDA graph support is toggled via `graph_mode`:

- `Disabled` (default) – Standard eager execution.
- `Capture` – Records a training iteration to a CUDA graph; requires fixed batch
  shapes and drops/pads remainder batches accordingly.
- `Replay` – Reuses a previously captured graph. Attempting to replay without a
  capture triggers a descriptive error.

The runtime verifies that selected optimizers are graph-safe and pre-allocates
workspace buffers so capture does not allocate at runtime. When buffering is
enabled (`buffer_vram > 0`), a double-buffered CUDA stream prefetcher overlaps
data transfers with compute.

## Telemetry and monitoring

`Model::training_telemetry()` exposes:

- `EpochSnapshot` – epoch index, deferred train/test loss scalars, deferred
  step latency, improvement flags, elapsed time, and learning-rate snapshots.
- `DatasetLossSnapshot` – detailed metrics for validation/test sweeps when
  requested.

These values remain on the host and lazily materialise GPU tensors, making them
cheap to log or feed into [Docs/Plot](../plot/README.md). When `monitor` is `true`,
progress is streamed to the provided `std::ostream` with non-blocking CUDA event
handling to avoid stalling the training loop.

### Manual training loop (“semi Omni”)

Omni keeps the underlying LibTorch modules exposed, so you can orchestrate a
training loop manually when you need custom control flow (curriculum learning,
reinforcement updates, mixed dataloaders, etc.). The model still constructs the
network graph and owns the optimizer/loss descriptors, but you steer the forward
and backward passes yourself:

```cpp
Omni::Model model("ManualMLP");
model.add(Omni::Layer::FC({784, 512, /*bias=*/true}, Omni::Activation::ReLU));
model.add(Omni::Layer::FC({512, 256, /*bias=*/true}, Omni::Activation::ReLU));
model.add(Omni::Layer::FC({256, 10, /*bias=*/true}, Omni::Activation::Identity));

model.set_optimizer(Omni::Optimizer::Adam({.learning_rate = 1e-3}));
const auto cross_entropy = Omni::Loss::CrossEntropy({});
model.set_loss(cross_entropy);

for (auto epoch = 0; epoch < max_epochs; ++epoch) {
    for (auto [inputs, targets] : minibatches) {
        inputs = inputs.to(model.device());
        targets = targets.to(model.device());

        model.zero_grad();                                 // clear stale gradients
        auto logits = model.forward(inputs);               // forward pass through the DAG
        auto loss = Omni::Loss::Details::compute(          // reuse Omni loss helpers
            cross_entropy, logits, targets);
        loss.backward();                                   // populate gradients
        model.step();                                      // step optimizer (+ scheduler if set)
    }
}
```
When your minibatches originate on the host, you can overlap page-locking with
compute by calling `Omni::async_pin_memory` before transferring the tensors to
the device:

```cpp
auto stage_for_device = [&](torch::Tensor tensor) {
    auto pinned = Omni::async_pin_memory(std::move(tensor));
    auto host   = pinned.materialize();
    const bool non_blocking = model.device().is_cuda() && host.is_pinned();
    return host.to(model.device(), host.scalar_type(), non_blocking);
};

auto inputs  = stage_for_device(minibatch_inputs);
auto targets = stage_for_device(minibatch_targets);
```
- **Forward.** `model.forward` executes the compiled graph exactly as
  `Model::train` would; this keeps AMP, calibration hooks, and CUDA graph capture
  available when you enable them via `ForwardOptions`.
- **Backward.** Loss tensors expose the standard `.backward()` API. You can mix
  Omni descriptors with custom reductions, attach gradient hooks, or integrate
  reinforcement learning signals before calling backward.
- **Optimizer step.** `model.step()` respects global and local optimizers plus
  schedulers. Use it when you still want Omni to handle learning-rate policies;
  call `model.optimizer().step()` directly only if you intend to bypass
  scheduler bookkeeping.

Prefer the manual path when you need tight integration with bespoke data
pipelines, gradient accumulation strategies, or debugging hooks that do not fit
inside `Model::train`. The abstractions remain the same—you reuse layer builders,
loss factories, optimizers, and telemetry APIs—while keeping full control over
loop structure.


## Advanced hooks

- **Staging observer.** `Model::set_staging_observer` lets you inspect every
  batch transferred to the device (for debugging augmentations or data quality).
- **Memory format.** Before the first epoch, Omni propagates the requested memory
  format to convolutional layers and residual projections so weight tensors match
  the layout of incoming batches.
- **Regularization integration.** Regularisation descriptors registered via
  [Docs/Regularization](../regularization/README.md) are evaluated inside the training
  step; penalties participate in AMP and CUDA graph capture.

---

Combine `TrainOptions` with [Docs/LrScheduler](../lrscheduler/README.md) and
per-module [Docs/Local](../local/README.md) overrides to craft complex optimisation
schemes. After training, persist the state with [Docs/Save & Load](../saveload/README.md).