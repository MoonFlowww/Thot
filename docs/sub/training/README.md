# Training Loop

Thot's `Model::train` orchestrates dataset streaming, optimisation, and telemetry
capture. You can either pass a packed dataset (`std::vector` of `{inputs,
targets}` pairs) or raw tensors. Training requires an optimizer and a loss
descriptor to be set beforehand (see [Optimizer](../optimizer/README.md) and
[Loss](../loss/README.md)).

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
| `memory_format` | Request `torch::MemoryFormat::ChannelsLast`; Thot only applies it when convolutional layers and CUDA are present. |

Validation/test splits are supplied as `std::vector<torch::Tensor>{inputs,
targets}` to preserve ownership and allow Thot to reuse contiguous host buffers.

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

- `EpochSnapshot` – epoch index, deferred train/test loss scalars, improvement
  flags, elapsed time, and learning-rate snapshots.
- `DatasetLossSnapshot` – detailed metrics for validation/test sweeps when
  requested.

These values remain on the host and lazily materialise GPU tensors, making them
cheap to log or feed into [Plot](../plot/README.md). When `monitor` is `true`,
progress is streamed to the provided `std::ostream` with non-blocking CUDA event
handling to avoid stalling the training loop.

## Advanced hooks

- **Staging observer.** `Model::set_staging_observer` lets you inspect every
  batch transferred to the device (for debugging augmentations or data quality).
- **Memory format.** Before the first epoch, Thot propagates the requested memory
  format to convolutional layers and residual projections so weight tensors match
  the layout of incoming batches.
- **Regularization integration.** Regularisation descriptors registered via
  [Regularization](../regularization/README.md) are evaluated inside the training
  step; penalties participate in AMP and CUDA graph capture.

---

Combine `TrainOptions` with [LrScheduler](../lrscheduler/README.md) and
per-module [Local](../local/README.md) overrides to craft complex optimisation
schemes. After training, persist the state with [Save & Load](../saveload/README.md).