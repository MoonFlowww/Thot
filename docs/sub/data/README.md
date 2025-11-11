# Data Loading, Manipulation, and Diagnostics

The `Thot::Data` toolbox wraps dataset ingestion, augmentation, repair, and
analysis helpers so you can prepare tensors before handing them to
[Training](../training/README.md) or [Evaluation](../evaluation/README.md).
Factories live under `src/data/details` and are split into four main groups:
`Load`, `Manipulation`, `DimReduction`, and `Check`/`Repair`.

## Dataset Loaders

Every loader is templated on `BufferVRAM`, allowing you to pin the resulting
tensors directly to the device selected by `Core::DevicePolicy`. Leave the flag
at the default (`false`) to receive CPU tensors.

- **`Load::CIFAR10(root, train_fraction, test_fraction, normalise)`** – Parses
  the binary CIFAR-10 batches, optionally downsamples splits, converts images to
  `float32`, and normalises to `[0, 1]` when requested. The helper returns a
  4-tuple `(train_images, train_labels, test_images, test_labels)` ready for
  batching.
- **`Load::MNIST(root, train_fraction, test_fraction, normalise)`** – Reads the
  IDX files distributed by Yann LeCun's site, validates magic numbers and
  shapes, and exposes the digit dataset as `(N, 1, 28, 28)` tensors.
- **`Load::ETTh(csv_path, train_fraction, test_fraction, normalise)`** – Imports
  the Electricity Transformer Temperature (ETT) CSV files, infers feature
  columns, and yields contiguous float tensors for forecasting experiments.
- **`Load::PTBXL(root, low_resolution, train_fraction, normalise, multilabel)`**
  – Handles the PTB-XL ECG archive, resolving `.hea/.dat` pairs, mapping SCP
  statements to the five superclasses, and optionally constructing multilabel
  targets with configurable thresholds.

All loaders guard against missing files and mismatch between data/label counts,
throwing detailed exceptions when directories are incomplete.

## Manipulation and Augmentation

The manipulation module operates directly on tensors and keeps augmentations
deterministic by accepting optional seeds where appropriate.

- **`Flip`** – Mirrors spatial axes based on symbolic tokens (`"x"`, `"y"`,
  `"z"`) or explicit indices. Frequency controls how many batch elements are
  augmented.
- **`Cutout`** – Stochastically masks rectangular patches, either with random
  noise or a constant fill value, and concatenates the augmented samples to the
  original batch.
- **`Shuffle` / `Fraction`** – Shuffle pairs of tensors with a shared
  permutation, or extract a fraction of the leading samples while keeping the
  original order intact.
- **`Upsample`** – Thin wrapper around `torch::nn::functional::interpolate` with
  scale-factor semantics.
- **`Grayscale`** – Converts RGB tensors to single-channel luminance using
  configurable weights.

### Normalisation pipelines

Located under `Data::Normalization`, these routines reshape arbitrary tensor
layouts so the temporal dimension is processed correctly:

- `Zscore` and `EWZscore` compute rolling or exponentially-weighted z-scores.
- `RobustZscore` uses median/MAD estimates for outlier-resistant scaling.
- `Demean` subtracts running or global means without touching variance.
- `StandardizeToTarget` re-scales any series to a desired mean/std pair.

## Dimensionality Reduction

When prototyping anomaly detection or latent factor models, leverage
`Data::DimReduction`:

- **`RPCA`** – Robust PCA via the inexact augmented Lagrange multiplier method.
  Returns a `(low_rank, sparse)` pair that isolates structured signal from
  transients.
- **`PCA`** – Classical PCA with optional centering and whitening. The helper
  reports principal components, explained variance, singular values, mean, and
  the transformed dataset.

Both routines accept the same `BufferVRAM` template argument as the loaders to
avoid redundant device transfers.

## Data Quality and Repair

- **`Check::Imbalance`** – Produces per-class counts, distributions,
  entropies, and KL divergence between train/test splits.
- **`Check::Shuffled`** – Heuristic to flag datasets that might still be ordered
  by label (useful for verifying randomisation before calling
  [Training](../training/README.md)).
- **`Check::Size`** – Convenience printer for tensor shapes.

For structured datasets with discrete attributes, `Data::Repair::HoloClean`
implements a lightweight version of the HoloClean probabilistic repair engine:
register functional dependencies, ingest records, and let the factor graph
suggest repairs above a chosen probability threshold.

---

Pair these utilities with [Links](../links/README.md) to route pre-processed
inputs into complex model branches, or combine them with local scopes from
[Local](../local/README.md) when different data streams need bespoke training
policies.
