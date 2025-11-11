# Regularization Descriptors

Regularizers in Thot are descriptors attached either globally through
`Model::set_regularization` or locally via [Local](../local/README.md). Each
descriptor bundles an option struct and a `Details::*` implementation that knows
how to accumulate penalties during [Training](../training/README.md).

## Weight penalties and sparsity

- **`Regularization::L1` / `L2`** – Classic Lasso/Ridge terms configurable per
  parameter tensor.
- **`ElasticNet`** – Combines L1 and L2 components with independent weights.
- **`GroupLasso` / `StructuredL2`** – Enforce sparsity at the group or channel
  level by aggregating weights into structured norms.
- **`MaxNorm` / `SpectralNorm` / `Orthogonality`** – Constrain norms of weight
  matrices, optionally projecting onto orthogonal manifolds.
- **`L0HardConcrete`** – Stochastic gate based on the hard concrete
  distribution for pruning entire parameters.

## Information-theoretic and decorrelation terms

- **`DeCov`** – Penalises correlated activations using covariance matrices.
- **`CenteringVariance`** – Drives activations towards zero mean and unit
  variance.
- **`JacobianNorm`** – Controls sensitivity by regularising the Jacobian.
- **`KLSparsity`** – Encourages sparse activations through KL divergence against
  a target activation frequency.

## Continual learning and Bayesian ensembling

- **`EWC`, `MAS`, `SI`** – Elastic Weight Consolidation, Memory Aware Synapses,
  and Synaptic Intelligence help preserve knowledge across tasks.
- **`SWA`, `SWAG`, `FGE`, `SFGE`** – Stochastic Weight Averaging and its Gaussian
  / Fast / Snapshot variants for ensemble-like posterior exploration.
- **`NuclearNorm`** – Low-rank regularisation for matrix parameters.

## Adversarial and robustness regularizers

- **`TRADES` / `VAT`** – Adversarial robustness objectives using Kullback-Leibler
  divergence or virtual adversarial perturbations.
- **`WGANGP`, `R1`, `R2`** – Gradient penalties for GAN critics, configurable per
  iteration.
- **`SI`**, **`MaxNorm`**, and related constraints cooperate with the CUDA graph
  safety checks inside `Model::compute_regularization_penalty`.

Every descriptor records whether it participates in CUDA graph capture; the core
engine pre-allocates storage so penalties can be computed without dynamic memory
allocations during the training step. As with other descriptors, the full
configuration is serialised by [Save & Load](../saveload/README.md).

---

Combine regularizers with [Loss](../loss/README.md) and
[Optimizer](../optimizer/README.md) descriptors to tailor the training signal to
your experiment.