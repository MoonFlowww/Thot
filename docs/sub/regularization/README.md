# Regularization Descriptors

Regularizers in Thot are descriptors attached either globally through
`Model::set_regularization` or locally via [Local](../local/README.md). Each
descriptor bundles an option struct and a `Details::*` implementation that knows
how to accumulate penalties during [Training](../training/README.md).

## Global and local usage

Attach penalties to the whole model by passing a vector of descriptors to
`Model::set_regularization`:

```cpp
model.set_regularization({ /*vector field*/
    Thot::Regularization::L2({.coefficient = 5e-5}),
    Thot::Regularization::ElasticNet({
        .l1_coefficient = 2e-6,
        .l2_coefficient = 5e-6,
    }),
});
```

To scope a regularizer to a single layer or block, provide it inside the
`regularization` field of a [`Thot::LocalConfig`](../local/README.md):

```cpp
Thot::LocalConfig decoder_scope{
    .regularization = { /*vector field*/
        Thot::Regularization::MaxNorm({
            .coefficient = 1.0,
            .max_norm = 3.0,
            .dim = 0,
        })
    }
};

model.add(Thot::Layer::FC({512, 256, /*bias*/ true},
                          Thot::Activation::SiLU,
                          Thot::Initialization::KaimingUniform,
                          decoder_scope),
          "decoder_fc");
```

Both global and local descriptors can be mixed; penalties are accumulated once
per training step while respecting CUDA graph capture requirements.


## Weight penalties and sparsity

- **`Regularization::L1` / `L2`** – Classic Lasso/Ridge terms configurable per
  parameter tensor.
  - `coefficient` (`double`, default `0.0`) – scale applied to the L1/L2 norm.
- **`ElasticNet`** – Combines L1 and L2 components with independent weights.
  - `l1_coefficient` (`double`, default `0.0`)
  - `l2_coefficient` (`double`, default `0.0`)
- **`GroupLasso` / `StructuredL2`** – Enforce sparsity at the group or channel
  level by aggregating weights into structured norms.
    - `coefficient` (`double`, default `0.0`)
    - `group_dim` (`std::int64_t`, default `0`) – dimension along which groups are
      formed.
    - `epsilon` (`double`, default `1e-8`, GroupLasso only) – numerical stability
      term for norm aggregation.
- **`MaxNorm` / `SpectralNorm` / `Orthogonality`** – Constrain norms of weight
  matrices, optionally projecting onto orthogonal manifolds.
    - `coefficient` (`double`, default `0.0`)
    - `max_norm` (`double`, default `1.0`, MaxNorm only) – target upper bound for
      the norm.
    - `dim` (`std::int64_t`, default `0`, MaxNorm only) – dimension reduced before
      computing the norm.
    - `target` (`double`, default `1.0`, SpectralNorm only) – desired largest
      singular value.
- **`L0HardConcrete`** – Stochastic gate based on the hard concrete distribution
  for pruning entire parameters.
    - `coefficient` (`double`, default `0.0`)
    - `beta` (`double`, default `2.0 / 3.0`)
    - `gamma` (`double`, default `-0.1`)
    - `zeta` (`double`, default `1.1`)
- **`NuclearNorm`** – Low-rank regularisation for matrix parameters.
    - `strength` (`double`, default `0.0`)

## Information-theoretic and decorrelation terms

- **`DeCov`** – Penalises correlated activations using covariance matrices.
    - `coefficient` (`double`, default `0.0`)
    - `epsilon` (`double`, default `1e-5`) – diagonal jitter when inverting/normalising.
- **`CenteringVariance`** – Drives activations towards zero mean and unit
  variance.
    - `coefficient` (`double`, default `0.0`)
    - `target_std` (`double`, default `1.0`)
- **`JacobianNorm`** – Controls sensitivity by regularising the Jacobian.
    - `coefficient` (`double`, default `0.0`)
- **`KLSparsity`** – Encourages sparse activations through KL divergence against
  a target activation frequency.
    - `coefficient` (`double`, default `0.0`)
    - `target` (`double`, default `0.05`)
    - `epsilon` (`double`, default `1e-6`)

## Continual learning and Bayesian ensembling

- **`EWC`, `MAS`, `SI`** – Elastic Weight Consolidation, Memory Aware Synapses,
  and Synaptic Intelligence help preserve knowledge across tasks.
    - `strength` (`double`, default `0.0`, EWC & MAS)
    - `damping` (`double`, default `1e-3`, SI only) – stabiliser for importance
      accumulation.
- **`SWA`, `SWAG`, `FGE`, `SFGE`** – Stochastic Weight Averaging and its Gaussian
  / Fast / Snapshot variants for ensemble-like posterior exploration.
    - `coefficient` (`double`, default `0.0`, SWA/SWAG/FGE/SFGE)
    - `variance_epsilon` (`double`, default `1e-8`, SWAG only) – variance floor for
      numerical stability.
    - `start_step` (`std::size_t`, default `0`, SWAG only) – training step to begin
      collecting snapshots.
    - `accumulation_stride` (`std::size_t`, default `1`, SWAG only) – interval
      between collected snapshots.
    - `max_snapshots` (`std::size_t`, default `0`, SWAG only) – cap on stored
      parameter snapshots.
- **`NuclearNorm`** (see above) also fits this category when used for low-rank
  priors.

## Adversarial and robustness regularizers

- **`TRADES` / `VAT`** – Adversarial robustness objectives using
  Kullback–Leibler divergence or virtual adversarial perturbations.
    - `coefficient` (`double`, default `0.0`)
- **`WGANGP`, `R1`, `R2`** – Gradient penalties for GAN critics, configurable per
  iteration.
    - `coefficient` (`double`, default `0.0`)
    - `target` (`double`, default `1.0`, WGANGP only) – desired gradient norm.

Every descriptor records whether it participates in CUDA graph capture; the core
engine pre-allocates storage so penalties can be computed without dynamic memory
allocations during the training step. As with other descriptors, the full
configuration is serialised by [Save & Load](../saveload/README.md).

---

Combine regularizers with [Loss](../loss/README.md) and
[Optimizer](../optimizer/README.md) descriptors to tailor the training signal to
your experiment.