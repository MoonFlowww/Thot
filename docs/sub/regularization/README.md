# Regularization Descriptors
Regularizers in Nott are descriptors that wrap an option struct together with a
`Details::*` implementation. Descriptors can be attached either globally through
`Model::set_regularization` or scoped to individual layers via
[`Nott::LocalConfig`](../local/README.md). During
[Docs/Training](../training/README.md) the engine evaluates every descriptor once per
step while remaining compatible with CUDA graph capture.

## Attaching descriptors

Apply regularizers to the whole model by passing a vector of descriptors to

`Model::set_regularization`:

```cpp
model.set_regularization({ /*vector field*/
    Nott::Regularization::L2({.coefficient = 5e-5}),
    Nott::Regularization::ElasticNet({
        .l1_coefficient = 2e-6,
        .l2_coefficient = 5e-6,
    }),
});
```

To localise penalties, include them in the `regularization` field of a
[`Nott::LocalConfig`](../local/README.md):

```cpp
Nott::LocalConfig decoder_scope{
    .regularization = { /*vector field*/
        Nott::Regularization::MaxNorm({
            .coefficient = 1.0,
            .max_norm = 3.0,
            .dim = 0,
        }),
    },
};

model.add(Nott::Layer::FC({512, 256, /*bias*/ true},
                          Nott::Activation::SiLU,
                          Nott::Initialization::KaimingUniform,
                          decoder_scope),
          "decoder_fc");
```

Both global and local descriptors can be mixed freely. The runtime aggregates
penalties without extra allocations and records CUDA-graph compatibility for
replay-safe training loops.

## Descriptor catalogue
The array below enumerates every factory provided by
[`src/regularization/regularization.hpp`](../../../src/regularization/regularization.hpp)
so you can spot the available options at a glance:

```cpp
constexpr std::array kAllRegularizers{
    Nott::Regularization::L1(),
    Nott::Regularization::ElasticNet(),
    Nott::Regularization::GroupLasso(),
    Nott::Regularization::StructuredL2(),
    Nott::Regularization::L0HardConcrete(),
    Nott::Regularization::Orthogonality(),
    Nott::Regularization::SpectralNorm(),
    Nott::Regularization::MaxNorm(),
    Nott::Regularization::KLSparsity(),
    Nott::Regularization::DeCov(),
    Nott::Regularization::CenteringVariance(),
    Nott::Regularization::JacobianNorm(),
    Nott::Regularization::WGANGP(),
    Nott::Regularization::R1(),
    Nott::Regularization::R2(),
    Nott::Regularization::TRADES(),
    Nott::Regularization::VAT(),
    Nott::Regularization::L2(),
    Nott::Regularization::EWC(),
    Nott::Regularization::MAS(),
    Nott::Regularization::SI(),
    Nott::Regularization::NuclearNorm(),
    Nott::Regularization::SWA(),
    Nott::Regularization::SWAG(),
    Nott::Regularization::FGE(),
    Nott::Regularization::SFGE(),
};
```
| Descriptor | Category | Key options (default) |
| --- | --- | --- |
| `L1` | Weight penalty | `coefficient` (`double`, `0.0`) |
| `L2` | Weight penalty | `coefficient` (`double`, `0.0`) |
| `ElasticNet` | Weight penalty | `l1_coefficient` (`double`, `0.0`), `l2_coefficient` (`double`, `0.0`) |
| `GroupLasso` | Structured sparsity | `coefficient` (`double`, `0.0`), `group_dim` (`std::int64_t`, `0`), `epsilon` (`double`, `1e-8`) |
| `StructuredL2` | Structured sparsity | `coefficient` (`double`, `0.0`), `group_dim` (`std::int64_t`, `0`) |
| `L0HardConcrete` | Sparsity/pruning | `coefficient` (`double`, `0.0`), `beta` (`double`, `2.0/3.0`), `gamma` (`double`, `-0.1`), `zeta` (`double`, `1.1`) |
| `Orthogonality` | Matrix constraint | `coefficient` (`double`, `0.0`) |
| `SpectralNorm` | Matrix constraint | `coefficient` (`double`, `0.0`), `target` (`double`, `1.0`) |
| `MaxNorm` | Matrix constraint | `coefficient` (`double`, `0.0`), `max_norm` (`double`, `1.0`), `dim` (`std::int64_t`, `0`) |
| `KLSparsity` | Activation sparsity | `coefficient` (`double`, `0.0`), `target` (`double`, `0.05`), `epsilon` (`double`, `1e-6`) |
| `DeCov` | Activation decorrelation | `coefficient` (`double`, `0.0`), `epsilon` (`double`, `1e-5`) |
| `CenteringVariance` | Activation centring | `coefficient` (`double`, `0.0`), `target_std` (`double`, `1.0`) |
| `JacobianNorm` | Sensitivity control | `coefficient` (`double`, `0.0`) |
| `WGANGP` | Adversarial robustness | `coefficient` (`double`, `0.0`), `target` (`double`, `1.0`) |
| `R1` | Adversarial robustness | `coefficient` (`double`, `0.0`) |
| `R2` | Adversarial robustness | `coefficient` (`double`, `0.0`) |
| `TRADES` | Adversarial robustness | `coefficient` (`double`, `0.0`) |
| `VAT` | Adversarial robustness | `coefficient` (`double`, `0.0`) |
| `EWC` | Continual learning | `strength` (`double`, `0.0`) |
| `MAS` | Continual learning | `strength` (`double`, `0.0`) |
| `SI` | Continual learning | `strength` (`double`, `0.0`), `damping` (`double`, `1e-3`) |
| `NuclearNorm` | Low-rank modelling | `strength` (`double`, `0.0`) |
| `SWA` | Ensemble averaging | `coefficient` (`double`, `0.0`) |
| `SWAG` | Ensemble averaging | `coefficient` (`double`, `0.0`), `variance_epsilon` (`double`, `1e-8`), `start_step` (`std::size_t`, `0`), `accumulation_stride` (`std::size_t`, `1`), `max_snapshots` (`std::size_t`, `0`) |
| `FGE` | Ensemble averaging | `coefficient` (`double`, `0.0`) |
| `SFGE` | Ensemble averaging | `coefficient` (`double`, `0.0`) |

## Choosing and combining penalties

- **Weight and sparsity terms** (`L1`, `L2`, `ElasticNet`, `GroupLasso`,
  `StructuredL2`, `L0HardConcrete`) encourage compact parameters through magnitude
  or group-based shrinkage.
- **Matrix constraints** (`Orthogonality`, `SpectralNorm`, `MaxNorm`,
  `NuclearNorm`) steer weight matrices towards bounded or orthogonal structure.
- **Activation-focused descriptors** (`KLSparsity`, `DeCov`, `CenteringVariance`,
  `JacobianNorm`) regulate the statistics of hidden activations.
- **Robustness objectives** (`WGANGP`, `R1`, `R2`, `TRADES`, `VAT`) add
  adversarial gradients or penalties for generative adversarial training.
- **Continual-learning utilities** (`EWC`, `MAS`, `SI`) preserve knowledge when
  training across tasks.
- **Averaging-based ensembles** (`SWA`, `SWAG`, `FGE`, `SFGE`) maintain running
  statistics or snapshots for Bayesian-style ensembling.

Descriptors are serialised alongside other components by
[Docs/Save & Load](../saveload/README.md) and integrate with
[Docs/Loss](../loss/README.md) and [Docs/Optimizer](../optimizer/README.md) descriptors to
shape the training signal for your experiments.
