# Optimizer Descriptors

Optimizers are expressed as descriptors so they can be registered globally via
`Model::set_optimizer` or attached to block-level scopes through
[Docs/Local](../local/README.md). The variant defined in `src/optimizer/optimizer.hpp`
covers first-order methods, adaptive optimizers, and experimental research
optimizers. All descriptors expose an `.options` struct mirroring the fields of
their underlying implementation.

## Classic methods

| Helper | Key options | Notes |
| --- | --- | --- |
| `Optimizer::SGD` | `learning_rate`, `momentum`, `dampening`, `weight_decay`, `nesterov` | Vanilla SGD with momentum/Nesterov switches. |
| `Optimizer::RMSprop` | `learning_rate`, `alpha`, `eps`, `centered`, `momentum`, `weight_decay` | Torch's RMSprop variant with centred updates. |
| `Optimizer::Adagrad` | `learning_rate`, `lr_decay`, `weight_decay`, `eps` | Supports accumulator initialisation and decay. |

## Adam family

| Helper | Key options | Highlights |
| --- | --- | --- |
| `Optimizer::Adam` | `learning_rate`, `beta1`, `beta2`, `eps`, `weight_decay`, `amsgrad` | Drop-in replacement for `torch::optim::Adam`. |
| `Optimizer::AdamW` | Same as Adam plus decoupled `weight_decay` defaulting to `1e-2`. | Uses decoupled weight decay by default. |
| `Optimizer::Lion` | `learning_rate`, `beta1`, `beta2`, `weight_decay` | Implements the sign-based Lion optimiser. |
| `Optimizer::LAMB` | `learning_rate`, `beta1`, `beta2`, `eps`, `weight_decay`, `clamp_value`, `adapts_lr` | Layer-wise adaptive moments for large batch training. |
| `Optimizer::Adafactor` | `learning_rate`, `beta1`, `decay_rate`, `weight_decay`, `clip_threshold`, `factored`, `scale_parameter` | Memory-efficient second-order approximation suitable for transformers. |

## Sophia and Muon variants

| Helper | Key options | Highlights |
| --- | --- | --- |
| `Optimizer::SophiaG` / `SophiaH` | `learning_rate`, `beta1`, `beta2`, `rho`, `weight_decay`, `use_prox`, `adaptive_clip` | Implements Sophia's curvature-approximation updates (G/H variants). |
| `Optimizer::Muon` | `learning_rate`, `beta1`, `beta2`, `eps`, `weight_decay`, `clip_threshold` | Gradient-free Muon optimiser with sign descent and gradient clipping. |
| `Optimizer::AdaMuon` | Adds `beta3` for adaptive scaling | Adaptive Muon variant for non-stationary landscapes. |
| `Optimizer::MuonManifold` | Extends Muon with `riemannian` toggles and manifold-aware projections | Targets optimisation on Stiefel/Grassmann manifolds. |

### Warmup integration

Adam-based optimizers expose `ensure_state_initialized()` hooks that Thot calls
when CUDA graph capture or warmup schedules are used. This keeps the internal
state tensors resident on the correct device before the first optimisation step.

---

Pair these optimizers with [Docs/LrScheduler](../lrscheduler/README.md) descriptors
and [Docs/Regularization](../regularization/README.md) penalties to fully define the
training regime configured through [Docs/Training](../training/README.md).