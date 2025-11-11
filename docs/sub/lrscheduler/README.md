# Learning Rate Schedulers

Schedulers in `Thot::LrScheduler` are thin descriptors layered on top of LibTorch
optimizers. Pair them with [Optimizer](../optimizer/README.md) descriptors via
`Model::set_optimizer` or `LocalConfig` to control the learning rate schedule per
module.

## Cosine Annealing with Warmup

`LrScheduler::CosineAnnealing` mirrors the widely used cosine decay policy with
an optional linear warmup stage. The options struct exposes:

- `T_max` – Number of scheduler steps covering one cosine period. Must be
  positive.
- `eta_min` – Minimum learning rate reached at the trough. Defaults to `0.0`.
- `warmup_steps` – Number of linear warmup iterations executed before the cosine
  cycle begins.
- `warmup_start_factor` – Initial scale applied to the base learning rate when
  warmup is enabled (0 → start from zero, 1 → start from the base rate).

The scheduler validates that optimizer parameter groups remain constant after
construction and updates each group's `lr` in lockstep. Use it alongside the
telemetry exposed by [Training](../training/README.md) to plot learning-rate
trajectories with [Plot](../plot/README.md).

---

Additional schedulers can be added by implementing `Details::Scheduler` and
expanding the `Descriptor` variant, following the same pattern as
`CosineAnnealing`.