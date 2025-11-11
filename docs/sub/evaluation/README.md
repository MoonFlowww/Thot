# Evaluation Pipelines

`Thot::Evaluation` provides templated `Evaluate` helpers for supervised models.
Each invocation streams batches through the network, accumulates metrics, and
prints human-readable tables if enabled. Two descriptors exist today:
`Evaluation::Classification` and `Evaluation::Timeseries`.

## Common workflow

```cpp
auto report = Thot::Evaluation::Evaluate(
    model, inputs, targets,
    Thot::Evaluation::Classification,
    {Thot::Metric::Classification::Accuracy,
     Thot::Metric::Classification::F1,
     Thot::Metric::Classification::ExpectedCalibrationError},
    {.batch_size = 128, .calibration_bins = 20});
    
// Or
auto report = model.evaluate(test_images, test_labels, Thot::Evaluation::Classification,{
    Thot::Metric::Classification::Accuracy,
    Thot::Metric::Classification::F1,
    Thot::Metric::Classification::ExpectedCalibrationError}, 
    {.batch_size = 64, .calibration_bins = 20});
```

- **Batching and buffering.** Both descriptors accept `Options` with
  `batch_size` and `buffer_vram`. When buffering is enabled, inputs/targets are
  staged on the host and copied to the device chunk-by-chunk, mirroring the
  streaming settings used by [Training](../training/README.md).
- **Console output.** Set `print_summary`/`print_per_class` (classification) or
  `print_summary` (timeseries) to control pretty-printed tables. Streams default
  to `std::cout`, but you can forward them anywhere via `Options::stream`.
- **Frame style.** Pick box-drawing vs. ASCII separators through
  `Utils::Terminal::FrameStyle`.

The resulting `Report` structures expose raw numbers so you can feed them into
custom dashboards or the [Plot](../plot/README.md) reliability tools.

## Classification descriptor

Designed for multi-class and multilabel classification:

- Accepts any mix of `Metric::Classification::Descriptor` entries, including
  calibration metrics, ROC/PR scores, top-k accuracy, class prevalence, and
  drift diagnostics.
- Aggregates support-weighted and macro averages in `Report::summary`, while the
  `per_class` matrix holds each metric per label. `labels` preserves the mapping
  from column index to class id.
- Computes calibration histograms with configurable `calibration_bins` and keeps
  the raw bin contents inside the telemetry hooks driving DET/ROC plots.

When `print_per_class` is `true`, the runner renders a confusion-matrix-style
table sorted by label id. Use this to spot overconfident or undertrained
classes before saving checkpoints with [Save & Load](../saveload/README.md).

## Timeseries descriptor

Specialised for regression and forecasting:

- Supports error-based metrics (`MAE`, `RMSE`, `SMAPE`, `MAPE`), statistical
  tests (`LjungBox`, `DurbinWatson`, `JarqueBera`), distributional distances
  (`Wasserstein`, `MMD`), and probabilistic scores (`CRPS`, `Pinball`,
  `PredictionIntervalCoverage`).
- Tracks the number of evaluated series and individual time points inside the
  `Report` so you can normalise metrics manually when composing multiple runs.
- Ensures the model is in evaluation mode and restores the training flag after
  the pass, mirroring the safety checks used elsewhere in the framework.

Both evaluation modes throw descriptive exceptions if tensor shapes disagree or
if the requested metric list is empty, helping catch mistakes early in the
experiment lifecycle.

---

For post-training calibration, call `Model::calibrate` (see
[Training](../training/README.md)) and visualise the resulting reliability curves
with [Plot](../plot/README.md).