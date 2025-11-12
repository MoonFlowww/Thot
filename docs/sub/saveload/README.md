# Saving and Loading Models

Thot serialises models as a pair of files inside the target directory:

- `architecture.json` – human-readable description of the module graph, including
  block/layer descriptors, `LocalConfig` overrides, loss/optimizer/regularisation
  descriptors, and named ports used by [Links](../links/README.md).
- `parameters.binary` – TorchScript archive containing module parameters and
  buffers.

Invoke `Model::save(path)` after building your network; Thot ensures the target
folder exists and will prompt before overwriting when the directory already
exists. Confirming the prompt causes the existing files to be rewritten in
place—Thot does not create temporary files or offer atomic guarantees for the
update. During `Model::load`:

1. `architecture.json` is parsed and replayed by calling `Model::add()` for each
   recorded module, preserving insertion order and human-readable names.
2. Global metadata such as `model_name`, local optimizers, and `LocalConfig`
   attachments are restored before parameters are loaded.
3. `parameters.binary` is read into a validation archive to verify every
   parameter/buffer key and tensor shape matches the freshly constructed model.
   Thot raises explicit errors when shapes diverge, preventing silent corruption.
4. Once validated, tensors are loaded into the module and `configure_step_impl()`
   is invoked to rebuild execution plans, CUDA graphs, and regularisation state.

Because descriptors are stored verbatim, saving after calling
`Model::set_optimizer`, `Model::set_loss`, or attaching [Regularization](../regularization/README.md)
ensures checkpoints carry full training intent. Tensor devices are preserved; if
you wish to load on a different device, call `model.to_device(...)` after
loading.

For reproducible experiments, pair checkpoints with telemetry exported from
[Training](../training/README.md) and reliability plots generated via
[Plot](../plot/README.md).

For full implementation details, see `Model::save` in [`src/core.hpp`](../../../src/core.hpp);
it delegates JSON persistence to `write_json_file` in [`src/common/save_load.hpp`](../../../src/common/save_load.hpp),
which streams directly to the destination path.