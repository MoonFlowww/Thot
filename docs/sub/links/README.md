## Routing with `Model::links`

### Ports, joins, and policies
Custom routing is defined as a list of `Thot::LinkSpec` pairs. Each pair links a
source port to a target port. Ports can reference inputs, outputs, modules, or
join buffers:

* `Port::Input(...)` and `Port::Output(...)` address model terminals.
* `Port::Module(...)` references a layer or block by name (or by numeric `#`
  index when unnamed).
* `Port::Join(...)` materialises a named aggregation buffer that can operate in
  `Strict` (pass-through), `Broadcast` (elementwise sum), or `Stack` (tensor
  concatenation) mode. The builder offers overloads for referencing an existing
  join by name, for declaring a multi-source join through an initializer-list of
  module names, and for pinning the stack dimension either via a dedicated
  parameter or an inline `dim=`/`axis=` attribute.【F:src/common/graph.hpp†L116-L220】

When the same join is mentioned multiple times, the compiler verifies that the
merge policy and concatenation dimension stay consistent, ensuring that shared
aggregation points behave predictably. [Code: Join policy impl](../../../src/core.hpp#L1891-L1916)

### Building the graph
`Model::links` accepts either the legacy `(specs, bool)` signature or the newer
`LinkParams` structure. The latter exposes three knobs that travel alongside the
vector of `LinkSpec`s:

* `LinkParams::enable_graph_capture` turns CUDA graphs on for the compiled
  wiring plan (no effect when the spec list is empty).【F:src/core.hpp†L788-L803】
* `LinkParams::inputs` is an alias → index table for addressing multiple exposed
  inputs.
* `LinkParams::outputs` performs the same role for outputs.【F:src/core.hpp†L779-L803】

The engine uses the highest referenced index from those maps to decide how many
`@input[k]`/`@output[k]` terminals to materialise and ensures the zeroth entry is
present even when the maps are left empty.【F:src/core.hpp†L801-L862】 Aliases are
consulted when resolving `Port::Input("alias")` or `Port::Output("alias")` and
fall back to numeric selectors like `@input[1]` or direct indices such as
`#0`. Requests for unknown aliases raise an exception so mis-spellings are
caught early.【F:src/core.hpp†L967-L1012】

The helper maps make it easy to express readable routing. For instance, to wire
an auxiliary loss head you can bind names to indices and still use numbered
ports when convenient:

```cpp
model.links(
    {/* link specs */},
    {
        .inputs = {{"HUFL", 0}, {"HULL", 1}, {"MUFL", 2}, {"MULL", 3}, {"LUFL", 4}, {"LULL", 5}}, // ETTH1 inputs name with their idx
        .outputs = {{"logits", 0}, {"aux_loss", 1}},
        .enable_graph_capture = true,
    }
);
```

During compilation the engine automatically adds join edges for aggregated ports
and disallows illegal connections (for example, trying to source from an output
port or feeding multiple producers into a non-join consumer). It also preserves
implicit sequential connections for modules that share the same registration
name, so naming a chain `"stem"` keeps the intra-stem wiring intact.【[Code: Graph compilation impl](../../../src/core.hpp#L1041-L1122)】

Finally, the compiled graph must be acyclic and every node needs a producer. A
topological sort enforces these conditions, and additional logic stacks multiple
terminal outputs into an automatic `Stack` join so the model still returns a
single tensor when several outputs are exposed.【[Code: Topological sort & output stacking impl](../../../src/core.hpp#L1126-L1239)】

If you clear the specification vector, the compiled routing is dropped and the
model falls back to its default linear ordering (each `.add()` feeds the next
module).【[Routing reset impl](../../../src/core.hpp#L789-L804)】

### Runtime behaviour
At execution time, join nodes collect tensors from their producers and apply the
selected merge policy before forwarding the result downstream. Strict joins act
like wires, broadcast joins sum tensors, and stack joins concatenate along the
configured dimension.【[Join policy runtime impl](../../../src/core.hpp#L1891-L1916)】

### Example from the test suite
The CIFAR example shows how to rewire a staged vision encoder by naming layers
and explicitly linking the branches into a Vision Transformer head, while also
opting into CUDA graph capture for the final plan.【[CIFAR routing example impl](../../../test/classification/images/cifar10.cpp#L144-L165)】

## Worked example
The snippet below fuses local configuration with custom routing. A dedicated
AdamW optimizer is scoped to the `head` classifier, and a `Stack` join merges two
branches before the head runs:

```cpp
//model.add(..., "_name_"); Name is optional
model.add(Thot::Layer::Conv2d({3, 32, {3, 3}}), "ConvEntry");
model.add(Thot::Layer::Conv2d({32, 64, {3, 3}}), "ConvPath1");
model.add(Thot::Layer::MaxPool2d({{2, 2}}), "MpPath2");
model.add(Thot::Layer::FC({64 * 16 * 16, 10}, Thot::Activation::Identity, Thot::Initialization::HeNormal, head_local), "FCExit");

model.links({
    //Level 1 (Input)
    Thot::LinkSpec{Thot::Port::Input("@input"), Thot::Port::Module("ConvEntry")},

    //Level 2 (Multi Channels)
    Thot::LinkSpec{Thot::Port::Module("ConvEntry"), Thot::Port::Module("ConvPath1")}, // path #1
    Thot::LinkSpec{Thot::Port::Module("ConvEntry"), Thot::Port::Module("MpPath2")}, // path #2

    //Level 3 (Merge Channels)
    Thot::LinkSpec{Thot::Port::Join({"ConvPath1", "MpPath2"}, Thot::MergePolicy::Stack), Thot::Port::Module("FCExit")}, // join

    //Level 4 (Output)
    Thot::LinkSpec{Thot::Port::Module("FCExit"), Thot::Port::Output("@output")},
});
```

The alias mapping in `LinkParams` exposes the head output as `"logits"` while
still stacking additional outputs automatically if more are declared.