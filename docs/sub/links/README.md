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
  concatenation) mode. The stack mode accepts an optional `dim`/`axis`
  attribute to choose the concatenation dimension.【F:src/common/graph.hpp†L24-L220】【F:src/core.hpp†L1891-L1916】

When the same join is mentioned multiple times, the compiler verifies that the
merge policy and concatenation dimension stay consistent, ensuring that shared
aggregation points behave predictably.【[Code: Join policy impl](../../../src/core.hpp#L1891-L1916)】

### Building the graph
`Model::links` accepts either the legacy `(specs, bool)` signature or the newer
`LinkParams` structure. The latter lets you:

* Opt into CUDA graph capture for the linked execution plan.
* Map human-readable aliases to multi-input/multi-output indices.
* Describe the actual routing edges through `std::vector<LinkSpec>`.【F:src/core.hpp†L778-L812】【F:src/core.hpp†L789-L1012】

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
opting into CUDA graph capture for the final plan.【[CIFAR routing example impl](../../../test/cifar.cpp#L144-L165)】

## Worked example
The snippet below fuses local configuration with custom routing. A dedicated
AdamW optimizer is scoped to the `head` classifier, and a `Stack` join merges two
branches before the head runs:

```cpp
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
    Thot::LinkSpec{Thot::Port::join({"ConvPath1", "MpPath2"}, Thot::MergePolicyKind::Concat), Thot::Port::Module("FCExit")}, // join

    //Level 4 (Output)
    Thot::LinkSpec{Thot::Port::Module("FCExit"), Thot::Port::Output("@output")},
});
```

The alias mapping in `LinkParams` exposes the head output as `"logits"` while
still stacking additional outputs automatically if more are declared.