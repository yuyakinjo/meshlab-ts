# meshlab-ts

A from-scratch TypeScript port of [MeshLab](https://github.com/cnr-isti-vclab/meshlab), running on
[Bun](https://bun.sh) with zero runtime dependencies.

The goal is **API compatibility with MeshLab's C++ architecture** — the same `FilterPlugin`,
`MeshDocument`, `RichParameterList`, `PluginManager` and `CMeshO` concepts, under the same names —
so that any recipe expressed against MeshLab or PyMeshLab can be expressed here without a Python
runtime.

## Status

**Tier 0 is complete**: mesh kernel, adjacency, plugin framework, filter registry and PLY/STL I/O.
Filters themselves are implemented tier by tier from here.

All **282 MeshLab filters are registered from day one** — the number and the names are extracted
from the C++ sources rather than transcribed. The ones without an implementation yet throw
`MLNotImplementedException` when applied, so a missing filter is never mistaken for a filter that
did nothing.

```bash
bun run bin/meshlab-ts list --class Cleaning
bun run bin/meshlab-ts info "Close Holes"
bun run bin/meshlab-ts apply "Remove Duplicate Vertices" in.stl -o out.stl
```

Filters can be named either the way MeshLab shows them (`"Close Holes"`) or the way PyMeshLab does
(`meshing_close_holes`); both resolve to the same filter.

## Development

```bash
bun install
bun run reference:clone   # clones MeshLab + VCGLib into .reference/ (gitignored, read-only spec)
bun run typecheck
bun run lint
bun test
```

`.reference/` is upstream MeshLab and VCGLib, used as a behavioural specification and as the source
of truth for filter names and parameter defaults. No code is copied from it.

## Design notes

- **Index-based structure-of-arrays.** `CMeshO` stores geometry in typed arrays and refers to
  vertices and faces by numeric index. C++ VCGLib's `PointerUpdater` machinery exists only because
  it holds raw pointers into a reallocating `std::vector`; with indices it is unnecessary.
- **Lazy deletion, like VCG.** Deleting sets a `DELETED` flag and decrements `vn`/`fn`; the slot
  stays. `vertSize !== vn` in general. Compaction is an explicit step that returns an old→new remap.
- **Optional components** (VCG's "Ocf") are driven by one channel-descriptor table, allocated on
  demand by `MeshModel.updateDataMask()`, exactly as MeshLab's `MM_*` mask does.
- **Preconditions and postconditions are enforced by the framework.** In C++ this is the GUI's job,
  which means headless callers lose it. Here `FilterExecutor` checks `getPreConditions()`,
  satisfies `getRequirements()`, and applies `postCondition()` around every filter.
- **Unknown parameters are an error.** `applyFilter(doc, "Close Holes", { maxholesizze: 30 })`
  raises rather than running the filter with a default and reporting success.
- **The filter catalogue is generated, not written.** `bun run stub:gen` reads the filter names,
  PyMeshLab names, classes and descriptions straight out of `.reference/meshlab`. 281 of the 282
  filters override `pythonFilterName` upstream, so deriving those names instead of reading them
  would have got almost all of them wrong.

## Testing

Tests are built on mathematics rather than on recorded output, because the whole point of the
project is not needing a Python install:

- **Analytic meshes** whose properties follow from closed forms — a torus is genus 1, an
  icosahedron has area 5√3a², a Möbius strip is not orientable. The builders are themselves
  verified first, against an independent naive implementation.
- **Structural invariants** checked after every mutation: live counts against deleted flags, channel
  lengths against capacity, FF rings closing, every face corner appearing exactly once in its
  vertex's VF chain.
- **Property-based tests** cross-validating the kernel's sorted-edge and intrusive-ring algorithms
  against straightforward hash-map versions, over random triangle soup including non-manifold,
  degenerate and disconnected inputs.

`bun run test:prop:deep` raises the fuzzing to 5000 cases per property. There is also an opt-in
`bun run golden:regen`, which compares against a real PyMeshLab in Docker; it refuses to run
without `MESHLAB_TS_ALLOW_GOLDEN_REGEN=1` and is never run by CI.

## License

GPL-3.0-or-later. MeshLab and VCGLib are GPL-2.0-or-later; this project mirrors their architecture
closely enough that copyleft is the honest and safe choice.
