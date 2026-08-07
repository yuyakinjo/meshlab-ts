# meshlab-ts

A from-scratch TypeScript port of [MeshLab](https://github.com/cnr-isti-vclab/meshlab), running on
[Bun](https://bun.sh) with zero runtime dependencies.

The goal is **API compatibility with MeshLab's C++ architecture** — the same `FilterPlugin`,
`MeshDocument`, `RichParameterList`, `PluginManager` and `CMeshO` concepts, under the same names —
so that any recipe expressed against MeshLab or PyMeshLab can be expressed here without a Python
runtime.

## Status

Tier 0 (kernel, plugin framework, filter registry, PLY/STL I/O). Filters are being implemented tier
by tier. **Every MeshLab filter is present in the registry from day one**; the ones that are not
implemented yet throw `MLNotImplementedException` rather than silently doing nothing.

```bash
bun run bin/meshlab-ts list          # every registered filter, with its class and status
bun run bin/meshlab-ts info "Close Holes"
```

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

## License

GPL-3.0-or-later. MeshLab and VCGLib are GPL-2.0-or-later; this project mirrors their architecture
closely enough that copyleft is the honest and safe choice.
