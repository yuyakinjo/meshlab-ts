# meshlab-ts

A from-scratch TypeScript port of [MeshLab](https://github.com/cnr-isti-vclab/meshlab), running on
[Bun](https://bun.sh) with zero runtime dependencies.

The goal is **API compatibility with MeshLab's C++ architecture** — the same `FilterPlugin`,
`MeshDocument`, `RichParameterList`, `PluginManager` and `CMeshO` concepts, under the same names —
so that any recipe expressed against MeshLab or PyMeshLab can be expressed here without a Python
runtime.

## Status

**Tiers 0 to 2 are complete, and Tier 3 is under way.** 157 of MeshLab's 282 filters are
implemented — enough to take a broken STL from a 3D scanner or a bad export and turn it into a
printable solid, and to go from a raw point cloud to a watertight surface:

- **filter_clean** (11) — welding, degenerate and duplicate removal, isolated pieces,
  non-manifold edge and vertex repair, T-vertices
- **filter_meshing** (17) — re-orientation, hole closing, QEM decimation, transforms,
  point-cloud normal estimation and smoothing, Loop/Butterfly/midpoint subdivision,
  isotropic explicit remeshing, clustering decimation, principal curvature directions
- **filter_select** (22) — selection and deletion, dilation and erosion, selection by
  vertex/face quality, by colour in RGB or HSV, by view angle, by edge length, by connectivity,
  by triangle shape and fold, and by local outlier probability
- **filter_measure** (8) — topological and geometric measures, the area and perimeter of a
  selection, quad-mesh measures over faux-tagged triangles, and per-vertex/per-face quality
  statistics and histograms
- **filter_unsharp** (15) — Laplacian, Taubin, HC and scale-dependent smoothing, normal
  recomputation in four weighting schemes, normal and quality normalisation and smoothing,
  unsharp masking of geometry/normals/colour/quality, linear morphing
- **filter_layer** (13) — flatten, duplicate, delete, rename, prune non-visible layers, move
  the selected faces or vertices to a new layer, split into connected components, plus the
  raster layers and their cameras in Bundler `.out` and Agisoft `.xml`
- **filter_sampling** (7) — Montecarlo, stratified, clustered, Poisson-disk and element
  sampling, point-cloud simplification, Hausdorff distance
- **filter_screened_poisson** (1) — Screened Poisson surface reconstruction
- **filter_texture** (8 of 9) — per-vertex/per-wedge UV conversion with seam splitting, flat-plane
  and trivial per-triangle parametrisation, texture assignment, and baking vertex colour, normals,
  quality or another mesh's texture into a texture map
- **filter_mls** (8) — APSS and RIMLS moving least squares surfaces: projection, marching-cubes
  iso-surface extraction, curvature colouring, radius-from-density, small-component selection
- **filter_func** (12) — expression-driven filters: conditional vertex and face selection,
  per-vertex/face quality, colour, normal and geometric functions, a grid generator, an
  implicit-surface extractor and user-defined refinement
- **filter_colorproc** (22) — colour fill, invert, desaturate, levels, brightness/contrast/gamma,
  thresholding, colourisation, white balance, noise, quality-to-colour ramps, quality clamping,
  colour and quality transfer between vertices and faces, random and per-component labelling,
  Laplacian colour smoothing, triangle-shape metrics, discrete curvatures
- **filter_create** (13) — the platonic solids, box, sphere, sphere cap, cone/cylinder, torus,
  annulus, spherical point clouds, and a plane fitted to a selection

**PLY, STL, OBJ and OFF** are read and written, and **PNG** for textures, including OBJ's negative indices and all four
of its face-corner forms, and OFF's `C`/`N` header prefixes.

All **282 filters are registered from day one** — the names are extracted from the C++ sources
rather than transcribed. The 125 without an implementation yet throw `MLNotImplementedException`
when applied, so a missing filter is never mistaken for a filter that did nothing.

```bash
bun run bin/meshlab-ts list --implemented
bun run bin/meshlab-ts info "Close Holes"
bun run bin/meshlab-ts apply "Remove Duplicate Vertices" in.stl -o out.stl
bun run bin/meshlab-ts script repair.mlx broken.stl -o fixed.ply
```

A creation filter needs no input, and `--save` names the extra per-vertex channels to write —
without it only geometry goes into the file, which for a point cloud means losing the normals
that make it reconstructable:

```bash
bun run bin/meshlab-ts apply Sphere -o sphere.ply --param subdiv=4
bun run bin/meshlab-ts apply "Montecarlo Sampling" sphere.ply -o cloud.ply --param SampleNum=20000
bun run bin/meshlab-ts apply "Compute normals for point sets" cloud.ply -o oriented.ply --save normals
bun run bin/meshlab-ts apply "Surface Reconstruction: Screened Poisson" oriented.ply -o solid.ply
```

Filters can be named either the way MeshLab shows them (`"Close Holes"`) or the way PyMeshLab does
(`meshing_close_holes`); both resolve to the same filter.

## Using it as a dependency

```bash
bun add github:yuyakinjo/meshlab-ts
```

The package ships TypeScript source rather than a build, so a consuming
`tsconfig.json` needs:

```jsonc
{ "compilerOptions": { "allowImportingTsExtensions": true } }
```

Everything is re-exported from the root: filters through `MeshLabKernel`, and
the algorithms (`Clean`, `Hole`, `Smooth`, `Inertia`, `quadricSimplification`),
the I/O (`readStl`, `writeStl`, `readPly`, `writePly`) and `FilterScript`
directly, for callers who would rather not go through the filter layer.

## Repairing an STL

`.mlx` scripts exported from MeshLab run as they are. The order below is the one the end-to-end
test uses, and each step is there for a reason:

```ts
new FilterScript()
  .add("Remove Duplicate Vertices")                 // STL is an unwelded soup; nothing else
                                                    // can see a surface until this runs
  .add("Remove Zero Area Faces")
  .add("Remove Duplicate Faces")
  .add("Remove Isolated pieces (wrt Diameter)", { MinComponentDiag: 0.5 })
  .add("Remove Unreferenced Vertices")
  .add("Repair non Manifold Edges")                 // orientability needs edge-manifoldness
  .add("Re-Orient all faces coherently")
  .add("Close Holes", { MaxHoleSize: 100 })         // a hole's boundary is only well defined
                                                    // on a manifold surface
  .add("Invert Faces Orientation", { forceFlip: false })  // face outward, not inward
  .add("Select None")                               // Close Holes leaves its new faces selected
  .run(kernel, doc);
```

Two things that bite in practice, both faithful to MeshLab:

- **`Close Holes` leaves the faces it created selected.** Several filters default "operate on the
  selection" to "true if anything is selected", so without a following `Select None` the next
  filter silently confines itself to a few cap triangles.
- **Writing to STL un-welds the mesh again.** It is a triangle soup format; a repaired 202-vertex
  solid comes back as 1200 vertices. Write PLY if the sharing matters, or weld again on load.

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
- **Local operations live in one place.** `edge_ops.ts` holds the edge collapse, the edge flip
  and the link condition, because QEM decimation and isotropic remeshing both need them and both
  get them subtly wrong on their own.
- **PNG is the only image format.** Textures are read and written through a codec built on
  `node:zlib` — 8-bit greyscale, RGB, palette, grey+alpha and RGBA on the way in, always RGBA on
  the way out. Interlaced and 16-bit files are refused by name rather than half-decoded, and a
  JPEG is refused outright: a wrong-looking texture is much harder to diagnose than a clear error.
- **MLS surfaces have compact support, and say so.** APSS and RIMLS weight each sample by
  `(1 - d²/r²)⁴` inside its own radius and by nothing outside it, so a query far from the cloud
  has no answer at all. Every entry point returns null there rather than extrapolating, and the
  filters report it as "out of range"; widening `FilterScale` is the knob that exists for it.
- **The abstract domain never loses a vertex.** `parametrization/abstract_domain.ts` simplifies a
  mesh while every original vertex stays pinned inside some face of the coarse domain by a
  barycentric coordinate. A collapse flattens the affected star, records where each pin sits,
  collapses, flattens the smaller star *reusing the first one's boundary layout* so the two share
  a coordinate system, and re-pins. A collapse that would strand a pin, or that folds the star
  when flattened, is undone rather than approximated.
- **Two flattening weights, because neither one wins.** `parametrization/harmonic.ts` offers mean
  value and cotangent weights for mapping a disk into the plane. Mean value weights are always
  positive, so Tutte's theorem applies and the result can never fold; cotangent weights minimise
  Dirichlet energy and so preserve angles better, but an obtuse triangle contributes a negative
  weight and the guarantee is gone. The default is mean value and the result reports whether it
  actually came out unfolded, so a caller wanting the conformal map can check and fall back.
- **Quad meshes are triangle meshes.** VCGLib tags the diagonals introduced by triangulation as
  "faux", so a quad is two triangles sharing a faux edge. Every algorithm here keeps working on a
  quad mesh without knowing it is one. One deliberate divergence lives in `bit_quad.ts`'s caller:
  MeshLab zeroes the quad count whenever `CountBitLargePolygons` is positive, which for a clean
  quad mesh is always — so upstream reports zero quads for every quad mesh. We zero it only when
  that count actually differs from the plain polygon count, which is the condition its own comment
  describes.
- **Rasters carry a camera, not an image.** A `RasterModel` is a `Shot` plus the path to its
  photograph; nothing decodes pixels. What is parsed is the PNG or JPEG header, because Bundler
  stores no image size and expects the reader to recover the viewport from the image itself.
- **The expression dialect is muParser's, not JavaScript's.** `filter_func` hands user formulas
  to muParser upstream, so `^` is exponentiation and right-associative, a unary sign binds
  tighter than `+`/`-` but looser than `^` (`-2^2` is -4, `-2*3` is -6), `log` is the natural
  logarithm, and comparisons yield 1 or 0 rather than booleans. `rnd` is the one omission: a
  filter whose output changed between runs would make every downstream hash useless.
- **Screened Poisson is reimplemented, not ported.** MeshLab vendors 15k lines of Kazhdan's
  PoissonRecon; this is a trilinear multigrid solve with marching tetrahedra instead of degree-2
  B-splines with conjugate gradients. The parameters and their meanings match, so the output is
  geometrically equivalent rather than bit-identical — a sampled sphere, torus and box each come
  back watertight, at the right genus, within a few percent of their true volume.

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
