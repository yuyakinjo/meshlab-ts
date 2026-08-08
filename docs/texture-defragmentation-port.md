# Porting `filter_texture_defragmentation`

A survey written before deciding whether to implement it, because this is the
largest single item left in the project and the earlier estimate that it was one
of "two light remaining filters" was wrong.

## What the filter does

Photo-reconstructed models arrive with a texture atlas cut into hundreds of small
charts. Every cut is a seam, every seam costs texel bleed and a visible line, and
the packing wastes most of the image. The filter merges charts back together
wherever the merge can be done without introducing distortion or overlap, then
repacks and re-renders the atlas.

It is Maggiordomo, Cignoni and Tarini, *Texture Defragmentation for
Photo-Reconstructed 3D Models* (2021), vendored into MeshLab as a 6,935-line
sub-library. It is not a thin wrapper over VCGLib the way most plugins are.

## Shape of the algorithm

```
charts  = decompose the mesh by UV seams
seams   = the seam network, split at junctions, clustered per chart pair
queue   = every clustered seam, keyed by a cost derived from its 3D length

while queue is not empty:
    take the cheapest seam
    compute the rigid/similarity transform matching one side's UVs to the other
    build a "shell": the local patch around the seam, in UV space
    run 2D ARAP on the shell with the far boundary pinned
    accept if: no local overlap, no global overlap, distortion within tolerance,
               shell genus still 0
    otherwise reject, penalise, and put it back

repack the surviving charts into a new atlas
resample the old texture images through the new parametrization
```

The accept/reject test is the hard part and the bulk of the code: `seam_remover.cpp`
alone is 1,660 lines, most of it bookkeeping for retries, penalties and the
`CheckStatus` cases (`FAIL_LOCAL_OVERLAP`, `FAIL_GLOBAL_OVERLAP_BEFORE`,
`FAIL_GLOBAL_OVERLAP_AFTER_OPT`, `FAIL_DISTORTION_LOCAL/GLOBAL`, `FAIL_TOPOLOGY`,
`FAIL_NUMERICAL_ERROR`).

## What is already here

More than expected. The two things that looked like blockers are not.

| Needed | Have | Where |
| --- | --- | --- |
| Sparse linear solve for ARAP | Jacobi-preconditioned CG | `vcg/math/sparse.ts` |
| Cotangent weights | used by the harmonic map | `vcg/complex/parametrization/harmonic.ts` |
| Local/global rotation fitting | cubic stylization does the 3D case | `meshlabplugins/filter_cubization` |
| Texture resampling | software rasteriser + pull-push fill | `meshlabplugins/filter_texture/rastering.ts` |
| Texture images | PNG read/write, `MeshModel.textures` | `vcg/space/image/`, `ml_document/mesh_model.ts` |
| UV seam handling | seam-aware vertex splitting | `vcg/complex/attribute_seam.ts` |
| Disk flattening | harmonic, with both weightings | `parametrization/harmonic.ts` |

**The OpenGL dependency is not a blocker.** Upstream calls `RenderTexture` through
a GL context only to rasterise the resampled atlas; `rasteriseFace` plus
`pullPushFill` already do that in software, and are already used by the four
`Transfer: … to Texture` filters.

## What would have to be written

| Module | New lines (est.) | Notes |
| --- | --- | --- |
| `chart_graph.ts` | ~250 | charts from UV seams, adjacency, cached areas and border lengths |
| `seams.ts` | ~300 | seam network, split at junctions, clustered per chart pair |
| `matching.ts` | ~150 | 2D least-squares rigid / similarity / affine fit between two point sets |
| `arap2d.ts` | ~300 | 2D ARAP, cotangent Laplacian, pinned vertices, over `solveCG` |
| `shell.ts` | ~250 | the local optimization patch, hole filling, scaffolding |
| `intersection2.ts` | ~200 | segment-set self-intersection and cross-intersection, grid-accelerated |
| `greedy.ts` | ~500 | the priority queue, the accept/reject checks, penalties and retries |
| `packing.ts` | ~400 | outline extraction, rasterised-outline packing, texture size search |
| `atlas_render.ts` | ~150 | resample through the new parametrization — mostly reuse |
| plugin + tests | ~400 | |
| **total** | **~2,900** | |

For scale: Screened Poisson, the largest thing reimplemented so far, was of the
same order. This is three to four batches of the size the project has been
running at, not one.

## Suggested staging

Each stage is independently testable, so a stall at any point still leaves
working, tested code rather than a half-filter.

**Stage A — the self-contained mathematics. Done.** `math/mat2.ts`,
`space/intersection2.ts`, `parametrization/matching2.ts`,
`parametrization/arap2d.ts`, with 29 tests. These had the crispest invariants in
the port and were held to them: the matching fits recover a known similarity
*exactly* from noiseless points and are the least-squares optimum under noise,
the grid-accelerated intersection agrees with brute force on random inputs, and
ARAP is exactly zero on an isometry with its fitting energy decreasing every
single iteration. Two things worth recording:
the three matching fits all collapse to one closed-form 2×2 polar
decomposition, so upstream's three separate Eigen paths became one function;
and `SparseMatrix.pin` turned out to be unusable twice on one matrix, which is
exactly what a two-coordinate solve needs — `pinMulti` now does both at once,
and the old failure mode was silent.

**Stage B — the structure.** `chart_graph.ts`, `seams.ts`, `shell.ts`. Testable by
combinatorics: chart count against connected components of the UV-cut mesh, seam
endpoints against vertices of seam-valence ≠ 2, `Σ chart faces = mesh faces`.

**Stage C — the driver.** `greedy.ts`. The stage with no exact oracle. Tested by
monotone properties instead: total UV border length strictly decreases, chart
count never increases, face count never changes, and a mesh whose atlas is
already a single chart is left untouched.

**Stage D — output.** `packing.ts`, `atlas_render.ts`, the plugin, end-to-end
tests: no two charts overlap in the packed atlas, every face has a non-degenerate
UV triangle, and the resampled texture agrees with the original when sampled
through both parametrizations.

## Divergences to expect

- **Packing quality.** Upstream uses VCGLib's `RasterizedOutline2Packer` with a
  permutation search over chart orderings and 16 rotations. A simpler packer
  gives a valid but larger atlas. That is a quality difference, not a correctness
  one — but it must be logged rather than left for the user to discover from a
  file size.
- **Which merges get accepted.** The greedy driver's penalty and retry
  bookkeeping is heuristic; a reimplementation will not accept exactly the same
  set of merges in the same order. The invariants above are what can be
  promised, not chart-for-chart agreement with MeshLab.
- **`timelimit`.** Upstream's parameter stops the optimization by wall clock,
  which makes the output machine-dependent. Recommendation: implement it, but
  treat `0` (unlimited) as the only tested path and say so, the way
  `Extratcoordw` and `overlapFlag` are handled elsewhere.
- **Non-manifold input.** Upstream warns and continues with "seam topology may be
  unreliable". Worth deciding deliberately rather than inheriting: a warning that
  precedes a wrong atlas is close to no warning at all.

## Recommendation

Feasible, and nothing about it is blocked — but it is a multi-batch project on its
own, and it is the only remaining filter of that size. It is also not on
ishijishi's path: nothing in the STL-repair or point-cloud-to-surface pipelines
touches a texture atlas.

So: worth doing if the goal is 269/282, worth skipping if the goal is a library
that covers what ishijishi and its neighbours actually run. Stage A is the
cheapest way to find out — the three modules in it are useful on their own
(2D ARAP in particular is what a future `Parametrization: ARAP` would need), and
they are the part with real oracles.
