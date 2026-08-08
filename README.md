# meshlab-ts

A from-scratch TypeScript port of [MeshLab](https://github.com/cnr-isti-vclab/meshlab), running on
[Bun](https://bun.sh) with zero runtime dependencies.

The goal is **API compatibility with MeshLab's C++ architecture** — the same `FilterPlugin`,
`MeshDocument`, `RichParameterList`, `PluginManager` and `CMeshO` concepts, under the same names —
so that any recipe expressed against MeshLab or PyMeshLab can be expressed here without a Python
runtime.

## Status

**Tiers 0 to 2 are complete, and Tier 3 is under way.** 269 of MeshLab's 282 filters are
implemented — enough to take a broken STL from a 3D scanner or a bad export and turn it into a
printable solid, and to go from a raw point cloud to a watertight surface:

- **filter_clean** (15, complete) — welding, degenerate and duplicate removal, isolated pieces,
  non-manifold edge and vertex repair, T-vertices, removal by quality, wedge-UV merging,
  mismatched-border snapping, and ball-pivoting surface reconstruction
- **filter_meshing** (37, complete) — re-orientation, hole closing, QEM decimation, the transform
  family (scale, centre, rotate, freeze, reset, explicit matrix, translation/rotation/scale,
  inversion, axis flip and swap, principal-axis alignment, rotate-to-fit-a-plane), point-cloud
  normal estimation and smoothing, Loop/Butterfly/midpoint subdivision, isotropic explicit
  remeshing, clustering decimation, principal curvature directions, crease-edge selection,
  polygon-to-triangle conversion, planar sections, selection perimeters, crease polylines and
  cylindrical unwrapping, quad-dominant pairing, 4-8 quad refinement, LS3 Loop subdivision and
  vertex attribute seams, smart triangle pairing, and Catmull-Clark and Doo-Sabin subdivision
- **filter_select** (24, complete) — selection and deletion, dilation and erosion, selection by
  vertex/face quality, by colour in RGB or HSV, by view angle, by edge length, by connectivity,
  by triangle shape and fold, by local outlier probability, by self-intersection, and by
  texture seam
- **filter_measure** (8) — topological and geometric measures, the area and perimeter of a
  selection, quad-mesh measures over faux-tagged triangles, and per-vertex/per-face quality
  statistics and histograms
- **filter_unsharp** (21, complete) — Laplacian, Taubin, HC, scale-dependent, depth-constrained
  and feature-preserving two-step smoothing, normal recomputation in four weighting schemes plus
  per-polygon normals, normal and quality normalisation and smoothing, unsharp masking of
  geometry/normals/colour/quality, linear morphing, directional geometry preservation, cutting
  along crease edges, and scalar harmonic fields
- **filter_layer** (13) — flatten, duplicate, delete, rename, prune non-visible layers, move
  the selected faces or vertices to a new layer, split into connected components, plus the
  raster layers and their cameras in Bundler `.out` and Agisoft `.xml`
- **filter_sampling** (14, complete) — Montecarlo, stratified, clustered, Poisson-disk, element,
  texel and regular-recursive sampling, point-cloud simplification, Hausdorff and reference-mesh
  distance, vertex attribute transfer, uniform volumetric resampling, and Voronoi and disk vertex
  colouring
- **filter_qhull** (4, complete) — convex hull, alpha complex and alpha shape, Voronoi filtering,
  and hidden-point removal, over a quickhull and a Bowyer-Watson Delaunay tetrahedralization
  written here rather than linked from Qhull, and a seam-preserving textured decimation
- **filter_screened_poisson** (1) — Screened Poisson surface reconstruction
- **filter_texture** (9, complete) — per-vertex/per-wedge UV conversion with seam splitting,
  flat-plane, trivial per-triangle and Voronoi-atlas parametrisation, texture assignment, and baking vertex colour, normals,
  quality or another mesh's texture into a texture map
- **filter_embree** (5) — the ray-traced measures under the names a script written against Embree
  uses, plus geometric face re-orientation and visible-face selection
- **filter_mesh_booleans** (4) — union, intersection, difference and XOR, volumetrically
- **filter_color_projection** (3) — project the registered photographs onto the vertices or into
  a texture, with a depth test and angle/distance/border weighting
- **filter_mesh_alpha_wrap** (1) — a watertight shell around any input, however broken
- **filter_cubization** (1) — Liu and Jacobson's cubic stylization
- **filter_developability** (1) — Stein, Grinspun and Crane's developability optimization, which
  pushes a surface toward pieces that can be cut flat and folded
- **filter_texture_defragmentation** (1) — Maggiordomo, Cignoni and Tarini's atlas
  defragmentation: merge the charts a photo-reconstruction cut apart, repack, and resample the
  texture through the new parametrization
- **filter_camera** (8) — set a mesh's or a raster's camera, move any or all of them, measure
  vertex quality from a camera, and re-orient normals to face one
- **filter_dirt** (2) — dust accumulation and point-cloud movement over a surface
- **filter_sdfgpu** (3) — shape diameter function, depth complexity and volumetric obscurance,
  by ray casting rather than by depth peeling
- **filter_icp** (3) — point-to-plane ICP between two meshes or across every visible layer, and
  an overlap measure
- **filter_ao** (1) — ambient occlusion
- **filter_voronoi** (4) — geodesic Voronoi sampling with Lloyd relaxation, volumetric sampling,
  Voronoi scaffolding and solid wireframes
- **filter_fractal** (3) — fractal terrain, fractal displacement and crater generation, over the
  five Musgrave fractal functions
- **filter_createiso** (1) — a noisy isosurface
- **filter_geodesic** (4) — distance along the surface from a point, the selection or the border,
  by Dijkstra or by the heat method
- **filter_trioptimize** (3) — edge flipping for triangle shape or for curvature, and a Laplacian
  smooth that refuses to move the surface
- **filter_parametrization** (2) — fixed-boundary harmonic and least-squares conformal maps
- **filter_quality** (1) — quality through an adjustable transfer function into colour
- **filter_sample** (1) — random vertex displacement
- **filter_isoparametrization** (4) — Pietroni, Tarini and Cignoni's abstract-domain
  parametrisation: build the coarse domain, remesh uniformly through it, build an atlased mesh,
  and transfer it between aligned meshes
- **filter_mls** (8) — APSS and RIMLS moving least squares surfaces: projection, marching-cubes
  iso-surface extraction, curvature colouring, radius-from-density, small-component selection
- **filter_func** (18, complete) — expression-driven filters: conditional vertex and face
  selection, per-vertex/face quality, colour, normal and geometric functions, per-vertex and
  per-wedge texture coordinates, named custom scalar and point attributes, a grid generator, an
  implicit-surface extractor and user-defined refinement
- **filter_colorproc** (28, complete) — colour fill, invert, desaturate, levels,
  brightness/contrast/gamma, thresholding, colourisation, white balance, noise, histogram
  equalisation, Perlin colouring, per-layer scattering, quality-to-colour ramps, quality clamping
  and saturation, colour and quality transfer between vertices, faces, the mesh and its texture,
  random and per-component labelling, Laplacian colour smoothing, triangle-shape metrics,
  discrete curvatures
- **filter_create** (13) — the platonic solids, box, sphere, sphere cap, cone/cylinder, torus,
  annulus, spherical point clouds, and a plane fitted to a selection

**PLY, STL, OBJ and OFF** are read and written, and **PNG** for textures, including OBJ's negative indices and all four
of its face-corner forms, and OFF's `C`/`N` header prefixes.

All **282 filters are registered from day one** — the names are extracted from the C++ sources
rather than transcribed. The 17 without an implementation yet throw `MLNotImplementedException`
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
- **Smart triangle pairing is augmenting-path matching in disguise.** Greedy pairing always
  strands some triangles; a sphere of 320 faces keeps 20 of them. The way out is that a quad's
  diagonal can be flipped, which moves the "unpaired" state one quad along — the near half marries
  the lonely triangle and the far half becomes lonely in its turn — so two lonely triangles can
  walk toward each other until they meet. That is an augmenting path, and it inherits the same
  limit: an odd face count admits no perfect matching whatever the connectivity, which is why one
  border face is split for parity first. Every closed test mesh comes out exactly half quads and
  no triangles.
- **A polygon is the faux-edge representation, not a second mesh type.** Upstream keeps a
  separate `PolyMesh` and converts to and from it for Catmull-Clark and Doo-Sabin. Here a polygon
  is already what the library has — triangles joined by hidden edges — so the bridge is two
  functions: one walks a group's unhidden boundary into an ordered ring, the other fans a ring
  back into triangles with the interior diagonals hidden. Both schemes then read polygons, compute
  polygons, and write polygons; neither touches a triangle.
- **LS3 Loop uses the normals; plain Loop cannot.** Both split the same edges with the same
  weights. The difference is what the weights are applied to: Loop averages the neighbouring
  *positions*, LS3 fits an algebraic sphere to the neighbouring positions *and normals* and takes
  the point on it. On a sphere with true normals, one pass of LS3 lands within 1e-5 of the real
  surface where Loop is 1.7% inside it. It does not improve monotonically with iterations, and
  that is not a defect: after the first pass the normals come from the new geometry rather than
  from the original surface, so what it converges to drifts a little as the mesh does.
- **A quad is two triangles and a hidden edge.** Making a mesh quad-dominant therefore moves
  nothing and adds nothing — it only decides which edges to hide, so `2·quads + triangles` always
  equals the face count and the operation is exactly reversible. The refining variant does add
  geometry, and its count is forced: splitting every triangle at its centroid makes each original
  *edge* the diagonal of a quad, so a closed mesh yields exactly as many quads as it had edges.
- **Edges are a real container, not degenerate triangles.** A polyline has vertices and edges and
  no faces, and three filters produce exactly that, so `CMeshO` grew an edge domain alongside the
  vertex and face ones. It costs nothing on an ordinary mesh — the arrays are empty — and it
  travels through the same generic grow/reset/compact passes everything else does. The catch,
  found immediately by a test, is that *anything which renumbers vertices must renumber the edges
  pointing at them*: both `compactVertexVector` and `removeDuplicateVertex` had to learn about
  them, and welding a polyline is precisely the operation that would otherwise break.
- **`Align to Principal Axis` orders its axes oppositely in its two modes.** Both take the
  eigenvalues ascending, but a covariance eigenvalue grows with the spread along its axis while a
  moment of inertia shrinks with it — so the point mode puts the longest axis on Z and the
  inertia mode puts it on X. Upstream has the same asymmetry, because it too simply sorts
  whichever matrix it was handed. Reproduced rather than reconciled, since scripts depend on the
  actual behaviour; both directions are pinned by tests.
- **A mirroring transform re-winds the faces.** A flip has determinant -1, so applying it to the
  coordinates alone would leave every face wound backwards and the solid inside out. The
  transform compensates, which is why a flip preserves the signed volume instead of negating it.
- **Qhull is replaced, not bound.** MeshLab links Qhull for a convex hull and a Delaunay
  triangulation; both are written here instead — quickhull and Bowyer-Watson, in
  `vcg/space/`. Neither uses exact predicates: degeneracy is handled with a scale-relative
  epsilon, which is enough for the point clouds these filters see and keeps the code readable.
  The cost is that a nearly-cospherical patch may come back triangulated differently than Qhull
  would do it. The *set* of hull vertices is the same either way, and that is what all four
  callers actually use.
- **An alpha shape cannot quite reach the convex hull through the slider.** It sweeps toward the
  hull as alpha grows, but the parameter is capped at the bounding-box diagonal while the sliver
  tetrahedra lying against the hull have circumradii an order of magnitude larger. That range is
  upstream's, so the behaviour matches; the exact statement — that the boundary of the *whole*
  tetrahedralization is the convex hull, face for face — is tested against the tetrahedralization
  itself instead.
- **Ball pivoting interpolates; Screened Poisson approximates.** Both reconstruct a surface from
  a point cloud, and which one is right depends on how much the points are trusted. Ball pivoting
  uses the input points and only those, so a clean scan comes back exactly as measured and a
  noisy one comes back noisy. Poisson fits an implicit function and re-samples it, which smooths
  the noise away but moves every point and invents new ones. The ball's radius is the whole
  parameter: gaps wider than it stay holes, detail finer than it is bridged over.
- **Self-intersection reports crossings, not coplanar overlaps.** Every edge of each triangle is
  tested against the other, which finds any pair that genuinely passes through — but two coplanar
  triangles that overlap without an edge piercing the other's interior are not reported. That
  matches upstream, and it is the right call for a mesh: two coplanar faces are far more often a
  legitimate shared edge than a defect.
- **Two filters ask for their own channel rather than declaring it a requirement.** Listing
  `MM_WEDGTEXCOORD` has the framework allocate zeroed coordinates for a mesh that has none, and
  the filter then quietly answers a question about a parametrization that does not exist —
  "no seams", or a colour sampled at (0, 0). `Select Vertex Texture Seams` and
  `Transfer Color: Texture to Vertex` check for the channel themselves so they can say which of
  the two situations they found.
- **The generated filter table swallowed the code after each switch.** The extractor gave the
  last `case` of a dispatch everything up to the end of the function, which in several plugins
  meant an `Error on Foo::filterName()` fallback string — so `Generate Scalar Harmonic Field` was
  registered under a name a script would never type, and thirty-odd descriptions ended in
  `Unknown Filter`. Cutting each arm at the first statement boundary fixed all of them at once.
  Two of the names had already been implemented in their mangled form; a generated table is only
  as trustworthy as its extractor, which is the argument for reviewing its diff.
- **The crease cut allocates one vertex per wedge, not one per crease crossing.** Upstream
  allocates on the crossing itself, so a closed fan claims one more vertex than it writes and the
  spare is never referenced — eight orphans on a cube. Deferring the allocation to the moment a
  wedge is actually written costs nothing and leaves none.
- **Quality saturation and the harmonic field both need every component.** See the note on
  saturation below; the harmonic solve simply refuses a mesh in more than one piece, since a
  second component has no path to either boundary condition and its values would be arbitrary.
- **Booleans are volumetric, and never fail.** Upstream uses exact predicates on the triangles;
  this builds a signed distance field per operand and combines them — `min` for union, `max` for
  intersection, `max(A, -B)` for difference. Two self-intersecting or non-manifold inputs give a
  watertight result all the same, where an exact method would refuse. What it gives up is the
  sharp seam: a crease finer than a grid cell is rounded off, and the resolution is the dial.
- **Marching tetrahedra orients against the field, not by volume.** Orienting a component by its
  signed volume is wrong the moment there is more than one: a hollow shell's inner surface
  encloses a cavity, and turning it "outwards" makes the two surfaces stop describing one solid.
  Reading the field's gradient is unambiguous however many pieces come out.
- **Alpha wrap is a morphological closing, not a Delaunay refinement.** CGAL builds it by
  refining a triangulation; here the input is dilated by `alpha + offset` and eroded by `alpha`
  on a distance-field grid, which is what "rolling a ball" means. The distance is *unsigned*, so
  the dilation of a closed surface is a hollow shell until the interior is flooded and filled —
  without that step the wrap comes back with an inner surface as well as an outer one. The grid
  resolution sets the accuracy, where CGAL guarantees a bound.
- **A camera's pose is a view point, not a translation.** `Shot.Extrinsics.tra` is where the
  camera *is*, following VCGLib, not the `-R·c` a world-to-camera matrix would store. Moving a
  camera is therefore a transform of that point plus a rotation of its axes — and the rotation
  composes on the *right*, since `rot` maps world to camera. Multiplying on the left instead
  leaves cameras somewhere plausible and pointing the wrong way.
- **Ray casting replaces the GPU.** `filter_ao` and `filter_sdfgpu` are named for the depth
  buffers MeshLab renders; there is no GPU here, so they trace rays against a BVH instead. The
  quantities are the same and if anything more accurate — a depth buffer quantises directions to
  its own resolution — but the peeling iteration count and its tolerance are gone, because those
  parameters exist to work around a rasteriser.
- **Solids are built through a field, not from geometry.** The Voronoi scaffolding and the solid
  wireframe both define a distance function and run marching tetrahedra on it, rather than
  emitting cylinders and spheres. Constructing them directly is faster and leaves a
  self-intersecting mess at every joint where several struts meet; the field has the joins built
  in.
- **One sparse solver, used four ways.** `math/sparse.ts` is a Jacobi-preconditioned conjugate
  gradient in a page: the heat-method geodesic, least-squares conformal maps and any Poisson
  problem on a mesh all reduce to it, and none of them needs a factorisation. Its `pin` moves a
  Dirichlet condition to the right-hand side *symmetrically*, because simply zeroing the row
  would break the symmetry CG depends on.
- **The isoparametrisation's inverse map is a projection.** Upstream keeps the fine triangulation
  restricted to each domain face and inverts the barycentric map through it exactly — most of
  `iso_parametrization.h`'s two thousand lines. Here a domain sample is evaluated on the coarse
  domain and projected onto the nearest point of the surface. The result lands on the surface and
  is uniform in domain space, which is what the remeshing needs, but it is not bit-comparable
  with MeshLab. Likewise the atlas assigns a straddling face to the slot holding most of it and
  clamps the strays, where upstream cuts the face; the count of straddling faces is reported.
- **The abstract domain never loses a vertex.** `parametrization/abstract_domain.ts` simplifies a
  mesh while every original vertex stays pinned inside some face of the coarse domain by a
  barycentric coordinate. A collapse flattens the affected star, records where each pin sits,
  collapses, flattens the smaller star *reusing the first one's boundary layout* so the two share
  a coordinate system, and re-pins. A collapse that would strand a pin, or that folds the star
  when flattened, is undone rather than approximated.
- **The developability energy and its gradient are two different functions, as upstream.**
  MeshLab ships `filter_developability` with `FILTERDEVELOPABILITY_AVOID_BRANCHING` defined, which
  makes the reported energy a *maximum* over pairs of face normals while the gradient it descends
  stays that of the *sum* over pairs. That is reproduced rather than repaired: it is what every
  MeshLab user's copy does, and the line search accepting a step by the max while moving along the
  sum is part of the filter's actual behaviour. Its sliver-removal pass is one genuine repair —
  upstream reads `faceAngles[-1]` there, which is undefined behaviour, so the corner it plainly
  meant is used instead.
- **Textured decimation guards the seams instead of extending the quadric.** MeshLab's
  `Quadric Edge Collapse Decimation (with texture)` carries a 5-dimensional quadric, weighting the
  UV coordinates against the geometry through its `Extratcoordw` parameter. Here the collapse keeps
  the ordinary 3D quadric and instead *refuses* any collapse touching a UV seam, interpolating the
  surviving vertex's coordinates along the collapsed edge. The result preserves the
  parametrisation exactly rather than approximately, at the cost of decimating a little less near
  a seam; `Extratcoordw` is accepted and warned about rather than silently ignored.
- **The atlas defragmenter is reimplemented, not ported, and diverges in four places.**
  MeshLab vendors this as a 6,935-line sub-library; the port is in
  `vcg/complex/parametrization/` (`chart_graph`, `seams`, `shell`, `arap2d`, `matching2`,
  `defragment`, `packing`) and is documented stage by stage in
  [docs/texture-defragmentation-port.md](docs/texture-defragmentation-port.md). The four
  differences a user can observe: a merge is computed on a copy and committed only if it passes,
  rather than performed destructively and unwound on rejection — so a refused merge provably
  leaves nothing behind; the neighbourhood a merge may relax is a fixed number of face rings
  rather than a distance threshold; the packer rasterises each chart's triangles instead of an
  extracted outline, and searches area-ordered with four rotations instead of permuting the chart
  order, which costs some atlas density (reported as `atlas_occupancy`) and buys determinism; and
  the new texture is rasterised in software rather than through an OpenGL context. `timelimit`
  is refused rather than honoured — a wall-clock bound would mean the same input did not give the
  same atlas twice — with a deterministic `maxMoves` offered in its place.
- **A Voronoi chart that will not flatten still gets real coordinates.** `Parametrization: Voronoi
  Atlas` partitions the mesh geodesically and flattens each region as a disk, but a region is not
  guaranteed to *be* a disk. Rather than leave such a region's coordinates untouched — which looks
  like a working parametrisation while collapsing the whole chart onto one texel — its faces fall
  back to a per-triangle layout inside the region's own cell, and the filter reports how many
  regions and faces took that path. Upstream's `overlapFlag`, which duplicates a border ring for
  bleeding, is not implemented and throws.
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
- **Custom attributes are a list beside the channel table, not a map on the side.** A filter can
  name a new per-vertex or per-face attribute at run time, so it cannot be a row in the static
  channel table. It still has to grow, reset and compact exactly as a built-in channel does — an
  attribute that stopped travelling with its vertex through a compaction would be far worse than
  one that never existed. So the three generic passes in `components.ts` walk the attribute list
  right after the table, and nothing else in the codebase knows the difference.
- **Histogram equalisation reproduces upstream's off-by-one baseline.** VCGLib subtracts `cdf[0]`
  — how many vertices are *exactly* zero — where the textbook subtracts the smallest cdf value
  actually present. The visible consequences are that the darkest colour does not come out black
  and that a uniformly coloured mesh equalises to white. Both are wrong in the abstract and right
  for MeshLab, and matching them keeps a `.mlx` doing what it did before.
- **Quality saturation restarts at every component.** Upstream seeds its traversal at vertex 0
  alone, so on a mesh with islands every component but the first comes back unsaturated. Since
  the filter's whole contract is a bound on the gradient *everywhere*, that is a bug rather than
  a convention, and this reseeds at each unvisited vertex.
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
