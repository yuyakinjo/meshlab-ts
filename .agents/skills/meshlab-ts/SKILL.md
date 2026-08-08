---
name: meshlab-ts
description: Process, repair and measure 3D meshes with meshlab-ts, a TypeScript port of MeshLab that runs on Bun with no Python. Use when the task involves fixing a broken or non-manifold STL, making a mesh watertight/printable, converting between STL/PLY/OBJ/OFF, reconstructing a surface from a point cloud (Screened Poisson), decimating/subdividing/smoothing a mesh, measuring volume, area, genus or dimensions, or running a MeshLab .mlx filter script without installing MeshLab or PyMeshLab.
---

# meshlab-ts: MeshLab filters from the command line

269 of MeshLab's 282 filters, addressable by name, on Bun. Anything a MeshLab
or PyMeshLab recipe does — repair, remesh, reconstruct, measure — runs here
without a Python runtime. Behaviour is differentially tested against real
PyMeshLab; the differences that exist are documented in the package README.

## Run it

```bash
bunx meshlab-ts@latest list --implemented     # no install needed
```

In a project, prefer a pinned dependency (`bun add meshlab-ts`) and plain
`bunx meshlab-ts`. Requires Bun ≥ 1.3.

## Always check a filter before applying it

Parameter names are case-sensitive and an unknown name **throws** rather than
being ignored — so read the schema first, then apply:

```bash
bunx meshlab-ts list --class Cleaning         # find candidates (or: list holes)
bunx meshlab-ts info "Close Holes"            # parameters, defaults, tooltips
bunx meshlab-ts apply "Close Holes" in.ply -o out.ply --param MaxHoleSize=100
```

Filters answer to both naming schemes: `"Close Holes"` (MeshLab) and
`meshing_close_holes` (PyMeshLab) are the same filter. 13 filters are
registered but unimplemented (GL/network/exotic-format ones) and throw
`MLNotImplementedException` — that error means "pick another route", not "retry".

## Command reference

```
meshlab-ts list [--class <name>] [--implemented|--todo] [--json] [pattern]
meshlab-ts info <filter>
meshlab-ts apply <filter> [input] -o <output> [--param key=value ...] [--save <channels>]
meshlab-ts script <script.mlx|.json> <input> -o <output> [--save <channels>]
meshlab-ts formats
```

Parameter value conventions (`--param key=value`):

- **Booleans**: `true` / `false`
- **Percentage-typed parameters** (shown as `RichPercentage` by `info`): pass an
  **absolute length** in mesh units, not a percent
- **Enums**: the option's name (quoted) or its index — `--param rotAxis="Z axis"`
- **Creation filters** (Sphere, Torus, Box…) take no input file

Filters that measure rather than modify print their results to stdout
(`mesh_volume`, `surface_area`, …); the output mesh is written unchanged.

## Recipe: repair a broken STL into a printable solid

This order is the one the library's end-to-end test uses; each step exists for
a reason and the order matters (orientability needs manifold edges; a hole
boundary is only well defined on a manifold surface):

```bash
cat > repair.json <<'EOF'
{ "filters": [
  { "filterName": "Remove Zero Area Faces", "params": {} },
  { "filterName": "Remove Duplicate Faces", "params": {} },
  { "filterName": "Remove Isolated pieces (wrt Diameter)", "params": { "MinComponentDiag": 0.5 } },
  { "filterName": "Remove Unreferenced Vertices", "params": {} },
  { "filterName": "Repair non Manifold Edges", "params": {} },
  { "filterName": "Re-Orient all faces coherently", "params": {} },
  { "filterName": "Close Holes", "params": { "MaxHoleSize": 100 } },
  { "filterName": "Invert Faces Orientation", "params": { "forceFlip": false } },
  { "filterName": "Select None", "params": {} }
] }
EOF
bunx meshlab-ts script repair.json broken.stl -o fixed.ply
```

`.mlx` scripts exported from the MeshLab GUI run as they are:
`bunx meshlab-ts script exported.mlx in.stl -o out.ply`.

Three things that bite in practice, all faithful to MeshLab:

- **STL is welded on load** (duplicate vertices unified, as MeshLab does).
  The raw soup is only reachable through the library API:
  `kernel.loadMesh(doc, path, { unify_vertices: false })` — the CLI's
  `--param` goes to the filter, not the loader.
- **`Close Holes` leaves its new faces selected**, and several filters default
  "operate on selection" to true when anything is selected — always follow
  with `Select None`.
- **Writing STL un-welds again** (it is a soup format). Keep intermediate
  results in PLY; convert to STL only as the final step.

## Recipe: point cloud → watertight solid

```bash
bunx meshlab-ts apply "Compute normals for point sets" cloud.ply -o oriented.ply --save normals
bunx meshlab-ts apply "Surface Reconstruction: Screened Poisson" oriented.ply -o solid.ply
```

- `--save normals` is mandatory on the first step: a point cloud written
  without normals is unusable for reconstruction downstream.
- Poisson needs a normal on **every** point; if it refuses, add
  `--param preClean=true`.
- For noisy scans, raise `--param samplesPerNode=4` (more smoothing). The
  reconstruction stores sample density in per-vertex quality, so
  low-confidence regions can be trimmed afterwards with
  `"Select by Vertex Quality"` followed by `"Delete Selected Faces and
  Vertices"`.

## Recipe: inspect a mesh

```bash
bunx meshlab-ts apply "Compute Geometric Measures" in.ply -o /tmp/ignore.ply    # volume, area, barycenter
bunx meshlab-ts apply "Compute Topological Measures" in.ply -o /tmp/ignore.ply  # genus, holes, manifoldness
```

(`-o` is required and chooses the writer by extension — `/dev/null` is
rejected for having none. The written mesh is just the unchanged input.)

Read `mesh_volume` from stdout — it is absent when the mesh is not closed,
which is itself the answer to "is this watertight".

## Programmatic use (TypeScript on Bun)

For multi-step work inside a project, the library API beats shelling out:

```ts
import { MeshLabKernel, MeshDocument, FilterScript } from "meshlab-ts";

const kernel = MeshLabKernel.default();
const doc = new MeshDocument();
kernel.loadMesh(doc, "broken.stl");
kernel.applyFilter(doc, "Close Holes", { MaxHoleSize: 100 });
const out = kernel.applyFilter(doc, "Compute Geometric Measures");
console.log(out.mesh_volume);
kernel.saveMesh(doc, "fixed.ply");
```

Filter outputs come back as a plain object (`closed_holes`, `mesh_volume`, …)
— the same keys PyMeshLab returns. Exceptions carry the failing step's name;
never swallow them, a half-run pipeline reporting success is the worst outcome.
