"""
Generates golden fixtures from a real PyMeshLab.

The only Python in this repository, and it is never executed by CI or by
`bun test`. It runs when a person deliberately asks — via Docker or a local
venv, see `scripts/regen-golden.ts` — and writes one JSON per case into
`test/fixtures/golden/`. Those files are checked in, and
`test/golden/golden.test.ts` compares this library against them on every
ordinary test run, no Python required.

It writes the same `MeshSummary` shape that `test/helpers/mesh_summary.ts`
produces. The two must agree on *formatting*, not just mathematics: the
geometry hash is a hash of decimal strings, so both sides render doubles as
fixed nine-decimal strings (`:.9f` / `toFixed(9)`), which is bit-reproducible
across the two languages where `%g`'s exponent switching is not. Negative
zero is the one case they render differently, and both sides normalise it.

Parameters in CASES use the C++ RichParameterList names, exactly as the
TypeScript side uses them. PyMeshLab's names are those same names lowercased
(plus a trailing underscore when the result is a Python keyword: `lambda_`),
and that translation happens here, so each case is written once.

Value markers:
    {"abs": x}   an absolute length  -> pymeshlab.PureValue(x)
    {"percent": x} percent of bbox diagonal -> pymeshlab.PercentageValue(x)
    {"enum": s}  an enum by option name -> the string itself
The TypeScript side reads the same markers: RichPercentage takes the raw
absolute number, and RichEnum accepts the option name directly.
"""

import hashlib
import json
import keyword
import math
import os
import pathlib
import sys

import pymeshlab

FIXTURES = pathlib.Path(os.environ.get("FIXTURES_DIR", "/fixtures"))
MESHES = FIXTURES / "meshes"
GOLDEN = FIXTURES / "golden"
VERSION = os.environ.get("PYMESHLAB_VERSION", "unknown")

# (label, mesh file, pymeshlab filter name, params in C++ names, compare class)
#
# The compare class states how much agreement is *expected*, and the test
# holds each case to exactly that:
#   exact      — identical geometry, down to the digest of the coordinates.
#                Deterministic filters whose arithmetic has one possible order.
#   equivalent — identical topology (every integer), geometry within tight
#                tolerances. Deterministic filters whose floating-point
#                summation order legitimately differs between implementations.
#   loose      — same shape of result: counts within a few percent, area and
#                volume within a couple of percent. Filters built on heaps or
#                heuristics, where which element wins a tie is not specified.
CASES = [
    # --- pure topology, one right answer ---------------------------------
    ("cube-binary", "cube_binary.stl", "meshing_remove_duplicate_vertices", {}, "exact"),
    ("cube-ascii", "cube_ascii.stl", "meshing_remove_duplicate_vertices", {}, "exact"),
    ("sphere", "sphere.ply", "meshing_remove_duplicate_faces", {}, "exact"),
    ("sphere", "sphere.ply", "meshing_remove_unreferenced_vertices", {}, "exact"),
    ("torus", "torus.ply", "meshing_remove_null_faces", {}, "exact"),
    ("sphere", "sphere.ply", "meshing_re_orient_faces_coherently", {}, "exact"),
    ("torus", "torus.ply", "meshing_invert_face_orientation", {"forceFlip": True}, "exact"),
    ("holed-cube", "cube_with_holes.stl", "meshing_close_holes", {"MaxHoleSize": 30}, "exact"),
    (
        "islands",
        "islands.ply",
        "meshing_remove_connected_component_by_face_number",
        {"MinComponentSize": 25},
        "exact",
    ),
    ("cube-weld", "cube_binary.stl", "meshing_merge_close_vertices", {"Threshold": {"abs": 1e-4}}, "exact"),
    # --- exact geometry through deterministic arithmetic ------------------
    (
        "torus-translate",
        "torus.ply",
        "compute_matrix_from_translation",
        {"axisX": 1.5, "axisY": -2.0, "axisZ": 0.25},
        "exact",
    ),
    (
        "sphere-rot90",
        "sphere.ply",
        "compute_matrix_from_rotation",
        {"rotAxis": {"enum": "Z axis"}, "angle": 90.0},
        "exact",
    ),
    (
        "torus-scale",
        "torus.ply",
        "compute_matrix_from_scaling_or_normalization",
        {"axisX": 2.0, "uniformFlag": True},
        "exact",
    ),
    (
        "torus-func",
        "torus.ply",
        "compute_coord_by_function",
        {"x": "x+1", "y": "y*2", "z": "z-0.5"},
        "exact",
    ),
    (
        "sphere-midpoint",
        "sphere.ply",
        "meshing_surface_subdivision_midpoint",
        {"Iterations": 1, "Threshold": {"abs": 0.0}},
        "exact",
    ),
    (
        "tetra-midpoint",
        "tetra_binary.ply",
        "meshing_surface_subdivision_midpoint",
        {"Iterations": 2, "Threshold": {"abs": 0.0}},
        "exact",
    ),
    # --- same mathematics, implementation-defined summation order ---------
    (
        "sphere-loop",
        "sphere.ply",
        "meshing_surface_subdivision_loop",
        {"Iterations": 1, "Threshold": {"abs": 0.0}, "LoopWeight": {"enum": "Loop"}},
        "equivalent",
    ),
    (
        "sphere3-laplacian",
        "sphere3.ply",
        "apply_coord_laplacian_smoothing",
        {"stepSmoothNum": 3, "cotangentWeight": False, "Boundary": True},
        "equivalent",
    ),
    (
        "sphere3-laplacian-cot",
        "sphere3.ply",
        "apply_coord_laplacian_smoothing",
        {"stepSmoothNum": 2, "cotangentWeight": True, "Boundary": True},
        "equivalent",
    ),
    (
        "sphere3-taubin",
        "sphere3.ply",
        "apply_coord_taubin_smoothing",
        {"lambda": 0.5, "mu": -0.53, "stepSmoothNum": 10},
        "equivalent",
    ),
    ("torus-hull", "torus.ply", "generate_convex_hull", {}, "equivalent"),
    # --- heaps and heuristics: ties are implementation-defined ------------
    (
        "sphere3-qecd",
        "sphere3.ply",
        "meshing_decimation_quadric_edge_collapse",
        {"TargetFaceNum": 320},
        "loose",
    ),
    (
        "sphere3-clustering",
        "sphere3.ply",
        "meshing_decimation_clustering",
        {"Threshold": {"abs": 0.3}},
        "loose",
    ),
    (
        "fan-repair",
        "fan.ply",
        "meshing_repair_non_manifold_edges",
        {"method": {"enum": "Remove Faces"}},
        "loose",
    ),
]


def to_pymeshlab_params(params: dict) -> dict:
    """C++ names and value markers -> what pymeshlab's kwargs want."""
    out = {}
    for name, value in params.items():
        key = name.lower()
        if keyword.iskeyword(key):
            key += "_"
        if isinstance(value, dict):
            if "abs" in value:
                out[key] = pymeshlab.PureValue(value["abs"])
            elif "percent" in value:
                out[key] = pymeshlab.PercentageValue(value["percent"])
            elif "enum" in value:
                out[key] = value["enum"]
            else:
                raise ValueError(f"unknown marker in {name}: {value}")
        else:
            out[key] = value
    return out


def summarize(ms: "pymeshlab.MeshSet") -> dict:
    """The same digest `test/helpers/mesh_summary.ts` computes."""
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    def key(v) -> str:
        parts = []
        for c in v:
            # Through float32 explicitly: MeshLab stores coordinates as floats,
            # so vertex_matrix() hands back float32 values promoted to float64.
            # The TypeScript side rounds its doubles the same way (Math.fround)
            # before formatting, and the digests only compare because both do.
            import numpy as np
            text = f"{float(np.float32(c)):.9f}"
            parts.append("0.000000000" if text == "-0.000000000" else text)
        return ",".join(parts)

    vert_keys = sorted(key(v) for v in verts)
    face_keys = sorted("|".join(sorted(key(verts[i]) for i in f)) for f in faces)
    digest = hashlib.sha256(
        (
            "V{}:{}\nF{}:{}".format(
                len(vert_keys), ";".join(vert_keys), len(face_keys), ";".join(face_keys)
            )
        ).encode()
    ).hexdigest()

    topo = ms.get_topological_measures()
    geom = ms.get_geometric_measures()

    def finite(x):
        x = float(x)
        return x if math.isfinite(x) else None

    return {
        "vn": int(m.vertex_number()),
        "fn": int(m.face_number()),
        "en": int(topo.get("edges_number", -1)),
        "boundaryEdges": int(topo.get("boundary_edges", -1)),
        "components": int(topo.get("connected_components_number", -1)),
        "boundaryLoops": int(topo.get("number_holes", -1)),
        "genus": int(topo.get("genus", -1)),
        "nonManifoldEdges": int(topo.get("non_two_manifold_edges", -1)),
        "nonManifoldVertices": int(topo.get("non_two_manifold_vertices", -1)),
        # None rather than NaN: Python's json module happily writes NaN, and
        # strict JSON parsers (Bun's included) refuse to read it back.
        "area": finite(geom.get("surface_area", float("nan"))),
        "volume": finite(geom.get("mesh_volume", float("nan"))),
        "geometryHash": digest,
    }


def main() -> int:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    written = 0
    failed = 0
    for label, mesh_file, filter_name, params, compare in CASES:
        path = MESHES / mesh_file
        if not path.exists():
            print(f"skip {label}: no {path}", file=sys.stderr)
            continue

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(path))
        try:
            ms.apply_filter(filter_name, **to_pymeshlab_params(params))
        except Exception as error:  # noqa: BLE001 - report and continue
            print(f"FAIL {filter_name}/{label}: {error}", file=sys.stderr)
            failed += 1
            continue

        out_dir = GOLDEN / filter_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{label}.json"
        out_path.write_text(
            json.dumps(
                {
                    "input": mesh_file,
                    "filter": filter_name,
                    "params": params,
                    "compare": compare,
                    "meshlabVersion": VERSION,
                    "summary": summarize(ms),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"wrote {out_path.relative_to(FIXTURES)}")
        written += 1

    print(f"{written} golden fixture(s) written from pymeshlab {VERSION}, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
