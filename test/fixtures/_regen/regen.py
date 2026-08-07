"""
Generates golden fixtures from a real PyMeshLab.

The only Python in this repository, and it is never executed by CI or by
`bun test`. It runs inside the Docker container that `scripts/regen-golden.ts`
starts, and only when a person deliberately asks for it. Everything the normal
test suite checks is derived from mathematics instead, which is the point of
the project.

It writes the same `MeshSummary` shape that `test/helpers/mesh_summary.ts`
produces, so the two can be compared field by field.
"""

import hashlib
import json
import os
import pathlib
import sys

import pymeshlab

FIXTURES = pathlib.Path("/fixtures")
MESHES = FIXTURES / "meshes"
GOLDEN = FIXTURES / "golden"
VERSION = os.environ.get("PYMESHLAB_VERSION", "unknown")

# Which filters to capture, and with which parameters. Extend as filters are
# implemented on the TypeScript side; a filter with no entry here simply has no
# golden fixture, which is fine — the invariant tests still cover it.
CASES = [
    # (mesh file, pymeshlab filter name, params, case label)
    ("cube_binary.stl", "meshing_remove_duplicate_vertices", {}, "cube-soup"),
    ("cube_with_holes.stl", "meshing_remove_duplicate_vertices", {}, "holed-cube"),
    ("sphere.ply", "meshing_re_orient_faces_coherently", {}, "sphere"),
]


def summarize(ms: "pymeshlab.MeshSet") -> dict:
    """The same digest `test/helpers/mesh_summary.ts` computes."""
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    def key(v) -> str:
        return ",".join(f"{round(float(c), 9):.9g}" for c in v)

    vert_keys = sorted(key(v) for v in verts)
    face_keys = sorted("|".join(sorted(key(verts[i]) for i in f)) for f in faces)
    digest = hashlib.sha256(
        ("V{}:{}\nF{}:{}".format(
            len(vert_keys), ";".join(vert_keys), len(face_keys), ";".join(face_keys)
        )).encode()
    ).hexdigest()

    topo = ms.get_topological_measures()
    geom = ms.get_geometric_measures()
    return {
        "vn": int(m.vertex_number()),
        "fn": int(m.face_number()),
        "en": int(topo.get("edge_num", -1)),
        "boundaryEdges": int(topo.get("boundary_edge_num", -1)),
        "components": int(topo.get("connected_components_number", -1)),
        "boundaryLoops": int(topo.get("number_holes", -1)),
        "genus": int(topo.get("genus", -1)),
        "nonManifoldEdges": int(topo.get("non_two_manifold_edges", -1)),
        "nonManifoldVertices": int(topo.get("non_two_manifold_vertices", -1)),
        "area": float(geom.get("surface_area", float("nan"))),
        "volume": float(geom.get("mesh_volume", float("nan"))),
        "geometryHash": digest,
    }


def main() -> int:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    written = 0
    for mesh_file, filter_name, params, label in CASES:
        path = MESHES / mesh_file
        if not path.exists():
            print(f"skip {label}: no {path}", file=sys.stderr)
            continue

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(path))
        ms.apply_filter(filter_name, **params)

        out_dir = GOLDEN / filter_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{label}.json"
        out_path.write_text(
            json.dumps(
                {
                    "input": mesh_file,
                    "filter": filter_name,
                    "params": params,
                    "meshlabVersion": VERSION,
                    "summary": summarize(ms),
                    # Heap tie-breaks and RNG make exact geometry
                    # unreproducible for some filters; those set this false and
                    # are compared on topology and tolerances alone.
                    "exactGeometry": True,
                    "tolerances": {"area": 1e-6, "volume": 1e-6},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"wrote {out_path.relative_to(FIXTURES)}")
        written += 1

    print(f"{written} golden fixture(s) written from pymeshlab {VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
