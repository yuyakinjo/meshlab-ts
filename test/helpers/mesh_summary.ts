/**
 * A digest of a mesh that a *real* PyMeshLab can also compute.
 *
 * This is the meeting point of the differential tests: `_regen/regen.py` runs
 * a filter in genuine PyMeshLab and writes this same structure to a golden
 * file; `test/golden/golden.test.ts` runs the same filter here and compares.
 * The two implementations must therefore agree not only on the mathematics but
 * on the *formatting* — the geometry hash is a hash of decimal strings, and a
 * hash comparison across two languages is only meaningful if both render
 * doubles identically.
 *
 * Hence fixed nine-decimal formatting (`toFixed(9)` / Python's `:.9f`), not
 * `%g`: both languages implement correctly-rounded decimal conversion of IEEE
 * doubles, so fixed-point output is bit-for-bit reproducible across them,
 * while the exponent-switching rules of `%g` differ. The one disagreement
 * left is negative zero — Python renders `-0.000000000`, JavaScript does not —
 * so both sides normalise it away.
 *
 * The digest is deliberately independent of vertex and face *order*: keys are
 * sorted before hashing, and a face's corners are sorted within the face. Two
 * meshes with the same geometry reached through different allocation orders
 * hash the same, which is what lets an independently-implemented filter be
 * compared at all. The price is that orientation is invisible to the hash —
 * which is why the summary also carries the *signed* volume.
 */
import { createHash } from "node:crypto";
import {
	connectedComponents,
	countHoles,
	countNonManifoldEdgeFF,
	countNonManifoldVertexFF,
} from "../../src/vcg/complex/clean.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { faceFace } from "../../src/vcg/complex/update/topology.ts";

export interface MeshSummary {
	vn: number;
	fn: number;
	en: number;
	boundaryEdges: number;
	components: number;
	/** -1 when the mesh is not two-manifold, as PyMeshLab reports it. */
	boundaryLoops: number;
	/** -1 when the mesh is not two-manifold, as PyMeshLab reports it. */
	genus: number;
	nonManifoldEdges: number;
	nonManifoldVertices: number;
	area: number;
	/** Signed, so an inverted mesh is distinguishable from its original. */
	volume: number;
	geometryHash: string;
}

/** One coordinate, formatted exactly as `regen.py` formats it. */
function coordinateKey(value: number): string {
	// Through float32 first: MeshLab's CMeshO stores coordinates as floats
	// (Scalarm), so the strongest agreement any filter can achieve is equality
	// at float32 granularity. Hashing our doubles unrounded would report a
	// mismatch that is really just MeshLab's own storage.
	const text = Math.fround(value).toFixed(9);
	// Python writes "-0.000000000" for a negative zero; JavaScript never does.
	// Values that merely round to zero keep their sign in both languages, so
	// only the all-zeros string needs the fix — on both sides.
	return text === "-0.000000000" ? "0.000000000" : text;
}

export function summarizeMesh(m: CMeshO): MeshSummary {
	// --- the order-independent geometry digest -----------------------------
	const liveVertexKey = new Map<number, string>();
	const vertexKeys: string[] = [];
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		const key = `${coordinateKey(m.vx(v))},${coordinateKey(m.vy(v))},${coordinateKey(m.vz(v))}`;
		liveVertexKey.set(v, key);
		vertexKeys.push(key);
	}
	vertexKeys.sort();

	const faceKeys: string[] = [];
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		const corners = [0, 1, 2].map((k) => liveVertexKey.get(m.fv(f, k)) as string);
		corners.sort();
		faceKeys.push(corners.join("|"));
	}
	faceKeys.sort();

	const digest = createHash("sha256")
		.update(
			`V${vertexKeys.length}:${vertexKeys.join(";")}\nF${faceKeys.length}:${faceKeys.join(";")}`,
		)
		.digest("hex");

	// --- topology -----------------------------------------------------------
	// Unique undirected edges over the live faces, and how many faces each has.
	const incidence = new Map<number, number>();
	const stride = m.vertSize + 1;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) {
			const a = m.fv(f, k);
			const b = m.fv(f, (k + 1) % 3);
			const key = a < b ? a * stride + b : b * stride + a;
			incidence.set(key, (incidence.get(key) ?? 0) + 1);
		}
	}
	const en = incidence.size;
	let boundaryEdges = 0;
	for (const count of incidence.values()) if (count === 1) boundaryEdges++;

	faceFace(m);
	const components = connectedComponents(m).length;
	const nonManifoldEdges = countNonManifoldEdgeFF(m);
	const nonManifoldVertices = countNonManifoldVertexFF(m);
	const manifold = nonManifoldEdges === 0 && nonManifoldVertices === 0;

	// PyMeshLab reports -1 for both when the mesh is not two-manifold, and the
	// comparison must be like for like.
	const boundaryLoops = manifold ? countHoles(m) : -1;
	// χ = V − E + F, then 2c = χ + 2g + b for the genus total. V counts only
	// *referenced* vertices: MeshLab reports unreferenced ones in vn but keeps
	// them out of the genus, and a crease cut legitimately leaves orphans
	// behind — including them would assign a sphere of six cut faces genus -4.
	const referenced = new Set<number>();
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) referenced.add(m.fv(f, k));
	}
	const genus = manifold
		? (2 * components - boundaryLoops - (referenced.size - en + m.fn)) / 2
		: -1;

	// --- geometry -------------------------------------------------------
	let area = 0;
	let volume = 0;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		const p = [0, 1, 2].map((k) => {
			const v = m.fv(f, k);
			return [m.vx(v), m.vy(v), m.vz(v)];
		});
		const e1 = [0, 1, 2].map((i) => p[1][i] - p[0][i]);
		const e2 = [0, 1, 2].map((i) => p[2][i] - p[0][i]);
		area +=
			Math.hypot(
				e1[1] * e2[2] - e1[2] * e2[1],
				e1[2] * e2[0] - e1[0] * e2[2],
				e1[0] * e2[1] - e1[1] * e2[0],
			) / 2;
		// Signed tetrahedron volumes against the origin: the divergence theorem,
		// which is also what MeshLab's mesh_volume computes.
		volume +=
			(p[0][0] * (p[1][1] * p[2][2] - p[2][1] * p[1][2]) -
				p[1][0] * (p[0][1] * p[2][2] - p[2][1] * p[0][2]) +
				p[2][0] * (p[0][1] * p[1][2] - p[1][1] * p[0][2])) /
			6;
	}

	return {
		vn: m.vn,
		fn: m.fn,
		en,
		boundaryEdges,
		components,
		boundaryLoops,
		genus,
		nonManifoldEdges,
		nonManifoldVertices,
		area,
		volume,
		geometryHash: digest,
	};
}
