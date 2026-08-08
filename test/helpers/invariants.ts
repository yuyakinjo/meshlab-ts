/**
 * Independent, deliberately naive computations of the properties the builders
 * declare, plus the assertions built on them.
 *
 * Nothing here calls into `src/vcg/complex/update` or `src/vcg/complex/clean`.
 * That is the point: these are a second implementation, written for clarity
 * rather than speed, so that agreement between them and the kernel is
 * evidence rather than a tautology. Hash maps and O(V·F) scans are fine —
 * the test meshes are small.
 */
import { expect } from "bun:test";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { CHANNELS, getChannel } from "../../src/vcg/complex/components.ts";
import type { MeshFacts } from "./mesh_builders.ts";

/** Live face indices, in order. */
export function liveFaces(m: CMeshO): number[] {
	const out: number[] = [];
	for (let f = 0; f < m.faceSize; f++) if (!m.isFaceD(f)) out.push(f);
	return out;
}

/** Live vertex indices, in order. */
export function liveVerts(m: CMeshO): number[] {
	const out: number[] = [];
	for (let v = 0; v < m.vertSize; v++) if (!m.isVertD(v)) out.push(v);
	return out;
}

const edgeKey = (a: number, b: number): string => (a < b ? `${a}_${b}` : `${b}_${a}`);

/** Undirected edge → the live faces incident on it. */
export function edgeFaceMap(m: CMeshO): Map<string, number[]> {
	const map = new Map<string, number[]>();
	for (const f of liveFaces(m)) {
		for (let k = 0; k < 3; k++) {
			const key = edgeKey(m.fv(f, k), m.fv(f, (k + 1) % 3));
			const hit = map.get(key);
			if (hit === undefined) map.set(key, [f]);
			else hit.push(f);
		}
	}
	return map;
}

export function countEdges(m: CMeshO): number {
	return edgeFaceMap(m).size;
}

export function countBoundaryEdges(m: CMeshO): number {
	let n = 0;
	for (const faces of edgeFaceMap(m).values()) if (faces.length === 1) n++;
	return n;
}

/**
 * True when some live face repeats a vertex index, e.g. (v, v, w).
 *
 * Such a face owns a self-edge (v, v), which has no direction, so "are these
 * two faces wound the same way along this edge" has no answer. Properties that
 * compare orientation between two implementations skip these meshes rather
 * than enshrine one arbitrary tie-break; `Remove Zero Area Faces` is what
 * removes them in practice.
 */
export function hasDegenerateFaces(m: CMeshO): boolean {
	for (const f of liveFaces(m)) {
		const a = m.fv(f, 0);
		const b = m.fv(f, 1);
		const c = m.fv(f, 2);
		if (a === b || b === c || a === c) return true;
	}
	return false;
}

export function countNonManifoldEdges(m: CMeshO): number {
	let n = 0;
	for (const faces of edgeFaceMap(m).values()) if (faces.length > 2) n++;
	return n;
}

export function isWatertight(m: CMeshO): boolean {
	if (m.fn === 0) return true;
	for (const faces of edgeFaceMap(m).values()) if (faces.length !== 2) return false;
	return true;
}

/** Connected components counted over face-to-face adjacency across shared edges. */
export function countComponents(m: CMeshO): number {
	const faces = liveFaces(m);
	if (faces.length === 0) return 0;
	const parent = new Map<number, number>(faces.map((f) => [f, f]));
	const find = (x: number): number => {
		let r = x;
		while (parent.get(r) !== r) r = parent.get(r) as number;
		let cur = x;
		while (parent.get(cur) !== r) {
			const next = parent.get(cur) as number;
			parent.set(cur, r);
			cur = next;
		}
		return r;
	};
	const union = (a: number, b: number) => {
		const ra = find(a);
		const rb = find(b);
		if (ra !== rb) parent.set(ra, rb);
	};
	for (const incident of edgeFaceMap(m).values()) {
		for (let i = 1; i < incident.length; i++) union(incident[0], incident[i]);
	}
	return new Set(faces.map(find)).size;
}

/**
 * Boundary loops, chained from the *undirected* boundary edges.
 *
 * Chaining by direction would be simpler but is wrong: on a non-orientable
 * surface such as the Möbius strip the boundary is a single loop whose
 * directed edges do not agree, so a directed walk sees a vertex with two
 * outgoing edges and gives up.
 *
 * Returns `undefined` when some boundary vertex has other than two incident
 * boundary edges — a pinch point or a non-manifold edge — where "how many
 * loops" has no answer.
 */
export function countBoundaryLoops(m: CMeshO): number | undefined {
	const incidence = edgeFaceMap(m);
	const neighbours = new Map<number, number[]>();
	const link = (a: number, b: number) => {
		const hit = neighbours.get(a);
		if (hit === undefined) neighbours.set(a, [b]);
		else hit.push(b);
	};
	for (const [key, faces] of incidence) {
		if (faces.length !== 1) continue;
		const [a, b] = key.split("_").map(Number);
		link(a, b);
		link(b, a);
	}
	if (neighbours.size === 0) return 0;
	for (const adj of neighbours.values()) if (adj.length !== 2) return undefined;

	const visited = new Set<number>();
	let loops = 0;
	for (const start of neighbours.keys()) {
		if (visited.has(start)) continue;
		loops++;
		let prev = -1;
		let cur = start;
		do {
			visited.add(cur);
			const adj = neighbours.get(cur) as number[];
			const next = adj[0] === prev ? adj[1] : adj[0];
			prev = cur;
			cur = next;
		} while (cur !== start);
	}
	return loops;
}

/**
 * True when every edge shared by exactly two faces is traversed in opposite
 * directions by them — i.e. the winding as given is already consistent.
 */
export function isCoherentlyOriented(m: CMeshO): boolean {
	const seen = new Map<string, number>();
	for (const f of liveFaces(m)) {
		for (let k = 0; k < 3; k++) {
			const a = m.fv(f, k);
			const b = m.fv(f, (k + 1) % 3);
			const key = edgeKey(a, b);
			const dir = a < b ? 1 : -1;
			const prev = seen.get(key);
			if (prev === undefined) seen.set(key, dir);
			else if (prev === dir) return false;
			else seen.set(key, 0); // matched pair
		}
	}
	return true;
}

/**
 * True when *some* consistent orientation exists, found by propagating a flip
 * bit across shared edges. Non-manifold edges are skipped, so this answers the
 * question for the manifold part only; callers should not ask it of a mesh with
 * non-manifold edges.
 */
export function isOrientable(m: CMeshO): boolean {
	const incidence = edgeFaceMap(m);
	const neighbours = new Map<number, Array<{ to: number; sameDir: boolean }>>();
	const dirOf = (f: number, a: number, b: number): boolean => {
		for (let k = 0; k < 3; k++) {
			if (m.fv(f, k) === a && m.fv(f, (k + 1) % 3) === b) return true;
		}
		return false;
	};
	for (const [key, faces] of incidence) {
		if (faces.length !== 2) continue;
		const [a, b] = key.split("_").map(Number);
		const [f0, f1] = faces;
		// Coherent when the two faces traverse the edge in opposite directions.
		const sameDir = dirOf(f0, a, b) === dirOf(f1, a, b);
		for (const [x, y] of [
			[f0, f1],
			[f1, f0],
		]) {
			const hit = neighbours.get(x);
			const entry = { to: y, sameDir };
			if (hit === undefined) neighbours.set(x, [entry]);
			else hit.push(entry);
		}
	}

	const flip = new Map<number, number>();
	for (const start of liveFaces(m)) {
		if (flip.has(start)) continue;
		flip.set(start, 0);
		const stack = [start];
		while (stack.length > 0) {
			const f = stack.pop() as number;
			const parity = flip.get(f) as number;
			for (const { to, sameDir } of neighbours.get(f) ?? []) {
				// `sameDir` means the neighbour must be flipped relative to f.
				const want = sameDir ? parity ^ 1 : parity;
				const known = flip.get(to);
				if (known === undefined) {
					flip.set(to, want);
					stack.push(to);
				} else if (known !== want) {
					return false;
				}
			}
		}
	}
	return true;
}

/** Total area of the live faces. */
export function surfaceArea(m: CMeshO): number {
	let total = 0;
	for (const f of liveFaces(m)) {
		const a = m.fv(f, 0);
		const b = m.fv(f, 1);
		const c = m.fv(f, 2);
		const ux = m.vx(b) - m.vx(a);
		const uy = m.vy(b) - m.vy(a);
		const uz = m.vz(b) - m.vz(a);
		const vx = m.vx(c) - m.vx(a);
		const vy = m.vy(c) - m.vy(a);
		const vz = m.vz(c) - m.vz(a);
		total += Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx) / 2;
	}
	return total;
}

/**
 * Signed volume by the divergence theorem. Meaningful only for a closed,
 * coherently oriented mesh; positive when the winding faces outward.
 */
export function signedVolume(m: CMeshO): number {
	let total = 0;
	for (const f of liveFaces(m)) {
		const a = m.fv(f, 0);
		const b = m.fv(f, 1);
		const c = m.fv(f, 2);
		total +=
			m.vx(a) * (m.vy(b) * m.vz(c) - m.vz(b) * m.vy(c)) -
			m.vy(a) * (m.vx(b) * m.vz(c) - m.vz(b) * m.vx(c)) +
			m.vz(a) * (m.vx(b) * m.vy(c) - m.vy(b) * m.vx(c));
	}
	return total / 6;
}

/** Everything computable, gathered for comparison against a builder's claims. */
export function computeFacts(m: CMeshO): MeshFacts {
	const en = countEdges(m);
	const nonManifoldEdges = countNonManifoldEdges(m);
	const chi = m.vn - en + m.fn;
	const watertight = isWatertight(m);
	const facts: MeshFacts = {
		vn: m.vn,
		fn: m.fn,
		en,
		chi,
		components: countComponents(m),
		watertight,
		nonManifoldEdges,
	};
	const loops = countBoundaryLoops(m);
	if (loops !== undefined) facts.boundaryLoops = loops;
	if (nonManifoldEdges === 0) {
		facts.orientable = isOrientable(m);
		facts.coherentlyOriented = isCoherentlyOriented(m);
		if (watertight && facts.orientable && m.fn > 0) {
			// chi = 2 - 2g per closed orientable component.
			facts.genus = (2 * facts.components - chi) / 2;
		}
	}
	return facts;
}

/** Checks every property the builder declared. Undeclared ones are ignored. */
export function assertFacts(m: CMeshO, expected: MeshFacts, label = "mesh"): void {
	const actual = computeFacts(m);
	for (const key of Object.keys(expected) as Array<keyof MeshFacts>) {
		const want = expected[key];
		if (want === undefined) continue;
		const got = actual[key];
		if (key === "area" || key === "volume") continue; // handled below
		expect(got, `${label}.${key}`).toBe(want);
	}
	if (expected.area !== undefined) {
		expect(surfaceArea(m), `${label}.area`).toBeCloseTo(expected.area, 9);
	}
	if (expected.volume !== undefined) {
		expect(signedVolume(m), `${label}.volume`).toBeCloseTo(expected.volume, 9);
	}
}

/**
 * The single highest-yield assertion in the suite: every structural invariant
 * the mesh kernel promises, checked at once.
 *
 * Run after every mutation in tests. It catches capacity drift, stale counters,
 * broken adjacency and dangling references long before they surface as a wrong
 * geometric answer.
 */
export function assertAllocatorConsistent(m: CMeshO, label = "mesh"): void {
	// Counters agree with the deleted flags.
	let liveV = 0;
	for (let v = 0; v < m.vertSize; v++) if (!m.isVertD(v)) liveV++;
	let liveF = 0;
	for (let f = 0; f < m.faceSize; f++) if (!m.isFaceD(f)) liveF++;
	expect(liveV, `${label}: vn must equal the number of undeleted vertices`).toBe(m.vn);
	expect(liveF, `${label}: fn must equal the number of undeleted faces`).toBe(m.fn);

	expect(m.vertSize, `${label}: vertSize must not exceed vertCap`).toBeLessThanOrEqual(m.vertCap);
	expect(m.faceSize, `${label}: faceSize must not exceed faceCap`).toBeLessThanOrEqual(m.faceCap);

	// Every allocated channel is exactly cap * arity long.
	for (const desc of CHANNELS) {
		const arr = getChannel(m, desc.key);
		if (arr === null) {
			expect(desc.optional, `${label}: mandatory channel ${desc.key} is missing`).toBe(true);
			continue;
		}
		const cap = desc.domain === "vert" ? m.vertCap : desc.domain === "face" ? m.faceCap : m.edgeCap;
		expect(arr.length, `${label}: channel ${desc.key} length`).toBe(cap * desc.arity);
	}

	// Live faces reference live, in-range vertices.
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) {
			const v = m.fv(f, k);
			expect(v, `${label}: face ${f} corner ${k} vertex index`).toBeGreaterThanOrEqual(0);
			expect(v, `${label}: face ${f} corner ${k} vertex index`).toBeLessThan(m.vertSize);
			expect(m.isVertD(v), `${label}: face ${f} references deleted vertex ${v}`).toBe(false);
		}
	}

	if (m.ffFace !== null) assertFFConsistent(m, label);
	if (m.vfHeadFace !== null) assertVFConsistent(m, label);
}

/**
 * FF adjacency must be a set of disjoint rings, one per undirected edge.
 *
 * Not an involution: VCGLib links *all* faces sharing an edge into a cycle, so
 * a border is a 1-cycle, a manifold edge a 2-cycle, and an edge shared by k
 * faces a k-cycle. Every member of a ring must agree on which two vertices the
 * edge joins, and the ring must close.
 */
export function assertFFConsistent(m: CMeshO, label = "mesh"): void {
	const endpoints = (f: number, e: number): string => {
		const a = m.fv(f, e);
		const b = m.fv(f, (e + 1) % 3);
		return a < b ? `${a}_${b}` : `${b}_${a}`;
	};

	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			const want = endpoints(f, e);
			let cf = f;
			let ce = e;
			let steps = 0;
			do {
				const nf = m.ffp(cf, ce);
				const ne = m.ffi(cf, ce);
				expect(nf, `${label}: ff(${cf},${ce}) target`).toBeGreaterThanOrEqual(0);
				expect(nf, `${label}: ff(${cf},${ce}) target`).toBeLessThan(m.faceSize);
				expect(m.isFaceD(nf), `${label}: ff(${cf},${ce}) points at deleted face ${nf}`).toBe(false);
				expect(ne, `${label}: ff(${cf},${ce}) edge index`).toBeLessThan(3);
				expect(endpoints(nf, ne), `${label}: ff ring from (${f},${e}) changes edge`).toBe(want);
				cf = nf;
				ce = ne;
				expect(++steps, `${label}: ff ring from (${f},${e}) does not close`).toBeLessThanOrEqual(
					m.faceSize * 3,
				);
			} while (cf !== f || ce !== e);
		}
	}
}

/** Every live face corner must appear exactly once in its vertex's VF chain. */
export function assertVFConsistent(m: CMeshO, label = "mesh"): void {
	const seen = new Set<string>();
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		let f = m.vfHeadFace![v];
		let i = m.vfHeadIndex![v];
		let guard = 0;
		while (f !== -1) {
			expect(m.isFaceD(f), `${label}: VF chain of vertex ${v} includes deleted face ${f}`).toBe(
				false,
			);
			expect(m.fv(f, i), `${label}: VF chain of vertex ${v} reached corner (${f},${i})`).toBe(v);
			const key = `${f}_${i}`;
			expect(seen.has(key), `${label}: corner (${f},${i}) appears twice in VF chains`).toBe(false);
			seen.add(key);
			const nf = m.vfNextFace![3 * f + i];
			const ni = m.vfNextIndex![3 * f + i];
			f = nf;
			i = ni;
			expect(++guard, `${label}: VF chain of vertex ${v} does not terminate`).toBeLessThan(
				m.faceSize * 3 + 2,
			);
		}
	}

	let corners = 0;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		corners += 3;
	}
	expect(seen.size, `${label}: VF chains must cover every live face corner`).toBe(corners);
}

/** Squared distance from a point to a triangle, clamped to the triangle. */
function pointTriangleDistanceSquared(
	px: number,
	py: number,
	pz: number,
	ax: number,
	ay: number,
	az: number,
	bx: number,
	by: number,
	bz: number,
	cx: number,
	cy: number,
	cz: number,
): number {
	// Ericson's closest-point-on-triangle, by Voronoi region.
	const abx = bx - ax;
	const aby = by - ay;
	const abz = bz - az;
	const acx = cx - ax;
	const acy = cy - ay;
	const acz = cz - az;
	const apx = px - ax;
	const apy = py - ay;
	const apz = pz - az;

	const d1 = abx * apx + aby * apy + abz * apz;
	const d2 = acx * apx + acy * apy + acz * apz;
	let qx: number;
	let qy: number;
	let qz: number;

	if (d1 <= 0 && d2 <= 0) {
		[qx, qy, qz] = [ax, ay, az];
	} else {
		const bpx = px - bx;
		const bpy = py - by;
		const bpz = pz - bz;
		const d3 = abx * bpx + aby * bpy + abz * bpz;
		const d4 = acx * bpx + acy * bpy + acz * bpz;
		if (d3 >= 0 && d4 <= d3) {
			[qx, qy, qz] = [bx, by, bz];
		} else {
			const cpx = px - cx;
			const cpy = py - cy;
			const cpz = pz - cz;
			const d5 = abx * cpx + aby * cpy + abz * cpz;
			const d6 = acx * cpx + acy * cpy + acz * cpz;
			if (d6 >= 0 && d5 <= d6) {
				[qx, qy, qz] = [cx, cy, cz];
			} else {
				const vc = d1 * d4 - d3 * d2;
				const vb = d5 * d2 - d1 * d6;
				const va = d3 * d6 - d5 * d4;
				if (vc <= 0 && d1 >= 0 && d3 <= 0) {
					const t = d1 / (d1 - d3);
					[qx, qy, qz] = [ax + abx * t, ay + aby * t, az + abz * t];
				} else if (vb <= 0 && d2 >= 0 && d6 <= 0) {
					const t = d2 / (d2 - d6);
					[qx, qy, qz] = [ax + acx * t, ay + acy * t, az + acz * t];
				} else if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
					const t = (d4 - d3) / (d4 - d3 + (d5 - d6));
					[qx, qy, qz] = [bx + (cx - bx) * t, by + (cy - by) * t, bz + (cz - bz) * t];
				} else {
					const denom = 1 / (va + vb + vc);
					const v = vb * denom;
					const w = vc * denom;
					[qx, qy, qz] = [ax + abx * v + acx * w, ay + aby * v + acy * w, az + abz * v + acz * w];
				}
			}
		}
	}
	const dx = px - qx;
	const dy = py - qy;
	const dz = pz - qz;
	return dx * dx + dy * dy + dz * dz;
}

/**
 * The one-sided Hausdorff distance from `a`'s vertices to `b`'s surface.
 *
 * Brute force, which is fine for test-sized meshes and keeps this a genuinely
 * independent check rather than a reuse of whatever spatial index the kernel
 * happens to have.
 */
export function hausdorffDistance(a: CMeshO, b: CMeshO): number {
	const faces = liveFaces(b);
	if (faces.length === 0) return Number.POSITIVE_INFINITY;
	let worst = 0;
	for (const v of liveVerts(a)) {
		let best = Number.POSITIVE_INFINITY;
		const px = a.vx(v);
		const py = a.vy(v);
		const pz = a.vz(v);
		for (const f of faces) {
			const i = b.fv(f, 0);
			const j = b.fv(f, 1);
			const k = b.fv(f, 2);
			const d = pointTriangleDistanceSquared(
				px,
				py,
				pz,
				b.vx(i),
				b.vy(i),
				b.vz(i),
				b.vx(j),
				b.vy(j),
				b.vz(j),
				b.vx(k),
				b.vy(k),
				b.vz(k),
			);
			if (d < best) best = d;
			if (best === 0) break;
		}
		if (best > worst) worst = best;
	}
	return Math.sqrt(worst);
}

/** The symmetric Hausdorff distance between two meshes. */
export function symmetricHausdorff(a: CMeshO, b: CMeshO): number {
	return Math.max(hausdorffDistance(a, b), hausdorffDistance(b, a));
}

/** Applying `fn` twice must produce the same mesh as applying it once. */
export function assertIdempotent(
	build: () => CMeshO,
	fn: (m: CMeshO) => void,
	label = "operation",
): void {
	const once = build();
	fn(once);
	const twice = build();
	fn(twice);
	fn(twice);
	expect(geometryDigest(twice), `${label} is not idempotent`).toBe(geometryDigest(once));
}

/**
 * A digest of the live geometry that ignores index order.
 *
 * We do not promise to reproduce MeshLab's vertex numbering, so comparisons
 * are made on the set of positions and the set of triangles-as-vertex-position
 * triples, not on the arrays themselves.
 */
export function geometryDigest(m: CMeshO, decimals = 9): string {
	const round = (x: number) => (Object.is(x, -0) ? 0 : Number(x.toFixed(decimals)));
	const key = (v: number) => `${round(m.vx(v))},${round(m.vy(v))},${round(m.vz(v))}`;
	const verts = liveVerts(m).map(key).sort();
	const faces = liveFaces(m)
		.map((f) => [key(m.fv(f, 0)), key(m.fv(f, 1)), key(m.fv(f, 2))].sort().join("|"))
		.sort();
	return `V${verts.length}:${verts.join(";")}\nF${faces.length}:${faces.join(";")}`;
}
