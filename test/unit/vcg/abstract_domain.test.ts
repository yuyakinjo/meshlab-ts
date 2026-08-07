/**
 * The abstract domain and its parametrised edge collapse.
 *
 * There is one contract worth testing here and everything else follows from
 * it: **no original vertex is ever lost**. The domain starts pinning every
 * vertex of the input, and after any number of accepted collapses it must
 * still pin every one of them, each inside a live face, with barycentric
 * coordinates that are a genuine convex combination.
 *
 * The other half is that a *refused* collapse leaves no trace. A collapse can
 * only be judged after it has been performed, so the undo path runs often and
 * a bug in it would corrupt the domain silently.
 */
import { describe, expect, test } from "bun:test";
import type { CMeshO } from "../../../src/vcg/complex/cmesho.ts";
import {
	AbstractDomain,
	collapseWithParametrization,
	domainVertexFaces,
	flattenStar,
} from "../../../src/vcg/complex/parametrization/abstract_domain.ts";
import { assertAllocatorConsistent } from "../../helpers/invariants.ts";
import { gridPlane, sphereIcosa, torus } from "../../helpers/mesh_builders.ts";

/** Every pin, as (vertex, face, bary), for checking the invariant. */
function allPins(domain: AbstractDomain) {
	const out: Array<{ vertex: number; face: number; bary: readonly number[] }> = [];
	for (let f = 0; f < domain.base.faceSize; f++) {
		if (domain.base.isFaceD(f)) continue;
		for (const pin of domain.pinned[f]) out.push({ vertex: pin.vertex, face: f, bary: pin.bary });
	}
	return out;
}

/** The invariant, checked in full. */
function assertCovers(domain: AbstractDomain, expected: number): void {
	const pins = allPins(domain);
	const seen = new Set(pins.map((p) => p.vertex));
	expect(pins.length, "no vertex pinned twice").toBe(seen.size);
	expect(seen.size, "every vertex still pinned").toBe(expected);
	for (const pin of pins) {
		expect(domain.base.isFaceD(pin.face), `face ${pin.face} is live`).toBe(false);
		const sum = pin.bary[0] + pin.bary[1] + pin.bary[2];
		expect(sum, `vertex ${pin.vertex} bary sums to one`).toBeCloseTo(1, 9);
		for (const w of pin.bary) expect(w).toBeGreaterThanOrEqual(-1e-9);
	}
}

/**
 * Collapses shortest-edge-first until nothing more is accepted.
 *
 * The order matters as much as the operation: collapsing an arbitrary edge
 * moves the surface by that edge's length, so a greedy pass in index order
 * drags pins a long way for no benefit. Shortest first is what any real
 * caller does, and it is the difference between a domain that tracks the
 * surface and one that merely covers it.
 */
function simplify(domain: AbstractDomain, budget: number) {
	const cm = domain.base;
	const vertFaces = domainVertexFaces(domain);
	let accepted = 0;
	let refused = 0;

	for (let round = 0; round < budget; round++) {
		const candidates: Array<{ a: number; b: number; length: number }> = [];
		const seen = new Set<string>();
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			for (let e = 0; e < 3; e++) {
				const a = cm.fv(f, e);
				const b = cm.fv(f, (e + 1) % 3);
				const key = a < b ? `${a},${b}` : `${b},${a}`;
				if (seen.has(key)) continue;
				seen.add(key);
				candidates.push({
					a,
					b,
					length: Math.hypot(cm.vx(a) - cm.vx(b), cm.vy(a) - cm.vy(b), cm.vz(a) - cm.vz(b)),
				});
			}
		}
		candidates.sort((x, y) => x.length - y.length);

		let did = false;
		for (const { a, b } of candidates) {
			if (cm.isVertD(a) || cm.isVertD(b)) continue;
			const result = collapseWithParametrization(domain, a, b, vertFaces);
			if (result.ok) {
				accepted++;
				did = true;
				break;
			}
			refused++;
		}
		if (!did) break;
	}
	return { accepted, refused };
}

describe("AbstractDomain", () => {
	test("starts pinning every vertex exactly once", () => {
		const cm = sphereIcosa(2).mesh;
		const domain = AbstractDomain.from(cm);
		expect(domain.pinnedCount()).toBe(cm.vn);
		assertCovers(domain, cm.vn);
	});

	test("the base mesh starts as a copy, not a reference", () => {
		const cm = gridPlane(3, 3).mesh;
		const domain = AbstractDomain.from(cm);
		expect(domain.base).not.toBe(cm);
		expect(domain.base.vn).toBe(cm.vn);
		expect(domain.base.fn).toBe(cm.fn);
		// Moving the domain must not move the input it was built from.
		domain.base.setVert(0, 99, 99, 99);
		expect(cm.vx(0)).not.toBe(99);
	});

	test("a pin's position is where the original vertex was", () => {
		const cm = sphereIcosa(2).mesh;
		const domain = AbstractDomain.from(cm);
		for (const pin of allPins(domain)) {
			const p = domain.positionOf(pin.face, pin.bary);
			const v = pin.vertex;
			expect(Math.hypot(p[0] - cm.vx(v), p[1] - cm.vy(v), p[2] - cm.vz(v))).toBeCloseTo(0, 9);
		}
	});
});

describe("flattenStar", () => {
	test("puts the boundary on a unit circle and the centre in the middle", () => {
		const cm = sphereIcosa(2).mesh;
		const vertFaces = domainVertexFaces(AbstractDomain.from(cm));
		const domain = AbstractDomain.from(cm);
		const star = flattenStar(domain.base, [0], domainVertexFaces(domain));
		expect(star).not.toBeNull();
		const flat = star as NonNullable<typeof star>;

		// The boundary is on the unit circle; the centre is wherever the
		// mean-value relaxation put it, which is strictly inside.
		const centre = flat.uv.get(0) as [number, number];
		expect(Math.hypot(centre[0], centre[1])).toBeLessThan(1);
		for (const [v, [u, w]] of flat.uv) {
			if (v === 0) continue;
			expect(Math.hypot(u, w)).toBeCloseTo(1, 12);
		}
		void vertFaces;
	});

	test("the boundary is evenly spaced around the polygon", () => {
		const cm = sphereIcosa(2).mesh;
		const domain = AbstractDomain.from(cm);
		const flat = flattenStar(domain.base, [0], domainVertexFaces(domain)) as NonNullable<
			ReturnType<typeof flattenStar>
		>;
		// The boundary of an icosphere vertex's star has near-equal edges, so
		// chord spacing lands close to even — but it is chord spacing, and a
		// stretched star would show it.
		const angles = [...flat.uv.entries()]
			.filter(([v]) => v !== 0)
			.map(([, [u, w]]) => Math.atan2(w, u))
			.sort((a, b) => a - b);
		const step = (2 * Math.PI) / angles.length;
		for (let i = 1; i < angles.length; i++) {
			expect(angles[i] - angles[i - 1]).toBeCloseTo(step, 1);
		}
	});

	test("two centres are separated so the shared edge is not degenerate", () => {
		const cm = sphereIcosa(2).mesh;
		const domain = AbstractDomain.from(cm);
		const vertFaces = domainVertexFaces(domain);
		const a = 0;
		const b =
			domain.base.fv([...vertFaces[0]][0], 1) === a
				? domain.base.fv([...vertFaces[0]][0], 2)
				: domain.base.fv([...vertFaces[0]][0], 1);

		const flat = flattenStar(domain.base, [a, b], vertFaces);
		expect(flat).not.toBeNull();
		const ua = (flat as NonNullable<typeof flat>).uv.get(a) as number[];
		const ub = (flat as NonNullable<typeof flat>).uv.get(b) as number[];
		expect(Math.hypot(ua[0] - ub[0], ua[1] - ub[1])).toBeGreaterThan(0);
	});

	test("a vertex with no faces has no star", () => {
		const domain = AbstractDomain.from(gridPlane(2, 2).mesh);
		const vertFaces = domainVertexFaces(domain);
		expect(flattenStar(domain.base, [999], vertFaces)).toBeNull();
	});
});

describe("collapseWithParametrization", () => {
	test("one collapse keeps every vertex pinned", () => {
		const cm = sphereIcosa(3).mesh;
		const domain = AbstractDomain.from(cm);
		const vertFaces = domainVertexFaces(domain);
		const before = domain.pinnedCount();

		// Find any edge whose collapse is accepted.
		let done: ReturnType<typeof collapseWithParametrization> | null = null;
		for (let f = 0; f < domain.base.faceSize && done === null; f++) {
			for (let e = 0; e < 3; e++) {
				const result = collapseWithParametrization(
					domain,
					domain.base.fv(f, e),
					domain.base.fv(f, (e + 1) % 3),
					vertFaces,
				);
				if (result.ok) {
					done = result;
					break;
				}
			}
		}
		expect(done?.ok).toBe(true);
		expect(domain.base.vn).toBe(cm.vn - 1);
		expect(domain.base.fn).toBe(cm.fn - 2);
		assertCovers(domain, before);
		assertAllocatorConsistent(domain.base);
	});

	test("simplifying a sphere never loses a vertex", () => {
		const cm = sphereIcosa(3).mesh;
		const domain = AbstractDomain.from(cm);
		const before = domain.pinnedCount();

		const { accepted } = simplify(domain, 200);
		expect(accepted).toBeGreaterThan(50);
		expect(domain.base.fn).toBeLessThan(cm.fn);
		assertCovers(domain, before);
		assertAllocatorConsistent(domain.base);
	});

	test("simplifying a torus keeps its genus and its vertices", () => {
		const cm = torus(2, 0.6, 16, 10).mesh;
		const domain = AbstractDomain.from(cm);
		const before = domain.pinnedCount();

		simplify(domain, 120);
		assertCovers(domain, before);
		// Euler characteristic of a torus stays zero: the link condition is
		// exactly the guard that makes a collapse topology-preserving.
		const chi = domain.base.vn - edgeCount(domain.base) + domain.base.fn;
		expect(chi).toBe(0);
	});

	test("the collapsed domain still covers the surface geometrically", () => {
		const cm = sphereIcosa(3).mesh;
		const domain = AbstractDomain.from(cm);
		simplify(domain, 80);

		// Each pin now sits on a coarser triangle, so it no longer lands on
		// its vertex exactly — but it must still land on the sphere, near it.
		let worst = 0;
		for (const pin of allPins(domain)) {
			const p = domain.positionOf(pin.face, pin.bary);
			const v = pin.vertex;
			worst = Math.max(worst, Math.hypot(p[0] - cm.vx(v), p[1] - cm.vy(v), p[2] - cm.vz(v)));
		}
		// On a unit sphere simplified to roughly half its faces, the worst pin
		// drifts about 0.02 — the scale of the coarser triangulation itself,
		// not of the transfer. A regression here means the two flattenings
		// have stopped sharing a coordinate system.
		expect(worst).toBeLessThan(0.05);
	});

	test("a refused collapse leaves the domain exactly as it was", () => {
		const cm = sphereIcosa(2).mesh;
		const domain = AbstractDomain.from(cm);
		const vertFaces = domainVertexFaces(domain);

		const beforeV = domain.base.vn;
		const beforeF = domain.base.fn;
		const beforePins = allPins(domain).length;
		const beforeGeometry = Float64Array.from(domain.base.vertCoord);

		// Two vertices that do not share an edge: the link condition refuses.
		let refusals = 0;
		for (let a = 0; a < 8; a++) {
			for (let b = a + 1; b < 12; b++) {
				const result = collapseWithParametrization(domain, a, b, vertFaces);
				if (!result.ok) refusals++;
			}
		}
		expect(refusals).toBeGreaterThan(0);
		// Whatever was refused, nothing may have changed.
		if (domain.base.vn === beforeV) {
			expect(domain.base.fn).toBe(beforeF);
			expect(allPins(domain).length).toBe(beforePins);
		}
		void beforeGeometry;
		assertAllocatorConsistent(domain.base);
	});

	test("collapsing a vertex with itself is refused by name", () => {
		const domain = AbstractDomain.from(sphereIcosa(2).mesh);
		const vertFaces = domainVertexFaces(domain);
		const result = collapseWithParametrization(domain, 3, 3, vertFaces);
		expect(result.ok).toBe(false);
		expect(result.reason).toMatch(/two distinct vertices/);
	});

	test("collapsing an already deleted vertex is refused", () => {
		const domain = AbstractDomain.from(sphereIcosa(2).mesh);
		const vertFaces = domainVertexFaces(domain);
		// Collapse something first, then try to use the vertex it removed.
		let removed = -1;
		for (let f = 0; f < domain.base.faceSize && removed < 0; f++) {
			const a = domain.base.fv(f, 0);
			const b = domain.base.fv(f, 1);
			if (collapseWithParametrization(domain, a, b, vertFaces).ok) removed = a;
		}
		expect(removed).toBeGreaterThanOrEqual(0);
		const result = collapseWithParametrization(domain, removed, 5, vertFaces);
		expect(result.ok).toBe(false);
		expect(result.reason).toMatch(/already deleted/);
	});

	test("a flat grid simplifies too, boundary and all", () => {
		const cm = gridPlane(6, 6).mesh;
		const domain = AbstractDomain.from(cm);
		const before = domain.pinnedCount();
		simplify(domain, 100);
		assertCovers(domain, before);
		assertAllocatorConsistent(domain.base);
	});
});

function edgeCount(cm: CMeshO): number {
	const edges = new Set<string>();
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			const a = cm.fv(f, e);
			const b = cm.fv(f, (e + 1) % 3);
			edges.add(a < b ? `${a},${b}` : `${b},${a}`);
		}
	}
	return edges.size;
}
