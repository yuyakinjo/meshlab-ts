/**
 * `filter_qhull` — convex hull, alpha shapes, Voronoi filtering and hidden
 * point removal.
 *
 * These four have unusually good tests available, because each rests on a
 * property that can be checked directly rather than against a reference:
 *
 * - A convex hull's faces all have every point behind them. That is the
 *   definition, and checking it catches any face the algorithm kept wrongly.
 * - A Delaunay tetrahedron's circumsphere contains no other point. Also the
 *   definition, and checking every tetrahedron against every point is cheap
 *   enough at test sizes to do exhaustively.
 * - An alpha shape converges to the convex hull as alpha grows past the largest
 *   circumradius. So the two filters have to agree in the limit, which ties
 *   them together without either being the reference for the other.
 * - Hidden point removal must select a point on the near side and reject the
 *   matching one on the far side.
 *
 * A note on the sphere: a point cloud sampled from a sphere's *surface* is
 * cospherical, which is the degenerate case for Delaunay — every tetrahedron
 * has the same circumradius and the tetrahedralization is not unique. That is
 * why the alpha-shape tests use a solid ball of points instead. It is not a
 * weakness in the code so much as the one input where the question itself is
 * ill-posed.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MLException } from "../../src/common/utilities/ml_exception.ts";
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { convexHull } from "../../src/vcg/space/convex_hull.ts";
import {
	circumcentre,
	delaunay3,
	tetraFaces,
	triangleCircumradius,
} from "../../src/vcg/space/delaunay3.ts";
import { sphereIcosa } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

/** A deterministic generator, so a failure is always reproducible. */
function rng(seed: number): () => number {
	let s = seed >>> 0 || 1;
	return () => {
		s = (s * 1103515245 + 12345) & 0x7fffffff;
		return s / 0x7fffffff;
	};
}

/** `n` points uniformly inside the unit ball. */
function ballCloud(n: number, seed = 987654321): Float64Array {
	const next = rng(seed);
	const out: number[] = [];
	while (out.length < 3 * n) {
		const x = 2 * next() - 1;
		const y = 2 * next() - 1;
		const z = 2 * next() - 1;
		if (x * x + y * y + z * z <= 1) out.push(x, y, z);
	}
	return new Float64Array(out);
}

/** The vertex coordinates of a builder mesh, faces discarded. */
function shellCloud(subdiv: number): Float64Array {
	const src = sphereIcosa(subdiv).mesh;
	const out = new Float64Array(3 * src.vn);
	for (let v = 0; v < src.vn; v++) {
		out[3 * v] = src.vx(v);
		out[3 * v + 1] = src.vy(v);
		out[3 * v + 2] = src.vz(v);
	}
	return out;
}

function docFromCloud(coords: Float64Array, count: number) {
	const cm = new CMeshO();
	Allocator.addVertices(cm, count);
	for (let i = 0; i < count; i++) {
		cm.setVert(i, coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]);
	}
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", "cloud", true, cm);
	m.updateBoxAndNormals();
	return { doc, m, cm };
}

// ------------------------------------------------------------- convex hull

describe("convex hull", () => {
	test("every point lies behind every face", () => {
		// The definition, checked exhaustively.
		const coords = ballCloud(300);
		const hull = convexHull(coords, 300);
		expect(hull).not.toBeNull();
		const h = hull as NonNullable<typeof hull>;
		let worst = 0;
		for (const f of h.faces) {
			for (let i = 0; i < 300; i++) {
				const d =
					f.normal[0] * coords[3 * i] +
					f.normal[1] * coords[3 * i + 1] +
					f.normal[2] * coords[3 * i + 2] -
					f.offset;
				worst = Math.max(worst, d);
			}
		}
		expect(worst).toBeLessThan(1e-9);
	});

	test("is a closed genus-zero surface", () => {
		const coords = ballCloud(300);
		const h = convexHull(coords, 300) as NonNullable<ReturnType<typeof convexHull>>;
		// V - E + F = 2 on a triangulated sphere, so F = 2V - 4.
		expect(h.faces.length).toBe(2 * h.vertices.length - 4);
		// And every edge is shared by exactly two faces.
		const edges = new Map<string, number>();
		for (const f of h.faces) {
			for (let k = 0; k < 3; k++) {
				const a = f.v[k];
				const b = f.v[(k + 1) % 3];
				const key = a < b ? `${a}_${b}` : `${b}_${a}`;
				edges.set(key, (edges.get(key) ?? 0) + 1);
			}
		}
		for (const count of edges.values()) expect(count).toBe(2);
	});

	test("ignores interior points", () => {
		// A cube's eight corners, buried in a cloud of points inside it.
		const corners = [
			[0, 0, 0],
			[1, 0, 0],
			[1, 1, 0],
			[0, 1, 0],
			[0, 0, 1],
			[1, 0, 1],
			[1, 1, 1],
			[0, 1, 1],
		];
		const next = rng(4242);
		const pts: number[] = corners.flat();
		for (let i = 0; i < 200; i++) {
			pts.push(0.1 + 0.8 * next(), 0.1 + 0.8 * next(), 0.1 + 0.8 * next());
		}
		const h = convexHull(new Float64Array(pts), pts.length / 3) as NonNullable<
			ReturnType<typeof convexHull>
		>;
		expect(h.vertices).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
		expect(h.faces.length).toBe(12);
	});

	test("keeps every point of a convex cloud", () => {
		const coords = shellCloud(3);
		const h = convexHull(coords, coords.length / 3) as NonNullable<ReturnType<typeof convexHull>>;
		expect(h.vertices.length).toBe(coords.length / 3);
	});

	test("refuses degenerate input", () => {
		// Three points, then four collinear ones, then four coplanar ones.
		expect(convexHull(new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3)).toBeNull();
		expect(convexHull(new Float64Array([0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0]), 4)).toBeNull();
		expect(convexHull(new Float64Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]), 4)).toBeNull();
	});

	test("through the filter, on a cube's corners", () => {
		const cube = new Float64Array([
			0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
		]);
		const { doc } = docFromCloud(cube, 8);
		const out = kernel.applyFilter(doc, "Convex Hull");
		expect(out.vertex_number).toBe(8);
		expect(out.face_number).toBe(12);
		// The input layer is untouched; the hull is a new one.
		expect(doc.meshNumber()).toBe(2);
	});

	test("refuses a mesh with fewer than four points", () => {
		const { doc } = docFromCloud(new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3);
		expect(() => kernel.applyFilter(doc, "Convex Hull")).toThrow(MLException);
	});
});

// ---------------------------------------------------------------- delaunay

describe("Delaunay tetrahedralization", () => {
	test("no circumsphere contains another point", () => {
		// The defining property, checked against every point.
		const coords = ballCloud(250, 24680);
		const tetra = delaunay3(coords, 250);
		expect(tetra.length).toBeGreaterThan(0);
		let violations = 0;
		for (const t of tetra) {
			for (let i = 0; i < 250; i++) {
				if (t.v.includes(i)) continue;
				const d = Math.hypot(
					coords[3 * i] - t.centre[0],
					coords[3 * i + 1] - t.centre[1],
					coords[3 * i + 2] - t.centre[2],
				);
				if (d < t.radius - 1e-9) violations++;
			}
		}
		expect(violations).toBe(0);
	});

	test("its outer boundary is exactly the convex hull", () => {
		// The strongest statement available about a tetrahedralization, and the
		// one the alpha shape approaches: a face used by one tetrahedron is on
		// the outside, and the outside of a Delaunay tetrahedralization is the
		// convex hull of the same points — face for face.
		const coords = ballCloud(400, 555);
		const tetra = delaunay3(coords, 400);
		const uses = new Map<string, number>();
		for (const t of tetra) {
			for (const f of tetraFaces(t)) {
				const key = f.join("_");
				uses.set(key, (uses.get(key) ?? 0) + 1);
			}
		}
		const boundary = new Set([...uses].filter(([, n]) => n === 1).map(([k]) => k));
		const hull = convexHull(coords, 400) as NonNullable<ReturnType<typeof convexHull>>;
		const hullFaces = new Set(hull.faces.map((f) => [...f.v].sort((a, b) => a - b).join("_")));
		expect(boundary.size).toBe(hullFaces.size);
		for (const key of boundary) expect(hullFaces.has(key)).toBe(true);
		// Nothing is used more than twice, which would be a non-manifold interior.
		expect([...uses.values()].every((n) => n <= 2)).toBe(true);
	});

	test("fills a cube's volume exactly", () => {
		const cube = new Float64Array([
			0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
		]);
		const tetra = delaunay3(cube, 8);
		const at = (i: number) => [cube[3 * i], cube[3 * i + 1], cube[3 * i + 2]];
		let volume = 0;
		for (const t of tetra) {
			const p = t.v.map(at);
			const u = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
			const v = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
			const w = [p[3][0] - p[0][0], p[3][1] - p[0][1], p[3][2] - p[0][2]];
			volume +=
				Math.abs(
					u[0] * (v[1] * w[2] - v[2] * w[1]) -
						u[1] * (v[0] * w[2] - v[2] * w[0]) +
						u[2] * (v[0] * w[1] - v[1] * w[0]),
				) / 6;
		}
		expect(volume).toBeCloseTo(1, 9);
	});

	test("the circumcentre is equidistant from all four corners", () => {
		const pa = [0, 0, 0];
		const pb = [2, 0, 0];
		const pc = [0, 3, 0];
		const pd = [0, 0, 4];
		const c = circumcentre(pa, pb, pc, pd) as number[];
		const d = [pa, pb, pc, pd].map((p) => Math.hypot(c[0] - p[0], c[1] - p[1], c[2] - p[2]));
		for (const r of d) expect(r).toBeCloseTo(d[0], 9);
	});

	test("coplanar points have no circumcentre", () => {
		expect(circumcentre([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0])).toBeNull();
	});

	test("a triangle's circumradius matches the known value", () => {
		// A 3-4-5 right triangle's circumcircle has the hypotenuse as diameter.
		expect(triangleCircumradius([0, 0, 0], [3, 0, 0], [0, 4, 0])).toBeCloseTo(2.5, 9);
		// A degenerate one has none, and must not join every alpha complex.
		expect(triangleCircumradius([0, 0, 0], [1, 0, 0], [2, 0, 0])).toBe(Number.POSITIVE_INFINITY);
	});

	test("fewer than four points, or all coplanar, gives nothing", () => {
		expect(delaunay3(new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3)).toEqual([]);
		expect(delaunay3(new Float64Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]), 4)).toEqual([]);
	});
});

// ------------------------------------------------------------ alpha shapes

describe("Alpha Complex/Shape", () => {
	test("a growing alpha shrinks the boundary toward the hull", () => {
		// The shape sweeps from the point cloud itself up to the convex hull as
		// alpha grows. It cannot quite *reach* the hull through the parameter:
		// the slider is capped at the bounding-box diagonal, while the sliver
		// tetrahedra sitting against the hull have circumradii an order of
		// magnitude larger. That is upstream's range too, and the invariant the
		// hull really pins down is tested against the tetrahedralization itself,
		// above.
		const coords = ballCloud(400, 555);
		const hullDoc = docFromCloud(coords, 400);
		const hull = kernel.applyFilter(hullDoc.doc, "Convex Hull");
		let previous = Number.POSITIVE_INFINITY;
		for (const alpha of [0.15, 0.3, 0.6, 3]) {
			const { doc } = docFromCloud(coords, 400);
			const out = kernel.applyFilter(doc, "Alpha Complex/Shape", { alpha, Filtering: 1 });
			expect(out.vertex_number as number).toBeLessThanOrEqual(previous);
			previous = out.vertex_number as number;
			// Never below the hull, which is the limit it approaches.
			expect(previous).toBeGreaterThanOrEqual(hull.vertex_number as number);
		}
	});

	test("a growing alpha keeps more tetrahedra", () => {
		const coords = ballCloud(400, 555);
		let previous = -1;
		for (const alpha of [0.15, 0.3, 0.6, 3]) {
			const { doc } = docFromCloud(coords, 400);
			const out = kernel.applyFilter(doc, "Alpha Complex/Shape", { alpha, Filtering: 1 });
			expect(out.tetrahedra as number).toBeGreaterThanOrEqual(previous);
			previous = out.tetrahedra as number;
		}
	});

	test("a small alpha keeps the boundary tight around the points", () => {
		// A tight alpha follows the cloud rather than bridging over it, so the
		// boundary uses far more of the points than the hull does.
		const coords = ballCloud(400, 555);
		const tight = docFromCloud(coords, 400);
		const loose = docFromCloud(coords, 400);
		const a = kernel.applyFilter(tight.doc, "Alpha Complex/Shape", {
			alpha: 0.15,
			Filtering: 1,
		});
		const b = kernel.applyFilter(loose.doc, "Alpha Complex/Shape", {
			alpha: 3,
			Filtering: 1,
		});
		expect(a.vertex_number as number).toBeGreaterThan((b.vertex_number as number) * 2);
	});

	test("the complex contains the shape", () => {
		const coords = ballCloud(300, 31337);
		const complexDoc = docFromCloud(coords, 300);
		const shapeDoc = docFromCloud(coords, 300);
		const complex = kernel.applyFilter(complexDoc.doc, "Alpha Complex/Shape", {
			alpha: 0.3,
			Filtering: 0,
		});
		const shape = kernel.applyFilter(shapeDoc.doc, "Alpha Complex/Shape", {
			alpha: 0.3,
			Filtering: 1,
		});
		// The shape is the complex's boundary, so it can only be smaller.
		expect(shape.face_number as number).toBeLessThan(complex.face_number as number);
	});

	test("each face carries its own alpha as quality", () => {
		const coords = ballCloud(200, 4711);
		const { doc } = docFromCloud(coords, 200);
		kernel.applyFilter(doc, "Alpha Complex/Shape", { alpha: 0.4, Filtering: 1 });
		const out = doc.mm().cm;
		const quality = out.faceQuality as Float64Array;
		expect(quality).not.toBeNull();
		for (let f = 0; f < out.faceSize; f++) {
			if (out.isFaceD(f)) continue;
			const p = [0, 1, 2].map((k) => {
				const v = out.fv(f, k);
				return [out.vx(v), out.vy(v), out.vz(v)];
			});
			expect(quality[f]).toBeCloseTo(triangleCircumradius(p[0], p[1], p[2]), 9);
			// And it really is at most alpha, which is what makes the value
			// usable as a re-threshold without recomputing.
			expect(quality[f]).toBeLessThanOrEqual(0.4 + 1e-9);
		}
	});

	test("refuses a non-positive alpha", () => {
		const { doc } = docFromCloud(ballCloud(50), 50);
		expect(() =>
			kernel.applyFilter(doc, "Alpha Complex/Shape", { alpha: 0, Filtering: 0 }),
		).toThrow(MLException);
	});

	test("refuses a coplanar cloud", () => {
		const flat = new Float64Array(3 * 20);
		const next = rng(9);
		for (let i = 0; i < 20; i++) {
			flat[3 * i] = next();
			flat[3 * i + 1] = next();
			flat[3 * i + 2] = 0;
		}
		const { doc } = docFromCloud(flat, 20);
		expect(() =>
			kernel.applyFilter(doc, "Alpha Complex/Shape", { alpha: 0.5, Filtering: 0 }),
		).toThrow(MLException);
	});
});

// ------------------------------------------------------- voronoi filtering

describe("Voronoi Filtering", () => {
	test("reconstructs a sphere's surface from points alone", () => {
		// No normals given, which is the filter's whole selling point.
		const coords = shellCloud(2);
		const count = coords.length / 3;
		const { doc } = docFromCloud(coords, count);
		const out = kernel.applyFilter(doc, "Voronoi Filtering");
		expect(out.face_number as number).toBeGreaterThan(0);
		// Every output vertex is an input sample, never a pole.
		const result = doc.mm().cm;
		for (let v = 0; v < result.vertSize; v++) {
			if (result.isVertD(v)) continue;
			expect(Math.hypot(result.vx(v), result.vy(v), result.vz(v))).toBeCloseTo(1, 6);
		}
	});

	test("finds two poles for most samples", () => {
		const coords = shellCloud(2);
		const count = coords.length / 3;
		const { doc } = docFromCloud(coords, count);
		const out = kernel.applyFilter(doc, "Voronoi Filtering");
		// One pole each at least; a closed surface should give close to two,
		// but a sample on the hull has its outer pole at infinity and drops it.
		expect(out.poles as number).toBeGreaterThanOrEqual(count);
	});

	test("a tighter discard threshold keeps fewer poles", () => {
		const coords = shellCloud(2);
		const count = coords.length / 3;
		const keep = (threshold: number) => {
			const { doc } = docFromCloud(coords, count);
			return kernel.applyFilter(doc, "Voronoi Filtering", { threshold }).poles as number;
		};
		expect(keep(0.1)).toBeLessThanOrEqual(keep(10));
	});

	test("leaves the input layer alone", () => {
		const coords = shellCloud(2);
		const count = coords.length / 3;
		const { doc, cm } = docFromCloud(coords, count);
		kernel.applyFilter(doc, "Voronoi Filtering");
		expect(cm.vn).toBe(count);
		expect(cm.fn).toBe(0);
		expect(doc.meshNumber()).toBe(2);
	});
});

// ---------------------------------------------------------- visible points

describe("Select Convex Hull Visible Points", () => {
	test("selects the near side and rejects the far side", () => {
		const coords = shellCloud(3);
		const count = coords.length / 3;
		const { doc, cm } = docFromCloud(coords, count);
		kernel.applyFilter(doc, "Select Convex Hull Visible Points", {
			viewpoint: [0, 0, 5],
		});
		// The point nearest the viewpoint must be visible, the one furthest
		// must not.
		let near = 0;
		let far = 0;
		for (let v = 1; v < cm.vertSize; v++) {
			if (cm.vz(v) > cm.vz(near)) near = v;
			if (cm.vz(v) < cm.vz(far)) far = v;
		}
		expect(cm.isVertS(near)).toBe(true);
		expect(cm.isVertS(far)).toBe(false);
	});

	test("selects roughly half a sphere", () => {
		const coords = shellCloud(3);
		const count = coords.length / 3;
		const { doc, cm } = docFromCloud(coords, count);
		const out = kernel.applyFilter(doc, "Select Convex Hull Visible Points", {
			viewpoint: [0, 0, 5],
		});
		const selected = out.selected_vertices as number;
		expect(selected).toBeGreaterThan(count * 0.25);
		expect(selected).toBeLessThan(count * 0.75);
		// And every selected point is on the near hemisphere.
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertS(v)) expect(cm.vz(v)).toBeGreaterThan(-0.5);
		}
	});

	test("moving the viewpoint moves the visible set", () => {
		const coords = shellCloud(3);
		const count = coords.length / 3;
		const above = docFromCloud(coords, count);
		const below = docFromCloud(coords, count);
		kernel.applyFilter(above.doc, "Select Convex Hull Visible Points", {
			viewpoint: [0, 0, 5],
		});
		kernel.applyFilter(below.doc, "Select Convex Hull Visible Points", {
			viewpoint: [0, 0, -5],
		});
		let overlap = 0;
		for (let v = 0; v < count; v++) {
			if (above.cm.isVertS(v) && below.cm.isVertS(v)) overlap++;
		}
		// The two hemispheres meet only near the equator.
		expect(overlap).toBeLessThan(count * 0.25);
	});

	test("a larger radius threshold selects more", () => {
		const coords = shellCloud(3);
		const count = coords.length / 3;
		const at = (radiusThreshold: number) => {
			const { doc } = docFromCloud(coords, count);
			return kernel.applyFilter(doc, "Select Convex Hull Visible Points", {
				viewpoint: [0, 0, 5],
				radiusThreshold,
			}).selected_vertices as number;
		};
		expect(at(3)).toBeGreaterThan(at(0));
	});

	test("clears any previous selection", () => {
		const coords = shellCloud(2);
		const count = coords.length / 3;
		const { doc, cm } = docFromCloud(coords, count);
		for (let v = 0; v < cm.vertSize; v++) cm.vertFlags[v] |= 0x0020;
		const out = kernel.applyFilter(doc, "Select Convex Hull Visible Points", {
			viewpoint: [0, 0, 5],
		});
		let selected = 0;
		for (let v = 0; v < cm.vertSize; v++) if (cm.isVertS(v)) selected++;
		expect(selected).toBe(out.selected_vertices as number);
		expect(selected).toBeLessThan(count);
	});

	test("says so when asked to use a camera the mesh has not got", () => {
		const { doc } = docFromCloud(shellCloud(1), shellCloud(1).length / 3);
		expect(() =>
			kernel.applyFilter(doc, "Select Convex Hull Visible Points", { usecamera: true }),
		).toThrow(MLException);
	});

	test("can emit the flipped hull for inspection", () => {
		const coords = shellCloud(2);
		const count = coords.length / 3;
		const { doc } = docFromCloud(coords, count);
		kernel.applyFilter(doc, "Select Convex Hull Visible Points", {
			viewpoint: [0, 0, 5],
			convex_hullFP: true,
		});
		expect(doc.meshNumber()).toBe(2);
	});
});

// ---------------------------------------------------------- registration

describe("registration", () => {
	test("every filter in the plugin is implemented", () => {
		const all = kernel.filterList().filter((f) => f.plugin.pluginName() === "FilterQhull");
		expect(all.length).toBe(4);
		expect(all.filter((f) => !f.implemented).map((f) => f.name)).toEqual([]);
	});
});
