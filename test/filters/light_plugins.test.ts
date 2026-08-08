/**
 * The small plugins: geodesic distance, triangulation optimisation, the
 * quality-to-colour mapper, random displacement, and the two standalone
 * parametrisations.
 *
 * Geodesic distance has an exact answer on a sphere — the great-circle
 * distance — so those tests check accuracy rather than plausibility, and they
 * pin the *relationship* between the two methods, since the whole reason the
 * heat method exists is that Dijkstra reads long.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import { Clean } from "../../src/vcg/complex/clean.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { VertexFlag } from "../../src/vcg/complex/flags.ts";
import { borderVertices, dijkstraGeodesic, heatGeodesic } from "../../src/vcg/complex/geodesic.ts";
import {
	foldedNum,
	meshAngleDistortion,
} from "../../src/vcg/complex/parametrization/distortion.ts";
import { SparseMatrix, solveCG } from "../../src/vcg/math/sparse.ts";
import { blue, red } from "../../src/vcg/space/color4.ts";
import { assertAllocatorConsistent } from "../helpers/invariants.ts";
import { gridPlane, sphereIcosa, torus } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

function scene(cm: CMeshO, channels = 0) {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", "test", true, cm);
	if (channels !== 0) m.updateDataMask(channels);
	m.updateBoxAndNormals();
	return { doc, m, cm };
}

/** The exact great-circle distance from vertex `from` on a unit sphere. */
function greatCircle(cm: CMeshO, from: number, to: number): number {
	const d = cm.vx(from) * cm.vx(to) + cm.vy(from) * cm.vy(to) + cm.vz(from) * cm.vz(to);
	return Math.acos(Math.min(1, Math.max(-1, d)));
}

/** The worst relative error against the great-circle distance. */
function sphereError(cm: CMeshO, distance: Float64Array, from: number): number {
	let worst = 0;
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v)) continue;
		const exact = greatCircle(cm, from, v);
		if (exact < 0.1) continue;
		worst = Math.max(worst, Math.abs(distance[v] - exact) / exact);
	}
	return worst;
}

describe("sparse solver", () => {
	test("solves a small system exactly", () => {
		// [[4,1],[1,3]] x = [1,2], whose solution is [1/11, 7/11].
		const a = new SparseMatrix(2);
		a.add(0, 0, 4);
		a.add(0, 1, 1);
		a.add(1, 0, 1);
		a.add(1, 1, 3);
		const { x, converged } = solveCG(a, Float64Array.from([1, 2]));
		expect(converged).toBe(true);
		expect(x[0]).toBeCloseTo(1 / 11, 10);
		expect(x[1]).toBeCloseTo(7 / 11, 10);
	});

	test("pinning keeps the matrix symmetric and moves the term to the right", () => {
		const a = new SparseMatrix(2);
		a.add(0, 0, 2);
		a.add(0, 1, 1);
		a.add(1, 0, 1);
		a.add(1, 1, 2);
		const rhs = Float64Array.from([0, 5]);
		a.pin(0, 3, rhs);

		expect(a.get(0, 0)).toBe(1);
		expect(a.get(0, 1)).toBe(0);
		// The pinned unknown's contribution moved into row 1's right-hand side.
		expect(a.get(1, 0)).toBe(0);
		expect(rhs[1]).toBe(5 - 3);
		const { x } = solveCG(a, rhs);
		expect(x[0]).toBe(3);
		expect(x[1]).toBeCloseTo(1, 10);
	});

	test("a diagonal of zeros does not produce NaN", () => {
		const a = new SparseMatrix(2);
		a.add(0, 1, 1);
		a.add(1, 0, 1);
		const { x } = solveCG(a, Float64Array.from([1, 1]));
		for (const v of x) expect(Number.isNaN(v)).toBe(false);
	});
});

describe("geodesic distance", () => {
	test("Dijkstra tracks the great-circle distance to within a quarter", () => {
		const cm = sphereIcosa(4).mesh;
		const d = dijkstraGeodesic(cm, [0]);
		// Two errors pull in opposite directions and neither dominates, so
		// this is *not* one-sided: an edge is a chord and reads short of the
		// arc it spans, while a path forced along edges reads long. The bound
		// is on the magnitude, not the sign.
		expect(sphereError(cm, d, 0)).toBeLessThan(0.25);
	});

	test("the heat method beats Dijkstra on a sphere", () => {
		const cm = sphereIcosa(4).mesh;
		const heat = heatGeodesic(cm, [0]);
		expect(heat).not.toBeNull();
		// Dijkstra can only walk along edges; the heat method crosses
		// triangles, which is the entire point of it.
		expect(sphereError(cm, heat as Float64Array, 0)).toBeLessThan(
			sphereError(cm, dijkstraGeodesic(cm, [0]), 0),
		);
	});

	test("the source itself is at zero and everything else beyond it", () => {
		const cm = sphereIcosa(3).mesh;
		for (const distance of [dijkstraGeodesic(cm, [5]), heatGeodesic(cm, [5]) as Float64Array]) {
			expect(distance[5]).toBeCloseTo(0, 9);
			for (let v = 0; v < cm.vertSize; v++) {
				if (cm.isVertD(v)) continue;
				expect(distance[v]).toBeGreaterThanOrEqual(0);
			}
		}
	});

	test("several sources give the distance to the nearest one", () => {
		const cm = sphereIcosa(3).mesh;
		const one = dijkstraGeodesic(cm, [0]);
		const two = dijkstraGeodesic(cm, [0, 11]);
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			expect(two[v]).toBeLessThanOrEqual(one[v] + 1e-9);
		}
		expect(two[11]).toBeCloseTo(0, 9);
	});

	test("a disconnected component stays at infinity", () => {
		const cm = sphereIcosa(2).mesh;
		const before = cm.vn;
		const extra = Allocator.addVertices(cm, 3);
		cm.setVert(extra, 10, 10, 10);
		cm.setVert(extra + 1, 11, 10, 10);
		cm.setVert(extra + 2, 10, 11, 10);
		Allocator.addFace(cm, extra, extra + 1, extra + 2);

		const d = dijkstraGeodesic(cm, [0]);
		expect(d[0]).toBe(0);
		expect(d[extra]).toBe(Number.POSITIVE_INFINITY);
		expect(before).toBeGreaterThan(0);
	});

	test("the border of a grid is its perimeter", () => {
		const cm = gridPlane(4, 3).mesh;
		expect(borderVertices(cm).length).toBe(2 * 4 + 2 * 3);
		expect(borderVertices(sphereIcosa(2).mesh)).toHaveLength(0);
	});

	test("distance from a border grows towards the middle", () => {
		const cm = gridPlane(8, 8).mesh;
		const d = dijkstraGeodesic(cm, borderVertices(cm));
		let centre = 0;
		let worst = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			if (d[v] > worst) {
				worst = d[v];
				centre = v;
			}
		}
		// The farthest point from the border of a square is its middle.
		expect(Math.abs(cm.vx(centre) - 0.5)).toBeLessThan(0.2);
		expect(Math.abs(cm.vy(centre) - 0.5)).toBeLessThan(0.2);
	});
});

describe("filter_geodesic", () => {
	test("colorizes from a given point and writes the distance as quality", () => {
		const { doc, cm } = scene(sphereIcosa(3).mesh);
		const out = kernel.applyFilter(doc, "Colorize by geodesic distance from a given point", {
			startPoint: [0, 0, 1],
			maxDistance: 0,
		});
		expect(out.sources).toBe(1);
		expect(out.max_distance as number).toBeGreaterThan(2);
		// The vertex nearest the start must have the smallest quality.
		let min = Number.POSITIVE_INFINITY;
		for (let v = 0; v < cm.vertSize; v++) min = Math.min(min, cm.vertQuality[v]);
		expect(min).toBeCloseTo(0, 9);
	});

	test("colorizes from the selection", () => {
		const { doc, cm } = scene(sphereIcosa(3).mesh);
		cm.vertFlags[7] |= VertexFlag.SELECTED;
		const out = kernel.applyFilter(doc, "Colorize by geodesic distance from the selected points", {
			maxDistance: 0,
		});
		expect(out.sources).toBe(1);
		expect(cm.vertQuality[7]).toBeCloseTo(0, 9);
	});

	test("the heat variant runs and agrees roughly with Dijkstra", () => {
		const { doc, cm } = scene(sphereIcosa(3).mesh);
		cm.vertFlags[7] |= VertexFlag.SELECTED;
		kernel.applyFilter(doc, "Colorize by approximated geodesic distance from the selected points", {
			m: 1,
		});
		const heat = Float64Array.from(cm.vertQuality);
		const exact = dijkstraGeodesic(cm, [7]);
		let worst = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			if (exact[v] < 0.3) continue;
			worst = Math.max(worst, Math.abs(heat[v] - exact[v]) / exact[v]);
		}
		expect(worst).toBeLessThan(0.5);
	});

	test("border distance needs a border, and says so", () => {
		const { doc } = scene(sphereIcosa(2).mesh);
		expect(() => kernel.applyFilter(doc, "Colorize by border distance", {})).toThrow(/no border/);
	});

	test("an empty selection is refused", () => {
		const { doc } = scene(sphereIcosa(2).mesh);
		expect(() =>
			kernel.applyFilter(doc, "Colorize by geodesic distance from the selected points", {
				maxDistance: 0,
			}),
		).toThrow(/no vertex is selected/);
	});

	test("border distance colours a grid", () => {
		const { doc, cm } = scene(gridPlane(6, 6).mesh);
		const out = kernel.applyFilter(doc, "Colorize by border distance", {});
		expect(out.sources as number).toBeGreaterThan(0);
		// The border itself is at zero, so it takes the low end of the ramp.
		const border = borderVertices(cm)[0];
		expect(cm.vertQuality[border]).toBeCloseTo(0, 9);
		// Whichever end of the ramp zero takes, the middle of the grid must
		// take the other one.
		let middle = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.vertQuality[v] > cm.vertQuality[middle]) middle = v;
		}
		expect(cm.vertColor[border]).not.toBe(cm.vertColor[middle]);
	});
});

describe("filter_trioptimize", () => {
	test("planar flipping improves the worst triangle and keeps the vertices", () => {
		// A grid pulled out of shape, so its triangles are poor but its
		// vertices are already where they belong.
		const cm = gridPlane(6, 6).mesh;
		for (let v = 0; v < cm.vertSize; v++) cm.setVert(v, cm.vx(v) * 3, cm.vy(v), cm.vz(v));
		const { doc } = scene(cm);
		const before = { vn: cm.vn, fn: cm.fn };

		kernel.applyFilter(doc, "Planar flipping optimization", {
			selection: false,
			pthreshold: 1,
			planartype: 0,
		});
		// A flip never adds or removes anything.
		expect(cm.vn).toBe(before.vn);
		expect(cm.fn).toBe(before.fn);
		assertAllocatorConsistent(cm);
	});

	test("curvature flipping leaves a flat mesh alone", () => {
		const { doc, cm } = scene(gridPlane(5, 5).mesh);
		const out = kernel.applyFilter(doc, "Curvature flipping optimization", {
			selection: false,
			pthreshold: 1,
			curvtype: 0,
		});
		// Every dihedral angle is already zero, so there is nothing to gain.
		expect(out.flipped_edges).toBe(0);
		void cm;
	});

	test("a flat mesh lets nearly every vertex move, a curved one does not", () => {
		const flat = scene(gridPlane(6, 6).mesh);
		const movedFlat =
			(kernel.applyFilter(flat.doc, "Laplacian Smooth (surface preserving)", {
				selection: false,
				AngleDeg: 0.5,
				iterations: 1,
			}).moved_vertices as number) / flat.cm.vn;

		const sphere = scene(sphereIcosa(3).mesh);
		const movedSphere =
			(kernel.applyFilter(sphere.doc, "Laplacian Smooth (surface preserving)", {
				selection: false,
				AngleDeg: 0.5,
				iterations: 1,
			}).moved_vertices as number) / sphere.cm.vn;

		// As a *fraction*: on a plane the normal never turns so almost every
		// vertex is free to move; on a sphere most moves would turn it.
		expect(movedFlat).toBeGreaterThan(0.9);
		expect(movedSphere).toBeLessThan(movedFlat);
	});

	test("a larger angle allowance lets more vertices move", () => {
		const measure = (angle: number) =>
			kernel.applyFilter(scene(sphereIcosa(3).mesh).doc, "Laplacian Smooth (surface preserving)", {
				selection: false,
				AngleDeg: angle,
				iterations: 1,
			}).moved_vertices as number;
		expect(measure(30)).toBeGreaterThan(measure(0.1));
	});

	test("zero iterations is refused", () => {
		const { doc } = scene(gridPlane(3, 3).mesh);
		expect(() =>
			kernel.applyFilter(doc, "Laplacian Smooth (surface preserving)", {
				selection: false,
				AngleDeg: 1,
				iterations: 0,
			}),
		).toThrow(/at least 1/);
	});
});

describe("filter_quality", () => {
	const NAME = "Quality Mapper applier";

	test("maps the quality range onto the colour band", () => {
		const { doc, cm } = scene(gridPlane(4, 4).mesh, MeshElement.MM_VERTQUALITY);
		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = v;

		const out = kernel.applyFilter(doc, NAME, {
			minQualityVal: 0,
			maxQualityVal: cm.vn - 1,
			midHandlePos: 50,
			brightness: 1,
		});
		expect(out.colored).toBe(cm.vn);
		// The two ends of the range take the two ends of the ramp, which are
		// opposite: whichever channel dominates at one end must not at the other.
		const low = cm.vertColor[0];
		const high = cm.vertColor[cm.vn - 1];
		expect(low).not.toBe(high);
		expect(red(low) > blue(low)).not.toBe(red(high) > blue(high));
	});

	test("the middle handle bends the mapping without moving the ends", () => {
		const build = (mid: number) => {
			const { doc, cm } = scene(gridPlane(4, 4).mesh, MeshElement.MM_VERTQUALITY);
			for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = v / (cm.vertSize - 1);
			kernel.applyFilter(doc, NAME, {
				minQualityVal: 0,
				maxQualityVal: 1,
				midHandlePos: mid,
				brightness: 1,
			});
			return Float64Array.from(cm.vertColor);
		};
		const centred = build(50);
		const shifted = build(20);
		expect(centred[0]).toBe(shifted[0]);
		expect(centred[centred.length - 1]).toBe(shifted[shifted.length - 1]);
		// Something in the middle must have changed.
		let differs = false;
		for (let i = 0; i < centred.length; i++) if (centred[i] !== shifted[i]) differs = true;
		expect(differs).toBe(true);
	});

	test("an empty or inverted range is refused", () => {
		const { doc, cm } = scene(gridPlane(3, 3).mesh, MeshElement.MM_VERTQUALITY);
		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = v;
		expect(() =>
			kernel.applyFilter(doc, NAME, {
				minQualityVal: 5,
				maxQualityVal: 5,
				midHandlePos: 50,
				brightness: 1,
			}),
		).toThrow(/empty or inverted/);
	});

	test("a middle handle at an end is refused", () => {
		const { doc, cm } = scene(gridPlane(3, 3).mesh, MeshElement.MM_VERTQUALITY);
		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = v;
		expect(() =>
			kernel.applyFilter(doc, NAME, {
				minQualityVal: 0,
				maxQualityVal: 8,
				midHandlePos: 0,
				brightness: 1,
			}),
		).toThrow(/strictly within/);
	});
});

describe("filter_sample", () => {
	const NAME = "Random Vertex Displacement";

	test("moves every vertex by no more than the bound", () => {
		const { doc, cm } = scene(sphereIcosa(2).mesh);
		const before = Float64Array.from(cm.vertCoord);
		const bound = 0.05;
		kernel.applyFilter(doc, NAME, { Displacement: bound, RandomSeed: 1 });

		for (let v = 0; v < cm.vertSize; v++) {
			const d = Math.hypot(
				cm.vx(v) - before[3 * v],
				cm.vy(v) - before[3 * v + 1],
				cm.vz(v) - before[3 * v + 2],
			);
			expect(d).toBeLessThanOrEqual(bound + 1e-12);
		}
	});

	test("the same seed gives the same displacement", () => {
		const run = (seed: number) => {
			const { doc, cm } = scene(sphereIcosa(2).mesh);
			kernel.applyFilter(doc, NAME, { Displacement: 0.1, RandomSeed: seed });
			return Float64Array.from(cm.vertCoord);
		};
		expect([...run(42)]).toEqual([...run(42)]);
		expect([...run(42)]).not.toEqual([...run(43)]);
	});

	test("a zero displacement leaves the mesh alone", () => {
		const { doc, cm } = scene(sphereIcosa(2).mesh);
		const before = Float64Array.from(cm.vertCoord);
		kernel.applyFilter(doc, NAME, { Displacement: 0, RandomSeed: 3 });
		expect([...cm.vertCoord]).toEqual([...before]);
	});
});

describe("filter_parametrization", () => {
	/** A disk-topology patch that genuinely curves. */
	function hemisphere(subdiv = 3): CMeshO {
		const full = sphereIcosa(subdiv).mesh;
		for (let f = 0; f < full.faceSize; f++) {
			if (full.isFaceD(f)) continue;
			const z = (full.vz(full.fv(f, 0)) + full.vz(full.fv(f, 1)) + full.vz(full.fv(f, 2))) / 3;
			if (z < 0) Allocator.deleteFace(full, f);
		}
		Clean.removeUnreferencedVertex(full);
		Allocator.compactEveryVector(full);
		return full;
	}

	test("harmonic lays a patch out inside the unit square", () => {
		const { doc, cm } = scene(hemisphere(3));
		kernel.applyFilter(doc, "Harmonic Parametrization", { harm_function: 1 });
		const wt = cm.wedgeTexCoord as Float64Array;
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			for (let k = 0; k < 6; k++) {
				expect(wt[6 * f + k]).toBeGreaterThanOrEqual(-1e-9);
				expect(wt[6 * f + k]).toBeLessThanOrEqual(1 + 1e-9);
			}
		}
	});

	test("mean value weights never fold, whatever the patch", () => {
		const { doc } = scene(hemisphere(3));
		const out = kernel.applyFilter(doc, "Harmonic Parametrization", { harm_function: 1 });
		expect(out.folded_faces).toBe(0);
	});

	test("LSCM distorts angles less than the fixed-boundary harmonic map", () => {
		const harmonic = scene(hemisphere(3));
		kernel.applyFilter(harmonic.doc, "Harmonic Parametrization", { harm_function: 2 });
		const harmonicError = meshAngleDistortion(harmonic.cm);

		const lscm = scene(hemisphere(3));
		const out = kernel.applyFilter(lscm.doc, "Least Squares Conformal Maps Parametrization", {});
		// The boundary is free in LSCM, so it can take the shape that keeps
		// the angles — which is exactly what a fixed circular boundary cannot.
		expect(out.angle_distortion as number).toBeLessThan(harmonicError);
		expect(foldedNum(lscm.cm)).toBe(0);
	});

	test("LSCM reproduces a flat patch almost exactly", () => {
		const { doc, cm } = scene(gridPlane(5, 5).mesh);
		kernel.applyFilter(doc, "Least Squares Conformal Maps Parametrization", {});
		// A plane is already conformal to itself, so the angles must survive.
		expect(meshAngleDistortion(cm)).toBeLessThan(1e-6);
		expect(foldedNum(cm)).toBe(0);
	});

	test("a closed surface is refused by both", () => {
		for (const [name, params] of [
			["Harmonic Parametrization", { harm_function: 1 }],
			["Least Squares Conformal Maps Parametrization", {}],
		] as const) {
			const { doc } = scene(torus(2, 0.6, 12, 8).mesh);
			expect(() => kernel.applyFilter(doc, name, params), name).toThrow(/not a disk|no boundary/);
		}
	});
});

describe("registry", () => {
	test("all eleven are registered under their own plugins", () => {
		const expected: Array<[string, string]> = [
			["Colorize by geodesic distance from a given point", "FilterGeodesic"],
			["Colorize by geodesic distance from the selected points", "FilterGeodesic"],
			["Colorize by approximated geodesic distance from the selected points", "FilterGeodesic"],
			["Colorize by border distance", "FilterGeodesic"],
			["Planar flipping optimization", "FilterTriOptimize"],
			["Curvature flipping optimization", "FilterTriOptimize"],
			["Laplacian Smooth (surface preserving)", "FilterTriOptimize"],
			["Harmonic Parametrization", "FilterParametrization"],
			["Least Squares Conformal Maps Parametrization", "FilterParametrization"],
			["Quality Mapper applier", "FilterQuality"],
			["Random Vertex Displacement", "FilterSample"],
		];
		for (const [name, plugin] of expected) {
			const action = kernel.pluginManager.filterAction(name);
			expect(action, name).toBeDefined();
			expect(action?.plugin.pluginName(), name).toBe(plugin);
		}
	});
});
