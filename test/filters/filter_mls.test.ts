/**
 * APSS and RIMLS.
 *
 * These are approximation methods, so the tests check what the mathematics
 * guarantees rather than particular numbers: the field vanishes on the
 * surface it was fitted to, its gradient is the normal, projection is
 * idempotent and lands on the analytic surface, and the mean curvature of a
 * sphere of radius r comes back as 1/r.
 *
 * A sphere is the sharpest case available. APSS fits algebraic spheres, so it
 * should reproduce one to within floating point — noticeably better than the
 * plane-based RIMLS, and the tests say so with different tolerances rather
 * than by settling on the looser one for both.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import { Apss, estimateRadii, Rimls } from "../../src/meshlabplugins/filter_mls/mls_surface.ts";
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { UpdateBounding } from "../../src/vcg/complex/update/bounding.ts";
import { cubePlusIslands, sphereIcosa } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

/**
 * An evenly spread oriented point cloud on a sphere, from the Fibonacci
 * spiral — no poles and no seam, unlike a lat/long grid.
 */
function sphereCloud(count: number, radius = 1): CMeshO {
	const cm = new CMeshO();
	Allocator.addVertices(cm, count);
	const golden = Math.PI * (3 - Math.sqrt(5));
	for (let i = 0; i < count; i++) {
		const y = 1 - (2 * i) / (count - 1);
		const r = Math.sqrt(Math.max(0, 1 - y * y));
		const theta = golden * i;
		const n = [Math.cos(theta) * r, y, Math.sin(theta) * r];
		cm.setVert(i, n[0] * radius, n[1] * radius, n[2] * radius);
		cm.vertNormal[3 * i] = n[0];
		cm.vertNormal[3 * i + 1] = n[1];
		cm.vertNormal[3 * i + 2] = n[2];
	}
	UpdateBounding.box(cm);
	return cm;
}

/** A flat square of points in the z = 0 plane, all normals pointing up. */
function planeCloud(n: number, size = 2): CMeshO {
	const cm = new CMeshO();
	Allocator.addVertices(cm, n * n);
	for (let j = 0; j < n; j++) {
		for (let i = 0; i < n; i++) {
			const v = j * n + i;
			cm.setVert(v, (i / (n - 1) - 0.5) * size, (j / (n - 1) - 0.5) * size, 0);
			cm.vertNormal[3 * v + 2] = 1;
		}
	}
	UpdateBounding.box(cm);
	return cm;
}

/** Two planes meeting at a right angle: the crease RIMLS is meant to keep. */
function creaseCloud(n: number): CMeshO {
	const cm = new CMeshO();
	Allocator.addVertices(cm, 2 * n * n);
	let at = 0;
	for (let j = 0; j < n; j++) {
		for (let i = 0; i < n; i++) {
			const u = i / (n - 1);
			const w = j / (n - 1) - 0.5;
			// The horizontal half, z = 0, x from 0 to 1.
			cm.setVert(at, u, w, 0);
			cm.vertNormal[3 * at + 2] = 1;
			at++;
			// The vertical half, x = 0, z from 0 to 1.
			cm.setVert(at, 0, w, u);
			cm.vertNormal[3 * at] = 1;
			at++;
		}
	}
	UpdateBounding.box(cm);
	return cm;
}

function scene(cm: CMeshO, label = "cloud") {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", label, true, cm);
	m.updateDataMask(MeshElement.MM_VERTNORMAL);
	return { doc, m, cm };
}

describe("radius estimation", () => {
	test("scales with the point spacing, not with the point count", () => {
		const sparse = estimateRadii(sphereCloud(500));
		const dense = estimateRadii(sphereCloud(2000));
		const mean = (r: Float64Array) => r.reduce((a, b) => a + b, 0) / r.length;

		// Four times the points on the same sphere halves the spacing.
		expect(mean(dense)).toBeLessThan(mean(sparse));
		expect(mean(sparse) / mean(dense)).toBeCloseTo(2, 0);
	});

	test("a uniform cloud gets a near-uniform radius", () => {
		const radii = estimateRadii(planeCloud(20));
		// Ignore the border, where a point has neighbours on one side only.
		const interior = [...radii].filter((_, v) => {
			const i = v % 20;
			const j = Math.floor(v / 20);
			return i > 3 && i < 16 && j > 3 && j < 16;
		});
		const mean = interior.reduce((a, b) => a + b, 0) / interior.length;
		for (const r of interior) expect(r / mean).toBeCloseTo(1, 1);
	});

	test("a single point still gets a usable radius", () => {
		const cm = new CMeshO();
		Allocator.addVertices(cm, 1);
		expect(estimateRadii(cm)[0]).toBeGreaterThan(0);
	});
});

describe("APSS", () => {
	test("the potential vanishes on the sphere it was fitted to", () => {
		const cm = sphereCloud(2000);
		const surface = new Apss(cm);
		for (const p of [
			[1, 0, 0],
			[0, 1, 0],
			[0, 0, -1],
			[0.577, 0.577, 0.577],
		]) {
			const value = surface.potential(p[0], p[1], p[2]);
			expect(value).not.toBeNull();
			expect(Math.abs(value as number)).toBeLessThan(0.01);
		}
	});

	test("the potential is a signed distance away from the surface", () => {
		// The weight has compact support, so a probe a fifth of a radius out
		// is outside every point's reach at the default scale. Widening the
		// filter is the parameter that exists for exactly this.
		const surface = new Apss(sphereCloud(2000));
		surface.filterScale = 8;
		expect(surface.potential(1.2, 0, 0) as number).toBeGreaterThan(0);
		expect(surface.potential(0.8, 0, 0) as number).toBeLessThan(0);
		expect(surface.potential(1.2, 0, 0) as number).toBeCloseTo(0.2, 1);
	});

	test("outside the support there is no surface to speak of", () => {
		// Not an error: it is what compact support means, and the filters
		// above report it as "out of range" rather than inventing a value.
		const surface = new Apss(sphereCloud(2000));
		expect(surface.potential(1.5, 0, 0)).toBeNull();
	});

	test("the gradient is the outward normal", () => {
		const surface = new Apss(sphereCloud(2000));
		for (const p of [
			[1, 0, 0],
			[0, -1, 0],
			[0, 0, 1],
		]) {
			const g = surface.gradient(p[0], p[1], p[2]);
			expect(g).not.toBeNull();
			const unit = g as number[];
			const length = Math.hypot(unit[0], unit[1], unit[2]);
			for (let k = 0; k < 3; k++) expect(unit[k] / length).toBeCloseTo(p[k], 2);
		}
	});

	test("projection lands on the sphere from either side", () => {
		const surface = new Apss(sphereCloud(2000));
		surface.filterScale = 8;
		for (const start of [
			[1.3, 0, 0],
			[0.75, 0.05, 0.05],
			[0.55, 0.55, 0.55],
		]) {
			const hit = surface.project(start[0], start[1], start[2]);
			expect(hit).not.toBeNull();
			const p = (hit as { point: number[] }).point;
			expect(Math.hypot(p[0], p[1], p[2])).toBeCloseTo(1, 2);
		}
	});

	test("projection is idempotent", () => {
		const surface = new Apss(sphereCloud(2000));
		surface.filterScale = 8;
		const first = surface.project(1.1, 0.05, -0.05) as { point: number[] };
		const second = surface.project(first.point[0], first.point[1], first.point[2]) as {
			point: number[];
		};
		for (let k = 0; k < 3; k++) expect(second.point[k]).toBeCloseTo(first.point[k], 4);
	});

	test("the fitted sphere's curvature is 1/r", () => {
		for (const radius of [0.5, 1, 3]) {
			const surface = new Apss(sphereCloud(2000, radius));
			surface.filterScale = 4;
			const h = surface.approxMeanCurvature(radius, 0, 0);
			expect(h, `radius ${radius}`).toBeCloseTo(1 / radius, 2);
		}
	});

	test("the mean curvature from the field agrees with the sphere's", () => {
		const surface = new Apss(sphereCloud(3000, 2));
		surface.filterScale = 6;
		const h = surface.meanCurvature(2, 0, 0) as number;
		expect(h).toBeCloseTo(0.5, 1);
	});

	test("a plane has zero curvature and reproduces itself", () => {
		const surface = new Apss(planeCloud(30));
		surface.filterScale = 8;
		expect(Math.abs(surface.potential(0.1, -0.2, 0) as number)).toBeLessThan(1e-6);
		expect(surface.potential(0, 0, 0.1) as number).toBeCloseTo(0.1, 3);
		expect(Math.abs(surface.approxMeanCurvature(0, 0, 0) as number)).toBeLessThan(0.05);
	});

	test("a spherical parameter of zero degenerates to a plane fit", () => {
		const surface = new Apss(sphereCloud(2000));
		surface.filterScale = 4;
		surface.sphericalParameter = 0;
		// A plane through the local neighbourhood still passes through the
		// surface, but it can no longer curve, so the fitted sphere is gone.
		expect(surface.approxMeanCurvature(1, 0, 0)).toBe(0);
		expect(Math.abs(surface.potential(1, 0, 0) as number)).toBeLessThan(0.02);
	});

	test("a query far from every point has no answer at all", () => {
		const surface = new Apss(sphereCloud(500));
		expect(surface.potential(100, 100, 100)).toBeNull();
		expect(surface.gradient(100, 100, 100)).toBeNull();
		expect(surface.project(100, 100, 100)).toBeNull();
	});

	test("an empty cloud is refused", () => {
		expect(() => new Apss(new CMeshO())).toThrow(/at least one point/);
	});
});

describe("RIMLS", () => {
	test("the potential vanishes on the sphere and signs correctly", () => {
		const surface = new Rimls(sphereCloud(2000));
		surface.filterScale = 8;
		// Looser than the APSS bound above, and necessarily so: RIMLS fits
		// planes, and a plane through a patch of sphere sits inside it. That
		// systematic bias with a wide filter is the reason APSS exists.
		expect(Math.abs(surface.potential(1, 0, 0) as number)).toBeLessThan(0.05);
		expect(surface.potential(1.2, 0, 0) as number).toBeGreaterThan(0);
		expect(surface.potential(0.85, 0, 0) as number).toBeLessThan(0);
	});

	test("projection lands on the sphere", () => {
		const surface = new Rimls(sphereCloud(3000));
		surface.filterScale = 8;
		const hit = surface.project(1.2, 0.1, 0) as { point: number[] };
		expect(Math.hypot(...hit.point)).toBeCloseTo(1, 1);
	});

	test("a plane is reproduced exactly", () => {
		const surface = new Rimls(planeCloud(30));
		surface.filterScale = 8;
		expect(Math.abs(surface.potential(0.1, -0.2, 0) as number)).toBeLessThan(1e-9);
		const hit = surface.project(0.1, -0.2, 0.2) as { point: number[]; normal: number[] };
		expect(hit.point[2]).toBeCloseTo(0, 6);
		expect(hit.normal[2]).toBeCloseTo(1, 6);
	});

	test("a sharp normal filter keeps the gradient off the bisector", () => {
		// Just above the horizontal face, close enough to the crease that both
		// planes are inside the support. Plain IMLS averages the two normals
		// and points along the 45 degree bisector; the reweighting is supposed
		// to pick a side, which shows up as one dominant component.
		const at = [0.04, 0, 0.03];
		const sharp = new Rimls(creaseCloud(30));
		sharp.filterScale = 6;
		sharp.sigmaN = 0.3;
		sharp.maxRefittingIters = 8;

		const smooth = new Rimls(creaseCloud(30));
		smooth.filterScale = 6;
		smooth.maxRefittingIters = 1; // plain IMLS

		const dominant = (g: number[] | null) => {
			const v = g as number[];
			const length = Math.hypot(v[0], v[1], v[2]);
			return Math.max(...v.map((c) => Math.abs(c / length)));
		};
		const sharpG = dominant(sharp.gradient(at[0], at[1], at[2]));
		const smoothG = dominant(smooth.gradient(at[0], at[1], at[2]));

		expect(smoothG).toBeLessThan(0.99);
		expect(sharpG).toBeGreaterThan(smoothG);
	});

	test("one refitting iteration is plain IMLS", () => {
		const once = new Rimls(sphereCloud(1500));
		once.filterScale = 6;
		once.maxRefittingIters = 1;
		const many = new Rimls(sphereCloud(1500));
		many.filterScale = 6;
		many.maxRefittingIters = 5;
		// On a smooth surface the reweighting has nothing to reject, so the
		// two must agree closely; a large gap would mean the weights are wrong.
		const a = once.potential(1.1, 0, 0) as number;
		const b = many.potential(1.1, 0, 0) as number;
		expect(Math.abs(a - b) / Math.abs(a)).toBeLessThan(0.05);
	});

	test("a query far from every point has no answer", () => {
		const surface = new Rimls(sphereCloud(500));
		expect(surface.potential(50, 50, 50)).toBeNull();
	});
});

describe("Estimate radius from density", () => {
	const NAME = "Estimate radius from density";

	test("writes the radius into the quality channel", () => {
		const { doc, cm } = scene(sphereCloud(1000));
		const out = kernel.applyFilter(doc, NAME, { NbNeighbors: 16 });

		expect(out.mean_radius as number).toBeGreaterThan(0);
		let min = Number.POSITIVE_INFINITY;
		for (let v = 0; v < cm.vertSize; v++) min = Math.min(min, cm.vertQuality[v]);
		expect(min).toBeGreaterThan(0);
	});

	test("more neighbours means a larger radius", () => {
		const small = scene(sphereCloud(1000));
		const large = scene(sphereCloud(1000));
		const a = kernel.applyFilter(small.doc, NAME, { NbNeighbors: 6 }).mean_radius as number;
		const b = kernel.applyFilter(large.doc, NAME, { NbNeighbors: 32 }).mean_radius as number;
		expect(b).toBeGreaterThan(a);
	});

	test("zero neighbours is refused", () => {
		const { doc } = scene(sphereCloud(200));
		expect(() => kernel.applyFilter(doc, NAME, { NbNeighbors: 0 })).toThrow(/at least 1/);
	});
});

describe("Select small disconnected component", () => {
	const NAME = "Select small disconnected component";

	test("selects the islands and leaves the main body alone", () => {
		const built = cubePlusIslands(3);
		const { doc, cm } = scene(built.mesh, "islands");
		const out = kernel.applyFilter(doc, NAME, { NbFaceRatio: 0.5, NonClosedOnly: false });

		expect(out.components).toBe(4);
		// The cube is 12 faces and each island is smaller; with the ratio at
		// 0.5 the threshold is 6, so the islands go and the cube stays.
		const selected: number[] = [];
		for (let f = 0; f < cm.faceSize; f++) if (cm.isFaceS(f)) selected.push(f);
		expect(selected.length).toBe(out.selected_faces as number);
		expect(selected.length).toBeGreaterThan(0);
		expect(selected.length).toBeLessThan(cm.fn);
		// No face of the cube — the first 12 — may be selected.
		for (const f of selected) expect(f).toBeGreaterThanOrEqual(12);
	});

	test("a ratio of zero selects nothing", () => {
		const { doc } = scene(cubePlusIslands(3).mesh, "islands");
		const out = kernel.applyFilter(doc, NAME, { NbFaceRatio: 0, NonClosedOnly: false });
		expect(out.selected_faces).toBe(0);
	});

	test("a single component is never small relative to itself", () => {
		const { doc } = scene(sphereIcosa(2).mesh, "sphere");
		const out = kernel.applyFilter(doc, NAME, { NbFaceRatio: 0.9, NonClosedOnly: false });
		expect(out.components).toBe(1);
		expect(out.selected_faces).toBe(0);
	});

	test("NonClosedOnly skips components with no boundary", () => {
		const { doc } = scene(cubePlusIslands(3).mesh, "islands");
		// Every component here is a closed box, so restricting to open ones
		// leaves nothing to consider at all.
		const out = kernel.applyFilter(doc, NAME, { NbFaceRatio: 0.9, NonClosedOnly: true });
		expect(out.components).toBe(0);
		expect(out.selected_faces).toBe(0);
	});
});

describe("MLS projection", () => {
	test("APSS pulls a noisy cloud back onto the sphere", () => {
		const cm = sphereCloud(1500);
		// Displace every point along its own normal, so the cloud is still
		// sphere-shaped but no longer on the sphere.
		let seed = 12345;
		const random = () => {
			seed = (seed * 1103515245 + 12345) & 0x7fffffff;
			return seed / 0x7fffffff - 0.5;
		};
		for (let v = 0; v < cm.vertSize; v++) {
			const jitter = 1 + random() * 0.1;
			cm.setVert(v, cm.vx(v) * jitter, cm.vy(v) * jitter, cm.vz(v) * jitter);
		}
		const before = radialSpread(cm);

		const { doc, m } = scene(cm);
		const out = kernel.applyFilter(doc, "MLS projection (APSS)", {
			ControlMesh: m.id(),
			ProxyMesh: m.id(),
			SelectionOnly: false,
			FilterScale: 3,
		});

		expect(out.projected).toBe(cm.vn);
		expect(radialSpread(cm)).toBeLessThan(before / 2);
	});

	test("a cloud already on the surface barely moves", () => {
		const cm = sphereCloud(1500);
		const original = Float64Array.from(cm.vertCoord);
		const { doc, m } = scene(cm);
		kernel.applyFilter(doc, "MLS projection (RIMLS)", {
			ControlMesh: m.id(),
			ProxyMesh: m.id(),
			SelectionOnly: false,
		});

		let worst = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			worst = Math.max(
				worst,
				Math.hypot(
					cm.vx(v) - original[3 * v],
					cm.vy(v) - original[3 * v + 1],
					cm.vz(v) - original[3 * v + 2],
				),
			);
		}
		expect(worst).toBeLessThan(0.05);
	});

	test("SelectionOnly leaves the unselected vertices where they were", () => {
		const cm = sphereCloud(800);
		for (let v = 0; v < cm.vertSize; v++)
			cm.setVert(v, cm.vx(v) * 1.2, cm.vy(v) * 1.2, cm.vz(v) * 1.2);
		const original = Float64Array.from(cm.vertCoord);
		const { doc, m } = scene(cm);
		// Select half of them.
		for (let v = 0; v < cm.vertSize; v += 2) cm.vertFlags[v] |= 0x0020;

		const control = doc.addNewMesh("", "control", false, sphereCloud(1500));
		control.updateDataMask(MeshElement.MM_VERTNORMAL);
		const out = kernel.applyFilter(doc, "MLS projection (APSS)", {
			ControlMesh: control.id(),
			ProxyMesh: m.id(),
			SelectionOnly: true,
			FilterScale: 3,
		});
		expect(out.projected as number).toBeLessThan(cm.vn);

		for (let v = 1; v < cm.vertSize; v += 2) {
			if (cm.isVertS(v)) continue;
			expect(cm.vx(v)).toBe(original[3 * v]);
		}
	});

	test("projecting onto a different control mesh resamples the proxy", () => {
		const control = sphereCloud(2000, 2);
		const proxy = sphereCloud(300, 1.8);
		const doc = new MeshDocument();
		const cmesh = doc.addNewMesh("", "control", false, control);
		const pmesh = doc.addNewMesh("", "proxy", true, proxy);
		cmesh.updateDataMask(MeshElement.MM_VERTNORMAL);
		pmesh.updateDataMask(MeshElement.MM_VERTNORMAL);

		kernel.applyFilter(doc, "MLS projection (APSS)", {
			ControlMesh: cmesh.id(),
			ProxyMesh: pmesh.id(),
			SelectionOnly: false,
			FilterScale: 8,
		});
		// The proxy started inside and must end on the radius-2 sphere.
		for (let v = 0; v < proxy.vertSize; v++) {
			expect(Math.hypot(proxy.vx(v), proxy.vy(v), proxy.vz(v))).toBeCloseTo(2, 1);
		}
	});
});

describe("Marching Cubes", () => {
	test("APSS reconstructs a closed sphere of the right size", () => {
		const { doc } = scene(sphereCloud(2000));
		const out = kernel.applyFilter(doc, "Marching Cubes (APSS)", {
			Resolution: 40,
			FilterScale: 3,
		});
		expect(out.face_number as number).toBeGreaterThan(100);

		const cm = doc.mm().cm;
		let min = Number.POSITIVE_INFINITY;
		let max = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			const r = Math.hypot(cm.vx(v), cm.vy(v), cm.vz(v));
			min = Math.min(min, r);
			max = Math.max(max, r);
		}
		// Every extracted vertex was projected onto the surface, so all of
		// them should sit on the unit sphere, not merely near it on average.
		expect(min).toBeGreaterThan(0.9);
		expect(max).toBeLessThan(1.1);
	});

	test("RIMLS extracts a surface too", () => {
		const { doc } = scene(sphereCloud(1500));
		const out = kernel.applyFilter(doc, "Marching Cubes (RIMLS)", {
			Resolution: 30,
			FilterScale: 3,
		});
		expect(out.face_number as number).toBeGreaterThan(100);
	});

	test("a finer grid gives more triangles", () => {
		const coarse = scene(sphereCloud(1500));
		const fine = scene(sphereCloud(1500));
		const a = kernel.applyFilter(coarse.doc, "Marching Cubes (APSS)", {
			Resolution: 20,
			FilterScale: 3,
		}).face_number as number;
		const b = kernel.applyFilter(fine.doc, "Marching Cubes (APSS)", {
			Resolution: 40,
			FilterScale: 3,
		}).face_number as number;
		expect(b).toBeGreaterThan(a);
	});

	test("a degenerate resolution is refused", () => {
		const { doc } = scene(sphereCloud(500));
		expect(() => kernel.applyFilter(doc, "Marching Cubes (APSS)", { Resolution: 1 })).toThrow(
			/at least 2/,
		);
	});
});

describe("Colorize curvature", () => {
	test("APSS colours a sphere and records the curvature as quality", () => {
		const { doc, cm } = scene(sphereCloud(1500, 2));
		const out = kernel.applyFilter(doc, "Colorize curvature (APSS)", {
			CurvatureType: 4, // the fitted sphere's own curvature
			SelectionOnly: false,
			FilterScale: 3,
		});

		expect(out.colorized).toBe(cm.vn);
		// Radius 2 everywhere, so the curvature should be 0.5 everywhere.
		for (let v = 0; v < cm.vertSize; v++) {
			expect(cm.vertQuality[v]).toBeCloseTo(0.5, 1);
		}
	});

	test("the mean curvature of a plane is near zero", () => {
		const { doc, cm } = scene(planeCloud(25));
		kernel.applyFilter(doc, "Colorize curvature (RIMLS)", {
			CurvatureType: 0,
			SelectionOnly: false,
		});
		let worst = 0;
		for (let v = 0; v < cm.vertSize; v++) worst = Math.max(worst, Math.abs(cm.vertQuality[v]));
		expect(worst).toBeLessThan(0.5);
	});

	test("the curvature types that need principal directions say so", () => {
		const { doc } = scene(sphereCloud(500));
		for (const type of [1, 2, 3]) {
			expect(() =>
				kernel.applyFilter(doc, "Colorize curvature (APSS)", {
					CurvatureType: type,
					SelectionOnly: false,
				}),
			).toThrow(/principal curvatures/);
		}
	});

	test("only APSS offers the fitted-sphere curvature", () => {
		// It is not a runtime check but a parameter one: the RIMLS enum simply
		// has no fifth entry, so the value is rejected before the filter runs.
		const apss = kernel.pluginManager.filterAction("Colorize curvature (APSS)");
		const rimls = kernel.pluginManager.filterAction("Colorize curvature (RIMLS)");
		const { m } = scene(sphereCloud(200));

		expect(() =>
			apss?.plugin.initParameterList(apss.id, m).applyPlain({ CurvatureType: 4 }),
		).not.toThrow();
		expect(() =>
			rimls?.plugin.initParameterList(rimls.id, m).applyPlain({ CurvatureType: 4 }),
		).toThrow(/outside 0\.\.3/);
	});
});

describe("registry", () => {
	test("all eight are implemented under FilterMLS", () => {
		for (const name of [
			"MLS projection (APSS)",
			"MLS projection (RIMLS)",
			"Marching Cubes (APSS)",
			"Marching Cubes (RIMLS)",
			"Colorize curvature (APSS)",
			"Colorize curvature (RIMLS)",
			"Estimate radius from density",
			"Select small disconnected component",
		]) {
			const action = kernel.pluginManager.filterAction(name);
			expect(action, name).toBeDefined();
			expect(action?.plugin.pluginName(), name).toBe("FilterMLS");
		}
	});
});

/** The spread of the radii, as a stand-in for how far off the sphere a cloud is. */
function radialSpread(cm: CMeshO): number {
	let sum = 0;
	let count = 0;
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v)) continue;
		sum += Math.hypot(cm.vx(v), cm.vy(v), cm.vz(v));
		count++;
	}
	const mean = sum / count;
	let sq = 0;
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v)) continue;
		sq += (Math.hypot(cm.vx(v), cm.vy(v), cm.vz(v)) - mean) ** 2;
	}
	return Math.sqrt(sq / count);
}
