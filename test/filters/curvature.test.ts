/**
 * Curvature: the scalar pair and the tensor.
 *
 * Curvature has closed forms on the shapes the builders make, which is what
 * makes it testable without a reference implementation. A sphere of radius R
 * has mean curvature 1/R and Gaussian 1/R² *everywhere*; a cylinder of radius
 * R has principal curvatures 1/R and 0, so its Gaussian curvature is zero
 * despite being visibly curved; a flat sheet has neither.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import { MLNotImplementedException } from "../../src/common/utilities/ml_exception.ts";
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { Platonic } from "../../src/vcg/complex/create/platonic.ts";
import {
	CurvatureMapping,
	CurvatureType,
	curvatureToScalar,
	discreteCurvature,
	meanAndGaussian,
	principalCurvatures,
	principalDirections,
	principalDirectionsFitting,
} from "../../src/vcg/complex/curvature.ts";
import { gridPlane } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

/** A geodesic sphere of the given radius. */
function sphere(radius = 1, subdiv = 4): CMeshO {
	const m = Platonic.sphere(subdiv);
	for (let v = 0; v < m.vertSize; v++) {
		m.setVert(v, m.vx(v) * radius, m.vy(v) * radius, m.vz(v) * radius);
	}
	return m;
}

/**
 * A cylinder wall about the Y axis, closed into a tube with no caps.
 *
 * `Platonic.cone` puts vertices only on its two rims, so there is nowhere on
 * it that is away from both a crease and a boundary — no good for sampling
 * the wall's curvature. This one has `rings` bands of interior vertices.
 */
function cylinderWall(radius: number, height: number, sides = 64, rings = 20): CMeshO {
	const m = new CMeshO();
	Allocator.addVertices(m, sides * (rings + 1));
	for (let r = 0; r <= rings; r++) {
		const y = -height / 2 + (r * height) / rings;
		for (let s = 0; s < sides; s++) {
			const a = (2 * Math.PI * s) / sides;
			m.setVert(r * sides + s, radius * Math.cos(a), y, radius * Math.sin(a));
		}
	}
	Allocator.addFaces(m, sides * rings * 2);
	let f = 0;
	for (let r = 0; r < rings; r++) {
		for (let s = 0; s < sides; s++) {
			const next = (s + 1) % sides;
			const a = r * sides + s;
			const b = r * sides + next;
			const c = (r + 1) * sides + s;
			const d = (r + 1) * sides + next;
			m.setFace(f++, a, c, b);
			m.setFace(f++, b, c, d);
		}
	}
	return m;
}

/** Mean and spread of the live entries of a per-vertex array. */
function summarise(m: CMeshO, values: Float64Array) {
	const live: number[] = [];
	for (let v = 0; v < m.vertSize; v++) if (!m.isVertD(v)) live.push(values[v]);
	const mean = live.reduce((a, b) => a + b, 0) / live.length;
	return { mean, min: Math.min(...live), max: Math.max(...live) };
}

/** A mesh with the curvature-direction channel enabled. */
function withCurvDir(cm: CMeshO) {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", "m", true, cm);
	m.updateDataMask(MeshElement.MM_VERTCURVDIR);
	m.updateBoxAndNormals();
	return { doc, m, cm: m.cm };
}

describe("mean and Gaussian curvature", () => {
	test("a sphere of radius R has H = 1/R and K = 1/R² everywhere", () => {
		for (const radius of [0.5, 1, 3]) {
			const m = sphere(radius);
			const { mean, gaussian } = meanAndGaussian(m);
			const h = summarise(m, mean);
			const k = summarise(m, gaussian);
			expect(h.mean, `H at r=${radius}`).toBeCloseTo(1 / radius, 3);
			expect(k.mean, `K at r=${radius}`).toBeCloseTo(1 / radius ** 2, 2);
			// "Everywhere" is the point: a sphere is the one surface with no
			// variation at all, so the spread bounds the method's noise.
			expect((h.max - h.min) / h.mean, `H spread at r=${radius}`).toBeLessThan(0.01);
		}
	});

	test("refining converges on the analytic value", () => {
		let previous = Number.POSITIVE_INFINITY;
		for (const subdiv of [2, 3, 4]) {
			const m = sphere(1, subdiv);
			const { gaussian } = meanAndGaussian(m);
			const error = Math.abs(summarise(m, gaussian).mean - 1);
			expect(error, `subdiv ${subdiv}`).toBeLessThan(previous);
			previous = error;
		}
	});

	test("a cylinder is curved but has zero Gaussian curvature", () => {
		// The test that separates the two quantities: a cylinder bends in one
		// direction only, so K = k1·k2 = (1/R)·0 = 0 while H is 1/(2R).
		const m = cylinderWall(1, 4);
		const { mean, gaussian } = meanAndGaussian(m);
		// Away from the two open ends, where the one-ring is not a disc.
		const K: number[] = [];
		const H: number[] = [];
		for (let v = 0; v < m.vertSize; v++) {
			if (m.isVertD(v) || Math.abs(m.vy(v)) > 1.5) continue;
			K.push(gaussian[v]);
			H.push(mean[v]);
		}
		expect(K.length).toBeGreaterThan(100);
		for (const k of K) expect(Math.abs(k)).toBeLessThan(0.05);
		expect(H.reduce((a, b) => a + b, 0) / H.length).toBeCloseTo(0.5, 1);
	});

	test("a flat sheet has no curvature of either kind", () => {
		// Only the interior: a boundary vertex has no closed one-ring, so its
		// angle deficit is measured against the boundary rather than a turn.
		const m = gridPlane(8, 8).mesh;
		const { mean, gaussian } = meanAndGaussian(m);
		let interior = 0;
		for (let v = 0; v < m.vertSize; v++) {
			const x = m.vx(v);
			const y = m.vy(v);
			if (x <= 0 || x >= 1 || y <= 0 || y >= 1) continue;
			interior++;
			expect(Math.abs(mean[v]), `H at v${v}`).toBeLessThan(1e-9);
			expect(Math.abs(gaussian[v]), `K at v${v}`).toBeLessThan(1e-9);
		}
		expect(interior).toBeGreaterThan(20);
	});

	test("the sign distinguishes a dome from a bowl", () => {
		// Inverting the surface flips the mean curvature but not the Gaussian,
		// which is what makes H signed and K not.
		const dome = sphere(1, 3);
		const bowl = sphere(1, 3);
		for (let f = 0; f < bowl.faceSize; f++) {
			bowl.setFace(f, bowl.fv(f, 0), bowl.fv(f, 2), bowl.fv(f, 1));
		}
		expect(summarise(dome, meanAndGaussian(dome).mean).mean).toBeGreaterThan(0);
		expect(summarise(bowl, meanAndGaussian(bowl).mean).mean).toBeLessThan(0);
		expect(summarise(bowl, meanAndGaussian(bowl).gaussian).mean).toBeGreaterThan(0);
	});

	test("an unreferenced vertex reads zero, not NaN", () => {
		const m = sphere(1, 1);
		const lonely = Allocator.addVertex(m, 5, 5, 5);
		const { mean, gaussian } = meanAndGaussian(m);
		expect(mean[lonely]).toBe(0);
		expect(gaussian[lonely]).toBe(0);
	});
});

describe("the four discrete curvature types", () => {
	test("each reduces the pair the way its formula says", () => {
		// On a unit sphere H = K = 1, so ABS is 2 and RMS is sqrt(4-2).
		const m = sphere(1);
		for (const [type, expected] of [
			[CurvatureType.Mean, 1],
			[CurvatureType.Gaussian, 1],
			[CurvatureType.RMS, Math.SQRT2],
			[CurvatureType.ABS, 2],
		] as const) {
			expect(summarise(m, discreteCurvature(m, type)).mean, `type ${type}`).toBeCloseTo(
				expected,
				2,
			);
		}
	});

	test("RMS never comes back as NaN on a saddle", () => {
		// 4H² - 2K goes negative where the two estimates disagree, which they
		// do on a saddle; the radicand is floored rather than left to produce a
		// NaN that would then poison the colour ramp.
		const m = Platonic.torus(3, 1, 32, 16);
		const values = discreteCurvature(m, CurvatureType.RMS);
		for (let v = 0; v < m.vertSize; v++) {
			expect(Number.isFinite(values[v]), `v${v}`).toBe(true);
			expect(values[v]).toBeGreaterThanOrEqual(0);
		}
	});

	test("a torus has negative Gaussian curvature on its inner side", () => {
		// The one shape where the sign of K is visible: positive on the outside
		// of the tube, negative in the hole.
		const m = Platonic.torus(3, 1, 48, 24);
		const k = discreteCurvature(m, CurvatureType.Gaussian);
		let inner = 0;
		let outer = 0;
		for (let v = 0; v < m.vertSize; v++) {
			const distance = Math.hypot(m.vx(v), m.vy(v));
			if (distance < 2.5) inner += k[v] < 0 ? 1 : -1;
			if (distance > 3.5) outer += k[v] > 0 ? 1 : -1;
		}
		expect(inner).toBeGreaterThan(0);
		expect(outer).toBeGreaterThan(0);
	});

	test("the total Gaussian curvature of a closed surface follows Gauss-Bonnet", () => {
		// The integral of K over a closed surface is 2*pi*chi, whatever the
		// shape: 4*pi for a sphere, 0 for a torus. Nothing else in this file
		// checks the estimator globally rather than pointwise.
		const { gaussian: sphereK } = meanAndGaussian(sphere(1, 4));
		const sphereMesh = sphere(1, 4);
		expect(integrateOverVertices(sphereMesh, sphereK)).toBeCloseTo(4 * Math.PI, 0);

		const torus = Platonic.torus(3, 1, 48, 24);
		const { gaussian: torusK } = meanAndGaussian(torus);
		expect(Math.abs(integrateOverVertices(torus, torusK))).toBeLessThan(0.5);
	});
});

describe("the curvature tensor", () => {
	for (const [name, compute] of [
		["Taubin", principalDirections],
		["quadric fitting", principalDirectionsFitting],
	] as const) {
		test(`${name} gives both principal curvatures as 1/R on a sphere`, () => {
			// A sphere is the degenerate case for this test: k1 and k2 are equal,
			// so how the eigensolver splits the pair is pure noise — k1 lands a
			// little above 1/R, k2 the same amount below, and the size of that
			// split varies with the platform's libm (macOS measures the k1 mean
			// at 1.004, Linux at 1.010, for the identical mesh). The invariant
			// that does not depend on the split is their mean: it is the mean
			// curvature, and the noise cancels out of it. So the mean is held
			// tight and each curvature only to a band wider than the split.
			for (const radius of [1, 2]) {
				const { cm } = withCurvDir(sphere(radius));
				compute(cm);
				let k1Sum = 0;
				let k2Sum = 0;
				for (let v = 0; v < cm.vn; v++) {
					const p = principalCurvatures(cm, v);
					k1Sum += p.k1;
					k2Sum += p.k2;
				}
				const k1 = k1Sum / cm.vn;
				const k2 = k2Sum / cm.vn;
				expect(((k1 + k2) / 2) * radius, `${name} H at r=${radius}`).toBeCloseTo(1, 2);
				expect(Math.abs(k1 * radius - 1), `${name} k1 at r=${radius}`).toBeLessThan(0.03);
				expect(Math.abs(k2 * radius - 1), `${name} k2 at r=${radius}`).toBeLessThan(0.03);
			}
		});

		test(`${name} leaves the principal directions in the tangent plane`, () => {
			// They describe how the surface bends *along* itself, so a component
			// along the normal would be meaningless.
			const { cm } = withCurvDir(sphere(1, 3));
			compute(cm);
			const curv = cm.vertCurvDir;
			expect(curv).not.toBeNull();
			if (curv === null) return;
			for (let v = 0; v < cm.vn; v++) {
				for (const offset of [0, 3]) {
					const dot =
						curv[8 * v + offset] * cm.vertNormal[3 * v] +
						curv[8 * v + offset + 1] * cm.vertNormal[3 * v + 1] +
						curv[8 * v + offset + 2] * cm.vertNormal[3 * v + 2];
					expect(Math.abs(dot), `v${v} d${offset === 0 ? 1 : 2}`).toBeLessThan(1e-6);
				}
			}
		});

		test(`${name} leaves the two directions perpendicular to each other`, () => {
			const { cm } = withCurvDir(sphere(1, 3));
			compute(cm);
			const curv = cm.vertCurvDir;
			if (curv === null) return;
			for (let v = 0; v < cm.vn; v++) {
				let dot = 0;
				for (let a = 0; a < 3; a++) dot += curv[8 * v + a] * curv[8 * v + 3 + a];
				expect(Math.abs(dot), `v${v}`).toBeLessThan(1e-6);
			}
		});
	}

	test("quadric fitting finds the cylinder's two very different curvatures", () => {
		// A cylinder is the shape that tells a tensor method from a scalar one:
		// one direction bends at 1/R and the other not at all.
		const { cm } = withCurvDir(cylinderWall(1, 4));
		principalDirectionsFitting(cm);
		const flat: number[] = [];
		const round: number[] = [];
		for (let v = 0; v < cm.vn; v++) {
			if (Math.abs(cm.vy(v)) > 1.5) continue;
			const { k1, k2 } = principalCurvatures(cm, v);
			flat.push(Math.min(Math.abs(k1), Math.abs(k2)));
			round.push(Math.max(Math.abs(k1), Math.abs(k2)));
		}
		expect(flat.length).toBeGreaterThan(100);
		const avg = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
		expect(avg(flat)).toBeLessThan(0.05);
		expect(avg(round)).toBeCloseTo(1, 1);
	});

	test("the scalar mappings reduce the tensor as their formulas say", () => {
		const { cm } = withCurvDir(sphere(2));
		principalDirectionsFitting(cm);
		// On a sphere of radius 2, k1 = k2 = 0.5.
		const at = (mapping: number) => curvatureToScalar(cm, 0, mapping);
		expect(at(CurvatureMapping.Mean)).toBeCloseTo(0.5, 2);
		expect(at(CurvatureMapping.Gaussian)).toBeCloseTo(0.25, 2);
		expect(at(CurvatureMapping.MinCurvature)).toBeCloseTo(0.5, 2);
		expect(at(CurvatureMapping.MaxCurvature)).toBeCloseTo(0.5, 2);
		expect(at(CurvatureMapping.Curvedness)).toBeCloseTo(0.5, 2);
		// Shape index is +1 for a cap, whatever its radius.
		expect(at(CurvatureMapping.ShapeIndex)).toBeCloseTo(1, 2);
		expect(at(CurvatureMapping.None)).toBe(0);
	});

	test("shape index says what kind of shape a point is, not how curved", () => {
		// Koenderink's scale: +1 a cap, 0 a saddle, -1 a cup. It has to be the
		// same for two spheres of different radius, which is the whole point.
		for (const radius of [1, 5]) {
			const { cm } = withCurvDir(sphere(radius));
			principalDirectionsFitting(cm);
			expect(curvatureToScalar(cm, 0, CurvatureMapping.ShapeIndex), `r=${radius}`).toBeCloseTo(
				1,
				1,
			);
		}
	});

	test("refuses to run without the channel it writes into", () => {
		expect(() => principalDirections(sphere(1, 1))).toThrow();
		expect(() => principalDirectionsFitting(sphere(1, 1))).toThrow();
	});
});

describe("the filters", () => {
	test("Discrete Curvatures writes quality and colour", () => {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("", "s", true, sphere(1));
		m.updateBoxAndNormals();
		const out = kernel.applyFilter(doc, "Discrete Curvatures", { CurvatureType: 0 });
		expect(out.min_value as number).toBeCloseTo(1, 2);
		expect(out.max_value as number).toBeCloseTo(1, 2);
		for (let v = 0; v < m.cm.vn; v++) expect(m.cm.vertQuality[v]).toBeCloseTo(1, 2);
		expect(m.hasDataMask(MeshElement.MM_VERTCOLOR)).toBe(true);
	});

	test("Discrete Curvatures refuses a non-manifold mesh", () => {
		// A one-ring that is not a disc has no curvature; reporting a number
		// anyway would be worse than saying so.
		const doc = new MeshDocument();
		const cm = sphere(1, 1);
		// A third face on an existing edge makes that edge non-manifold.
		const extra = Allocator.addVertex(cm, 3, 3, 3);
		Allocator.addFace(cm, cm.fv(0, 0), cm.fv(0, 1), extra);
		const m = doc.addNewMesh("", "s", true, cm);
		m.updateBoxAndNormals();
		expect(() => kernel.applyFilter(doc, "Discrete Curvatures", {})).toThrow();
	});

	test("principal directions runs both implemented methods", () => {
		for (const Method of [0, 3]) {
			const doc = new MeshDocument();
			const m = doc.addNewMesh("", "s", true, sphere(1));
			m.updateBoxAndNormals();
			const out = kernel.applyFilter(doc, "Compute curvature principal directions", {
				Method,
				CurvColorMethod: 0,
			});
			expect(out.min_value as number, `method ${Method}`).toBeCloseTo(1, 1);
			expect(m.hasDataMask(MeshElement.MM_VERTCURVDIR), `method ${Method}`).toBe(true);
		}
	});

	test("the three methods it does not implement say so", () => {
		for (const Method of [1, 2, 4]) {
			const doc = new MeshDocument();
			const m = doc.addNewMesh("", "s", true, sphere(1, 2));
			m.updateBoxAndNormals();
			expect(
				() => kernel.applyFilter(doc, "Compute curvature principal directions", { Method }),
				`method ${Method}`,
			).toThrow(MLNotImplementedException);
		}
	});

	test("carries MeshLab's parameter defaults", () => {
		const list = kernel.initParameterList("Compute curvature principal directions");
		// The default is Quadric Fitting, not Taubin.
		expect(list.getParameterByName("Method").defaultValue.value).toBe(3);
		expect(list.getParameterByName("CurvColorMethod").defaultValue.value).toBe(0);
		expect(list.getParameterByName("Autoclean").defaultValue.value).toBe(true);
		expect(
			kernel.initParameterList("Discrete Curvatures").getParameterByName("CurvatureType")
				.defaultValue.value,
		).toBe(0);
	});

	test("are registered as MeshLab registers them", () => {
		for (const [name, pythonName, plugin] of [
			[
				"Compute curvature principal directions",
				"compute_curvature_principal_directions_per_vertex",
				"FilterMeshing",
			],
			["Discrete Curvatures", "compute_scalar_by_discrete_curvature_per_vertex", "FilterColorProc"],
		] as const) {
			const action = kernel.pluginManager.filterAction(name);
			expect(action, name).toBeDefined();
			if (!action) continue;
			expect(action.pythonName, name).toBe(pythonName);
			expect(action.plugin.pluginName(), name).toBe(plugin);
		}
	});
});

/** Integrates a per-vertex quantity over the mesh, one third of each face to each corner. */
function integrateOverVertices(m: CMeshO, values: Float64Array): number {
	const area = new Float64Array(m.vertSize);
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		const a = m.fv(f, 0);
		const b = m.fv(f, 1);
		const c = m.fv(f, 2);
		const u = [m.vx(b) - m.vx(a), m.vy(b) - m.vy(a), m.vz(b) - m.vz(a)];
		const v = [m.vx(c) - m.vx(a), m.vy(c) - m.vy(a), m.vz(c) - m.vz(a)];
		const doubleArea = Math.hypot(
			u[1] * v[2] - u[2] * v[1],
			u[2] * v[0] - u[0] * v[2],
			u[0] * v[1] - u[1] * v[0],
		);
		for (const w of [a, b, c]) area[w] += doubleArea / 6;
	}
	let total = 0;
	for (let v = 0; v < m.vertSize; v++) if (!m.isVertD(v)) total += values[v] * area[v];
	return total;
}
