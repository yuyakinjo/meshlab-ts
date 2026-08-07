/**
 * The normal, quality and unsharp-mask half of `filter_unsharp`.
 *
 * Unsharp masking is `original + weight * (original - smoothed)`: take what a
 * low-pass threw away and add it back. That gives it a property worth testing
 * directly — with weight zero it is the identity, and with a positive weight
 * it moves the signal *away* from its smoothed version rather than toward it.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import { filterClassToString } from "../../src/common/plugins/filter_class.ts";
import { MLException } from "../../src/common/utilities/ml_exception.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { Platonic } from "../../src/vcg/complex/create/platonic.ts";
import { Smooth } from "../../src/vcg/complex/smooth.ts";
import { UpdateNormal } from "../../src/vcg/complex/update/normal.ts";
import { UpdateTopology } from "../../src/vcg/complex/update/topology.ts";
import { blue, green, red, rgba } from "../../src/vcg/space/color4.ts";
import { signedVolume } from "../helpers/invariants.ts";

const kernel = MeshLabKernel.default();

function scene(channels = 0, subdiv = 3) {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", "m", true, Platonic.sphere(subdiv));
	if (channels !== 0) m.updateDataMask(channels);
	m.updateBoxAndNormals();
	return { doc, m, cm: m.cm };
}

const lengthOfVertexNormal = (cm: CMeshO, v: number) =>
	Math.hypot(cm.vertNormal[3 * v], cm.vertNormal[3 * v + 1], cm.vertNormal[3 * v + 2]);

describe("recomputing and normalizing normals", () => {
	test("face normals come back as the plane of the face", () => {
		const { doc, cm } = scene();
		cm.faceNormal.fill(0);
		kernel.applyFilter(doc, "Re-Compute Face Normals", {});
		for (let f = 0; f < cm.fn; f++) {
			// Unnormalised: their length is twice the triangle's area.
			const length = Math.hypot(
				cm.faceNormal[3 * f],
				cm.faceNormal[3 * f + 1],
				cm.faceNormal[3 * f + 2],
			);
			expect(length, `f${f}`).toBeGreaterThan(0);
		}
		kernel.applyFilter(doc, "Normalize Face Normals", {});
		for (let f = 0; f < cm.fn; f++) {
			expect(
				Math.hypot(cm.faceNormal[3 * f], cm.faceNormal[3 * f + 1], cm.faceNormal[3 * f + 2]),
				`f${f}`,
			).toBeCloseTo(1, 12);
		}
	});

	test("all four weighting schemes point outward on a sphere", () => {
		// Where they differ is on irregular triangulations; on a sphere every
		// scheme has to agree with the radial direction.
		for (const weightMode of [0, 1, 2, 3]) {
			const { doc, cm } = scene();
			cm.vertNormal.fill(0);
			kernel.applyFilter(doc, "Re-Compute Vertex Normals", { weightMode });
			for (let v = 0; v < cm.vn; v++) {
				const dot =
					cm.vertNormal[3 * v] * cm.vx(v) +
					cm.vertNormal[3 * v + 1] * cm.vy(v) +
					cm.vertNormal[3 * v + 2] * cm.vz(v);
				expect(dot, `mode ${weightMode} v${v}`).toBeGreaterThan(0.99);
				expect(lengthOfVertexNormal(cm, v), `mode ${weightMode} v${v}`).toBeCloseTo(1, 9);
			}
		}
	});

	test("the schemes disagree where the triangulation is uneven", () => {
		// If they all produced the same answer there would be no reason for
		// MeshLab to offer four of them.
		const results = [0, 1, 2, 3].map((weightMode) => {
			const doc = new MeshDocument();
			const m = doc.addNewMesh("", "m", true, Platonic.cone(1, 2, 3, 7));
			m.updateBoxAndNormals();
			kernel.applyFilter(doc, "Re-Compute Vertex Normals", { weightMode });
			return Array.from(m.cm.vertNormal.subarray(0, m.cm.vn * 3));
		});
		const same = (a: number[], b: number[]) => a.every((x, i) => Math.abs(x - b[i]) < 1e-9);
		expect(same(results[0], results[1])).toBe(false);
		expect(same(results[1], results[2])).toBe(false);
		expect(same(results[2], results[3])).toBe(false);
	});

	test("Nelson Max weighting is exact on a sphere, whatever the triangulation", () => {
		// The property Max's weights were designed for: a polyhedron inscribed
		// in a sphere gets the sphere's own normal at every vertex.
		const m = Platonic.sphere(2);
		UpdateNormal.perVertexNelsonMaxWeighted(m);
		UpdateNormal.normalizePerVertex(m);
		for (let v = 0; v < m.vn; v++) {
			const dot =
				m.vertNormal[3 * v] * m.vx(v) +
				m.vertNormal[3 * v + 1] * m.vy(v) +
				m.vertNormal[3 * v + 2] * m.vz(v);
			expect(dot, `v${v}`).toBeCloseTo(1, 9);
		}
	});

	test("normalizing an already-unit normal changes nothing", () => {
		const { doc, cm } = scene();
		kernel.applyFilter(doc, "Normalize Vertex Normals", {});
		const before = Array.from(cm.vertNormal.subarray(0, cm.vn * 3));
		kernel.applyFilter(doc, "Normalize Vertex Normals", {});
		for (let i = 0; i < before.length; i++) {
			expect(cm.vertNormal[i]).toBeCloseTo(before[i], 12);
		}
	});
});

describe("the Laplacian smoothers this half adds", () => {
	test("smoothing a uniform quality leaves it uniform", () => {
		const m = Platonic.sphere(2);
		m.vertQuality.fill(7);
		Smooth.vertexQualityLaplacian(m, 5);
		for (let v = 0; v < m.vn; v++) expect(m.vertQuality[v]).toBeCloseTo(7, 12);
	});

	test("smoothing pulls an isolated quality spike toward its neighbours", () => {
		const m = Platonic.sphere(2);
		m.vertQuality.fill(0);
		m.vertQuality[0] = 100;
		Smooth.vertexQualityLaplacian(m, 1);
		expect(m.vertQuality[0]).toBe(0);
		// And the neighbours have taken some of it.
		let touched = 0;
		for (let v = 1; v < m.vn; v++) if (m.vertQuality[v] > 0) touched++;
		expect(touched).toBeGreaterThan(3);
	});

	test("quality smoothing is a Jacobi step, not order-dependent", () => {
		// Written into a scratch buffer and copied back, so the answer cannot
		// depend on which vertex happened to be visited first.
		const a = Platonic.sphere(2);
		for (let v = 0; v < a.vertSize; v++) a.vertQuality[v] = v % 7;
		const b = Platonic.sphere(2);
		for (let v = 0; v < b.vertSize; v++) b.vertQuality[v] = v % 7;
		Smooth.vertexQualityLaplacian(a, 3);
		Smooth.vertexQualityLaplacian(b, 1);
		Smooth.vertexQualityLaplacian(b, 1);
		Smooth.vertexQualityLaplacian(b, 1);
		for (let v = 0; v < a.vn; v++)
			expect(a.vertQuality[v], `v${v}`).toBeCloseTo(b.vertQuality[v], 12);
	});

	test("colour smoothing averages each channel and stays in range", () => {
		const m = Platonic.sphere(2);
		for (let v = 0; v < m.vertSize; v++)
			m.vertColor[v] = v % 2 === 0 ? rgba(255, 0, 0) : rgba(0, 0, 255);
		Smooth.vertexColorLaplacian(m, 3);
		for (let v = 0; v < m.vn; v++) {
			expect(red(m.vertColor[v])).toBeGreaterThanOrEqual(0);
			expect(red(m.vertColor[v])).toBeLessThanOrEqual(255);
			// Green was never present, so it cannot appear.
			expect(green(m.vertColor[v]), `v${v}`).toBe(0);
		}
	});

	test("face normal smoothing averages across shared edges only", () => {
		const m = Platonic.sphere(2);
		UpdateNormal.perFaceNormalized(m);
		UpdateTopology.faceFace(m);
		const before = Array.from(m.faceNormal.subarray(0, m.fn * 3));
		Smooth.faceNormalLaplacianFF(m, 1);
		let moved = 0;
		for (let i = 0; i < before.length; i++)
			if (Math.abs(m.faceNormal[i] - before[i]) > 1e-12) moved++;
		expect(moved).toBeGreaterThan(0);
		// Averaging unit vectors that all point outward keeps them outward.
		for (let f = 0; f < m.fn; f++) {
			const length = Math.hypot(
				m.faceNormal[3 * f],
				m.faceNormal[3 * f + 1],
				m.faceNormal[3 * f + 2],
			);
			expect(length, `f${f}`).toBeGreaterThan(0.9);
		}
	});
});

describe("unsharp masking", () => {
	test("a zero weight and unit original weight is the identity", () => {
		// result = original * 1 + (original - smoothed) * 0.
		const { doc, cm } = scene();
		const before = Array.from(cm.vertCoord.subarray(0, cm.vn * 3));
		kernel.applyFilter(doc, "UnSharp Mask Geometry", { weight: 0, weightOrig: 1, iterations: 5 });
		for (let i = 0; i < before.length; i++) expect(cm.vertCoord[i]).toBeCloseTo(before[i], 12);
	});

	test("zero iterations leaves nothing for the mask to add back", () => {
		// With no smoothing the difference is zero, so any weight is a no-op.
		const { doc, cm } = scene();
		const before = Array.from(cm.vertCoord.subarray(0, cm.vn * 3));
		kernel.applyFilter(doc, "UnSharp Mask Geometry", { weight: 5, iterations: 0 });
		for (let i = 0; i < before.length; i++) expect(cm.vertCoord[i]).toBeCloseTo(before[i], 12);
	});

	test("sharpening geometry moves away from the smoothed shape, not toward it", () => {
		// Laplacian smoothing shrinks a sphere; the mask has to overshoot the
		// other way, past where it started.
		const original = signedVolume(Platonic.sphere(3));
		const smoothed = Platonic.sphere(3);
		Smooth.vertexCoordLaplacian(smoothed, 5);
		expect(signedVolume(smoothed)).toBeLessThan(original);

		const { doc, cm } = scene();
		kernel.applyFilter(doc, "UnSharp Mask Geometry", { weight: 0.3, iterations: 5 });
		expect(signedVolume(cm)).toBeGreaterThan(original);
	});

	test("a bigger weight sharpens harder", () => {
		const base = signedVolume(Platonic.sphere(3));
		let previous = base;
		for (const weight of [0.1, 0.3, 0.8]) {
			const { doc, cm } = scene();
			kernel.applyFilter(doc, "UnSharp Mask Geometry", { weight, iterations: 5 });
			const volume = signedVolume(cm);
			expect(volume, `weight ${weight}`).toBeGreaterThan(previous);
			previous = volume;
		}
	});

	test("quality sharpening exaggerates the range it was given", () => {
		const { doc, cm } = scene(MeshElement.MM_VERTQUALITY);
		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = cm.vz(v);
		const spread = (a: Float64Array, n: number) => {
			let min = Number.POSITIVE_INFINITY;
			let max = Number.NEGATIVE_INFINITY;
			for (let v = 0; v < n; v++) {
				min = Math.min(min, a[v]);
				max = Math.max(max, a[v]);
			}
			return max - min;
		};
		const before = spread(cm.vertQuality, cm.vn);
		kernel.applyFilter(doc, "UnSharp Mask Quality", { weight: 0.5, iterations: 5 });
		expect(spread(cm.vertQuality, cm.vn)).toBeGreaterThan(before);
	});

	test("colour sharpening clamps rather than wrapping", () => {
		// Pushing a channel past 255 has to saturate; wrapping would turn a
		// bright red vertex black.
		const { doc, cm } = scene(MeshElement.MM_VERTCOLOR);
		for (let v = 0; v < cm.vertSize; v++) {
			cm.vertColor[v] = cm.vz(v) > 0 ? rgba(255, 255, 255) : rgba(0, 0, 0);
		}
		kernel.applyFilter(doc, "UnSharp Mask Color", { weight: 2, iterations: 5 });
		for (let v = 0; v < cm.vn; v++) {
			for (const channel of [red, green, blue]) {
				expect(channel(cm.vertColor[v]), `v${v}`).toBeGreaterThanOrEqual(0);
				expect(channel(cm.vertColor[v]), `v${v}`).toBeLessThanOrEqual(255);
			}
		}
	});

	test("the normal mask leaves the geometry alone", () => {
		const { doc, cm } = scene();
		const before = Array.from(cm.vertCoord.subarray(0, cm.vn * 3));
		kernel.applyFilter(doc, "UnSharp Mask Normals", { weight: 0.5, iterations: 5 });
		for (let i = 0; i < before.length; i++) expect(cm.vertCoord[i]).toBe(before[i]);
	});

	test("the normal mask survives its own postcondition", () => {
		// The framework recomputes normals after a filter that says it changed
		// geometry; this one has to declare that it did not, or the mask it
		// just applied would be thrown away.
		const { doc, cm } = scene();
		UpdateNormal.perFaceNormalized(cm);
		const before = Array.from(cm.faceNormal.subarray(0, cm.fn * 3));
		kernel.applyFilter(doc, "UnSharp Mask Normals", { weight: 0.8, iterations: 5 });
		let moved = 0;
		for (let i = 0; i < before.length; i++)
			if (Math.abs(cm.faceNormal[i] - before[i]) > 1e-9) moved++;
		expect(moved).toBeGreaterThan(0);
	});

	test("a negative iteration count is refused", () => {
		const { doc } = scene();
		expect(() => kernel.applyFilter(doc, "UnSharp Mask Geometry", { iterations: -1 })).toThrow(
			MLException,
		);
	});
});

describe("Vertex Linear Morphing", () => {
	/** Two spheres with matching vertex order, the target twice the size. */
	function morphScene() {
		const doc = new MeshDocument();
		const source = doc.addNewMesh("", "source", true, Platonic.sphere(2));
		const target = Platonic.sphere(2);
		for (let v = 0; v < target.vertSize; v++) {
			target.setVert(v, target.vx(v) * 2, target.vy(v) * 2, target.vz(v) * 2);
		}
		const targetModel = doc.addNewMesh("", "target", false, target);
		doc.setCurrentMesh(source.id());
		source.updateBoxAndNormals();
		return { doc, source, targetId: targetModel.id() };
	}

	test("interpolates from the source to the target", () => {
		for (const [percent, radius] of [
			[0, 1],
			[50, 1.5],
			[100, 2],
		] as const) {
			const { doc, source, targetId } = morphScene();
			kernel.applyFilter(doc, "Vertex Linear Morphing", {
				TargetMesh: targetId,
				PercentMorph: percent,
			});
			const cm = source.cm;
			for (let v = 0; v < cm.vn; v++) {
				expect(Math.hypot(cm.vx(v), cm.vy(v), cm.vz(v)), `${percent}% v${v}`).toBeCloseTo(
					radius,
					9,
				);
			}
		}
	});

	test("extrapolates beyond the two ends rather than clamping", () => {
		// Which is why the slider runs from -150 to 250 rather than 0 to 100.
		const { doc, source, targetId } = morphScene();
		kernel.applyFilter(doc, "Vertex Linear Morphing", { TargetMesh: targetId, PercentMorph: 200 });
		expect(Math.hypot(source.cm.vx(0), source.cm.vy(0), source.cm.vz(0))).toBeCloseTo(3, 9);
	});

	test("refuses two meshes that cannot correspond", () => {
		const doc = new MeshDocument();
		const source = doc.addNewMesh("", "source", true, Platonic.sphere(2));
		const target = doc.addNewMesh("", "target", false, Platonic.sphere(3));
		doc.setCurrentMesh(source.id());
		expect(() =>
			kernel.applyFilter(doc, "Vertex Linear Morphing", { TargetMesh: target.id() }),
		).toThrow(MLException);
	});
});

describe("registration", () => {
	test("matches the upstream catalogue", () => {
		for (const [name, pythonName, cls] of [
			["Re-Compute Face Normals", "compute_normal_per_face", "Normal"],
			["Re-Compute Vertex Normals", "compute_normal_per_vertex", "Normal"],
			["Normalize Face Normals", "apply_normal_normalization_per_face", "Normal"],
			["Normalize Vertex Normals", "apply_normal_normalization_per_vertex", "Normal"],
			["Smooth Face Normals", "apply_normal_smoothing_per_face", "Smoothing"],
			["Smooth Vertex Quality", "apply_scalar_smoothing_per_vertex", "Smoothing"],
			["UnSharp Mask Geometry", "apply_coord_unsharp_mask", "Smoothing"],
			["UnSharp Mask Normals", "apply_normal_unsharp_mask_per_vertex", "Smoothing"],
			["UnSharp Mask Color", "apply_color_unsharp_mask_per_vertex", "VertexColoring|Smoothing"],
			["UnSharp Mask Quality", "apply_scalar_unsharp_mask_per_vertex", "Smoothing"],
			["Vertex Linear Morphing", "compute_coord_linear_morphing", "Smoothing"],
		] as const) {
			const action = kernel.pluginManager.filterAction(name);
			expect(action, name).toBeDefined();
			if (!action) continue;
			expect(action.pythonName, name).toBe(pythonName);
			expect(filterClassToString(action.filterClass), name).toBe(cls);
			expect(action.plugin.pluginName(), name).toBe("FilterUnsharp");
		}
	});

	test("carries MeshLab's parameter defaults", () => {
		for (const name of [
			"UnSharp Mask Geometry",
			"UnSharp Mask Normals",
			"UnSharp Mask Color",
			"UnSharp Mask Quality",
		]) {
			const list = kernel.initParameterList(name);
			expect(list.getParameterByName("weight").defaultValue.value, name).toBeCloseTo(0.3, 6);
			expect(list.getParameterByName("weightOrig").defaultValue.value, name).toBe(1);
			expect(list.getParameterByName("iterations").defaultValue.value, name).toBe(5);
		}
		// Only the normal mask offers a recompute.
		expect(kernel.initParameterList("UnSharp Mask Normals").hasParameter("recalc")).toBe(true);
		expect(kernel.initParameterList("UnSharp Mask Geometry").hasParameter("recalc")).toBe(false);
		expect(
			kernel.initParameterList("Re-Compute Vertex Normals").getParameterByName("weightMode")
				.defaultValue.value,
		).toBe(0);
	});

	test("the ones needing a solver, a polygon or an attribute stay unimplemented", () => {
		for (const name of [
			"Re-Compute Per-Polygon Face Normals",
			"TwoStep Smooth",
			"Depth Smooth",
			"Directional Geometry Preservation",
			"Cut mesh along crease edges",
		]) {
			expect(kernel.filterAction(name).implemented, name).toBe(false);
		}
	});
});
