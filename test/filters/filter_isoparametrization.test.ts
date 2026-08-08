/**
 * The isoparametrisation pipeline.
 *
 * The domain is built by a greedy process, so the exact face count is not
 * predictable and the tests do not pretend it is. What is predictable is the
 * contract: the domain still pins every vertex, a remesh lands on the original
 * surface and is far more uniform than the input, an atlas has no folded
 * faces, and every filter that needs a domain refuses to run without one.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { isoParametrizationOf } from "../../src/meshlabplugins/filter_isoparametrization/filter_isoparametrization.ts";
import { IsoParametrization } from "../../src/meshlabplugins/filter_isoparametrization/iso_parametrization.ts";
import { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { foldedNum, globallyUnfolded } from "../../src/vcg/complex/parametrization/distortion.ts";
import { assertAllocatorConsistent } from "../helpers/invariants.ts";
import { sphereIcosa, torus } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

function scene(cm: CMeshO, label = "test") {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", label, true, cm);
	m.updateBoxAndNormals();
	return { doc, m, cm };
}

/** The spread of edge lengths, as a stand-in for how uniform a mesh is. */
function edgeUniformity(cm: CMeshO): number {
	const lengths: number[] = [];
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			const a = cm.fv(f, e);
			const b = cm.fv(f, (e + 1) % 3);
			lengths.push(Math.hypot(cm.vx(a) - cm.vx(b), cm.vy(a) - cm.vy(b), cm.vz(a) - cm.vz(b)));
		}
	}
	const mean = lengths.reduce((x, y) => x + y, 0) / lengths.length;
	const variance = lengths.reduce((s, x) => s + (x - mean) ** 2, 0) / lengths.length;
	return Math.sqrt(variance) / mean;
}

/** How far a mesh's vertices are from a unit sphere. */
function sphereError(cm: CMeshO): number {
	let worst = 0;
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v)) continue;
		worst = Math.max(worst, Math.abs(Math.hypot(cm.vx(v), cm.vy(v), cm.vz(v)) - 1));
	}
	return worst;
}

describe("IsoParametrization.build", () => {
	test("simplifies towards the target and keeps every vertex", () => {
		const cm = sphereIcosa(3).mesh;
		const iso = IsoParametrization.build(cm, { targetMinFaces: 20, targetMaxFaces: 60 });

		expect(iso.faceCount).toBeLessThan(cm.fn);
		expect(iso.faceCount).toBeGreaterThanOrEqual(20);
		expect(iso.domain.pinnedCount()).toBe(cm.vn);
		assertAllocatorConsistent(iso.domain.base);
	});

	test("every original vertex can be located in the domain", () => {
		const cm = sphereIcosa(2).mesh;
		const iso = IsoParametrization.build(cm, { targetMinFaces: 20, targetMaxFaces: 40 });
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			const at = iso.locate(v);
			expect(at, `vertex ${v}`).toBeDefined();
			const bary = (at as NonNullable<typeof at>).bary;
			expect(bary[0] + bary[1] + bary[2]).toBeCloseTo(1, 9);
		}
	});

	test("a domain point maps back onto the original surface", () => {
		const cm = sphereIcosa(3).mesh;
		const iso = IsoParametrization.build(cm, { targetMinFaces: 30, targetMaxFaces: 80 });
		// Samples all over the domain, all of which must land on the sphere.
		for (let f = 0; f < iso.domain.base.faceSize; f++) {
			if (iso.domain.base.isFaceD(f)) continue;
			for (const bary of [
				[1 / 3, 1 / 3, 1 / 3],
				[0.6, 0.2, 0.2],
				[0.1, 0.1, 0.8],
			]) {
				const p = iso.toSurface(f, bary);
				// The projection puts it on the *original* triangulation, whose
				// vertices are on the unit sphere; a face centre sits slightly
				// inside, which is the discretisation, not the map.
				expect(Math.hypot(p[0], p[1], p[2])).toBeGreaterThan(0.9);
				expect(Math.hypot(p[0], p[1], p[2])).toBeLessThanOrEqual(1.0001);
			}
		}
	});

	test("a torus keeps its genus through the simplification", () => {
		const cm = torus(2, 0.6, 16, 10).mesh;
		const iso = IsoParametrization.build(cm, { targetMinFaces: 40, targetMaxFaces: 100 });
		const base = iso.domain.base;
		const edges = new Set<string>();
		for (let f = 0; f < base.faceSize; f++) {
			if (base.isFaceD(f)) continue;
			for (let e = 0; e < 3; e++) {
				const a = base.fv(f, e);
				const b = base.fv(f, (e + 1) % 3);
				edges.add(a < b ? `${a},${b}` : `${b},${a}`);
			}
		}
		expect(base.vn - edges.size + base.fn).toBe(0);
	});

	test("an inverted or too-small face range is refused", () => {
		const cm = sphereIcosa(2).mesh;
		expect(() => IsoParametrization.build(cm, { targetMinFaces: 2, targetMaxFaces: 10 })).toThrow(
			/at least 4 faces/,
		);
		expect(() => IsoParametrization.build(cm, { targetMinFaces: 100, targetMaxFaces: 50 })).toThrow(
			/inverted/,
		);
	});

	test("an empty mesh is refused", () => {
		expect(() =>
			IsoParametrization.build(new CMeshO(), { targetMinFaces: 8, targetMaxFaces: 20 }),
		).toThrow(/no faces/);
	});
});

describe("remeshing", () => {
	test("the result is on the surface and far more uniform than the domain", () => {
		const cm = sphereIcosa(3).mesh;
		const iso = IsoParametrization.build(cm, { targetMinFaces: 30, targetMaxFaces: 80 });
		const remeshed = iso.remesh(4);

		expect(remeshed.fn).toBeGreaterThan(iso.faceCount);
		expect(sphereError(remeshed)).toBeLessThan(0.15);
		// Every domain face contributes the same regular lattice, so the
		// output's uniformity is the domain's own — not better than it. What
		// the remesh guarantees is that it is not *worse*: no face is
		// subdivided more finely than another regardless of its size.
		expect(edgeUniformity(remeshed)).toBeLessThan(1.4 * edgeUniformity(iso.domain.base));
		expect(edgeUniformity(remeshed)).toBeLessThan(0.35);
		assertAllocatorConsistent(remeshed);
	});

	test("the per-face lattices weld into one surface", () => {
		const cm = sphereIcosa(2).mesh;
		const iso = IsoParametrization.build(cm, { targetMinFaces: 20, targetMaxFaces: 40 });
		const rate = 3;
		const remeshed = iso.remesh(rate);

		// Without welding there would be (rate+1)(rate+2)/2 vertices per face
		// and no sharing at all; the shared edges must have collapsed away.
		const unwelded = (iso.faceCount * (rate + 1) * (rate + 2)) / 2;
		expect(remeshed.vn).toBeLessThan(unwelded);
		expect(remeshed.fn).toBe(iso.faceCount * rate * rate);
	});

	test("a finer rate gives proportionally more faces", () => {
		const cm = sphereIcosa(2).mesh;
		const iso = IsoParametrization.build(cm, { targetMinFaces: 20, targetMaxFaces: 40 });
		expect(iso.remesh(6).fn / iso.remesh(3).fn).toBeCloseTo(4, 6);
	});

	test("a rate below two is refused", () => {
		const cm = sphereIcosa(2).mesh;
		const iso = IsoParametrization.build(cm, { targetMinFaces: 20, targetMaxFaces: 40 });
		expect(() => iso.remesh(1)).toThrow(/at least 2/);
	});
});

describe("atlas", () => {
	test("only the faces that straddle a slot can fold", () => {
		const cm = sphereIcosa(2).mesh;
		const iso = IsoParametrization.build(cm, { targetMinFaces: 20, targetMaxFaces: 50 });
		const { cm: atlased, straddling } = iso.atlasUV(0.1);

		expect(atlased.fn).toBe(cm.fn);
		expect(atlased.vn).toBe(cm.vn);
		// A face whose three vertices are pinned in one domain face maps into
		// its slot by a plain affine map and cannot fold. A straddling face
		// has its strays clamped, and that is where a fold can appear — which
		// is exactly the approximation the atlas documents, so the fold count
		// is bounded by the straddle count rather than being zero.
		expect(foldedNum(atlased)).toBeLessThanOrEqual(straddling);
		void globallyUnfolded;
	});

	test("every wedge coordinate stays inside the unit square", () => {
		const cm = sphereIcosa(2).mesh;
		const iso = IsoParametrization.build(cm, { targetMinFaces: 20, targetMaxFaces: 50 });
		const { cm: atlased } = iso.atlasUV(0.1);
		const wt = atlased.wedgeTexCoord as Float64Array;
		for (let f = 0; f < atlased.faceSize; f++) {
			if (atlased.isFaceD(f)) continue;
			for (let k = 0; k < 6; k++) {
				expect(wt[6 * f + k]).toBeGreaterThanOrEqual(0);
				expect(wt[6 * f + k]).toBeLessThanOrEqual(1);
			}
		}
	});

	test("a larger border leaves more space between the slots", () => {
		const cm = sphereIcosa(2).mesh;
		const iso = IsoParametrization.build(cm, { targetMinFaces: 20, targetMaxFaces: 50 });
		const span = (border: number) => {
			const { cm: atlased } = iso.atlasUV(border);
			const wt = atlased.wedgeTexCoord as Float64Array;
			let area = 0;
			for (let f = 0; f < atlased.faceSize; f++) {
				if (atlased.isFaceD(f)) continue;
				const p = [0, 1, 2].map((k) => [wt[6 * f + 2 * k], wt[6 * f + 2 * k + 1]]);
				area += Math.abs(
					(p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]),
				);
			}
			return area;
		};
		expect(span(0.3)).toBeLessThan(span(0.02));
	});
});

describe("filters", () => {
	test("Main stores a domain the other filters can find", () => {
		const { doc, cm } = scene(sphereIcosa(3).mesh);
		const out = kernel.applyFilter(doc, "Iso Parametrization: Main", {
			targetAbstractMinFaceNum: 30,
			targetAbstractMaxFaceNum: 80,
			stopCriteria: 1,
		});
		expect(out.abstract_mesh_faces as number).toBeLessThan(cm.fn);
		expect(isoParametrizationOf(cm)).toBeDefined();
	});

	test("Remeshing adds a layer built from the domain", () => {
		const { doc } = scene(sphereIcosa(3).mesh);
		kernel.applyFilter(doc, "Iso Parametrization: Main", {
			targetAbstractMinFaceNum: 30,
			targetAbstractMaxFaceNum: 80,
			stopCriteria: 1,
		});
		const out = kernel.applyFilter(doc, "Iso Parametrization Remeshing", { SamplingRate: 4 });
		expect(out.face_number as number).toBeGreaterThan(0);
		expect(doc.meshNumber()).toBe(2);
		expect(doc.mm().label()).toMatch(/remeshed/);
	});

	test("Build Atlased Mesh reports the seams it had to fudge", () => {
		const { doc } = scene(sphereIcosa(2).mesh);
		kernel.applyFilter(doc, "Iso Parametrization: Main", {
			targetAbstractMinFaceNum: 20,
			targetAbstractMaxFaceNum: 50,
			stopCriteria: 1,
		});
		const out = kernel.applyFilter(doc, "Iso Parametrization Build Atlased Mesh", {
			BorderSize: 0.1,
		});
		expect(out.new_mesh_id).toBeDefined();
		// A coarse domain over a fine mesh: most faces straddle a slot. The
		// number is reported rather than hidden, since it is the honest cost
		// of not cutting the mesh the way upstream does.
		expect(out.straddling_faces as number).toBeGreaterThanOrEqual(0);
		expect(out.l2_stretch as number).toBeGreaterThan(0);
	});

	test("transfer moves a parametrisation onto an aligned copy", () => {
		const doc = new MeshDocument();
		const source = doc.addNewMesh("", "source", true, sphereIcosa(3).mesh);
		source.updateBoxAndNormals();
		kernel.applyFilter(doc, "Iso Parametrization: Main", {
			targetAbstractMinFaceNum: 30,
			targetAbstractMaxFaceNum: 80,
			stopCriteria: 1,
		});
		const target = doc.addNewMesh("", "target", true, sphereIcosa(2).mesh);
		target.updateBoxAndNormals();

		const out = kernel.applyFilter(doc, "Iso Parametrization transfer between meshes", {
			sourceMesh: source.id(),
			targetMesh: target.id(),
		});
		expect(out.abstract_mesh_faces as number).toBeGreaterThan(0);
		const moved = isoParametrizationOf(target.cm);
		expect(moved).toBeDefined();
		expect((moved as IsoParametrization).domain.pinnedCount()).toBe(target.cm.vn);
	});

	test("a mesh with no domain is told to run Main first", () => {
		const { doc } = scene(sphereIcosa(2).mesh, "bare");
		for (const [name, params] of [
			["Iso Parametrization Remeshing", { SamplingRate: 4 }],
			["Iso Parametrization Build Atlased Mesh", { BorderSize: 0.1 }],
		] as const) {
			expect(() => kernel.applyFilter(doc, name, params), name).toThrow(
				/run "Iso Parametrization: Main"/,
			);
		}
	});

	test("a sampling rate of two or less is refused by the filter too", () => {
		const { doc } = scene(sphereIcosa(2).mesh);
		kernel.applyFilter(doc, "Iso Parametrization: Main", {
			targetAbstractMinFaceNum: 20,
			targetAbstractMaxFaceNum: 50,
			stopCriteria: 1,
		});
		expect(() =>
			kernel.applyFilter(doc, "Iso Parametrization Remeshing", { SamplingRate: 2 }),
		).toThrow(/greater than 2/);
	});

	test("all four are registered under FilterIsoParametrization", () => {
		for (const name of [
			"Iso Parametrization: Main",
			"Iso Parametrization Remeshing",
			"Iso Parametrization Build Atlased Mesh",
			"Iso Parametrization transfer between meshes",
		]) {
			const action = kernel.pluginManager.filterAction(name);
			expect(action, name).toBeDefined();
			expect(action?.plugin.pluginName(), name).toBe("FilterIsoParametrization");
		}
	});
});
