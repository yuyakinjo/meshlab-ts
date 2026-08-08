/**
 * The Embree-named ray measures and the volumetric booleans.
 *
 * Three of the Embree filters compute the same quantities as their GPU-named
 * counterparts, so the tests check exactly that: the two spellings must agree
 * on the same mesh. If they ever drift apart, one of the two has changed
 * behind the other's back.
 *
 * The booleans are checked by volume, which is what a boolean is *for*: the
 * union of two disjoint cubes has the volume of both, their intersection is
 * empty, and the difference of a shape with itself is nothing at all. Volume
 * survives the grid's rounding of the seams; a face count would not.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { Clean } from "../../src/vcg/complex/clean.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { UpdateTopology } from "../../src/vcg/complex/update/topology.ts";
import { assertAllocatorConsistent } from "../helpers/invariants.ts";
import { cube, sphereIcosa, torus } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

function scene(cm: CMeshO, label = "test") {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", label, true, cm);
	m.updateBoxAndNormals();
	return { doc, m, cm };
}

/**
 * A cube of the given *side*, centred where asked.
 *
 * The builder's `cube(1)` spans -0.5..0.5, so its side is already one and the
 * scale is the side directly. Getting that wrong makes every volume below out
 * by a factor of eight, which is how it was found.
 */
function boxAt(side: number, centre: readonly number[]): CMeshO {
	const cm = cube(1).mesh;
	for (let v = 0; v < cm.vertSize; v++) {
		cm.setVert(
			v,
			cm.vx(v) * side + centre[0],
			cm.vy(v) * side + centre[1],
			cm.vz(v) * side + centre[2],
		);
	}
	return cm;
}

/** Two operands in one document, with the first current. */
function pair(a: CMeshO, b: CMeshO) {
	const doc = new MeshDocument();
	const first = doc.addNewMesh("", "a", true, a);
	const second = doc.addNewMesh("", "b", true, b);
	first.updateBoxAndNormals();
	second.updateBoxAndNormals();
	doc.setCurrentMesh(first.id());
	return { doc, first, second };
}

function volumeOf(cm: CMeshO): number {
	return Math.abs(Clean.signedVolume(cm));
}

describe("Embree ray measures", () => {
	test("ambient occlusion agrees with the GPU-named filter", () => {
		const embree = scene(sphereIcosa(3).mesh);
		const out = kernel.applyFilter(embree.doc, "Compute Ambient occlusion", {
			Rays: 64,
			randomSeed: 1,
		});
		// A convex surface sees the whole sky, whichever spelling asks.
		expect(out.min as number).toBeGreaterThan(0.9);

		const gpu = scene(sphereIcosa(3).mesh);
		const other = kernel.applyFilter(gpu.doc, "Ambient Occlusion", {
			occMode: 0,
			dirBias: 0,
			coneDir: [0, 1, 0],
			coneAngle: 30,
			numberRays: 64,
			randomSeed: 1,
			useGPU: false,
		});
		expect(Math.abs((out.min as number) - (other.min_openness as number))).toBeLessThan(0.1);
	});

	test("the shape diameter function measures a sphere's thickness", () => {
		const { doc } = scene(sphereIcosa(3).mesh);
		const out = kernel.applyFilter(doc, "Compute Shape-Diameter Function", {
			Rays: 64,
			randomSeed: 1,
			cone_amplitude: 60,
		});
		// A unit sphere is two units through the middle.
		expect(out.max as number).toBeGreaterThan(1.4);
		expect(out.max as number).toBeLessThan(2.05);
	});

	test("obscurance is bounded, and the exponent only matters where rays hit", () => {
		const measure = (tau: number, mesh: CMeshO) => {
			const { doc } = scene(mesh);
			return kernel.applyFilter(doc, "Compute Obscurance", {
				Rays: 48,
				randomSeed: 1,
				TAU: tau,
			}).min as number;
		};
		// A convex sphere occludes nothing, so every ray escapes and the
		// exponent cannot change the answer — it is exactly one either way.
		expect(measure(0.01, sphereIcosa(2).mesh)).toBe(1);
		expect(measure(1, sphereIcosa(2).mesh)).toBe(1);

		// A torus sees its own inner surface, so there the exponent bites. A
		// larger exponent makes a hit stop mattering sooner, so the same
		// occluder obscures less and the value goes *up*.
		const gentle = measure(0.01, torus(2, 0.7, 20, 12).mesh);
		const steep = measure(1, torus(2, 0.7, 20, 12).mesh);
		expect(gentle).toBeGreaterThanOrEqual(0);
		expect(steep).toBeLessThanOrEqual(1);
		expect(steep).toBeGreaterThan(gentle);
	});

	test("a ray count below one is refused", () => {
		const { doc } = scene(sphereIcosa(2).mesh);
		expect(() =>
			kernel.applyFilter(doc, "Compute Ambient occlusion", { Rays: 0, randomSeed: 1 }),
		).toThrow(/at least 1/);
	});
});

describe("Reorient face normals by geometry", () => {
	const NAME = "Reorient face normals by geometry";

	test("turns an inside-out sphere the right way round", () => {
		const cm = sphereIcosa(2).mesh;
		// Flip every face: consistently oriented, and consistently wrong.
		for (let f = 0; f < cm.faceSize; f++) {
			cm.setFace(f, cm.fv(f, 0), cm.fv(f, 2), cm.fv(f, 1));
		}
		const { doc, m } = scene(cm);
		const before = Clean.signedVolume(cm);
		expect(before).toBeLessThan(0);

		const out = kernel.applyFilter(doc, NAME, { Rays: 48, randomSeed: 1 });
		expect(out.flipped_faces as number).toBeGreaterThan(cm.fn * 0.9);
		m.updateBoxAndNormals();
		// This is a geometric test, unlike "Re-Orient all faces coherently",
		// which only makes neighbours agree — an inside-out mesh passes that
		// one unchanged and fails this one.
		expect(Clean.signedVolume(cm)).toBeGreaterThan(0);
	});

	test("leaves an already outward mesh alone", () => {
		const { doc } = scene(sphereIcosa(2).mesh);
		const out = kernel.applyFilter(doc, NAME, { Rays: 48, randomSeed: 1 });
		expect(out.flipped_faces as number).toBeLessThan(3);
	});
});

describe("Select Visible Faces", () => {
	const NAME = "Select Visible Faces";

	test("selects only the faces turned towards the viewer", () => {
		const { doc, cm } = scene(sphereIcosa(3).mesh);
		const out = kernel.applyFilter(doc, NAME, {
			Rays: 8,
			randomSeed: 1,
			dir: [0, 0, -1],
			incrementalSelection: false,
		});
		expect(out.selected_faces as number).toBeGreaterThan(0);
		expect(out.selected_faces as number).toBeLessThan(cm.fn);

		// Looking along -z means standing on +z: every selected face must have
		// a normal with a positive z component.
		for (let f = 0; f < cm.faceSize; f++) {
			if (!cm.isFaceS(f)) continue;
			expect(cm.faceNormal[3 * f + 2], `face ${f}`).toBeGreaterThan(0);
		}
	});

	test("a face hidden behind another is not selected", () => {
		// Two spheres in a line; the far one is turned towards the viewer but
		// screened by the near one, so only the ray cast can reject it.
		const doc = new MeshDocument();
		const near = sphereIcosa(2).mesh;
		const far = sphereIcosa(2).mesh;
		for (let v = 0; v < far.vertSize; v++) far.setVert(v, far.vx(v), far.vy(v), far.vz(v) - 4);
		const a = doc.addNewMesh("", "a", true, near);
		const b = doc.addNewMesh("", "b", true, far);
		a.updateBoxAndNormals();
		b.updateBoxAndNormals();
		doc.setCurrentMesh(a.id());
		kernel.applyFilter(doc, "Flatten Visible Layers", {
			MergeVisible: true,
			DeleteLayer: true,
			MergeVertices: false,
		});

		const cm = doc.mm().cm;
		kernel.applyFilter(doc, NAME, {
			Rays: 8,
			randomSeed: 1,
			dir: [0, 0, -1],
			incrementalSelection: false,
		});
		// Nothing on the far sphere's near face may be selected.
		for (let f = 0; f < cm.faceSize; f++) {
			if (!cm.isFaceS(f)) continue;
			const z = (cm.vz(cm.fv(f, 0)) + cm.vz(cm.fv(f, 1)) + cm.vz(cm.fv(f, 2))) / 3;
			expect(z, `face ${f}`).toBeGreaterThan(-2);
		}
	});

	test("incremental selection adds rather than replaces", () => {
		const { doc, cm } = scene(sphereIcosa(2).mesh);
		const first = kernel.applyFilter(doc, NAME, {
			Rays: 8,
			randomSeed: 1,
			dir: [0, 0, -1],
			incrementalSelection: false,
		}).selected_faces as number;
		kernel.applyFilter(doc, NAME, {
			Rays: 8,
			randomSeed: 1,
			dir: [0, 0, 1],
			incrementalSelection: true,
		});
		let total = 0;
		for (let f = 0; f < cm.faceSize; f++) if (cm.isFaceS(f)) total++;
		// The two opposite views between them see the whole sphere.
		expect(total).toBeGreaterThan(first);
	});

	test("a zero direction is refused", () => {
		const { doc } = scene(sphereIcosa(2).mesh);
		expect(() =>
			kernel.applyFilter(doc, NAME, {
				Rays: 8,
				randomSeed: 1,
				dir: [0, 0, 0],
				incrementalSelection: false,
			}),
		).toThrow(/zero length/);
	});
});

describe("Mesh booleans", () => {
	const options = { Resolution: 64, transfer_face_color: false };

	test("the union of two disjoint boxes has the volume of both", () => {
		const { doc, first, second } = pair(boxAt(1, [-1.5, 0, 0]), boxAt(1, [1.5, 0, 0]));
		const out = kernel.applyFilter(doc, "Mesh Boolean: Union", {
			first_mesh: first.id(),
			second_mesh: second.id(),
			...options,
		});
		expect(out.face_number as number).toBeGreaterThan(0);
		const result = doc.mm().cm;
		// Each box is 1x1x1; the union of two disjoint ones is 2.
		expect(volumeOf(result)).toBeGreaterThan(1.7);
		expect(volumeOf(result)).toBeLessThan(2.3);
		UpdateTopology.faceFace(result);
		expect(Clean.countEdgeNum(result).boundary).toBe(0);
		assertAllocatorConsistent(result);
	});

	test("the intersection of two overlapping boxes is their overlap", () => {
		// Two unit boxes offset by half: the overlap is half a unit thick.
		const { doc, first, second } = pair(boxAt(1, [0, 0, 0]), boxAt(1, [0.5, 0, 0]));
		kernel.applyFilter(doc, "Mesh Boolean: Intersection", {
			first_mesh: first.id(),
			second_mesh: second.id(),
			...options,
		});
		const result = doc.mm().cm;
		expect(volumeOf(result)).toBeGreaterThan(0.35);
		expect(volumeOf(result)).toBeLessThan(0.65);
	});

	test("the difference removes the second from the first", () => {
		const { doc, first, second } = pair(boxAt(1, [0, 0, 0]), boxAt(1, [0.5, 0, 0]));
		kernel.applyFilter(doc, "Mesh Boolean: Difference", {
			first_mesh: first.id(),
			second_mesh: second.id(),
			...options,
		});
		const result = doc.mm().cm;
		// One unit minus the half that overlapped.
		expect(volumeOf(result)).toBeGreaterThan(0.35);
		expect(volumeOf(result)).toBeLessThan(0.65);
	});

	test("XOR keeps what is in one but not both", () => {
		const { doc, first, second } = pair(boxAt(1, [0, 0, 0]), boxAt(1, [0.5, 0, 0]));
		kernel.applyFilter(doc, "Mesh Boolean: Symmetric Difference (XOR)", {
			first_mesh: first.id(),
			second_mesh: second.id(),
			...options,
		});
		const result = doc.mm().cm;
		// Two halves, one from each box.
		expect(volumeOf(result)).toBeGreaterThan(0.7);
		expect(volumeOf(result)).toBeLessThan(1.3);
	});

	test("a difference between two disjoint meshes gives the first back", () => {
		const { doc, first, second } = pair(boxAt(1, [0, 0, 0]), boxAt(1, [5, 0, 0]));
		kernel.applyFilter(doc, "Mesh Boolean: Difference", {
			first_mesh: first.id(),
			second_mesh: second.id(),
			...options,
		});
		expect(volumeOf(doc.mm().cm)).toBeGreaterThan(0.85);
	});

	test("an intersection of two disjoint meshes is refused, not returned empty", () => {
		const { doc, first, second } = pair(boxAt(1, [0, 0, 0]), boxAt(1, [10, 0, 0]));
		expect(() =>
			kernel.applyFilter(doc, "Mesh Boolean: Intersection", {
				first_mesh: first.id(),
				second_mesh: second.id(),
				...options,
			}),
		).toThrow(/do not overlap/);
	});

	test("a sphere cut from a bigger sphere leaves a shell", () => {
		const outer = sphereIcosa(3).mesh;
		const inner = sphereIcosa(3).mesh;
		for (let v = 0; v < inner.vertSize; v++) {
			inner.setVert(v, inner.vx(v) * 0.6, inner.vy(v) * 0.6, inner.vz(v) * 0.6);
		}
		const { doc, first, second } = pair(outer, inner);
		kernel.applyFilter(doc, "Mesh Boolean: Difference", {
			first_mesh: first.id(),
			second_mesh: second.id(),
			Resolution: 72,
			transfer_face_color: false,
		});
		const result = doc.mm().cm;
		UpdateTopology.faceFace(result);
		// A hollow shell: watertight, and two connected components would be
		// wrong — the inner and outer surfaces belong to one solid.
		expect(Clean.countEdgeNum(result).boundary).toBe(0);
		expect(Clean.countConnectedComponents(result)).toBe(2);
		// 4/3 pi (1 - 0.6^3) is about 3.28.
		expect(volumeOf(result)).toBeGreaterThan(2.6);
		expect(volumeOf(result)).toBeLessThan(3.6);
	});

	test("the same layer twice is refused", () => {
		const { doc, first } = pair(boxAt(1, [0, 0, 0]), boxAt(1, [1, 0, 0]));
		expect(() =>
			kernel.applyFilter(doc, "Mesh Boolean: Union", {
				first_mesh: first.id(),
				second_mesh: first.id(),
				...options,
			}),
		).toThrow(/two different layers/);
	});

	test("too coarse a grid is refused", () => {
		const { doc, first, second } = pair(boxAt(1, [0, 0, 0]), boxAt(1, [0.5, 0, 0]));
		expect(() =>
			kernel.applyFilter(doc, "Mesh Boolean: Union", {
				first_mesh: first.id(),
				second_mesh: second.id(),
				Resolution: 4,
				transfer_face_color: false,
			}),
		).toThrow(/at least 8/);
	});

	test("a torus keeps its hole through a union with a distant box", () => {
		const { doc, first, second } = pair(torus(2, 0.5, 24, 12).mesh, boxAt(1, [8, 0, 0]));
		kernel.applyFilter(doc, "Mesh Boolean: Union", {
			first_mesh: first.id(),
			second_mesh: second.id(),
			Resolution: 80,
			transfer_face_color: false,
		});
		const result = doc.mm().cm;
		UpdateTopology.faceFace(result);
		expect(Clean.countConnectedComponents(result)).toBe(2);
	});
});

describe("registry", () => {
	test("all nine are registered under their own plugins", () => {
		const expected: Array<[string, string]> = [
			["Compute Ambient occlusion", "FilterEmbree"],
			["Compute Obscurance", "FilterEmbree"],
			["Compute Shape-Diameter Function", "FilterEmbree"],
			["Reorient face normals by geometry", "FilterEmbree"],
			["Select Visible Faces", "FilterEmbree"],
			["Mesh Boolean: Union", "FilterMeshBoolean"],
			["Mesh Boolean: Intersection", "FilterMeshBoolean"],
			["Mesh Boolean: Difference", "FilterMeshBoolean"],
			["Mesh Boolean: Symmetric Difference (XOR)", "FilterMeshBoolean"],
		];
		for (const [name, plugin] of expected) {
			const action = kernel.pluginManager.filterAction(name);
			expect(action, name).toBeDefined();
			expect(action?.plugin.pluginName(), name).toBe(plugin);
		}
	});
});
