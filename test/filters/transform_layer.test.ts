import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { filterClassToString } from "../../src/common/plugins/filter_class.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { Matrix44Ops } from "../../src/vcg/math/matrix44.ts";
import {
	assertAllocatorConsistent,
	computeFacts,
	geometryDigest,
	signedVolume,
	surfaceArea,
} from "../helpers/invariants.ts";
import { type BuiltMesh, cube, sphereIcosa, tetrahedron } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

function apply(
	built: BuiltMesh | CMeshO,
	name: string,
	params: Record<string, unknown> = {},
): { mesh: CMeshO; doc: MeshDocument; out: Record<string, unknown> } {
	const cm = "mesh" in built ? built.mesh : built;
	const doc = new MeshDocument();
	const m = doc.addNewMesh("test", "test", true, cm);
	m.updateBoxAndNormals();
	const out = kernel.applyFilter(doc, name, params);
	return { mesh: m.cm, doc, out };
}

describe("transforms", () => {
	test("all five are registered with upstream's names", () => {
		for (const [name, python, klass] of [
			["Transform: Scale, Normalize", "compute_matrix_from_scaling_or_normalization", "Normal"],
			["Transform: Translate, Center, set Origin", "compute_matrix_from_translation", "Normal"],
			["Transform: Rotate", "compute_matrix_from_rotation", "Normal"],
			// Rendered in canonical bit order, so Layer (0x200) precedes
			// Normal (0x800) whatever order the C++ source writes them in.
			["Matrix: Freeze Current Matrix", "apply_matrix_freeze", "Layer|Normal"],
			["Matrix: Reset Current Matrix", "set_matrix_identity", "Layer|Normal"],
		] as const) {
			const a = kernel.filterAction(name);
			expect(a.implemented, name).toBe(true);
			expect(a.pythonName, name).toBe(python);
			expect(filterClassToString(a.filterClass), name).toBe(klass);
		}
	});

	test("uniform scaling multiplies volume by the cube of the factor", () => {
		const { mesh } = apply(cube(1), "Transform: Scale, Normalize", { axisX: 3 });
		expect(signedVolume(mesh)).toBeCloseTo(27, 9);
		expect(surfaceArea(mesh)).toBeCloseTo(54, 9);
	});

	test("non-uniform scaling needs uniformFlag off", () => {
		const { mesh } = apply(cube(1), "Transform: Scale, Normalize", {
			axisX: 2,
			axisY: 3,
			axisZ: 4,
			uniformFlag: false,
		});
		expect(signedVolume(mesh)).toBeCloseTo(24, 9);
		expect(mesh.bbox.dimX).toBeCloseTo(2, 9);
		expect(mesh.bbox.dimY).toBeCloseTo(3, 9);
		expect(mesh.bbox.dimZ).toBeCloseTo(4, 9);
	});

	test("uniformFlag on uses the X value for all three axes", () => {
		const { mesh } = apply(cube(1), "Transform: Scale, Normalize", {
			axisX: 2,
			axisY: 99,
			axisZ: 99,
			uniformFlag: true,
		});
		expect(mesh.bbox.dimY).toBeCloseTo(2, 9);
	});

	test("Scale to Unit bbox fits the mesh inside the unit cube", () => {
		const { mesh } = apply(cube(7), "Transform: Scale, Normalize", { unitFlag: true });
		expect(mesh.bbox.maxDim).toBeCloseTo(1, 9);
	});

	test("scaling about the barycentre keeps the barycentre put", () => {
		const shifted = cube(1).mesh;
		for (let v = 0; v < shifted.vn; v++) {
			shifted.setVert(v, shifted.vx(v) + 10, shifted.vy(v), shifted.vz(v));
		}
		const { mesh } = apply(shifted, "Transform: Scale, Normalize", {
			axisX: 2,
			scaleCenter: "barycenter",
		});
		expect(mesh.bbox.center[0]).toBeCloseTo(10, 9);
	});

	test("translation moves the box and leaves the volume alone", () => {
		const { mesh } = apply(cube(2), "Transform: Translate, Center, set Origin", {
			axisX: 5,
			axisY: -3,
			axisZ: 1,
		});
		expect(mesh.bbox.center).toEqual([5, -3, 1]);
		expect(signedVolume(mesh)).toBeCloseTo(8, 9);
	});

	test("Center on Layer BBox puts the box centre at the origin", () => {
		const shifted = cube(2).mesh;
		for (let v = 0; v < shifted.vn; v++) {
			shifted.setVert(v, shifted.vx(v) + 10, shifted.vy(v) + 20, shifted.vz(v) + 30);
		}
		const { mesh } = apply(shifted, "Transform: Translate, Center, set Origin", {
			traslMethod: "Center on Layer BBox",
		});
		for (const c of mesh.bbox.center) expect(c).toBeCloseTo(0, 9);
	});

	test("rotation preserves volume, area and the topology", () => {
		const before = computeFacts(cube(2).mesh);
		const { mesh } = apply(cube(2), "Transform: Rotate", { rotAxis: "Z axis", angle: 37 });
		expect(signedVolume(mesh)).toBeCloseTo(8, 9);
		expect(surfaceArea(mesh)).toBeCloseTo(24, 9);
		expect(computeFacts(mesh).genus).toBe(before.genus);
	});

	test("rotating 90 degrees about Z swaps the x and y extents", () => {
		const box = cube(1).mesh;
		for (let v = 0; v < box.vn; v++) box.setVert(v, box.vx(v) * 4, box.vy(v), box.vz(v));
		const { mesh } = apply(box, "Transform: Rotate", { rotAxis: "Z axis", angle: 90 });
		expect(mesh.bbox.dimX).toBeCloseTo(1, 9);
		expect(mesh.bbox.dimY).toBeCloseTo(4, 9);
	});

	test("four 90-degree rotations return to the start", () => {
		const original = geometryDigest(sphereIcosa(1).mesh);
		let cm = sphereIcosa(1).mesh;
		for (let i = 0; i < 4; i++) {
			cm = apply(cm, "Transform: Rotate", { rotAxis: "X axis", angle: 90 }).mesh;
		}
		expect(geometryDigest(cm)).toBe(original);
	});

	test("a custom axis rotation works", () => {
		const { mesh } = apply(cube(2), "Transform: Rotate", {
			rotAxis: "custom axis",
			customAxis: [1, 1, 1],
			angle: 120,
		});
		// A 120-degree turn about the cube's body diagonal permutes its eight
		// corners among themselves, so the *vertex set* comes back identical.
		// The triangulation does not: each square face gets split along the
		// other diagonal, so comparing the full digest would be asserting
		// something that is not true.
		const cornerSet = (cm: CMeshO) =>
			Array.from({ length: cm.vn }, (_, v) =>
				[cm.vx(v), cm.vy(v), cm.vz(v)].map((x) => Number(x.toFixed(9))).join(","),
			).sort();
		expect(cornerSet(mesh)).toEqual(cornerSet(cube(2).mesh));
		expect(signedVolume(mesh)).toBeCloseTo(8, 9);
	});

	test("Freeze=false leaves the coordinates alone and records the matrix", () => {
		const before = geometryDigest(cube(1).mesh);
		const { mesh } = apply(cube(1), "Transform: Scale, Normalize", { axisX: 3, Freeze: false });
		expect(geometryDigest(mesh)).toBe(before);
		expect(Matrix44Ops.isIdentity(mesh.transformMatrix)).toBe(false);
		expect(mesh.transformMatrix[0]).toBeCloseTo(3, 9);
	});

	test("Matrix: Freeze bakes the pending matrix into the coordinates", () => {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("t", "t", true, cube(1).mesh);
		m.updateBoxAndNormals();
		kernel.applyFilter(doc, "Transform: Scale, Normalize", { axisX: 3, Freeze: false });
		expect(signedVolume(m.cm)).toBeCloseTo(1, 9); // not applied yet
		kernel.applyFilter(doc, "Matrix: Freeze Current Matrix");
		expect(signedVolume(m.cm)).toBeCloseTo(27, 9);
		expect(Matrix44Ops.isIdentity(m.cm.transformMatrix)).toBe(true);
	});

	test("Matrix: Reset discards the pending matrix without touching geometry", () => {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("t", "t", true, cube(1).mesh);
		m.updateBoxAndNormals();
		kernel.applyFilter(doc, "Transform: Scale, Normalize", { axisX: 3, Freeze: false });
		kernel.applyFilter(doc, "Matrix: Reset Current Matrix");
		expect(Matrix44Ops.isIdentity(m.cm.transformMatrix)).toBe(true);
		expect(signedVolume(m.cm)).toBeCloseTo(1, 9);
	});

	test("a mirroring scale reverses the winding so the solid stays outward", () => {
		// Negative scale on one axis is a reflection: the geometry mirrors but
		// the winding does not, which would turn a printable solid inside out.
		const { mesh } = apply(cube(2), "Transform: Scale, Normalize", {
			axisX: -1,
			uniformFlag: false,
		});
		expect(signedVolume(mesh)).toBeGreaterThan(0);
		expect(signedVolume(mesh)).toBeCloseTo(8, 9);
		expect(computeFacts(mesh).coherentlyOriented).toBe(true);
	});

	test("transforms leave the mesh structurally consistent", () => {
		for (const [name, params] of [
			["Transform: Scale, Normalize", { axisX: 2 }],
			["Transform: Translate, Center, set Origin", { axisX: 1 }],
			["Transform: Rotate", { rotAxis: "Y axis", angle: 45 }],
		] as const) {
			const { mesh } = apply(sphereIcosa(2), name, params);
			assertAllocatorConsistent(mesh, name);
			expect(computeFacts(mesh).watertight, name).toBe(true);
		}
	});
});

describe("filter_layer", () => {
	test("all four are registered as Layer", () => {
		for (const name of [
			"Flatten Visible Layers",
			"Duplicate Current layer",
			"Delete Current Mesh",
			"Rename Current Mesh",
		]) {
			const a = kernel.filterAction(name);
			expect(a.implemented, name).toBe(true);
			expect(filterClassToString(a.filterClass), name).toBe("Layer");
		}
	});

	test("Flatten Visible Layers has VARIABLE arity, unlike everything else so far", () => {
		expect(kernel.filterAction("Flatten Visible Layers").arity).toBe(3);
	});

	test("flattening merges every layer into one", () => {
		const doc = new MeshDocument();
		for (const built of [cube(1), tetrahedron(), sphereIcosa(1)]) {
			doc.addNewMesh("", built.name, true, built.mesh).updateBoxAndNormals();
		}
		const totalFaces = doc.fn();
		const out = kernel.applyFilter(doc, "Flatten Visible Layers");
		expect(out.merged_layers).toBe(3);
		expect(doc.meshNumber()).toBe(1);
		expect(doc.mm().cm.fn).toBe(totalFaces);
		expect(doc.mm().label()).toBe("Merged Mesh");
		assertAllocatorConsistent(doc.mm().cm);
	});

	test("DeleteLayer=false keeps the sources alongside the merge", () => {
		const doc = new MeshDocument();
		doc.addNewMesh("", "a", true, cube(1).mesh).updateBoxAndNormals();
		doc.addNewMesh("", "b", true, tetrahedron().mesh).updateBoxAndNormals();
		kernel.applyFilter(doc, "Flatten Visible Layers", { DeleteLayer: false });
		expect(doc.meshNumber()).toBe(3);
	});

	test("MergeVisible=true skips the hidden layers", () => {
		const doc = new MeshDocument();
		const hidden = doc.addNewMesh("", "hidden", true, cube(1).mesh);
		hidden.updateBoxAndNormals();
		hidden.setVisible(false);
		doc.addNewMesh("", "shown", true, tetrahedron().mesh).updateBoxAndNormals();

		const out = kernel.applyFilter(doc, "Flatten Visible Layers", { MergeVisible: true });
		expect(out.merged_layers).toBe(1);
		expect(out.faces).toBe(4); // the tetrahedron alone
	});

	test("MergeVertices welds coincident vertices across layers", () => {
		const doc = new MeshDocument();
		// Two copies of the same cube: welding should halve the vertex count.
		doc.addNewMesh("", "a", true, cube(1).mesh).updateBoxAndNormals();
		doc.addNewMesh("", "b", true, cube(1).mesh).updateBoxAndNormals();

		const welded = kernel.applyFilter(doc, "Flatten Visible Layers", { MergeVertices: true });
		expect(welded.vertices).toBe(8);
		expect(welded.faces).toBe(24); // faces are not deduplicated, only vertices
	});

	test("MergeVertices=false leaves both copies' vertices", () => {
		const doc = new MeshDocument();
		doc.addNewMesh("", "a", true, cube(1).mesh).updateBoxAndNormals();
		doc.addNewMesh("", "b", true, cube(1).mesh).updateBoxAndNormals();
		const out = kernel.applyFilter(doc, "Flatten Visible Layers", { MergeVertices: false });
		expect(out.vertices).toBe(16);
	});

	test("Duplicate makes an independent copy", () => {
		const doc = new MeshDocument();
		const original = doc.addNewMesh("", "orig", true, cube(1).mesh);
		original.updateBoxAndNormals();
		kernel.applyFilter(doc, "Duplicate Current layer");

		expect(doc.meshNumber()).toBe(2);
		const copy = doc.mm();
		expect(copy.label()).toBe("copy of orig");
		expect(geometryDigest(copy.cm)).toBe(geometryDigest(original.cm));

		// Independent: editing one must not touch the other.
		copy.cm.setVert(0, 99, 99, 99);
		expect(original.cm.vx(0)).not.toBe(99);
	});

	test("Delete removes the layer and moves the current pointer", () => {
		const doc = new MeshDocument();
		const keep = doc.addNewMesh("", "keep", true, cube(1).mesh);
		keep.updateBoxAndNormals();
		const victim = doc.addNewMesh("", "victim", true, tetrahedron().mesh);
		victim.updateBoxAndNormals();

		const out = kernel.applyFilter(doc, "Delete Current Mesh");
		expect(out.deleted_mesh_id).toBe(victim.id());
		expect(doc.meshNumber()).toBe(1);
		expect(doc.mm().label()).toBe("keep");
	});

	test("Rename changes the label and nothing else", () => {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("", "before", true, cube(1).mesh);
		m.updateBoxAndNormals();
		const before = geometryDigest(m.cm);
		const out = kernel.applyFilter(doc, "Rename Current Mesh", { newName: "after" });
		expect(out.name).toBe("after");
		expect(m.label()).toBe("after");
		expect(geometryDigest(m.cm)).toBe(before);
	});

	test("flattening two disjoint solids gives two components", () => {
		const doc = new MeshDocument();
		doc.addNewMesh("", "a", true, cube(1).mesh).updateBoxAndNormals();
		const far = tetrahedron().mesh;
		for (let v = 0; v < far.vn; v++) far.setVert(v, far.vx(v) + 100, far.vy(v), far.vz(v));
		doc.addNewMesh("", "b", true, far).updateBoxAndNormals();

		kernel.applyFilter(doc, "Flatten Visible Layers");
		const facts = computeFacts(doc.mm().cm);
		expect(facts.components).toBe(2);
		expect(facts.watertight).toBe(true);
	});
});
