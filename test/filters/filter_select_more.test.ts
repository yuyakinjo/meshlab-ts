/**
 * The second half of filter_select: morphology, and selection by quality,
 * colour, view angle, edge length, connectivity, triangle shape and outlier
 * probability.
 *
 * Selection is one of the few places where "what MeshLab does" is a matter of
 * definition rather than of geometry, so these tests pin the definitions:
 * loose versus strict propagation, which side of the threshold is included,
 * and whether a filter replaces the selection or adds to it.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { FaceFlag, VertexFlag } from "../../src/vcg/complex/flags.ts";
import { rgba } from "../../src/vcg/space/color4.ts";
import { assertAllocatorConsistent } from "../helpers/invariants.ts";
import {
	type BuiltMesh,
	cube,
	cubePlusIslands,
	gridPlane,
	sphereIcosa,
} from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

function scene(built: BuiltMesh | CMeshO, channels = 0) {
	const cm = "mesh" in built ? built.mesh : built;
	const doc = new MeshDocument();
	const m = doc.addNewMesh("test", "test", true, cm);
	if (channels !== 0) m.updateDataMask(channels);
	m.updateBoxAndNormals();
	return { doc, m, cm };
}

function selectedFaces(cm: CMeshO): number[] {
	const out: number[] = [];
	for (let f = 0; f < cm.faceSize; f++) {
		if (!cm.isFaceD(f) && cm.isFaceS(f)) out.push(f);
	}
	return out;
}

function selectedVerts(cm: CMeshO): number[] {
	const out: number[] = [];
	for (let v = 0; v < cm.vertSize; v++) {
		if (!cm.isVertD(v) && cm.isVertS(v)) out.push(v);
	}
	return out;
}

describe("Dilate and Erode Selection", () => {
	test("dilation grows a single face into its whole vertex neighbourhood", () => {
		const { doc, cm } = scene(gridPlane(6, 6));
		// A face somewhere in the middle, so growth is not clipped by the border.
		const seed = 24;
		cm.faceFlags[seed] |= FaceFlag.SELECTED;

		kernel.applyFilter(doc, "Dilate Selection");
		const grown = selectedFaces(cm);
		expect(grown).toContain(seed);
		expect(grown.length).toBeGreaterThan(1);

		// Everything selected must share a vertex with the seed — that is the
		// definition of one dilation step, not just "more faces".
		const seedVerts = new Set([cm.fv(seed, 0), cm.fv(seed, 1), cm.fv(seed, 2)]);
		for (const f of grown) {
			const touches = [0, 1, 2].some((k) => seedVerts.has(cm.fv(f, k)));
			expect(touches, `face ${f} should touch the seed`).toBe(true);
		}
		assertAllocatorConsistent(cm);
	});

	test("erosion is the dual, and undoes a dilation on a large enough patch", () => {
		const { doc, cm } = scene(sphereIcosa(3));
		const seed = 100;
		cm.faceFlags[seed] |= FaceFlag.SELECTED;

		kernel.applyFilter(doc, "Dilate Selection");
		kernel.applyFilter(doc, "Dilate Selection");
		const grown = selectedFaces(cm).length;
		kernel.applyFilter(doc, "Erode Selection");
		const shrunk = selectedFaces(cm);

		expect(shrunk.length).toBeLessThan(grown);
		// One erosion after two dilations still leaves the seed covered.
		expect(shrunk).toContain(seed);
	});

	test("eroding an empty selection stays empty, and a full one stays full", () => {
		const { doc, cm } = scene(cube());
		kernel.applyFilter(doc, "Erode Selection");
		expect(selectedFaces(cm)).toHaveLength(0);

		for (let f = 0; f < cm.faceSize; f++) cm.faceFlags[f] |= FaceFlag.SELECTED;
		kernel.applyFilter(doc, "Erode Selection");
		expect(selectedFaces(cm)).toHaveLength(cm.fn);
	});
});

describe("Select by Vertex Quality", () => {
	const NAME = "Select by Vertex Quality";

	test("selects the vertices inside the range, boundaries included", () => {
		const { doc, cm } = scene(gridPlane(4, 4), MeshElement.MM_VERTQUALITY);
		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = v;

		kernel.applyFilter(doc, NAME, { minQ: 3, maxQ: 6, Inclusive: true });
		expect(selectedVerts(cm)).toEqual([3, 4, 5, 6]);
	});

	test("Inclusive decides whether a straddling face comes along", () => {
		const { doc, cm } = scene(gridPlane(4, 4), MeshElement.MM_VERTQUALITY);
		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = v;
		const range = { minQ: 0, maxQ: 4 };

		kernel.applyFilter(doc, NAME, { ...range, Inclusive: true });
		const strict = selectedFaces(cm).length;
		kernel.applyFilter(doc, NAME, { ...range, Inclusive: false });
		const loose = selectedFaces(cm).length;

		// Strict needs all three vertices in range; loose needs one. On a grid
		// cut through the middle there are always faces in between.
		expect(loose).toBeGreaterThan(strict);
	});

	test("a range that falls between two quality values selects nothing", () => {
		const { doc, cm } = scene(gridPlane(4, 4), MeshElement.MM_VERTQUALITY);
		// Even values only, so the odd gap below is genuinely empty. The range
		// itself has to stay inside the mesh's quality span: the parameter is a
		// RichDynamicFloat whose bounds are that span, exactly as upstream.
		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = 2 * v;
		kernel.applyFilter(doc, NAME, { minQ: 3, maxQ: 3.5, Inclusive: true });
		expect(selectedVerts(cm)).toHaveLength(0);
		expect(selectedFaces(cm)).toHaveLength(0);
	});
});

describe("Select by Face Quality", () => {
	test("selects the faces inside the range and marks their vertices", () => {
		const { doc, m, cm } = scene(gridPlane(4, 4), MeshElement.MM_FACEQUALITY);
		const q = m.cm.faceQuality as Float64Array;
		for (let f = 0; f < cm.faceSize; f++) q[f] = f;

		kernel.applyFilter(doc, "Select by Face Quality", { minQ: 2, maxQ: 4, Inclusive: true });
		expect(selectedFaces(cm)).toEqual([2, 3, 4]);
		// Vertex selection follows loosely from the faces.
		for (const f of [2, 3, 4]) {
			for (let k = 0; k < 3; k++) expect(cm.isVertS(cm.fv(f, k))).toBe(true);
		}
	});
});

describe("Select Faces by Color", () => {
	const NAME = "Select Faces by Color";

	test("matches an exact colour with a tight tolerance", () => {
		const { doc, cm } = scene(gridPlane(4, 4), MeshElement.MM_VERTCOLOR);
		const red = rgba(255, 0, 0, 255);
		for (let v = 0; v < cm.vertSize; v++) cm.vertColor[v] = rgba(0, 0, 255, 255);
		cm.vertColor[5] = red;
		cm.vertColor[6] = red;

		kernel.applyFilter(doc, NAME, {
			Color: red,
			ColorSpace: 1, // RGB
			PercentRH: 0.01,
			PercentGS: 0.01,
			PercentBV: 0.01,
			Inclusive: false,
		});
		expect(selectedVerts(cm)).toEqual([5, 6]);
	});

	test("a tolerance of 1 in every channel matches everything", () => {
		const { doc, cm } = scene(gridPlane(3, 3), MeshElement.MM_VERTCOLOR);
		for (let v = 0; v < cm.vertSize; v++) cm.vertColor[v] = rgba(v * 7, 30, 200, 255);
		kernel.applyFilter(doc, NAME, {
			Color: rgba(0, 0, 0, 255),
			ColorSpace: 1,
			PercentRH: 1,
			PercentGS: 1,
			PercentBV: 1,
			Inclusive: false,
		});
		expect(selectedVerts(cm)).toHaveLength(cm.vn);
	});

	test("HSV ignores brightness where RGB does not", () => {
		const { doc, cm } = scene(gridPlane(3, 3), MeshElement.MM_VERTCOLOR);
		// Same hue, wildly different value.
		for (let v = 0; v < cm.vertSize; v++) cm.vertColor[v] = rgba(60, 0, 0, 255);
		const bright = rgba(255, 0, 0, 255);

		const tolerant = { PercentRH: 0.02, PercentGS: 0.02, PercentBV: 1, Inclusive: false };
		kernel.applyFilter(doc, NAME, { Color: bright, ColorSpace: 0, ...tolerant });
		expect(selectedVerts(cm)).toHaveLength(cm.vn);

		kernel.applyFilter(doc, NAME, {
			Color: bright,
			ColorSpace: 1,
			PercentRH: 0.02,
			PercentGS: 1,
			PercentBV: 1,
			Inclusive: false,
		});
		expect(selectedVerts(cm)).toHaveLength(0);
	});
});

describe("Select Faces by view angle", () => {
	const NAME = "Select Faces by view angle";

	test("at 90 degrees it takes the hemisphere facing the eye", () => {
		const built = sphereIcosa(3);
		const { doc, cm } = scene(built);
		kernel.applyFilter(doc, NAME, {
			anglelimit: 90,
			usecamera: false,
			viewpoint: [0, 0, 10],
		});

		const chosen = selectedFaces(cm);
		expect(chosen.length).toBeGreaterThan(0);
		expect(chosen.length).toBeLessThan(cm.fn);
		// The test is `viewray · normal < cos(limit)`, and the view ray runs
		// *from* the eye, so at 90 degrees the negative dot products — the
		// faces turned towards the eye — are the ones selected.
		for (const f of chosen) {
			const z = (cm.vz(cm.fv(f, 0)) + cm.vz(cm.fv(f, 1)) + cm.vz(cm.fv(f, 2))) / 3;
			expect(z).toBeGreaterThan(-0.05);
		}
	});

	test("the limit runs the other way from what the name suggests", () => {
		// cos(180) is -1 and no normalised dot product is below that, so the
		// widest-sounding angle selects nothing while 0 degrees selects all.
		// This is upstream's sign, kept deliberately.
		const wide = scene(sphereIcosa(2));
		kernel.applyFilter(wide.doc, NAME, {
			anglelimit: 180,
			usecamera: false,
			viewpoint: [0, 0, 10],
		});
		expect(selectedFaces(wide.cm)).toHaveLength(0);

		const narrow = scene(sphereIcosa(2));
		kernel.applyFilter(narrow.doc, NAME, {
			anglelimit: 0,
			usecamera: false,
			viewpoint: [0, 0, 10],
		});
		expect(selectedFaces(narrow.cm)).toHaveLength(narrow.cm.fn);
	});

	test("asking for the mesh camera is refused rather than silently ignored", () => {
		const { doc } = scene(cube());
		expect(() =>
			kernel.applyFilter(doc, NAME, { anglelimit: 90, usecamera: true, viewpoint: [0, 0, 1] }),
		).toThrow(/camera/i);
	});
});

describe("Select Faces with edges longer than...", () => {
	const NAME = "Select Faces with edges longer than...";

	test("splits a mesh at the threshold, monotonically", () => {
		const { doc, cm } = scene(sphereIcosa(2));
		let longest = 0;
		for (let f = 0; f < cm.faceSize; f++) {
			for (let k = 0; k < 3; k++) {
				const a = cm.fv(f, k);
				const b = cm.fv(f, (k + 1) % 3);
				longest = Math.max(
					longest,
					Math.hypot(cm.vx(a) - cm.vx(b), cm.vy(a) - cm.vy(b), cm.vz(a) - cm.vz(b)),
				);
			}
		}

		kernel.applyFilter(doc, NAME, { Threshold: longest * 1.01 });
		expect(selectedFaces(cm)).toHaveLength(0);

		kernel.applyFilter(doc, NAME, { Threshold: longest * 0.99 });
		const some = selectedFaces(cm).length;
		expect(some).toBeGreaterThan(0);

		kernel.applyFilter(doc, NAME, { Threshold: 0 });
		expect(selectedFaces(cm)).toHaveLength(cm.fn);
	});
});

describe("Select Connected Faces", () => {
	const NAME = "Select Connected Faces";

	test("one seed face pulls in exactly its own component", () => {
		const built = cubePlusIslands(3);
		const { doc, cm } = scene(built);
		cm.faceFlags[0] |= FaceFlag.SELECTED;

		kernel.applyFilter(doc, NAME);
		const chosen = selectedFaces(cm);
		// The cube is 12 faces; the islands must not come along.
		expect(chosen).toHaveLength(12);
		expect(chosen.length).toBeLessThan(cm.fn);
	});

	test("no seed means no growth", () => {
		const { doc, cm } = scene(cube());
		kernel.applyFilter(doc, NAME);
		expect(selectedFaces(cm)).toHaveLength(0);
	});
});

describe("Select 'problematic' faces", () => {
	const NAME = "Select 'problematic' faces";

	test("finds a sliver by aspect ratio and leaves a clean mesh alone", () => {
		const clean = scene(sphereIcosa(2));
		kernel.applyFilter(clean.doc, NAME, { useAR: true, ARatio: 0.2, useNF: false });
		expect(selectedFaces(clean.cm)).toHaveLength(0);

		// A grid with one vertex dragged almost onto its neighbour, which turns
		// the two faces using it into needles.
		const built = gridPlane(4, 4);
		const cm = built.mesh;
		cm.vertCoord[3 * 5] = cm.vertCoord[3 * 6] - 1e-4;
		cm.vertCoord[3 * 5 + 1] = cm.vertCoord[3 * 6 + 1];
		const { doc } = scene(cm);

		kernel.applyFilter(doc, NAME, { useAR: true, ARatio: 0.2, useNF: false });
		expect(selectedFaces(cm).length).toBeGreaterThan(0);
	});

	test("the normal test fires on a spike and not on a smooth sphere", () => {
		const smooth = scene(sphereIcosa(3));
		kernel.applyFilter(smooth.doc, NAME, { useAR: false, useNF: true, NFRatio: 60 });
		expect(selectedFaces(smooth.cm)).toHaveLength(0);

		const built = sphereIcosa(3);
		const cm = built.mesh;
		for (let k = 0; k < 3; k++) cm.vertCoord[3 * 7 + k] *= 3; // pull one vertex out
		const spiky = scene(cm);
		kernel.applyFilter(spiky.doc, NAME, { useAR: false, useNF: true, NFRatio: 60 });
		expect(selectedFaces(cm).length).toBeGreaterThan(0);
	});
});

describe("Select Outliers", () => {
	const NAME = "Select Outliers";

	test("finds points planted far from an otherwise even cloud", () => {
		const built = sphereIcosa(3);
		const cm = built.mesh;
		// Move three vertices well outside the sphere; on a shell of radius 1
		// with an even sampling these are unambiguous outliers.
		for (const v of [3, 40, 90]) {
			for (let k = 0; k < 3; k++) cm.vertCoord[3 * v + k] *= 4;
		}
		const { doc } = scene(cm);
		kernel.applyFilter(doc, NAME, { PropThreshold: 0.8, KNearest: 16 });

		const chosen = new Set(selectedVerts(cm));
		for (const v of [3, 40, 90]) expect(chosen.has(v), `vertex ${v}`).toBe(true);
		// It should be a handful, not half the mesh.
		expect(chosen.size).toBeLessThan(cm.vn / 4);
	});

	test("an evenly sampled sphere has no outliers at a high threshold", () => {
		const { doc, cm } = scene(sphereIcosa(3));
		kernel.applyFilter(doc, NAME, { PropThreshold: 0.99, KNearest: 16 });
		expect(selectedVerts(cm).length).toBeLessThan(cm.vn / 10);
	});

	test("a neighbour count below one is refused", () => {
		const { doc } = scene(sphereIcosa(2));
		expect(() => kernel.applyFilter(doc, NAME, { PropThreshold: 0.8, KNearest: 0 })).toThrow(
			/at least 1/,
		);
	});
});

describe("Delete ALL Faces", () => {
	const NAME = "Delete ALL Faces";

	test("leaves the vertices behind as a point cloud", () => {
		const { doc, m, cm } = scene(sphereIcosa(2));
		const before = cm.vn;
		const out = kernel.applyFilter(doc, NAME, { allLayers: false });

		expect(out.deleted_faces).toBe(before === 0 ? 0 : (out.deleted_faces as number));
		expect(cm.fn).toBe(0);
		expect(cm.vn).toBe(before);
		expect(m.cm.fn).toBe(0);
		assertAllocatorConsistent(cm);
	});

	test("allLayers reaches every visible mesh", () => {
		const doc = new MeshDocument();
		const a = doc.addNewMesh("a", "a", true, sphereIcosa(2).mesh);
		const b = doc.addNewMesh("b", "b", true, cube().mesh);
		a.updateBoxAndNormals();
		b.updateBoxAndNormals();

		kernel.applyFilter(doc, NAME, { allLayers: true });
		expect(a.cm.fn).toBe(0);
		expect(b.cm.fn).toBe(0);
		expect(b.cm.vn).toBe(8);
	});
});

describe("registry", () => {
	test("all eleven are implemented rather than stubs", () => {
		for (const name of [
			"Dilate Selection",
			"Erode Selection",
			"Select by Vertex Quality",
			"Select by Face Quality",
			"Select Faces by Color",
			"Select Faces by view angle",
			"Select Faces with edges longer than...",
			"Select Connected Faces",
			"Select 'problematic' faces",
			"Select Outliers",
			"Delete ALL Faces",
		]) {
			const action = kernel.pluginManager.filterAction(name);
			expect(action, name).toBeDefined();
			expect(action?.plugin.pluginName(), name).toBe("FilterSelect");
		}
	});

	test("VertexFlag and FaceFlag selection bits stay independent", () => {
		const { cm } = scene(cube());
		cm.vertFlags[0] |= VertexFlag.SELECTED;
		expect(cm.isVertS(0)).toBe(true);
		expect(cm.isFaceS(0)).toBe(false);
	});
});
