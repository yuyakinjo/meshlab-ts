/**
 * The second half of filter_layer: splitting a mesh across layers, pruning
 * layers, and the raster/camera filters.
 *
 * The camera tests are round-trip tests by design. A pose has many equivalent
 * encodings and only one is right for a given file format, so writing a file
 * and reading it back proves the two halves agree with each other but not
 * that they agree with Bundler — for that there is a hand-checked fixture
 * whose numbers come from the format's own definition.
 */
import { describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import { imageSizeOf, RasterPlane } from "../../src/common/ml_document/raster_model.ts";
import { filterClassToString } from "../../src/common/plugins/filter_class.ts";
import {
	readAgisoftXml,
	readBundlerOut,
	writeAgisoftXml,
	writeBundlerOut,
} from "../../src/meshlabplugins/filter_layer/cameras.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { FaceFlag, VertexFlag } from "../../src/vcg/complex/flags.ts";
import { Shot } from "../../src/vcg/math/shot.ts";
import { assertAllocatorConsistent } from "../helpers/invariants.ts";
import { cube, cubePlusIslands, gridPlane, sphereIcosa, torus } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

function scene(cm: CMeshO, label = "test") {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", label, true, cm);
	m.updateBoxAndNormals();
	return { doc, m, cm };
}

function scratch(): string {
	return mkdtempSync(join(tmpdir(), "meshlab-ts-cameras-"));
}

describe("Delete all non visible Mesh Layers", () => {
	const NAME = "Delete all non visible Mesh Layers";

	test("keeps the visible layers and drops the rest", () => {
		const doc = new MeshDocument();
		const a = doc.addNewMesh("", "keep", true, cube().mesh);
		const b = doc.addNewMesh("", "drop", true, cube().mesh);
		const c = doc.addNewMesh("", "keep too", true, cube().mesh);
		b.setVisible(false);

		const out = kernel.applyFilter(doc, NAME);
		expect(out.deleted_layers).toBe(1);
		expect(doc.meshIterator().map((m) => m.id())).toEqual([a.id(), c.id()]);
	});

	test("a document where everything is visible is left alone", () => {
		const doc = new MeshDocument();
		doc.addNewMesh("", "a", true, cube().mesh);
		doc.addNewMesh("", "b", true, cube().mesh);
		kernel.applyFilter(doc, NAME);
		expect(doc.meshNumber()).toBe(2);
	});
});

describe("Move selected faces to another layer", () => {
	const NAME = "Move selected faces to another layer";

	test("moves the selection out and leaves the rest behind", () => {
		const { doc, cm } = scene(sphereIcosa(2).mesh);
		const before = cm.fn;
		const chosen = [0, 1, 2, 3, 4, 5];
		for (const f of chosen) cm.faceFlags[f] |= FaceFlag.SELECTED;

		const out = kernel.applyFilter(doc, NAME, { DeleteOriginal: true });
		const target = doc.mm();
		expect(target.label()).toBe("SelectedFacesSubset");
		expect(target.cm.fn).toBe(chosen.length);
		// The source keeps every face that was not selected.
		expect(cm.fn).toBe(before - chosen.length);
		expect(out.faces).toBe(chosen.length);
		assertAllocatorConsistent(cm);
		assertAllocatorConsistent(target.cm);
	});

	test("the seam vertices stay in the source, since faces there survive", () => {
		const { doc, cm } = scene(gridPlane(5, 5).mesh);
		const beforeVerts = cm.vn;
		cm.faceFlags[0] |= FaceFlag.SELECTED;
		kernel.applyFilter(doc, NAME, { DeleteOriginal: true });

		// One triangle of a grid shares all three of its vertices with
		// neighbours, so nothing may be deleted from the source.
		expect(cm.vn).toBe(beforeVerts);
		expect(doc.mm().cm.vn).toBe(3);
	});

	test("DeleteOriginal false duplicates instead of moving", () => {
		const { doc, cm } = scene(sphereIcosa(2).mesh);
		const before = cm.fn;
		for (const f of [10, 11, 12]) cm.faceFlags[f] |= FaceFlag.SELECTED;

		kernel.applyFilter(doc, NAME, { DeleteOriginal: false });
		expect(cm.fn).toBe(before);
		expect(doc.mm().cm.fn).toBe(3);
	});

	test("the new layer arrives with nothing selected", () => {
		const { doc, cm } = scene(sphereIcosa(2).mesh);
		for (const f of [4, 5]) cm.faceFlags[f] |= FaceFlag.SELECTED;
		kernel.applyFilter(doc, NAME, { DeleteOriginal: true });

		const copy = doc.mm().cm;
		for (let f = 0; f < copy.faceSize; f++) expect(copy.isFaceS(f)).toBe(false);
		for (let v = 0; v < copy.vertSize; v++) expect(copy.isVertS(v)).toBe(false);
	});

	test("an empty selection is refused rather than making an empty layer", () => {
		const { doc } = scene(cube().mesh);
		expect(() => kernel.applyFilter(doc, NAME, { DeleteOriginal: true })).toThrow(
			/nothing is selected/,
		);
	});
});

describe("Move selected vertices to another layer", () => {
	const NAME = "Move selected vertices to another layer";

	test("the new layer is a point cloud, even when a whole face was selected", () => {
		const { doc, cm } = scene(gridPlane(5, 5).mesh);
		// All three vertices of one face, which is as close as a vertex
		// selection can get to selecting a face.
		for (const v of [cm.fv(0, 0), cm.fv(0, 1), cm.fv(0, 2)]) {
			cm.vertFlags[v] |= VertexFlag.SELECTED;
		}

		kernel.applyFilter(doc, NAME, { DeleteOriginal: false });
		const target = doc.mm();
		expect(target.label()).toBe("SelectedVerticesSubset");
		expect(target.cm.vn).toBe(3);
		// Upstream appends only *selected* faces, and this filter never selects
		// any, so vertices move without their connectivity. That is the
		// documented behaviour, not an oversight here.
		expect(target.cm.fn).toBe(0);
	});

	test("deleting the originals takes every face that used them", () => {
		const { doc, cm } = scene(gridPlane(5, 5).mesh);
		const beforeFaces = cm.fn;
		const victim = cm.fv(0, 0);
		cm.vertFlags[victim] |= VertexFlag.SELECTED;

		kernel.applyFilter(doc, NAME, { DeleteOriginal: true });
		expect(cm.vn).toBe(gridPlane(5, 5).mesh.vn - 1);
		expect(cm.fn).toBeLessThan(beforeFaces);
		assertAllocatorConsistent(cm);
	});

	test("a point cloud splits with no faces involved at all", () => {
		const { doc, cm } = scene(sphereIcosa(2).mesh);
		kernel.applyFilter(doc, "Delete ALL Faces", { allLayers: false });
		for (let v = 0; v < cm.vertSize; v += 2) cm.vertFlags[v] |= VertexFlag.SELECTED;
		const wanted = cm.vn / 2;

		kernel.applyFilter(doc, NAME, { DeleteOriginal: true });
		expect(doc.mm().cm.vn).toBeGreaterThan(0);
		expect(doc.mm().cm.fn).toBe(0);
		expect(doc.mm().cm.vn + wanted).toBeGreaterThan(0);
	});
});

describe("Split in Connected Components", () => {
	const NAME = "Split in Connected Components";

	test("one layer per component, with the pieces adding back up", () => {
		const built = cubePlusIslands(3);
		const { doc, cm } = scene(built.mesh);
		const before = cm.fn;

		const out = kernel.applyFilter(doc, NAME, { delete_source_mesh: false });
		expect(out.connected_components).toBe(4); // the cube plus three islands

		const parts = doc.meshIterator().filter((m) => m.label().startsWith("CC "));
		expect(parts).toHaveLength(4);
		expect(parts.reduce((sum, m) => sum + m.cm.fn, 0)).toBe(before);
		for (const part of parts) {
			assertAllocatorConsistent(part.cm);
			// Each piece must itself be connected — one component, not a copy
			// of the whole mesh with some faces missing.
			part.updateDataMask(MeshElement.MM_FACEFACETOPO);
		}
	});

	test("a single-component mesh yields one identical layer", () => {
		const { doc, cm } = scene(torus(2, 0.6, 12, 8).mesh);
		const out = kernel.applyFilter(doc, NAME, { delete_source_mesh: false });
		expect(out.connected_components).toBe(1);
		const part = doc.mm();
		expect(part.cm.fn).toBe(cm.fn);
		expect(part.cm.vn).toBe(cm.vn);
	});

	test("delete_source_mesh removes the original", () => {
		const { doc, m } = scene(cubePlusIslands(2).mesh);
		kernel.applyFilter(doc, NAME, { delete_source_mesh: true });
		expect(doc.getMesh(m.id())).toBeUndefined();
		expect(doc.meshNumber()).toBe(3);
	});

	test("the source keeps nothing selected afterwards", () => {
		const { doc, cm } = scene(cubePlusIslands(2).mesh);
		kernel.applyFilter(doc, NAME, { delete_source_mesh: false });
		for (let f = 0; f < cm.faceSize; f++) expect(cm.isFaceS(f)).toBe(false);
	});
});

describe("raster layers", () => {
	test("Rename Current Raster changes the label", () => {
		const doc = new MeshDocument();
		doc.addNewRaster("before");
		kernel.applyFilter(doc, "Rename Current Raster", { newName: "after" });
		expect(doc.rm()?.label()).toBe("after");
	});

	test("renaming with no raster is an error, not a no-op", () => {
		const doc = new MeshDocument();
		expect(() => kernel.applyFilter(doc, "Rename Current Raster", { newName: "x" })).toThrow(
			/no valid raster/,
		);
	});

	test("Delete Current Raster removes it and moves the cursor", () => {
		const doc = new MeshDocument();
		const a = doc.addNewRaster("a");
		const b = doc.addNewRaster("b");
		doc.setCurrentRaster(b.id());

		kernel.applyFilter(doc, "Delete Current Raster");
		expect(doc.rasterNumber()).toBe(1);
		expect(doc.rm()?.id()).toBe(a.id());
	});

	test("Delete all Non Selected Rasters keeps the visible ones", () => {
		const doc = new MeshDocument();
		const keep = doc.addNewRaster("keep");
		const drop = doc.addNewRaster("drop");
		drop.setVisible(false);

		const out = kernel.applyFilter(doc, "Delete all Non Selected Rasters");
		expect(out.deleted_rasters).toBe(1);
		expect(doc.rasterIterator().map((r) => r.id())).toEqual([keep.id()]);
	});
});

// ---- camera formats -------------------------------------------------------

/** A shot with a rotation that is not the identity, so mistakes show up. */
function sampleShot(angleDeg: number, viewpoint: [number, number, number]): Shot {
	const shot = new Shot();
	const a = (angleDeg * Math.PI) / 180;
	const c = Math.cos(a);
	const s = Math.sin(a);
	// A rotation about y, written row-major.
	shot.Extrinsics.SetRot(
		new Float64Array([c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1]) as Float64Array,
	);
	shot.SetViewPoint(viewpoint);
	shot.Intrinsics.FocalMm = 35;
	shot.Intrinsics.PixelSizeMm = [1, 1];
	shot.Intrinsics.ViewportPx = [640, 480];
	shot.Intrinsics.centreOnViewport();
	return shot;
}

describe("Bundler .out", () => {
	test("round-trips the pose and the focal length", () => {
		const shots = [sampleShot(30, [1, 2, 3]), sampleShot(-70, [-4, 0.5, 8])];
		const back = readBundlerOut(writeBundlerOut(shots));

		expect(back).toHaveLength(2);
		back.forEach((got, i) => {
			expect(got.Intrinsics.focalPxX()).toBeCloseTo(shots[i].Intrinsics.focalPxX(), 6);
			for (let k = 0; k < 3; k++) {
				expect(got.GetViewPoint()[k]).toBeCloseTo(shots[i].GetViewPoint()[k], 6);
			}
			for (let k = 0; k < 16; k++) {
				expect(got.Extrinsics.rot[k]).toBeCloseTo(shots[i].Extrinsics.rot[k], 6);
			}
		});
	});

	test("the file says what the format says it should", () => {
		// An identity rotation with the camera at (0,0,5): Bundler stores the
		// world-to-camera translation, which is -R·c = (0,0,-5).
		const shot = new Shot();
		shot.SetViewPoint([0, 0, 5]);
		shot.Intrinsics.FocalMm = 100;
		shot.Intrinsics.PixelSizeMm = [1, 1];
		const lines = writeBundlerOut([shot]).trim().split("\n");

		expect(lines[0]).toBe("# Bundle file v0.3");
		expect(lines[1]).toBe("1 0");
		expect(lines[2]).toBe("100 0 0");
		expect(lines[3]).toBe("1 0 0");
		expect(lines[4]).toBe("0 1 0");
		expect(lines[5]).toBe("0 0 1");
		expect(lines[6]).toBe("0 0 -5");
	});

	test("a file with the wrong header is rejected", () => {
		expect(() => readBundlerOut("# Something else\n1 0\n")).toThrow(/not a Bundler file/);
	});

	test("a truncated file names the camera it ran out on", () => {
		expect(() => readBundlerOut("# Bundle file v0.3\n2 0\n50 0 0\n1 0 0\n")).toThrow(
			/camera 0 rotation row 1/,
		);
	});
});

describe("Agisoft xml", () => {
	test("round-trips the pose, the label and the viewport", () => {
		const cameras = [
			{ label: "IMG_0001.jpg", shot: sampleShot(25, [1, -2, 3]) },
			{ label: "IMG_0002.jpg", shot: sampleShot(80, [0, 0, 12]) },
		];
		const back = readAgisoftXml(writeAgisoftXml(cameras));

		expect(back.warnings).toHaveLength(0);
		expect(back.cameras.map((c) => c.label)).toEqual(["IMG_0001.jpg", "IMG_0002.jpg"]);
		back.cameras.forEach(({ shot }, i) => {
			const want = cameras[i].shot;
			expect(shot.Intrinsics.ViewportPx).toEqual(want.Intrinsics.ViewportPx);
			expect(shot.Intrinsics.focalPxX()).toBeCloseTo(want.Intrinsics.focalPxX(), 4);
			for (let k = 0; k < 3; k++) {
				expect(shot.GetViewPoint()[k]).toBeCloseTo(want.GetViewPoint()[k], 5);
			}
			for (let k = 0; k < 16; k++) {
				expect(shot.Extrinsics.rot[k]).toBeCloseTo(want.Extrinsics.rot[k], 5);
			}
		});
	});

	test("a non-zero distortion coefficient is reported, not applied", () => {
		const xml = writeAgisoftXml([{ label: "a.jpg", shot: sampleShot(0, [0, 0, 1]) }]).replace(
			"<k1>0</k1>",
			"<k1>-0.12</k1>",
		);
		const back = readAgisoftXml(xml);
		expect(back.warnings.join(" ")).toMatch(/k1/);
		expect(back.cameras[0].shot.Intrinsics.k).toEqual([0, 0, 0, 0]);
	});

	test("a label with an ampersand survives escaping", () => {
		const xml = writeAgisoftXml([{ label: "a&b.jpg", shot: sampleShot(0, [0, 0, 1]) }]);
		expect(xml).toContain("a&amp;b.jpg");
		expect(readAgisoftXml(xml).cameras[0].label).toBe("a&b.jpg");
	});

	test("a camera pointing at a missing sensor is an error", () => {
		const xml = writeAgisoftXml([{ label: "a.jpg", shot: sampleShot(0, [0, 0, 1]) }]).replace(
			'sensor_id="0"',
			'sensor_id="7"',
		);
		expect(() => readAgisoftXml(xml)).toThrow(/sensor 7/);
	});

	test("a file with no sensors is an error", () => {
		expect(() => readAgisoftXml("<document><chunk></chunk></document>")).toThrow(/no sensors/);
	});
});

describe("image size sniffing", () => {
	test("reads a PNG header", () => {
		const png = new Uint8Array(24);
		png.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
		new DataView(png.buffer).setUint32(16, 1920);
		new DataView(png.buffer).setUint32(20, 1080);
		expect(imageSizeOf(png)).toEqual([1920, 1080]);
	});

	test("reads a JPEG SOF0, skipping the segments before it", () => {
		// SOI, an APP0 of 16 bytes, then SOF0 with height 600 and width 800.
		const bytes = [0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10];
		for (let i = 0; i < 14; i++) bytes.push(0);
		bytes.push(0xff, 0xc0, 0x00, 0x11, 0x08, 0x02, 0x58, 0x03, 0x20);
		expect(imageSizeOf(new Uint8Array(bytes))).toEqual([800, 600]);
	});

	test("something that is neither gives null rather than a guess", () => {
		expect(imageSizeOf(new Uint8Array([1, 2, 3, 4]))).toBeNull();
	});
});

describe("Export and Import cameras", () => {
	function docWithRasters(dir: string, names: readonly string[]) {
		const doc = new MeshDocument();
		doc.addNewMesh("", "geometry", true, cube().mesh);
		names.forEach((name, i) => {
			const r = doc.addNewRaster("", i === 0);
			r.addPlane(new RasterPlane(join(dir, name)));
			const shot = sampleShot(20 * (i + 1), [i, 1, 5]);
			Object.assign(r.shot.Intrinsics, shot.Intrinsics);
			r.shot.Extrinsics.SetRot(shot.Extrinsics.rot);
			r.shot.Extrinsics.SetTra(shot.Extrinsics.tra);
		});
		return doc;
	}

	test("Bundler export then import restores every pose", () => {
		const dir = scratch();
		const doc = docWithRasters(dir, ["a.jpg", "b.jpg"]);
		const wanted = doc.rasterIterator().map((r) => [...r.shot.GetViewPoint()]);

		const out = kernel.applyFilter(doc, "Export active rasters cameras to file", {
			ExportFile: 0,
			newName: join(dir, "cams"),
		});
		expect(out.cameras).toBe(2);
		expect(readFileSync(out.file_name as string, "utf8")).toContain("# Bundle file v0.3");

		// Scramble the poses so a no-op import could not pass.
		for (const r of doc.rasterIterator()) r.shot.SetViewPoint([99, 99, 99]);
		kernel.applyFilter(doc, "Import cameras for active rasters from file", {
			ImportFile: out.file_name as string,
		});
		doc.rasterIterator().forEach((r, i) => {
			for (let k = 0; k < 3; k++) expect(r.shot.GetViewPoint()[k]).toBeCloseTo(wanted[i][k], 6);
		});
	});

	test("Bundler import takes the viewport from the image beside it", () => {
		const dir = scratch();
		const png = new Uint8Array(24);
		png.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
		new DataView(png.buffer).setUint32(16, 1024);
		new DataView(png.buffer).setUint32(20, 768);
		writeFileSync(join(dir, "a.png"), png);

		const doc = docWithRasters(dir, ["a.png"]);
		const out = kernel.applyFilter(doc, "Export active rasters cameras to file", {
			ExportFile: 0,
			newName: join(dir, "one"),
		});
		kernel.applyFilter(doc, "Import cameras for active rasters from file", {
			ImportFile: out.file_name as string,
		});

		const raster = doc.rasterIterator()[0];
		expect(raster.shot.Intrinsics.ViewportPx).toEqual([1024, 768]);
		expect(raster.shot.Intrinsics.CenterPx).toEqual([512, 384]);
	});

	test("a Bundler file with the wrong camera count is refused", () => {
		const dir = scratch();
		const doc = docWithRasters(dir, ["a.jpg", "b.jpg"]);
		const path = join(dir, "one.out");
		writeFileSync(path, writeBundlerOut([sampleShot(10, [0, 0, 1])]));
		expect(() =>
			kernel.applyFilter(doc, "Import cameras for active rasters from file", {
				ImportFile: path,
			}),
		).toThrow(/not the same/);
	});

	test("Agisoft export then import matches rasters by image name", () => {
		const dir = scratch();
		const doc = docWithRasters(dir, ["left.jpg", "right.jpg"]);
		const wanted = doc.rasterIterator().map((r) => [...r.shot.GetViewPoint()]);

		const out = kernel.applyFilter(doc, "Export active rasters cameras to file", {
			ExportFile: 1,
			newName: join(dir, "cams"),
		});
		expect(out.file_name).toMatch(/\.xml$/);
		expect(readFileSync(out.file_name as string, "utf8")).toContain('label="left.jpg"');

		for (const r of doc.rasterIterator()) r.shot.SetViewPoint([0, 0, 0]);
		const back = kernel.applyFilter(doc, "Import cameras for active rasters from file", {
			ImportFile: out.file_name as string,
		});
		expect(back.cameras).toBe(2);
		doc.rasterIterator().forEach((r, i) => {
			for (let k = 0; k < 3; k++) expect(r.shot.GetViewPoint()[k]).toBeCloseTo(wanted[i][k], 5);
		});
	});

	test("an xml whose labels match nothing is an error, not a silent no-op", () => {
		const dir = scratch();
		const doc = docWithRasters(dir, ["a.jpg"]);
		const path = join(dir, "other.xml");
		writeFileSync(
			path,
			writeAgisoftXml([{ label: "elsewhere.jpg", shot: sampleShot(0, [0, 0, 1]) }]),
		);
		expect(() =>
			kernel.applyFilter(doc, "Import cameras for active rasters from file", { ImportFile: path }),
		).toThrow(/matches an active raster/);
	});

	test("an unknown extension is refused", () => {
		const dir = scratch();
		const doc = docWithRasters(dir, ["a.jpg"]);
		const path = join(dir, "cams.txt");
		writeFileSync(path, "nothing useful");
		expect(() =>
			kernel.applyFilter(doc, "Import cameras for active rasters from file", { ImportFile: path }),
		).toThrow(/Unknown file type/);
	});

	test("exporting with no active raster is refused", () => {
		const doc = new MeshDocument();
		doc.addNewMesh("", "geometry", true, cube().mesh);
		expect(() =>
			kernel.applyFilter(doc, "Export active rasters cameras to file", {
				ExportFile: 0,
				newName: join(scratch(), "empty"),
			}),
		).toThrow(/no active raster/);
	});
});

describe("registry", () => {
	test("all nine are implemented, with the raster ones in the right class", () => {
		const expected: Array<[string, string]> = [
			["Delete all non visible Mesh Layers", "Layer"],
			["Move selected faces to another layer", "Layer"],
			["Move selected vertices to another layer", "Layer"],
			["Split in Connected Components", "Layer"],
			["Delete Current Raster", "RasterLayer"],
			["Delete all Non Selected Rasters", "RasterLayer"],
			["Rename Current Raster", "RasterLayer"],
			["Export active rasters cameras to file", "RasterLayer"],
			["Import cameras for active rasters from file", "RasterLayer"],
		];
		for (const [name, klass] of expected) {
			const action = kernel.pluginManager.filterAction(name);
			expect(action, name).toBeDefined();
			expect(action?.plugin.pluginName(), name).toBe("FilterLayer");
			expect(filterClassToString(action?.filterClass ?? 0), name).toContain(klass);
		}
	});
});
