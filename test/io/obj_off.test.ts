/**
 * OBJ and OFF.
 *
 * Both are text formats, so the round trip is exact for the values and the
 * interesting cases are all in the parsing: OBJ's negative indices, its four
 * ways of writing a face corner, and n-gons; OFF's optional header prefixes
 * and its counts line that may or may not share the magic word's line.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import {
	MLIOException,
	MLNotImplementedException,
} from "../../src/common/utilities/ml_exception.ts";
import { readObj, writeObj } from "../../src/meshlabplugins/io_base/obj.ts";
import { readOff, writeOff } from "../../src/meshlabplugins/io_base/off.ts";
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { Platonic } from "../../src/vcg/complex/create/platonic.ts";
import { blue, green, red, rgba } from "../../src/vcg/space/color4.ts";
import { computeFacts, signedVolume } from "../helpers/invariants.ts";

const kernel = MeshLabKernel.default();
const bytes = (s: string) => new TextEncoder().encode(s);
const textOf = (b: Uint8Array) => new TextDecoder().decode(b);

describe("OBJ reading", () => {
	test("reads the plainest possible file", () => {
		const m = new CMeshO();
		readObj(m, bytes("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"));
		expect(m.vn).toBe(3);
		expect(m.fn).toBe(1);
		expect(m.vx(1)).toBe(1);
		expect([m.fv(0, 0), m.fv(0, 1), m.fv(0, 2)]).toEqual([0, 1, 2]);
	});

	test("indices are 1-based", () => {
		const m = new CMeshO();
		// A 0 index is not "the first vertex" — it is invalid, and reading it
		// as the first would silently shift the whole mesh.
		expect(() => readObj(m, bytes("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n"))).toThrow(MLIOException);
	});

	test("negative indices count back from the end", () => {
		const m = new CMeshO();
		readObj(m, bytes("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n"));
		expect([m.fv(0, 0), m.fv(0, 1), m.fv(0, 2)]).toEqual([0, 1, 2]);
	});

	test("negative indices are relative to what has been declared so far", () => {
		// Which is the whole point of them: two blocks can be concatenated with
		// no renumbering, and -1 means a different vertex in each.
		const m = new CMeshO();
		readObj(
			m,
			bytes("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -1 -2 -3\nv 5 5 5\nv 6 6 6\nv 7 7 7\nf -1 -2 -3\n"),
		);
		expect(m.fn).toBe(2);
		expect([m.fv(0, 0), m.fv(0, 1), m.fv(0, 2)]).toEqual([2, 1, 0]);
		expect([m.fv(1, 0), m.fv(1, 1), m.fv(1, 2)]).toEqual([5, 4, 3]);
	});

	test("all four corner forms name the same vertex", () => {
		for (const corner of ["1 2 3", "1/1 2/2 3/3", "1//1 2//2 3//3", "1/1/1 2/2/2 3/3/3"]) {
			const m = new CMeshO();
			readObj(
				m,
				bytes(
					`v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nvn 0 0 1\nvn 0 0 1\nvn 0 0 1\nf ${corner}\n`,
				),
			);
			expect([m.fv(0, 0), m.fv(0, 1), m.fv(0, 2)], corner).toEqual([0, 1, 2]);
		}
	});

	test("the doubled slash means a normal with no texture coordinate", () => {
		// Reading `1//1` as `1/1` would take the normal index for a texture
		// index and then find no normals at all.
		const m = new CMeshO();
		readObj(m, bytes("v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1//1 2//1 3//1\n"));
		expect(m.vertNormal[2]).toBeCloseTo(1, 12);
	});

	test("an n-gon is fanned into triangles", () => {
		const m = new CMeshO();
		readObj(m, bytes("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv -1 0.5 0\nf 1 2 3 4 5\n"));
		// Five corners fan into three triangles, all sharing the first corner.
		expect(m.fn).toBe(3);
		for (let f = 0; f < 3; f++) expect(m.fv(f, 0)).toBe(0);
	});

	test("comments, blank lines and unknown keywords are ignored", () => {
		const m = new CMeshO();
		readObj(
			m,
			bytes(
				"# a comment\n\nmtllib none.mtl\no thing\ng group\ns off\nusemtl red\nv 0 0 0 # trailing\nv 1 0 0\nv 0 1 0\nvp 0 0\nf 1 2 3\n",
			),
		);
		expect(m.vn).toBe(3);
		expect(m.fn).toBe(1);
	});

	test("a line continued with a backslash is one line", () => {
		const m = new CMeshO();
		readObj(m, bytes("v 0 0 \\\n0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"));
		expect(m.vn).toBe(3);
		expect(m.vz(0)).toBe(0);
	});

	test("reads the extended per-vertex colour", () => {
		const m = new CMeshO();
		const result = readObj(m, bytes("v 0 0 0 1 0 0\nv 1 0 0 0 1 0\nv 0 1 0 0 0 1\nf 1 2 3\n"));
		expect(result.mask & MeshElement.MM_VERTCOLOR).not.toBe(0);
		expect(red(m.vertColor[0])).toBe(255);
		expect(green(m.vertColor[1])).toBe(255);
		expect(blue(m.vertColor[2])).toBe(255);
	});

	test("reports only the channels the file actually had", () => {
		const plain = new CMeshO();
		const mask = readObj(plain, bytes("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")).mask;
		expect(mask & MeshElement.MM_VERTNORMAL).toBe(0);
		expect(mask & MeshElement.MM_VERTCOLOR).toBe(0);
		expect(mask & MeshElement.MM_FACEVERT).not.toBe(0);

		// A point cloud has no faces, and must not claim otherwise.
		const cloud = new CMeshO();
		const cloudMask = readObj(cloud, bytes("v 0 0 0\nv 1 0 0\n")).mask;
		expect(cloudMask & MeshElement.MM_FACEVERT).toBe(0);
		expect(cloud.vn).toBe(2);
	});

	test("says where the problem is", () => {
		const m = new CMeshO();
		expect(() => readObj(m, bytes("v 0 0 0\nv 1 0 0\nf 1 2 9\n"))).toThrow(MLIOException);
		expect(() => readObj(m, bytes("v 0 0\n"))).toThrow(MLIOException);
		expect(() => readObj(m, bytes("v 0 0 zero\n"))).toThrow(MLIOException);
		expect(() => readObj(m, bytes("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2\n"))).toThrow(MLIOException);
	});
});

describe("OFF reading", () => {
	const TRIANGLE = "OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n";

	test("reads the plainest possible file", () => {
		const m = new CMeshO();
		readOff(m, bytes(TRIANGLE));
		expect(m.vn).toBe(3);
		expect(m.fn).toBe(1);
		expect(m.vx(1)).toBe(1);
	});

	test("accepts the counts on the magic word's line", () => {
		const m = new CMeshO();
		readOff(m, bytes("OFF 3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n"));
		expect(m.vn).toBe(3);
		expect(m.fn).toBe(1);
	});

	test("ignores the edge count, which is conventionally wrong", () => {
		const m = new CMeshO();
		readOff(m, bytes("OFF\n3 1 999\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n"));
		expect(m.fn).toBe(1);
	});

	test("COFF carries colour, NOFF carries normals", () => {
		const withColor = new CMeshO();
		const cmask = readOff(
			withColor,
			bytes("COFF\n3 1 0\n0 0 0 255 0 0 255\n1 0 0 0 255 0 255\n0 1 0 0 0 255 255\n3 0 1 2\n"),
		).mask;
		expect(cmask & MeshElement.MM_VERTCOLOR).not.toBe(0);
		expect(red(withColor.vertColor[0])).toBe(255);

		const withNormal = new CMeshO();
		const nmask = readOff(
			withNormal,
			bytes("NOFF\n3 1 0\n0 0 0 0 0 1\n1 0 0 0 0 1\n0 1 0 0 0 1\n3 0 1 2\n"),
		).mask;
		expect(nmask & MeshElement.MM_VERTNORMAL).not.toBe(0);
		expect(withNormal.vertNormal[2]).toBe(1);
	});

	test("colours may be 0..1 floats or 0..255 integers", () => {
		// The format does not say which, so the reader decides from the values.
		const asFloat = new CMeshO();
		readOff(asFloat, bytes("COFF\n1 0 0\n0 0 0 1 0 0 1\n"));
		expect(red(asFloat.vertColor[0])).toBe(255);

		const asByte = new CMeshO();
		readOff(asByte, bytes("COFF\n1 0 0\n0 0 0 255 0 0 255\n"));
		expect(red(asByte.vertColor[0])).toBe(255);
	});

	test("an n-gon is fanned, as in OBJ", () => {
		const m = new CMeshO();
		readOff(m, bytes("OFF\n4 1 0\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n"));
		expect(m.fn).toBe(2);
	});

	test("comments are stripped", () => {
		const m = new CMeshO();
		readOff(m, bytes("OFF\n# how many\n3 1 0\n0 0 0 # origin\n1 0 0\n0 1 0\n3 0 1 2\n"));
		expect(m.vn).toBe(3);
	});

	test("refuses what it cannot read, and says why", () => {
		const m = new CMeshO();
		expect(() => readOff(m, bytes(""))).toThrow(MLIOException);
		expect(() => readOff(m, bytes("PLY\n3 1 0\n"))).toThrow(MLIOException);
		expect(() => readOff(m, bytes("OFF\n3 1 0\n0 0 0\n"))).toThrow(MLIOException);
		expect(() => readOff(m, bytes("OFF\n1 1 0\n0 0 0\n3 0 5 9\n"))).toThrow(MLIOException);
		expect(() => readOff(m, bytes("OFF\n-1 0 0\n"))).toThrow(MLIOException);
		// The higher-dimensional variants are named rather than mis-parsed.
		expect(() => readOff(m, bytes("4OFF\n3 1 0\n"))).toThrow(MLNotImplementedException);
	});
});

describe("round trips", () => {
	test("OBJ preserves geometry and topology exactly", () => {
		const original = Platonic.sphere(2);
		const back = new CMeshO();
		readObj(back, writeObj(original));
		expect(back.vn).toBe(original.vn);
		expect(back.fn).toBe(original.fn);
		for (let v = 0; v < original.vn; v++) {
			for (let k = 0; k < 3; k++) {
				expect(back.vertCoord[3 * v + k]).toBe(original.vertCoord[3 * v + k]);
			}
		}
		expect(signedVolume(back)).toBe(signedVolume(original));
		expect(computeFacts(back).genus).toBe(0);
	});

	test("OFF preserves geometry and topology exactly", () => {
		const original = Platonic.torus(3, 1, 16, 8);
		const back = new CMeshO();
		readOff(back, writeOff(original));
		expect(back.vn).toBe(original.vn);
		expect(back.fn).toBe(original.fn);
		expect(signedVolume(back)).toBe(signedVolume(original));
		expect(computeFacts(back).genus).toBe(1);
	});

	test("both keep colours when asked, and leave them out when not", () => {
		const original = Platonic.sphere(1);
		for (let v = 0; v < original.vertSize; v++) original.vertColor[v] = rgba(255, 128, 0);

		for (const [label, write, read] of [
			["OBJ", writeObj, readObj],
			["OFF", writeOff, readOff],
		] as const) {
			const withColor = new CMeshO();
			read(withColor, write(original, { saveColors: true }));
			expect(red(withColor.vertColor[0]), label).toBe(255);
			expect(blue(withColor.vertColor[0]), label).toBe(0);

			const without = new CMeshO();
			const mask = read(without, write(original, {})).mask;
			expect(mask & MeshElement.MM_VERTCOLOR, label).toBe(0);
		}
	});

	test("OFF's header prefix always matches what it wrote", () => {
		// A mismatch would make every vertex line split wrongly, which is the
		// one way to corrupt an OFF file beyond recovery.
		const m = Platonic.sphere(1);
		expect(textOf(writeOff(m, {})).split("\n")[0]).toBe("OFF");
		expect(textOf(writeOff(m, { saveColors: true })).split("\n")[0]).toBe("COFF");
		expect(textOf(writeOff(m, { saveNormals: true })).split("\n")[0]).toBe("NOFF");
		expect(textOf(writeOff(m, { saveColors: true, saveNormals: true })).split("\n")[0]).toBe(
			"CNOFF",
		);
	});

	test("writing skips deleted slots and renumbers the faces", () => {
		const m = Platonic.sphere(1);
		// Delete a vertex and everything using it, so the numbering has a gap.
		const doomed = 0;
		for (let f = 0; f < m.faceSize; f++) {
			if (m.isFaceD(f)) continue;
			if (m.fv(f, 0) === doomed || m.fv(f, 1) === doomed || m.fv(f, 2) === doomed) {
				Allocator.deleteFace(m, f);
			}
		}
		Allocator.deleteVertex(m, doomed);

		for (const [label, write, read] of [
			["OBJ", writeObj, readObj],
			["OFF", writeOff, readOff],
		] as const) {
			const back = new CMeshO();
			read(back, write(m, {}));
			expect(back.vn, label).toBe(m.vn);
			expect(back.fn, label).toBe(m.fn);
			// If the gap had not been closed, some face would point past the end.
			for (let f = 0; f < back.fn; f++) {
				for (let k = 0; k < 3; k++) expect(back.fv(f, k), label).toBeLessThan(back.vn);
			}
		}
	});

	test("a point cloud survives both formats", () => {
		const cloud = Platonic.pointCloudFrom([
			[0, 0, 1],
			[1, 0, 0],
			[0, 1, 0],
		]);
		for (const [label, write, read] of [
			["OBJ", writeObj, readObj],
			["OFF", writeOff, readOff],
		] as const) {
			const back = new CMeshO();
			read(back, write(cloud, {}));
			expect(back.vn, label).toBe(3);
			expect(back.fn, label).toBe(0);
		}
	});
});

describe("through the kernel", () => {
	test("both formats are offered for reading and writing", () => {
		const { input, output } = kernel.pluginManager.supportedExtensions();
		for (const ext of ["OBJ", "OFF"]) {
			expect(input, ext).toContain(ext);
			expect(output, ext).toContain(ext);
		}
	});

	test("a mesh survives a trip out to OBJ and back in", () => {
		const doc = new MeshDocument();
		doc.addNewMesh("", "m", true, Platonic.sphere(2));
		const data = kernel.serializeMesh(doc, "m.obj");
		const back = new MeshDocument();
		const read = kernel.openMeshData(back, "m.obj", data);
		expect(read.cm.vn).toBe(162);
		expect(read.cm.fn).toBe(320);
		expect(computeFacts(read.cm).watertight).toBe(true);
	});

	test("a mesh survives a trip out to OFF and back in", () => {
		const doc = new MeshDocument();
		doc.addNewMesh("", "m", true, Platonic.sphere(2));
		const data = kernel.serializeMesh(doc, "m.off");
		const back = new MeshDocument();
		const read = kernel.openMeshData(back, "m.off", data);
		expect(read.cm.vn).toBe(162);
		expect(read.cm.fn).toBe(320);
		expect(computeFacts(read.cm).watertight).toBe(true);
	});

	test("normals go out only when the mask asks", () => {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("", "m", true, Platonic.sphere(1));
		m.updateBoxAndNormals();
		expect(textOf(kernel.serializeMesh(doc, "m.obj"))).not.toContain("vn ");
		const withNormals = kernel.serializeMesh(
			doc,
			"m.obj",
			undefined,
			{},
			undefined,
			MeshElement.MM_VERTCOORD | MeshElement.MM_FACEVERT | MeshElement.MM_VERTNORMAL,
		);
		expect(textOf(withNormals)).toContain("vn ");
		// And the face lines switch to the v//vn form to reference them.
		expect(textOf(withNormals)).toContain("//");
	});

	test("neither format is offered a Binary switch, because neither has one", () => {
		const doc = new MeshDocument();
		doc.addNewMesh("", "m", true, Platonic.sphere(1));
		for (const path of ["m.obj", "m.off"]) {
			expect(() => kernel.serializeMesh(doc, path, undefined, { Binary: true }), path).toThrow();
		}
	});
});
