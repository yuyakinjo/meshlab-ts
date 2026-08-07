/**
 * PLY and STL round-trips.
 *
 * The key distinction under test: PLY preserves the mesh, STL does not. STL is
 * an unwelded triangle soup written in float32, so a cube goes out with 8
 * vertices and comes back with 36. Any assertion here that expects otherwise
 * would be asserting a bug.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import { MLIOException } from "../../src/common/utilities/ml_exception.ts";
import { parsePlyHeader, readPly, writePly } from "../../src/meshlabplugins/io_base/ply.ts";
import { isBinaryStl, readStl, writeStl } from "../../src/meshlabplugins/io_base/stl.ts";
import { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { assertAllocatorConsistent, geometryDigest, surfaceArea } from "../helpers/invariants.ts";
import { cube, gridPlane, sphereIcosa, tetrahedron, torus } from "../helpers/mesh_builders.ts";

const ROUND_TRIP_MESHES = [
	() => tetrahedron(),
	() => cube(2),
	() => sphereIcosa(2),
	() => torus(2, 0.6, 12, 8),
	() => gridPlane(3, 2),
];

describe("PLY", () => {
	for (const binary of [true, false]) {
		const label = binary ? "binary" : "ascii";
		test(`${label}: round-trips geometry exactly`, () => {
			for (const build of ROUND_TRIP_MESHES) {
				const { mesh, name } = build();
				const bytes = writePly(mesh, { binary });
				const back = new CMeshO();
				readPly(back, bytes);
				expect(back.vn, `${name} vertices`).toBe(mesh.vn);
				expect(back.fn, `${name} faces`).toBe(mesh.fn);
				expect(geometryDigest(back), name).toBe(geometryDigest(mesh));
				assertAllocatorConsistent(back, name);
			}
		});

		test(`${label}: writing is byte-stable`, () => {
			const { mesh } = sphereIcosa(2);
			const first = writePly(mesh, { binary });
			const back = new CMeshO();
			readPly(back, first);
			// Reading and rewriting must reproduce the file exactly, or the
			// reader and writer disagree about something.
			expect(Array.from(writePly(back, { binary }))).toEqual(Array.from(first));
		});
	}

	test("the header reports the format and the elements", () => {
		const header = parsePlyHeader(writePly(cube().mesh, { binary: true }));
		expect(header.format).toBe("binary_little_endian");
		expect(header.elements.map((e) => e.name)).toEqual(["vertex", "face"]);
		expect(header.elements[0].count).toBe(8);
		expect(header.elements[1].count).toBe(12);
	});

	test("normals, colours and quality survive when asked for", () => {
		const { mesh } = cube();
		for (let v = 0; v < mesh.vn; v++) {
			mesh.vertQuality[v] = v * 1.5;
			mesh.vertColor[v] = (0xff000000 | (v * 8)) >>> 0;
			mesh.vertNormal[3 * v] = 1;
		}
		const bytes = writePly(mesh, {
			binary: true,
			saveNormals: true,
			saveColors: true,
			saveQuality: true,
		});
		const back = new CMeshO();
		const { mask } = readPly(back, bytes);
		expect(mask & MeshElement.MM_VERTQUALITY).toBeTruthy();
		expect(mask & MeshElement.MM_VERTCOLOR).toBeTruthy();
		expect(mask & MeshElement.MM_VERTNORMAL).toBeTruthy();
		for (let v = 0; v < back.vn; v++) {
			expect(back.vertQuality[v]).toBe(v * 1.5);
			expect(back.vertColor[v]).toBe(mesh.vertColor[v]);
			expect(back.vertNormal[3 * v]).toBe(1);
		}
	});

	test("geometry-only files report only geometry in the mask", () => {
		const back = new CMeshO();
		const { mask } = readPly(back, writePly(cube().mesh, { binary: true }));
		expect(mask & MeshElement.MM_VERTCOLOR).toBeFalsy();
		expect(mask & MeshElement.MM_VERTQUALITY).toBeFalsy();
	});

	test("big-endian binary reads the same as little-endian", () => {
		// Hand-built, because the writer only emits little-endian.
		const header =
			"ply\nformat binary_big_endian 1.0\nelement vertex 3\n" +
			"property float x\nproperty float y\nproperty float z\n" +
			"element face 1\nproperty list uchar int vertex_indices\nend_header\n";
		const head = new TextEncoder().encode(header);
		const body = new Uint8Array(3 * 12 + 13);
		const view = new DataView(body.buffer);
		const coords = [0, 0, 0, 1, 0, 0, 0, 1, 0];
		for (let i = 0; i < 9; i++) view.setFloat32(i * 4, coords[i], false);
		view.setUint8(36, 3);
		for (let k = 0; k < 3; k++) view.setInt32(37 + k * 4, k, false);
		const file = new Uint8Array(head.length + body.length);
		file.set(head);
		file.set(body, head.length);

		const m = new CMeshO();
		readPly(m, file);
		expect(m.vn).toBe(3);
		expect(m.fn).toBe(1);
		expect([m.vx(1), m.vy(2)]).toEqual([1, 1]);
		expect(surfaceArea(m)).toBeCloseTo(0.5, 6);
	});

	test("an n-gon face is fan-triangulated", () => {
		const header =
			"ply\nformat ascii 1.0\nelement vertex 4\n" +
			"property float x\nproperty float y\nproperty float z\n" +
			"element face 1\nproperty list uchar int vertex_indices\nend_header\n";
		const body = "0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n";
		const m = new CMeshO();
		readPly(m, new TextEncoder().encode(header + body));
		expect(m.vn).toBe(4);
		expect(m.fn).toBe(2); // the quad became two triangles
		expect(surfaceArea(m)).toBeCloseTo(1, 9);
	});

	test("unmodelled per-vertex properties are skipped, not misread", () => {
		const header =
			"ply\nformat ascii 1.0\nelement vertex 3\n" +
			"property float x\nproperty float y\nproperty float z\n" +
			"property float confidence\nproperty uchar flags\n" +
			"element face 1\nproperty list uchar int vertex_indices\nend_header\n";
		const body = "0 0 0 0.5 7\n1 0 0 0.6 8\n0 1 0 0.7 9\n3 0 1 2\n";
		const m = new CMeshO();
		readPly(m, new TextEncoder().encode(header + body));
		expect(m.vn).toBe(3);
		expect([m.vx(1), m.vy(2)]).toEqual([1, 1]);
	});

	test("a malformed file is rejected with a clear message", () => {
		const enc = (s: string) => new TextEncoder().encode(s);
		expect(() => parsePlyHeader(enc("not a ply\n"))).toThrow(MLIOException);
		expect(() => parsePlyHeader(enc("ply\nformat ascii 1.0\n"))).toThrow(/end_header/);
		expect(() => parsePlyHeader(enc("ply\nelement vertex 1\nend_header\n"))).toThrow(/no format/);
		expect(() =>
			readPly(new CMeshO(), enc("ply\nformat ascii 1.0\nelement face 0\nend_header\n")),
		).toThrow(/no vertex element/);
	});

	test("a truncated data section is reported, not read as garbage", () => {
		const header =
			"ply\nformat ascii 1.0\nelement vertex 3\n" +
			"property float x\nproperty float y\nproperty float z\nend_header\n";
		expect(() => readPly(new CMeshO(), new TextEncoder().encode(`${header}0 0 0\n1 0\n`))).toThrow(
			/ended early/,
		);
	});
});

describe("STL", () => {
	test("binary and ascii are told apart by size, not by the 'solid' token", () => {
		const { mesh } = cube();
		expect(isBinaryStl(writeStl(mesh, { binary: true }))).toBe(true);
		expect(isBinaryStl(writeStl(mesh, { binary: false }))).toBe(false);
		// A binary file whose header happens to begin with "solid" is still
		// binary; the size check is what decides.
		const tricky = writeStl(mesh, { binary: true, header: "solid something" });
		expect(isBinaryStl(tricky)).toBe(true);
	});

	for (const binary of [true, false]) {
		const label = binary ? "binary" : "ascii";
		test(`${label}: preserves the surface as an unwelded soup`, () => {
			const { mesh } = cube(2);
			const back = new CMeshO();
			readStl(back, writeStl(mesh, { binary }));
			// STL shares nothing: 12 triangles become 36 vertices.
			expect(back.fn).toBe(mesh.fn);
			expect(back.vn).toBe(mesh.fn * 3);
			// float32 for binary, decimal text for ascii; either way the area
			// survives to single precision.
			expect(surfaceArea(back)).toBeCloseTo(surfaceArea(mesh), 5);
			assertAllocatorConsistent(back);
		});

		test(`${label}: writing reaches a byte-exact fixed point`, () => {
			// STL stores float32, so the *first* write is lossy: reading it
			// back and rewriting recomputes each facet normal from rounded
			// coordinates, and the normals differ in their last bits. From the
			// second write on nothing more is lost, so that is where byte
			// stability begins — asserting it of the first write would be
			// asserting that float32 is float64.
			const { mesh } = sphereIcosa(1);
			const first = writeStl(mesh, { binary });

			const second = new CMeshO();
			readStl(second, first);
			const secondBytes = writeStl(second, { binary });

			const third = new CMeshO();
			readStl(third, secondBytes);
			const thirdBytes = writeStl(third, { binary });

			expect(Array.from(thirdBytes)).toEqual(Array.from(secondBytes));
			// The loss is confined to rounding, not to the surface itself.
			expect(surfaceArea(third)).toBeCloseTo(surfaceArea(mesh), 5);
		});
	}

	test("the recomputed facet normal is written, not a trusted stored one", () => {
		const { mesh } = cube();
		const bytes = writeStl(mesh, { binary: true });
		const view = new DataView(bytes.buffer);
		// The first cube face is on +z, so its normal is (0, 0, 1).
		expect(view.getFloat32(84, true)).toBeCloseTo(0, 6);
		expect(view.getFloat32(88, true)).toBeCloseTo(0, 6);
		expect(view.getFloat32(92, true)).toBeCloseTo(1, 6);
	});

	test("an empty mesh writes a valid, readable file", () => {
		for (const binary of [true, false]) {
			const bytes = writeStl(new CMeshO(), { binary });
			const back = new CMeshO();
			readStl(back, bytes);
			expect(back.fn).toBe(0);
			expect(back.vn).toBe(0);
		}
	});

	test("a truncated binary file is reported", () => {
		const bytes = writeStl(cube().mesh, { binary: true });
		expect(() => readStl(new CMeshO(), bytes.subarray(0, 100))).toThrow(MLIOException);
	});
});

describe("the kernel's I/O front door", () => {
	test("round-trips a file through load and save", () => {
		const kernel = MeshLabKernel.default();
		const doc = new MeshDocument();
		const path = "test/fixtures/meshes/tetra_binary.ply";
		const m = kernel.loadMesh(doc, path);
		expect(m.cm.vn).toBe(4);
		expect(m.cm.fn).toBe(4);
		expect(m.fullName()).toBe(path);
		// Loading is not a modification.
		expect(m.meshModified()).toBe(false);
		// The box and normals are ready without the caller asking.
		expect(m.cm.bbox.isEmpty).toBe(false);

		const bytes = kernel.serializeMesh(doc, "out.ply");
		const back = new CMeshO();
		readPly(back, bytes);
		expect(geometryDigest(back)).toBe(geometryDigest(m.cm));
	});

	test("reads every fixture we ship", () => {
		const kernel = MeshLabKernel.default();
		for (const [file, verts, faces] of [
			["cube_binary.stl", 36, 12],
			["cube_ascii.stl", 36, 12],
			["cube_with_holes.stl", 30, 10],
			["tetra_ascii.ply", 4, 4],
			["tetra_binary.ply", 4, 4],
			["sphere.ply", 162, 320],
		] as const) {
			const doc = new MeshDocument();
			const m = kernel.loadMesh(doc, `test/fixtures/meshes/${file}`);
			expect(m.cm.vn, file).toBe(verts);
			expect(m.cm.fn, file).toBe(faces);
			assertAllocatorConsistent(m.cm, file);
		}
	});

	test("an unsupported extension names the ones that are supported", () => {
		const kernel = MeshLabKernel.default();
		const doc = new MeshDocument();
		expect(() => kernel.loadMesh(doc, "model.obj")).toThrow(/no plugin can read/);
		try {
			kernel.loadMesh(doc, "model.obj");
		} catch (err) {
			expect((err as Error).message).toContain("PLY");
			expect((err as Error).message).toContain("STL");
		}
	});

	test("a missing file is an MLIOException naming the path", () => {
		const kernel = MeshLabKernel.default();
		expect(() => kernel.loadMesh(new MeshDocument(), "nope.ply")).toThrow(MLIOException);
	});

	test("the Binary save parameter selects the encoding", () => {
		const kernel = MeshLabKernel.default();
		const doc = new MeshDocument();
		kernel.loadMesh(doc, "test/fixtures/meshes/tetra_binary.ply");
		const asciiBytes = kernel.serializeMesh(doc, "out.ply", undefined, { Binary: false });
		expect(new TextDecoder().decode(asciiBytes.subarray(0, 32))).toContain("format ascii");
		const binaryBytes = kernel.serializeMesh(doc, "out.ply", undefined, { Binary: true });
		expect(new TextDecoder().decode(binaryBytes.subarray(0, 40))).toContain("binary_little_endian");
	});
});
