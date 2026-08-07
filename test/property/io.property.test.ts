/**
 * I/O round-trips over arbitrary meshes.
 *
 * The two formats promise different things, and the properties say so
 * explicitly: PLY is lossless for geometry, STL preserves the surface but not
 * the vertex sharing and only to float32.
 */
import { describe, expect, test } from "bun:test";
import fc from "fast-check";
import { readPly, writePly } from "../../src/meshlabplugins/io_base/ply.ts";
import { readStl, writeStl } from "../../src/meshlabplugins/io_base/stl.ts";
import { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { arbTriSoup, propertyOptions } from "../helpers/arbitrary.ts";
import { assertAllocatorConsistent, geometryDigest, surfaceArea } from "../helpers/invariants.ts";
import { buildMesh } from "../helpers/mesh_builders.ts";

/**
 * Random coordinates are drawn from a sane magnitude range. float32 has ~7
 * decimal digits, so comparing an STL round-trip of a 1e300 coordinate would
 * be measuring the format's exponent range rather than our code.
 */
const arbPrintableSoup = () =>
	arbTriSoup().map(({ coords, faces }) => ({
		coords: coords.map((c) => Math.round(c * 1000) / 1000),
		faces,
	}));

describe("PLY round-trip properties", () => {
	for (const binary of [true, false]) {
		test(`${binary ? "binary" : "ascii"}: geometry survives exactly`, () => {
			fc.assert(
				fc.property(arbTriSoup(), ({ coords, faces }) => {
					const m = buildMesh(coords, faces);
					const back = new CMeshO();
					readPly(back, writePly(m, { binary }));
					expect(back.vn).toBe(m.vn);
					expect(back.fn).toBe(m.fn);
					// PLY writes float64, so this is exact, not approximate.
					expect(geometryDigest(back)).toBe(geometryDigest(m));
					assertAllocatorConsistent(back);
				}),
				propertyOptions,
			);
		});

		test(`${binary ? "binary" : "ascii"}: save(load(save(m))) === save(m)`, () => {
			fc.assert(
				fc.property(arbTriSoup(), ({ coords, faces }) => {
					const m = buildMesh(coords, faces);
					const once = writePly(m, { binary });
					const back = new CMeshO();
					readPly(back, once);
					expect(Array.from(writePly(back, { binary }))).toEqual(Array.from(once));
				}),
				propertyOptions,
			);
		});
	}

	test("attributes survive alongside geometry", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = buildMesh(coords, faces);
				for (let v = 0; v < m.vn; v++) m.vertQuality[v] = v * 0.25;
				const back = new CMeshO();
				readPly(back, writePly(m, { binary: true, saveQuality: true }));
				for (let v = 0; v < back.vn; v++) expect(back.vertQuality[v]).toBe(v * 0.25);
			}),
			propertyOptions,
		);
	});
});

describe("STL round-trip properties", () => {
	for (const binary of [true, false]) {
		const label = binary ? "binary" : "ascii";
		test(`${label}: the triangle count survives and vertices unweld`, () => {
			fc.assert(
				fc.property(arbPrintableSoup(), ({ coords, faces }) => {
					const m = buildMesh(coords, faces);
					const back = new CMeshO();
					readStl(back, writeStl(m, { binary }));
					expect(back.fn).toBe(m.fn);
					// STL shares nothing between triangles, by design.
					expect(back.vn).toBe(m.fn * 3);
					assertAllocatorConsistent(back);
				}),
				propertyOptions,
			);
		});

		test(`${label}: the surface area survives to float32 precision`, () => {
			fc.assert(
				fc.property(arbPrintableSoup(), ({ coords, faces }) => {
					const m = buildMesh(coords, faces);
					const back = new CMeshO();
					readStl(back, writeStl(m, { binary }));
					const want = surfaceArea(m);
					const got = surfaceArea(back);
					const tolerance = Math.max(1e-4, Math.abs(want) * 1e-5);
					expect(Math.abs(got - want)).toBeLessThanOrEqual(tolerance);
				}),
				propertyOptions,
			);
		});

		test(`${label}: rewriting after one round trip is byte-exact`, () => {
			fc.assert(
				fc.property(arbPrintableSoup(), ({ coords, faces }) => {
					const m = buildMesh(coords, faces);
					const second = new CMeshO();
					readStl(second, writeStl(m, { binary }));
					const secondBytes = writeStl(second, { binary });
					const third = new CMeshO();
					readStl(third, secondBytes);
					expect(Array.from(writeStl(third, { binary }))).toEqual(Array.from(secondBytes));
				}),
				propertyOptions,
			);
		});
	}
});
