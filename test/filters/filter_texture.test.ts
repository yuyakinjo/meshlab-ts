/**
 * PNG, rasterisation, and the eight implemented filter_texture filters.
 *
 * Texture work has a specific failure mode: everything looks plausible and is
 * off by a flip, a half-texel, or a channel. So the tests here pin the things
 * that would be invisible in a picture — that v = 0 is the bottom of the
 * image, that a texel's colour is the barycentric mix of its face's corners,
 * that a UV seam becomes a real cut in the mesh — rather than checking that
 * something non-empty came out.
 */
import { describe, expect, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import {
	faceUV,
	pullPushFill,
	rasteriseFace,
} from "../../src/meshlabplugins/filter_texture/rastering.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { blue, green, red, rgba } from "../../src/vcg/space/color4.ts";
import { Image } from "../../src/vcg/space/image/image.ts";
import { isPng, readPng, writePng } from "../../src/vcg/space/image/png.ts";
import { assertAllocatorConsistent } from "../helpers/invariants.ts";
import { cube, gridPlane, sphereIcosa } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

function scene(cm: CMeshO, channels = 0) {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", "test", true, cm);
	if (channels !== 0) m.updateDataMask(channels);
	m.updateBoxAndNormals();
	return { doc, m, cm };
}

function scratch(): string {
	return mkdtempSync(join(tmpdir(), "meshlab-ts-texture-"));
}

/** Fills the wedge UVs so the whole mesh sits inside 0..1. */
function flatten(doc: MeshDocument): void {
	kernel.applyFilter(doc, "Parametrization: Flat Plane", {
		projectionPlane: 0,
		aspectRatio: false,
		sideGutter: 0,
	});
}

describe("PNG", () => {
	test("round-trips every pixel exactly", () => {
		const image = new Image(37, 23);
		for (let y = 0; y < 23; y++) {
			for (let x = 0; x < 37; x++) {
				image.setPixel(x, y, rgba((x * 7) % 256, (y * 11) % 256, (x * y) % 256, 255));
			}
		}
		const back = readPng(writePng(image));
		expect(back.width).toBe(37);
		expect(back.height).toBe(23);
		expect([...back.data]).toEqual([...image.data]);
	});

	test("keeps the alpha channel", () => {
		const image = new Image(4, 4, rgba(10, 20, 30, 0));
		image.setPixel(2, 2, rgba(1, 2, 3, 128));
		const back = readPng(writePng(image));
		expect(back.pixel(0, 0)).toBe(rgba(10, 20, 30, 0));
		expect(back.pixel(2, 2)).toBe(rgba(1, 2, 3, 128));
	});

	test("a one-pixel image is still a valid PNG", () => {
		const image = new Image(1, 1, rgba(9, 9, 9, 255));
		const bytes = writePng(image);
		expect(isPng(bytes)).toBe(true);
		expect(readPng(bytes).pixel(0, 0)).toBe(rgba(9, 9, 9, 255));
	});

	test("something that is not a PNG is rejected", () => {
		expect(() => readPng(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9]))).toThrow(/not a PNG/);
		expect(isPng(new Uint8Array([1, 2, 3]))).toBe(false);
	});

	test("a corrupt row filter is named, not guessed at", () => {
		const bytes = writePng(new Image(8, 8, rgba(1, 1, 1, 255)));
		// Flip a byte in the IHDR's colour type to something unsupported.
		const copy = bytes.slice();
		copy[8 + 8 + 9] = 7;
		expect(() => readPng(copy)).toThrow(/colour type 7/);
	});
});

describe("Image sampling", () => {
	test("v = 0 is the bottom of the image, not the top", () => {
		const image = new Image(2, 2);
		image.setPixel(0, 0, rgba(255, 0, 0, 255)); // top-left
		image.setPixel(0, 1, rgba(0, 0, 255, 255)); // bottom-left
		image.setPixel(1, 0, rgba(255, 0, 0, 255));
		image.setPixel(1, 1, rgba(0, 0, 255, 255));

		// This is the flip that quietly turns every baked texture upside down
		// if it is missed anywhere.
		expect(blue(image.sample(0.5, 0))).toBe(255);
		expect(red(image.sample(0.5, 1))).toBe(255);
		// And 1 is the far edge rather than wrapping back to 0, which would
		// put a one-texel seam down two sides of every baked texture.
		expect(image.sample(0.5, 1)).not.toBe(image.sample(0.5, 0));
	});

	test("bilinear interpolation halfway between two texels", () => {
		const image = new Image(2, 1);
		image.setPixel(0, 0, rgba(0, 0, 0, 255));
		image.setPixel(1, 0, rgba(200, 100, 50, 255));
		const mid = image.sample(0.5, 0.5);
		expect(red(mid)).toBe(100);
		expect(green(mid)).toBe(50);
		expect(blue(mid)).toBe(25);
	});

	test("coordinates outside 0..1 wrap round", () => {
		const image = new Image(4, 4, rgba(7, 7, 7, 255));
		image.setPixel(0, 3, rgba(1, 2, 3, 255));
		expect(image.sample(-0.75, 0.25)).toBe(image.sample(0.25, 0.25));
		expect(image.sample(1.25, 0.25)).toBe(image.sample(0.25, 0.25));
		expect(image.sample(2.0, 0.5)).toBe(image.sample(0.0, 0.5));
	});

	test("a zero-sized image is refused", () => {
		expect(() => new Image(0, 8)).toThrow(/positive integer dimensions/);
	});
});

describe("rasterisation", () => {
	test("a triangle covering the whole square paints every texel once", () => {
		const { cm } = scene(gridPlane(1, 1).mesh, MeshElement.MM_WEDGTEXCOORD);
		// A right triangle over the lower-left half of UV space.
		const wt = cm.wedgeTexCoord as Float64Array;
		wt.set([0, 0, 1, 0, 0, 1], 0);

		const counts = new Map<string, number>();
		rasteriseFace(cm, 0, 16, 16, ({ x, y }) => {
			const key = `${x},${y}`;
			counts.set(key, (counts.get(key) ?? 0) + 1);
		});
		for (const n of counts.values()) expect(n).toBe(1);
		// Roughly half of a 16x16 grid.
		expect(counts.size).toBeGreaterThan(100);
		expect(counts.size).toBeLessThan(160);
	});

	test("the barycentric weights sum to one everywhere", () => {
		const { cm } = scene(gridPlane(1, 1).mesh, MeshElement.MM_WEDGTEXCOORD);
		(cm.wedgeTexCoord as Float64Array).set([0.1, 0.1, 0.9, 0.2, 0.3, 0.8], 0);
		rasteriseFace(cm, 0, 32, 32, ({ bary }) => {
			expect(bary[0] + bary[1] + bary[2]).toBeCloseTo(1, 10);
			for (const w of bary) expect(w).toBeGreaterThan(-1e-8);
		});
	});

	test("a degenerate triangle paints nothing rather than dividing by zero", () => {
		const { cm } = scene(gridPlane(1, 1).mesh, MeshElement.MM_WEDGTEXCOORD);
		(cm.wedgeTexCoord as Float64Array).set([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 0);
		let painted = 0;
		rasteriseFace(cm, 0, 16, 16, () => painted++);
		expect(painted).toBe(0);
	});
});

describe("pull-push fill", () => {
	test("spreads a single painted texel over the whole image", () => {
		const background = rgba(0, 0, 0, 0);
		const image = new Image(16, 16, background);
		image.setPixel(8, 8, rgba(200, 100, 50, 255));

		pullPushFill(image, background);
		let empty = 0;
		for (let y = 0; y < 16; y++) {
			for (let x = 0; x < 16; x++) if (image.pixel(x, y) === background) empty++;
		}
		expect(empty).toBe(0);
		// And the one real texel is untouched.
		expect(image.pixel(8, 8)).toBe(rgba(200, 100, 50, 255));
	});

	test("leaves an already full image alone", () => {
		const background = rgba(0, 0, 0, 0);
		const image = new Image(8, 8, rgba(3, 4, 5, 255));
		const before = [...image.data];
		pullPushFill(image, background);
		expect([...image.data]).toEqual(before);
	});

	test("the fill is a blend of the real colours, not a constant", () => {
		const background = rgba(0, 0, 0, 0);
		const image = new Image(32, 32, background);
		for (let y = 0; y < 32; y++) image.setPixel(0, y, rgba(255, 0, 0, 255));
		for (let y = 0; y < 32; y++) image.setPixel(31, y, rgba(0, 0, 255, 255));

		pullPushFill(image, background);
		// Near the red edge it must still be mostly red.
		expect(red(image.pixel(2, 16))).toBeGreaterThan(blue(image.pixel(2, 16)));
		expect(blue(image.pixel(29, 16))).toBeGreaterThan(red(image.pixel(29, 16)));
	});
});

describe("Parametrization: Flat Plane", () => {
	const NAME = "Parametrization: Flat Plane";

	test("fills the whole 0..1 square", () => {
		const { doc, cm } = scene(gridPlane(4, 4).mesh);
		kernel.applyFilter(doc, NAME, { projectionPlane: 0, aspectRatio: false, sideGutter: 0 });

		let minU = 1;
		let maxU = 0;
		let minV = 1;
		let maxV = 0;
		for (let f = 0; f < cm.faceSize; f++) {
			for (const [u, v] of faceUV(cm, f)) {
				minU = Math.min(minU, u);
				maxU = Math.max(maxU, u);
				minV = Math.min(minV, v);
				maxV = Math.max(maxV, v);
			}
		}
		expect(minU).toBeCloseTo(0, 10);
		expect(maxU).toBeCloseTo(1, 10);
		expect(minV).toBeCloseTo(0, 10);
		expect(maxV).toBeCloseTo(1, 10);
	});

	test("preserving the ratio leaves the shorter axis short", () => {
		// Twice as wide as it is tall, so v must span only half of 0..1.
		const cm = gridPlane(4, 4).mesh;
		for (let v = 0; v < cm.vertSize; v++) cm.vertCoord[3 * v] *= 2;
		const { doc } = scene(cm);
		kernel.applyFilter(doc, NAME, { projectionPlane: 0, aspectRatio: true, sideGutter: 0 });

		let maxV = 0;
		let maxU = 0;
		for (let f = 0; f < cm.faceSize; f++) {
			for (const [u, v] of faceUV(cm, f)) {
				maxU = Math.max(maxU, u);
				maxV = Math.max(maxV, v);
			}
		}
		expect(maxU).toBeCloseTo(1, 10);
		expect(maxV).toBeCloseTo(0.5, 10);
	});

	test("a gutter keeps the layout off the edges", () => {
		const { doc, cm } = scene(gridPlane(3, 3).mesh);
		kernel.applyFilter(doc, NAME, { projectionPlane: 0, aspectRatio: false, sideGutter: 0.1 });
		for (let f = 0; f < cm.faceSize; f++) {
			for (const [u, v] of faceUV(cm, f)) {
				expect(u).toBeGreaterThan(0.05);
				expect(u).toBeLessThan(0.95);
				expect(v).toBeGreaterThan(0.05);
				expect(v).toBeLessThan(0.95);
			}
		}
	});

	test("the projection plane chooses which axes are used", () => {
		// A grid in z = 0 projected onto YZ collapses onto one axis, and the
		// degenerate extent must not produce NaN coordinates.
		const { doc, cm } = scene(gridPlane(3, 3).mesh);
		kernel.applyFilter(doc, NAME, { projectionPlane: 2, aspectRatio: false, sideGutter: 0 });
		for (let f = 0; f < cm.faceSize; f++) {
			for (const [u, v] of faceUV(cm, f)) {
				expect(Number.isFinite(u)).toBe(true);
				expect(Number.isFinite(v)).toBe(true);
			}
		}
	});

	test("an out-of-range gutter is refused", () => {
		const { doc } = scene(gridPlane(2, 2).mesh);
		expect(() =>
			kernel.applyFilter(doc, NAME, { projectionPlane: 0, aspectRatio: false, sideGutter: 0.8 }),
		).toThrow(/side gutter/);
	});
});

describe("Parametrization: Trivial Per-Triangle", () => {
	const NAME = "Parametrization: Trivial Per-Triangle";

	test("every face gets a slot inside 0..1 and none overlap", () => {
		const { doc, cm } = scene(sphereIcosa(1).mesh);
		kernel.applyFilter(doc, NAME, { sidedim: 0, textdim: 1024, border: 2, method: 0 });

		const painted = new Set<string>();
		let collisions = 0;
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			for (const [u, v] of faceUV(cm, f)) {
				expect(u).toBeGreaterThanOrEqual(0);
				expect(u).toBeLessThanOrEqual(1);
				expect(v).toBeGreaterThanOrEqual(0);
				expect(v).toBeLessThanOrEqual(1);
			}
			rasteriseFace(cm, f, 512, 512, ({ x, y }) => {
				const key = `${x},${y}`;
				if (painted.has(key)) collisions++;
				painted.add(key);
			});
		}
		// The whole point of the border is that no two faces share a texel.
		expect(collisions).toBe(0);
		expect(painted.size).toBeGreaterThan(0);
	});

	test("space optimising gives the larger faces the earlier slots", () => {
		// Stretch one half of a grid so the areas differ a lot.
		const cm = gridPlane(4, 4).mesh;
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.vy(v) > 0) cm.setVert(v, cm.vx(v) * 4, cm.vy(v) * 4, cm.vz(v));
		}
		const { doc } = scene(cm);
		kernel.applyFilter(doc, NAME, { sidedim: 0, textdim: 1024, border: 2, method: 1 });

		// The first grid cell is at the top-left; whichever face landed there
		// must be at least as large as the one in the last cell.
		const areaOf = (f: number) => {
			const a = cm.fv(f, 0);
			const b = cm.fv(f, 1);
			const c = cm.fv(f, 2);
			const u = [cm.vx(b) - cm.vx(a), cm.vy(b) - cm.vy(a), cm.vz(b) - cm.vz(a)];
			const w = [cm.vx(c) - cm.vx(a), cm.vy(c) - cm.vy(a), cm.vz(c) - cm.vz(a)];
			return Math.hypot(
				u[1] * w[2] - u[2] * w[1],
				u[2] * w[0] - u[0] * w[2],
				u[0] * w[1] - u[1] * w[0],
			);
		};
		const topLeftness = (f: number) => {
			const uv = faceUV(cm, f);
			return Math.min(...uv.map(([u, v]) => u + (1 - v)));
		};
		const faces = [...Array(cm.faceSize).keys()].filter((f) => !cm.isFaceD(f));
		const first = faces.reduce((a, b) => (topLeftness(a) <= topLeftness(b) ? a : b));
		const last = faces.reduce((a, b) => (topLeftness(a) >= topLeftness(b) ? a : b));
		expect(areaOf(first)).toBeGreaterThanOrEqual(areaOf(last));
	});

	test("too few quads per line is refused with the number that would work", () => {
		const { doc } = scene(sphereIcosa(2).mesh);
		expect(() =>
			kernel.applyFilter(doc, NAME, { sidedim: 2, textdim: 1024, border: 2, method: 0 }),
		).toThrow(/at least \d+/);
	});

	test("an impossible border is refused", () => {
		const { doc } = scene(sphereIcosa(2).mesh);
		expect(() =>
			kernel.applyFilter(doc, NAME, { sidedim: 0, textdim: 64, border: 20, method: 0 }),
		).toThrow(/border is too much/);
	});
});

describe("UV channel conversion", () => {
	test("vertex to wedge copies each vertex's coordinate onto its corners", () => {
		const { doc, cm } = scene(gridPlane(3, 3).mesh, MeshElement.MM_VERTTEXCOORD);
		const vt = cm.vertTexCoord as Float64Array;
		for (let v = 0; v < cm.vertSize; v++) {
			vt[2 * v] = v / cm.vertSize;
			vt[2 * v + 1] = 1 - v / cm.vertSize;
		}

		kernel.applyFilter(doc, "Convert PerVertex UV into PerWedge UV");
		for (let f = 0; f < cm.faceSize; f++) {
			const uv = faceUV(cm, f);
			for (let k = 0; k < 3; k++) {
				const v = cm.fv(f, k);
				expect(uv[k][0]).toBe(vt[2 * v]);
				expect(uv[k][1]).toBe(vt[2 * v + 1]);
			}
		}
	});

	test("a seamless parametrisation survives a round trip without splitting", () => {
		const { doc, cm } = scene(gridPlane(3, 3).mesh);
		flatten(doc);
		const before = cm.vn;

		const out = kernel.applyFilter(doc, "Convert PerWedge UV into PerVertex UV");
		// A flat-plane mapping assigns each vertex one coordinate, so there is
		// no seam and nothing to cut.
		expect(out.split_vertices).toBe(0);
		expect(cm.vn).toBe(before);

		kernel.applyFilter(doc, "Convert PerVertex UV into PerWedge UV");
		assertAllocatorConsistent(cm);
	});

	test("a seam becomes a real cut in the mesh", () => {
		const { doc, cm } = scene(gridPlane(3, 3).mesh, MeshElement.MM_WEDGTEXCOORD);
		// Give every face its own corner of UV space, so every shared vertex
		// disagrees with itself.
		for (let f = 0; f < cm.faceSize; f++) {
			const base = (f % 4) * 0.2;
			(cm.wedgeTexCoord as Float64Array).set(
				[base, base, base + 0.1, base, base, base + 0.1],
				6 * f,
			);
		}
		const before = cm.vn;

		const out = kernel.applyFilter(doc, "Convert PerWedge UV into PerVertex UV");
		expect(out.split_vertices as number).toBeGreaterThan(0);
		expect(cm.vn).toBe(before + (out.split_vertices as number));
		assertAllocatorConsistent(cm);

		// Every corner's per-vertex UV must now match the wedge it came from.
		const vt = cm.vertTexCoord as Float64Array;
		for (let f = 0; f < cm.faceSize; f++) {
			const uv = faceUV(cm, f);
			for (let k = 0; k < 3; k++) {
				const v = cm.fv(f, k);
				expect(vt[2 * v]).toBeCloseTo(uv[k][0], 12);
				expect(vt[2 * v + 1]).toBeCloseTo(uv[k][1], 12);
			}
		}
	});

	test("splitting leaves the geometry where it was", () => {
		const { doc, cm } = scene(cube().mesh, MeshElement.MM_WEDGTEXCOORD);
		for (let f = 0; f < cm.faceSize; f++) {
			(cm.wedgeTexCoord as Float64Array).set([f / 12, 0, 1, f / 12, 0, 1], 6 * f);
		}
		kernel.applyFilter(doc, "Convert PerWedge UV into PerVertex UV");

		// Each copy sits exactly on top of the vertex it came from, so the
		// cube's corners are still eight distinct places.
		const places = new Set<string>();
		for (let v = 0; v < cm.vertSize; v++) {
			if (!cm.isVertD(v)) places.add(`${cm.vx(v)},${cm.vy(v)},${cm.vz(v)}`);
		}
		expect(places.size).toBe(8);
	});
});

describe("Set Texture", () => {
	const NAME = "Set Texture";

	test("attaches a dummy checkerboard", () => {
		const { doc, m } = scene(gridPlane(2, 2).mesh);
		flatten(doc);
		kernel.applyFilter(doc, NAME, {
			use_dummy_texture: true,
			dummy_img_size: 64,
			dummy_check_size: 8,
			dummy_type: 0,
			textName: "",
		});

		expect(m.cm.textures).toEqual(["dummy.png"]);
		const image = readPng(m.textures.get("dummy.png") as Uint8Array);
		expect(image.width).toBe(64);
		// A checker alternates every 8 px.
		expect(image.pixel(0, 0)).not.toBe(image.pixel(8, 0));
		expect(image.pixel(0, 0)).toBe(image.pixel(16, 0));
	});

	test("the grid pattern draws lines, not squares", () => {
		const { doc, m } = scene(gridPlane(2, 2).mesh);
		flatten(doc);
		kernel.applyFilter(doc, NAME, {
			use_dummy_texture: true,
			dummy_img_size: 32,
			dummy_check_size: 8,
			dummy_type: 1,
			textName: "",
		});
		const image = readPng(m.textures.get("dummy.png") as Uint8Array);
		// Lines on the multiples of 8, white in between.
		expect(red(image.pixel(8, 3))).toBe(0);
		expect(red(image.pixel(3, 3))).toBe(255);
	});

	test("loads a PNG from disk and remembers its name", () => {
		const dir = scratch();
		const path = join(dir, "checker.png");
		writeFileSync(path, writePng(new Image(16, 8, rgba(1, 2, 3, 255))));

		const { doc, m } = scene(gridPlane(2, 2).mesh);
		flatten(doc);
		const out = kernel.applyFilter(doc, NAME, { textName: path, use_dummy_texture: false });
		expect(out.width).toBe(16);
		expect(out.height).toBe(8);
		expect(m.cm.textures).toEqual(["checker.png"]);
	});

	test("a non-PNG file is refused rather than attached blindly", () => {
		const dir = scratch();
		const path = join(dir, "not-an-image.png");
		writeFileSync(path, "hello");
		const { doc } = scene(gridPlane(2, 2).mesh);
		flatten(doc);
		expect(() =>
			kernel.applyFilter(doc, NAME, { textName: path, use_dummy_texture: false }),
		).toThrow(/not one/);
	});

	test("no file and no dummy is an error", () => {
		const { doc } = scene(gridPlane(2, 2).mesh);
		flatten(doc);
		expect(() => kernel.applyFilter(doc, NAME, { textName: "", use_dummy_texture: false })).toThrow(
			/not specified/,
		);
	});
});

describe("Transfer: Vertex Color to Texture", () => {
	const NAME = "Transfer: Vertex Color to Texture";

	test("a uniformly coloured mesh gives a uniformly coloured texture", () => {
		const { doc, cm } = scene(gridPlane(4, 4).mesh, MeshElement.MM_VERTCOLOR);
		flatten(doc);
		for (let v = 0; v < cm.vertSize; v++) cm.vertColor[v] = rgba(10, 200, 30, 255);

		kernel.applyFilter(doc, NAME, {
			textName: "flat",
			textW: 64,
			textH: 64,
			overwrite: false,
			pullpush: false,
		});
		const image = readPng(doc.mm().textures.get("flat.png") as Uint8Array);
		// The interior is definitely covered; the very edge may not be.
		expect(image.pixel(32, 32)).toBe(rgba(10, 200, 30, 255));
	});

	test("a colour gradient survives into the texture in the right direction", () => {
		const { doc, cm } = scene(gridPlane(6, 6).mesh, MeshElement.MM_VERTCOLOR);
		flatten(doc);
		// Red increases with x, which the flat-plane mapping sends to u.
		let minX = Number.POSITIVE_INFINITY;
		let maxX = Number.NEGATIVE_INFINITY;
		for (let v = 0; v < cm.vertSize; v++) {
			minX = Math.min(minX, cm.vx(v));
			maxX = Math.max(maxX, cm.vx(v));
		}
		for (let v = 0; v < cm.vertSize; v++) {
			const t = (cm.vx(v) - minX) / (maxX - minX);
			cm.vertColor[v] = rgba(Math.round(t * 255), 0, 0, 255);
		}

		kernel.applyFilter(doc, NAME, {
			textName: "gradient",
			textW: 64,
			textH: 64,
			overwrite: false,
			pullpush: true,
		});
		const image = readPng(doc.mm().textures.get("gradient.png") as Uint8Array);
		expect(red(image.pixel(56, 32))).toBeGreaterThan(red(image.pixel(8, 32)));
	});

	test("without pull-push the unmapped texels stay empty", () => {
		const { doc, cm } = scene(gridPlane(2, 2).mesh, MeshElement.MM_VERTCOLOR);
		// A parametrisation using only a corner of the texture, so most of it
		// is never touched.
		kernel.applyFilter(doc, "Parametrization: Flat Plane", {
			projectionPlane: 0,
			aspectRatio: false,
			sideGutter: 0.4,
		});
		for (let v = 0; v < cm.vertSize; v++) cm.vertColor[v] = rgba(255, 255, 255, 255);

		kernel.applyFilter(doc, NAME, {
			textName: "sparse",
			textW: 32,
			textH: 32,
			overwrite: false,
			pullpush: false,
		});
		const bare = readPng(doc.mm().textures.get("sparse.png") as Uint8Array);
		expect(bare.pixel(0, 0)).toBe(rgba(0, 0, 0, 0));

		kernel.applyFilter(doc, NAME, {
			textName: "filled",
			textW: 32,
			textH: 32,
			overwrite: false,
			pullpush: true,
		});
		const filled = readPng(doc.mm().textures.get("filled.png") as Uint8Array);
		expect(filled.pixel(0, 0)).not.toBe(rgba(0, 0, 0, 0));
	});

	test("a mesh whose UVs cover no texel is an error, not an empty texture", () => {
		const { doc, cm } = scene(
			gridPlane(2, 2).mesh,
			MeshElement.MM_VERTCOLOR | MeshElement.MM_WEDGTEXCOORD,
		);
		// Every face collapsed to a point in texture space.
		(cm.wedgeTexCoord as Float64Array).fill(0.5);
		expect(() =>
			kernel.applyFilter(doc, NAME, {
				textName: "empty",
				textW: 16,
				textH: 16,
				overwrite: false,
				pullpush: false,
			}),
		).toThrow(/no texel/);
	});
});

describe("Transfer between meshes", () => {
	/** A parametrised, textured plane to sample from. */
	function texturedSource(doc: MeshDocument) {
		const m = doc.addNewMesh("", "source", true, gridPlane(4, 4).mesh);
		m.updateDataMask(MeshElement.MM_VERTCOLOR);
		m.updateBoxAndNormals();
		kernel.applyFilter(doc, "Parametrization: Flat Plane", {
			projectionPlane: 0,
			aspectRatio: false,
			sideGutter: 0,
		});
		// Left half red, right half blue, so a sample's side is unambiguous.
		const image = new Image(32, 32);
		for (let y = 0; y < 32; y++) {
			for (let x = 0; x < 32; x++) {
				image.setPixel(x, y, x < 16 ? rgba(255, 0, 0, 255) : rgba(0, 0, 255, 255));
			}
		}
		m.textures.clear();
		m.textures.set("src.png", writePng(image));
		m.cm.textures = ["src.png"];
		return m;
	}

	test("Texture to Vertex Color picks the colour under each vertex", () => {
		const doc = new MeshDocument();
		const source = texturedSource(doc);
		const target = doc.addNewMesh("", "target", true, gridPlane(5, 5).mesh);
		target.updateBoxAndNormals();

		const out = kernel.applyFilter(doc, "Transfer: Texture to Vertex Color (1 or 2 meshes)", {
			sourceMesh: source.id(),
			targetMesh: target.id(),
			upperBound: 1,
		});
		expect(out.colored as number).toBeGreaterThan(0);

		// A vertex on the left of the plane must come out red, one on the
		// right blue — that is the whole transfer, and it also pins the flip.
		const cm = target.cm;
		let minX = Number.POSITIVE_INFINITY;
		let maxX = Number.NEGATIVE_INFINITY;
		for (let v = 0; v < cm.vertSize; v++) {
			minX = Math.min(minX, cm.vx(v));
			maxX = Math.max(maxX, cm.vx(v));
		}
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			if (cm.vx(v) < minX + (maxX - minX) * 0.2) {
				expect(red(cm.vertColor[v]), `vertex ${v}`).toBeGreaterThan(200);
			} else if (cm.vx(v) > maxX - (maxX - minX) * 0.2) {
				expect(blue(cm.vertColor[v]), `vertex ${v}`).toBeGreaterThan(200);
			}
		}
	});

	test("a source with no texture is refused", () => {
		const doc = new MeshDocument();
		const source = doc.addNewMesh("", "bare", true, gridPlane(3, 3).mesh);
		source.updateBoxAndNormals();
		expect(() =>
			kernel.applyFilter(doc, "Transfer: Texture to Vertex Color (1 or 2 meshes)", {
				sourceMesh: source.id(),
				targetMesh: source.id(),
				upperBound: 1,
			}),
		).toThrow(/no texture/);
	});

	test("nothing within the search distance is an error, not a blank result", () => {
		const doc = new MeshDocument();
		const source = texturedSource(doc);
		const far = doc.addNewMesh("", "far", true, gridPlane(3, 3).mesh);
		for (let v = 0; v < far.cm.vertSize; v++) {
			far.cm.setVert(v, far.cm.vx(v) + 1000, far.cm.vy(v), far.cm.vz(v));
		}
		far.updateBoxAndNormals();

		expect(() =>
			kernel.applyFilter(doc, "Transfer: Texture to Vertex Color (1 or 2 meshes)", {
				sourceMesh: source.id(),
				targetMesh: far.id(),
				upperBound: 0.01,
			}),
		).toThrow(/within the search distance/);
	});

	test("Vertex Attributes to Texture bakes vertex colour", () => {
		const doc = new MeshDocument();
		const source = doc.addNewMesh("", "source", true, gridPlane(4, 4).mesh);
		source.updateDataMask(MeshElement.MM_VERTCOLOR);
		source.updateBoxAndNormals();
		for (let v = 0; v < source.cm.vertSize; v++) {
			source.cm.vertColor[v] = rgba(0, 255, 0, 255);
		}
		const target = doc.addNewMesh("", "target", true, gridPlane(3, 3).mesh);
		target.updateBoxAndNormals();
		doc.setCurrentMesh(target.id());
		kernel.applyFilter(doc, "Parametrization: Flat Plane", {
			projectionPlane: 0,
			aspectRatio: false,
			sideGutter: 0,
		});

		const out = kernel.applyFilter(doc, "Transfer: Vertex Attributes to Texture (1 or 2 meshes)", {
			sourceMesh: source.id(),
			targetMesh: target.id(),
			AttributeEnum: 0,
			upperBound: 1,
			textName: "baked",
			textW: 32,
			textH: 32,
			overwrite: false,
			pullpush: true,
		});
		expect(out.texels as number).toBeGreaterThan(0);
		const image = readPng(target.textures.get("baked.png") as Uint8Array);
		expect(green(image.pixel(16, 16))).toBeGreaterThan(200);
	});

	test("baking a normal map centres a flat surface on the encoding's midpoint", () => {
		const doc = new MeshDocument();
		const source = doc.addNewMesh("", "source", true, gridPlane(4, 4).mesh);
		source.updateBoxAndNormals();
		const target = doc.addNewMesh("", "target", true, gridPlane(3, 3).mesh);
		target.updateBoxAndNormals();
		doc.setCurrentMesh(target.id());
		kernel.applyFilter(doc, "Parametrization: Flat Plane", {
			projectionPlane: 0,
			aspectRatio: false,
			sideGutter: 0,
		});

		kernel.applyFilter(doc, "Transfer: Vertex Attributes to Texture (1 or 2 meshes)", {
			sourceMesh: source.id(),
			targetMesh: target.id(),
			AttributeEnum: 1,
			upperBound: 1,
			textName: "normals",
			textW: 32,
			textH: 32,
			overwrite: false,
			pullpush: true,
		});
		// A plane in z = 0 has the normal (0,0,±1), which encodes to a mid
		// red and green and a saturated blue.
		const image = readPng(target.textures.get("normals.png") as Uint8Array);
		const middle = image.pixel(16, 16);
		expect(Math.abs(red(middle) - 128)).toBeLessThan(4);
		expect(Math.abs(green(middle) - 128)).toBeLessThan(4);
		expect(blue(middle)).toBeGreaterThan(250);
	});

	test("transferring a texture the source does not have is refused", () => {
		const doc = new MeshDocument();
		const source = doc.addNewMesh("", "bare", true, gridPlane(3, 3).mesh);
		source.updateBoxAndNormals();
		const target = doc.addNewMesh("", "target", true, gridPlane(3, 3).mesh);
		target.updateBoxAndNormals();
		doc.setCurrentMesh(target.id());
		flatten(doc);

		expect(() =>
			kernel.applyFilter(doc, "Transfer: Vertex Attributes to Texture (1 or 2 meshes)", {
				sourceMesh: source.id(),
				targetMesh: target.id(),
				AttributeEnum: 3,
				upperBound: 1,
				textName: "x",
				textW: 16,
				textH: 16,
				overwrite: false,
				pullpush: false,
			}),
		).toThrow(/no texture to transfer/);
	});
});

describe("registry", () => {
	// Derived rather than listed: a hand-written list of names goes stale the
	// moment a filter lands, and it goes stale *silently* by continuing to pass.
	const actions = kernel.filterList().filter((f) => f.plugin.pluginName() === "FilterTexture");

	test("every FilterTexture filter is implemented", () => {
		expect(actions.length).toBe(9);
		for (const action of actions) {
			expect(action.implemented, action.name).toBe(true);
		}
	});

	test("each one has a python name", () => {
		for (const action of actions) {
			expect(action.pythonName, action.name).toMatch(/^[a-z][a-z0-9_]*$/);
		}
	});
});
