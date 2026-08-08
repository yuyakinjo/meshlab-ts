/**
 * Colour projection, alpha wrapping, and cubic stylization.
 *
 * Projection is checked with a synthetic camera and a two-colour image, so
 * every vertex has a known right answer: the half of the mesh the camera sees
 * takes the colour of the half of the image it lands in, and the far side
 * takes none at all. That last part is the depth test, which is the whole
 * difference between projecting and painting through the object.
 *
 * The wrap is checked by what a wrap guarantees — watertight, outside the
 * input, and closing gaps narrower than alpha — rather than by its triangle
 * count, which is a function of the grid.
 */
import { describe, expect, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { RasterPlane } from "../../src/common/ml_document/raster_model.ts";
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import { Clean } from "../../src/vcg/complex/clean.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { UpdateTopology } from "../../src/vcg/complex/update/topology.ts";
import { blue, red, rgba } from "../../src/vcg/space/color4.ts";
import { Image } from "../../src/vcg/space/image/image.ts";
import { readPng, writePng } from "../../src/vcg/space/image/png.ts";
import { assertAllocatorConsistent } from "../helpers/invariants.ts";
import { cube, sphereIcosa } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

function scene(cm: CMeshO, label = "test") {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", label, true, cm);
	m.updateBoxAndNormals();
	return { doc, m, cm };
}

function scratch(): string {
	return mkdtempSync(join(tmpdir(), "meshlab-ts-projection-"));
}

/**
 * A document with a sphere and one camera on the +z axis looking at it,
 * holding an image that is red on the left and blue on the right.
 */
function projectionScene(subdiv = 3) {
	const { doc, m, cm } = scene(sphereIcosa(subdiv).mesh, "sphere");
	const dir = scratch();
	const image = new Image(64, 64);
	for (let y = 0; y < 64; y++) {
		for (let x = 0; x < 64; x++) {
			image.setPixel(x, y, x < 32 ? rgba(255, 0, 0, 255) : rgba(0, 0, 255, 255));
		}
	}
	const path = join(dir, "view.png");
	writeFileSync(path, writePng(image));

	const raster = doc.addNewRaster("front");
	raster.addPlane(new RasterPlane(path));
	// Looking down -z from (0,0,5): the camera's own z axis points at the
	// mesh, its x to the right and its y up.
	raster.shot.Extrinsics.SetRot(
		Float64Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1]),
	);
	raster.shot.SetViewPoint([0, 0, 5]);
	raster.shot.Intrinsics.FocalMm = 40;
	raster.shot.Intrinsics.PixelSizeMm = [1, 1];
	raster.shot.Intrinsics.ViewportPx = [64, 64];
	raster.shot.Intrinsics.centreOnViewport();
	return { doc, m, cm, raster, path };
}

describe("Project current raster color to current mesh", () => {
	const NAME = "Project current raster color to current mesh";

	test("colours the near side and leaves the far side alone", () => {
		const { doc, cm } = projectionScene();
		const out = kernel.applyFilter(doc, NAME, {
			usedepth: true,
			deptheta: 0.01,
			onselection: false,
			blankColor: rgba(0, 255, 0, 255),
		});
		expect(out.colored as number).toBeGreaterThan(0);
		expect(out.not_seen as number).toBeGreaterThan(0);

		// The camera is on +z, so only the +z hemisphere may be painted; the
		// rest keeps the blank colour. Without the depth test the far side
		// would be painted too, which is the bug this pins.
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.vz(v) > 0.4) continue;
			if (cm.vz(v) < -0.6) {
				expect(cm.vertColor[v], `vertex ${v} on the far side`).toBe(rgba(0, 255, 0, 255));
			}
		}
	});

	test("left of the image lands on the left of the mesh", () => {
		const { doc, cm } = projectionScene();
		kernel.applyFilter(doc, NAME, {
			usedepth: true,
			deptheta: 0.01,
			onselection: false,
			blankColor: rgba(0, 0, 0, 0),
		});
		// The camera's x axis is world +x, so the red half of the image falls
		// on the negative-x side of the sphere.
		let negative = 0;
		let positive = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.vz(v) < 0.5) continue;
			if (cm.vx(v) < -0.2 && red(cm.vertColor[v]) > 200) negative++;
			if (cm.vx(v) > 0.2 && blue(cm.vertColor[v]) > 200) positive++;
		}
		expect(negative).toBeGreaterThan(0);
		expect(positive).toBeGreaterThan(0);
	});

	test("a transparent blank colour leaves the unseen vertices untouched", () => {
		const { doc, cm } = projectionScene();
		for (let v = 0; v < cm.vertSize; v++) cm.vertColor[v] = rgba(7, 8, 9, 255);
		kernel.applyFilter(doc, NAME, {
			usedepth: true,
			deptheta: 0.01,
			onselection: false,
			blankColor: rgba(0, 0, 0, 0),
		});
		let untouched = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.vertColor[v] === rgba(7, 8, 9, 255)) untouched++;
		}
		expect(untouched).toBeGreaterThan(0);
	});

	test("the depth test rejects a surface hidden behind another", () => {
		// Two spheres in a line from the camera. The rear one's near side
		// faces the camera, so the normal test alone would happily paint it —
		// on a single convex mesh the two tests agree, which is why this needs
		// an occluder to distinguish them.
		const { doc, raster } = projectionScene(2);
		const behind = sphereIcosa(2).mesh;
		for (let v = 0; v < behind.vertSize; v++) {
			behind.setVert(v, behind.vx(v), behind.vy(v), behind.vz(v) - 3);
		}
		const rear = doc.addNewMesh("", "rear", true, behind);
		rear.updateBoxAndNormals();
		doc.setCurrentMesh(doc.meshIterator()[0].id());
		kernel.applyFilter(doc, "Flatten Visible Layers", {
			MergeVisible: true,
			DeleteLayer: true,
			MergeVertices: false,
		});
		void raster;

		const seen = (usedepth: boolean) =>
			kernel.applyFilter(doc, NAME, {
				usedepth,
				deptheta: 0.01,
				onselection: false,
				blankColor: rgba(0, 0, 0, 0),
			}).colored as number;

		expect(seen(false)).toBeGreaterThan(seen(true));
	});

	test("with no raster it says so", () => {
		const { doc } = scene(sphereIcosa(2).mesh);
		expect(() =>
			kernel.applyFilter(doc, NAME, {
				usedepth: true,
				deptheta: 0.5,
				onselection: false,
				blankColor: rgba(0, 0, 0, 255),
			}),
		).toThrow(/no current raster/);
	});

	test("a missing image file is named rather than skipped", () => {
		const { doc } = scene(sphereIcosa(2).mesh);
		const raster = doc.addNewRaster("missing");
		raster.addPlane(new RasterPlane(join(scratch(), "nope.png")));
		expect(() =>
			kernel.applyFilter(doc, NAME, {
				usedepth: true,
				deptheta: 0.5,
				onselection: false,
				blankColor: rgba(0, 0, 0, 255),
			}),
		).toThrow(/cannot read/);
	});
});

describe("Project active rasters color to current mesh", () => {
	const NAME = "Project active rasters color to current mesh";

	test("blends every visible raster", () => {
		const { doc } = projectionScene();
		const out = kernel.applyFilter(doc, NAME, {
			usedepth: true,
			deptheta: 0.01,
			onselection: false,
			useangle: true,
			usedistance: true,
			useborders: true,
			usealpha: false,
		});
		expect(out.rasters).toBe(1);
		expect(out.colored as number).toBeGreaterThan(0);
	});

	test("the border weight fades the frame's edge out", () => {
		// With the border weight on, a vertex projecting near the edge of the
		// image contributes nothing, so fewer vertices end up coloured.
		const count = (useborders: boolean) => {
			const { doc } = projectionScene();
			return kernel.applyFilter(doc, NAME, {
				usedepth: true,
				deptheta: 0.01,
				onselection: false,
				useangle: false,
				usedistance: false,
				useborders,
				usealpha: false,
			}).colored as number;
		};
		expect(count(true)).toBeLessThanOrEqual(count(false));
	});
});

describe("Project active rasters color to current mesh, filling the texture", () => {
	const NAME = "Project active rasters color to current mesh, filling the texture";

	test("bakes the projection into a texture", () => {
		const { doc, m } = projectionScene(2);
		doc.setCurrentMesh(m.id());
		kernel.applyFilter(doc, "Parametrization: Trivial Per-Triangle", {
			sidedim: 0,
			textdim: 512,
			border: 2,
			method: 0,
		});
		const out = kernel.applyFilter(doc, NAME, {
			usedepth: true,
			deptheta: 0.01,
			onselection: false,
			useangle: true,
			usedistance: true,
			useborders: false,
			usealpha: false,
			textName: "projected",
			textW: 128,
			textH: 128,
			pullpush: true,
		});
		expect(out.texels as number).toBeGreaterThan(0);
		const image = readPng(m.textures.get("projected.png") as Uint8Array);
		expect(image.width).toBe(128);
	});

	test("a mesh with no parametrisation is told to make one", () => {
		const { doc } = projectionScene(2);
		expect(() =>
			kernel.applyFilter(doc, NAME, {
				usedepth: true,
				deptheta: 0.01,
				onselection: false,
				useangle: true,
				usedistance: true,
				useborders: false,
				usealpha: false,
				textName: "x",
				textW: 64,
				textH: 64,
				pullpush: false,
			}),
		).toThrow(/no texture coordinates/);
	});
});

describe("Alpha Wrap", () => {
	const NAME = "Alpha Wrap";

	test("produces a watertight shell around the input", () => {
		const { doc } = scene(sphereIcosa(2).mesh);
		const out = kernel.applyFilter(doc, NAME, {
			Alpha: 0.3,
			Offset: 0.05,
			Resolution: 48,
		});
		expect(out.face_number as number).toBeGreaterThan(100);

		const shell = doc.mm().cm;
		UpdateTopology.faceFace(shell);
		expect(Clean.countEdgeNum(shell).boundary).toBe(0);
		assertAllocatorConsistent(shell);
	});

	test("the shell lies outside the input", () => {
		const { doc } = scene(sphereIcosa(3).mesh);
		kernel.applyFilter(doc, NAME, { Alpha: 0.3, Offset: 0.08, Resolution: 48 });
		const shell = doc.mm().cm;
		let min = Number.POSITIVE_INFINITY;
		for (let v = 0; v < shell.vertSize; v++) {
			if (shell.isVertD(v)) continue;
			min = Math.min(min, Math.hypot(shell.vx(v), shell.vy(v), shell.vz(v)));
		}
		// A unit sphere wrapped with an offset: nothing may be inside it.
		expect(min).toBeGreaterThan(0.95);
	});

	test("it closes a gap narrower than alpha, and keeps one wider", () => {
		const wrapCount = (gap: number, alpha: number) => {
			// Two boxes separated by `gap`.
			const doc = new MeshDocument();
			const left = cube(1).mesh;
			for (let v = 0; v < left.vertSize; v++) {
				left.setVert(v, left.vx(v) - (1 + gap / 2), left.vy(v), left.vz(v));
			}
			const right = cube(1).mesh;
			for (let v = 0; v < right.vertSize; v++) {
				right.setVert(v, right.vx(v) + (1 + gap / 2), right.vy(v), right.vz(v));
			}
			const a = doc.addNewMesh("", "a", true, left);
			const b = doc.addNewMesh("", "b", true, right);
			a.updateBoxAndNormals();
			b.updateBoxAndNormals();
			doc.setCurrentMesh(a.id());
			kernel.applyFilter(doc, "Flatten Visible Layers", {
				MergeVisible: true,
				DeleteLayer: true,
				MergeVertices: false,
			});
			kernel.applyFilter(doc, NAME, { Alpha: alpha, Offset: 0.05, Resolution: 56 });
			const shell = doc.mm().cm;
			UpdateTopology.faceFace(shell);
			return Clean.countConnectedComponents(shell);
		};

		// A ball far bigger than the gap bridges it into one piece; a small
		// one rolls through and leaves two.
		expect(wrapCount(0.6, 1.2)).toBe(1);
		expect(wrapCount(1.6, 0.3)).toBe(2);
	});

	test("an alpha smaller than two grid cells is refused with the reason", () => {
		const { doc } = scene(sphereIcosa(2).mesh);
		expect(() =>
			kernel.applyFilter(doc, NAME, { Alpha: 0.005, Offset: 0.01, Resolution: 16 }),
		).toThrow(/smaller than two grid cells/);
	});

	test("a zero offset is refused", () => {
		const { doc } = scene(sphereIcosa(2).mesh);
		expect(() => kernel.applyFilter(doc, NAME, { Alpha: 0.3, Offset: 0, Resolution: 32 })).toThrow(
			/offset must be positive/,
		);
	});
});

describe("Cubic stylization", () => {
	const NAME = "Cubic stylization";

	test("zero cubeness leaves the mesh where it was", () => {
		const { doc, cm } = scene(sphereIcosa(3).mesh);
		const before = Float64Array.from(cm.vertCoord);
		kernel.applyFilter(doc, NAME, { lcubeness: 0, iterations: 5, applycol: false });
		let worst = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			worst = Math.max(
				worst,
				Math.hypot(
					cm.vx(v) - before[3 * v],
					cm.vy(v) - before[3 * v + 1],
					cm.vz(v) - before[3 * v + 2],
				),
			);
		}
		// With no cubeness term this is plain ARAP against its own rest pose,
		// whose optimum is the rest pose.
		expect(worst).toBeLessThan(0.02);
	});

	test("it keeps the connectivity and the vertex count", () => {
		const { doc, cm } = scene(sphereIcosa(3).mesh);
		const before = { vn: cm.vn, fn: cm.fn };
		kernel.applyFilter(doc, NAME, { lcubeness: 0.3, iterations: 8, applycol: false });
		expect(cm.vn).toBe(before.vn);
		expect(cm.fn).toBe(before.fn);
		assertAllocatorConsistent(cm);
	});

	test("stylizing a sphere pushes its normals onto the axes", () => {
		const axisAlignment = (cubeness: number) => {
			const { doc, cm } = scene(sphereIcosa(3).mesh);
			kernel.applyFilter(doc, NAME, { lcubeness: cubeness, iterations: 12, applycol: false });
			doc.mm().updateBoxAndNormals();
			// The mean of the largest normal component: 1 for a perfect cube,
			// about 0.75 for a sphere.
			let sum = 0;
			let count = 0;
			for (let v = 0; v < cm.vertSize; v++) {
				if (cm.isVertD(v)) continue;
				sum += Math.max(
					Math.abs(cm.vertNormal[3 * v]),
					Math.abs(cm.vertNormal[3 * v + 1]),
					Math.abs(cm.vertNormal[3 * v + 2]),
				);
				count++;
			}
			return sum / count;
		};
		expect(axisAlignment(0.5)).toBeGreaterThan(axisAlignment(0));
	});

	test("colorize records how far each vertex moved", () => {
		const { doc, cm } = scene(sphereIcosa(2).mesh);
		kernel.applyFilter(doc, NAME, { lcubeness: 0.4, iterations: 6, applycol: true });
		let max = 0;
		for (let v = 0; v < cm.vertSize; v++) max = Math.max(max, cm.vertQuality[v]);
		expect(max).toBeGreaterThan(0);
	});

	test("a negative cubeness is refused", () => {
		const { doc } = scene(sphereIcosa(2).mesh);
		expect(() =>
			kernel.applyFilter(doc, NAME, { lcubeness: -1, iterations: 5, applycol: false }),
		).toThrow(/cannot be negative/);
	});

	test("an unreferenced vertex does not make the solve singular", () => {
		const cm = sphereIcosa(2).mesh;
		const spare = Allocator.addVertices(cm, 1);
		cm.setVert(spare, 5, 5, 5);
		const { doc } = scene(cm);
		expect(() =>
			kernel.applyFilter(doc, NAME, { lcubeness: 0.2, iterations: 4, applycol: false }),
		).not.toThrow();
	});
});

describe("registry", () => {
	test("all five are registered under their own plugins", () => {
		const expected: Array<[string, string]> = [
			["Project current raster color to current mesh", "FilterColorProjection"],
			["Project active rasters color to current mesh", "FilterColorProjection"],
			[
				"Project active rasters color to current mesh, filling the texture",
				"FilterColorProjection",
			],
			["Alpha Wrap", "FilterMeshAlphaWrap"],
			["Cubic stylization", "FilterCubization"],
		];
		for (const [name, plugin] of expected) {
			const action = kernel.pluginManager.filterAction(name);
			expect(action, name).toBeDefined();
			expect(action?.plugin.pluginName(), name).toBe(plugin);
		}
	});
});
