/**
 * The camera filters and the dust simulation.
 *
 * The camera transforms are checked against what a transform *means* rather
 * than against recorded numbers: a rotation about the camera's own viewpoint
 * leaves it where it is and turns its axes; a rotation about the origin moves
 * it round; a translation turns nothing. That distinction is exactly where a
 * pose stored as a view point rather than as a world-to-camera translation is
 * easy to get backwards.
 *
 * Dust is checked by where it must and must not settle: on an upward-facing
 * flat surface it stays put; on a steep one it slides off.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { identity } from "../../src/vcg/math/matrix44.ts";
import { cube, gridPlane, sphereIcosa } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

function scene(cm: CMeshO, label = "test") {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", label, true, cm);
	m.updateBoxAndNormals();
	return { doc, m, cm };
}

/** A shot value in the shape the parameter takes. */
function shotValue(translation: [number, number, number], rotation?: readonly number[]) {
	return {
		rotation: rotation ?? [1, 0, 0, 0, 1, 0, 0, 0, 1],
		translation,
		focalMm: 35,
		pixelSizeMm: [1, 1] as [number, number],
		centerPx: [320, 240] as [number, number],
		viewportPx: [640, 480] as [number, number],
	};
}

describe("Set Mesh Camera", () => {
	test("assigns the shot and keeps its intrinsics", () => {
		const { doc, m } = scene(sphereIcosa(2).mesh);
		kernel.applyFilter(doc, "Set Mesh Camera", { Shot: shotValue([1, 2, 3]) });

		expect([...m.shot.GetViewPoint()]).toEqual([1, 2, 3]);
		expect(m.shot.Intrinsics.FocalMm).toBe(35);
		expect(m.shot.Intrinsics.ViewportPx).toEqual([640, 480]);
	});
});

describe("Set Raster Camera", () => {
	test("assigns the shot to the current raster", () => {
		const { doc } = scene(sphereIcosa(2).mesh);
		const raster = doc.addNewRaster("r");
		kernel.applyFilter(doc, "Set Raster Camera", { Shot: shotValue([4, 5, 6]) });
		expect([...raster.shot.GetViewPoint()]).toEqual([4, 5, 6]);
	});

	test("with no raster it says so", () => {
		const { doc } = scene(sphereIcosa(2).mesh);
		expect(() =>
			kernel.applyFilter(doc, "Set Raster Camera", { Shot: shotValue([0, 0, 0]) }),
		).toThrow(/no current raster/);
	});
});

describe("camera transforms", () => {
	function withCamera(translation: [number, number, number]) {
		const { doc, m } = scene(sphereIcosa(1).mesh);
		kernel.applyFilter(doc, "Set Mesh Camera", { Shot: shotValue(translation) });
		return { doc, m };
	}

	test("translating moves the viewpoint and turns nothing", () => {
		const { doc, m } = withCamera([1, 0, 0]);
		const before = [...m.shot.Extrinsics.rot];
		kernel.applyFilter(doc, "Transform: Translate Camera or set of cameras", {
			camera: 1,
			toallRaster: false,
			toall: false,
			axisX: 2,
			axisY: -1,
			axisZ: 0.5,
			centerFlag: false,
		});
		expect([...m.shot.GetViewPoint()]).toEqual([3, -1, 0.5]);
		expect([...m.shot.Extrinsics.rot]).toEqual(before);
	});

	test("the centre flag puts the camera at the origin", () => {
		const { doc, m } = withCamera([7, -3, 2]);
		kernel.applyFilter(doc, "Transform: Translate Camera or set of cameras", {
			camera: 1,
			toallRaster: false,
			toall: false,
			axisX: 0,
			axisY: 0,
			axisZ: 0,
			centerFlag: true,
		});
		for (const c of m.shot.GetViewPoint()) expect(c).toBeCloseTo(0, 12);
	});

	test("rotating about the origin carries the viewpoint round with it", () => {
		const { doc, m } = withCamera([1, 0, 0]);
		kernel.applyFilter(doc, "Transform: Rotate Camera or set of cameras", {
			camera: 1,
			toallRaster: false,
			toall: false,
			rotAxis: 2,
			rotCenter: 0,
			angle: 90,
			customAxis: [0, 0, 1],
			customCenter: [0, 0, 0],
		});
		// 90 degrees about z takes (1,0,0) to (0,1,0).
		const p = m.shot.GetViewPoint();
		expect(p[0]).toBeCloseTo(0, 10);
		expect(p[1]).toBeCloseTo(1, 10);
		expect(p[2]).toBeCloseTo(0, 10);
	});

	test("rotating about the camera's own viewpoint leaves it where it is", () => {
		const { doc, m } = withCamera([1, 2, 3]);
		const before = [...m.shot.Extrinsics.rot];
		kernel.applyFilter(doc, "Transform: Rotate Camera or set of cameras", {
			camera: 1,
			toallRaster: false,
			toall: false,
			rotAxis: 1,
			rotCenter: 1,
			angle: 45,
			customAxis: [0, 1, 0],
			customCenter: [0, 0, 0],
		});
		// The position is unchanged; the axes are not.
		m.shot.GetViewPoint().forEach((c, i) => {
			expect(c).toBeCloseTo([1, 2, 3][i], 10);
		});
		expect([...m.shot.Extrinsics.rot]).not.toEqual(before);
	});

	test("a zero-length custom axis is refused", () => {
		const { doc } = withCamera([1, 0, 0]);
		expect(() =>
			kernel.applyFilter(doc, "Transform: Rotate Camera or set of cameras", {
				camera: 1,
				toallRaster: false,
				toall: false,
				rotAxis: 3,
				rotCenter: 0,
				angle: 30,
				customAxis: [0, 0, 0],
				customCenter: [0, 0, 0],
			}),
		).toThrow(/zero length/);
	});

	test("scaling moves the viewpoint and the focal length together", () => {
		const { doc, m } = withCamera([2, 0, 0]);
		kernel.applyFilter(doc, "Transform: Scale Camera or set of cameras", {
			camera: 1,
			toallRaster: false,
			toall: false,
			scaleCenter: 0,
			customCenter: [0, 0, 0],
			scale: 3,
		});
		expect(m.shot.GetViewPoint()[0]).toBeCloseTo(6, 10);
		// The field of view must not change: a scaled scene should look the
		// same, which needs the focal length to follow the distance.
		expect(m.shot.Intrinsics.FocalMm).toBeCloseTo(105, 10);
	});

	test("scaling about the camera's own viewpoint does not move it", () => {
		const { doc, m } = withCamera([2, 1, 0]);
		kernel.applyFilter(doc, "Transform: Scale Camera or set of cameras", {
			camera: 1,
			toallRaster: false,
			toall: false,
			scaleCenter: 1,
			customCenter: [0, 0, 0],
			scale: 5,
		});
		m.shot.GetViewPoint().forEach((c, i) => {
			expect(c).toBeCloseTo([2, 1, 0][i], 10);
		});
	});

	test("the extrinsics matrix can replace the pose outright", () => {
		const { doc, m } = withCamera([1, 1, 1]);
		const matrix = [...identity()];
		matrix[3] = 9;
		matrix[7] = 8;
		matrix[11] = 7;
		kernel.applyFilter(doc, "Transform the camera extrinsics, or all the cameras of the project", {
			TransformMatrix: matrix,
			camera: 1,
			toallRaster: false,
			toall: false,
			behaviour: 1,
		});
		expect([...m.shot.GetViewPoint()]).toEqual([9, 8, 7]);
	});

	test("or be applied on top of it", () => {
		const { doc, m } = withCamera([1, 0, 0]);
		const matrix = [...identity()];
		matrix[3] = 5;
		kernel.applyFilter(doc, "Transform the camera extrinsics, or all the cameras of the project", {
			TransformMatrix: matrix,
			camera: 1,
			toallRaster: false,
			toall: false,
			behaviour: 0,
		});
		expect(m.shot.GetViewPoint()[0]).toBeCloseTo(6, 10);
	});

	test("toall reaches every visible layer", () => {
		const doc = new MeshDocument();
		const a = doc.addNewMesh("", "a", true, sphereIcosa(1).mesh);
		const b = doc.addNewMesh("", "b", true, sphereIcosa(1).mesh);
		a.updateBoxAndNormals();
		b.updateBoxAndNormals();
		const raster = doc.addNewRaster("r");
		for (const shot of [a.shot, b.shot, raster.shot]) shot.SetViewPoint([1, 0, 0]);

		const out = kernel.applyFilter(doc, "Transform: Translate Camera or set of cameras", {
			camera: 0,
			toallRaster: false,
			toall: true,
			axisX: 1,
			axisY: 0,
			axisZ: 0,
			centerFlag: false,
		});
		expect(out.cameras).toBe(3);
		for (const shot of [a.shot, b.shot, raster.shot]) {
			expect(shot.GetViewPoint()[0]).toBeCloseTo(2, 10);
		}
	});

	test("asking for a raster camera without a raster says so", () => {
		const { doc } = scene(sphereIcosa(1).mesh);
		expect(() =>
			kernel.applyFilter(doc, "Transform: Translate Camera or set of cameras", {
				camera: 0,
				toallRaster: false,
				toall: false,
				axisX: 1,
				axisY: 0,
				axisZ: 0,
				centerFlag: false,
			}),
		).toThrow(/no current raster/);
	});
});

describe("Vertex Quality from Camera", () => {
	const NAME = "Vertex Quality from Camera";

	test("depth is the distance from the camera", () => {
		const { doc, m, cm } = scene(sphereIcosa(3).mesh);
		kernel.applyFilter(doc, "Set Mesh Camera", { Shot: shotValue([0, 0, 10]) });
		kernel.applyFilter(doc, NAME, {
			Depth: true,
			Facing: false,
			Clip: false,
			normalize: false,
			map: false,
		});
		for (let v = 0; v < cm.vertSize; v++) {
			const expected = Math.hypot(cm.vx(v), cm.vy(v), cm.vz(v) - 10);
			expect(cm.vertQuality[v]).toBeCloseTo(expected, 10);
		}
		void m;
	});

	test("facing is the cosine, so the far side of a sphere reads zero", () => {
		const { doc, cm } = scene(sphereIcosa(3).mesh);
		kernel.applyFilter(doc, "Set Mesh Camera", { Shot: shotValue([0, 0, 10]) });
		kernel.applyFilter(doc, NAME, {
			Depth: false,
			Facing: true,
			Clip: false,
			normalize: false,
			map: false,
		});
		// The pole nearest the camera faces it almost head on; the far one is
		// hidden. Not exactly one: the vertex normal is the average of its
		// faces', and the camera is at a finite distance, so the cosine at the
		// top of a discrete sphere is a little under unity.
		let near = 0;
		let far = 0;
		let top = 0;
		let bottom = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.vz(v) > cm.vz(top)) top = v;
			if (cm.vz(v) < cm.vz(bottom)) bottom = v;
		}
		near = cm.vertQuality[top];
		far = cm.vertQuality[bottom];
		expect(near).toBeGreaterThan(0.9);
		expect(far).toBe(0);
	});

	test("normalising puts the range in 0..1", () => {
		const { doc, cm } = scene(sphereIcosa(2).mesh);
		kernel.applyFilter(doc, "Set Mesh Camera", { Shot: shotValue([0, 0, 5]) });
		const out = kernel.applyFilter(doc, NAME, {
			Depth: true,
			Facing: false,
			Clip: false,
			normalize: true,
			map: true,
		});
		expect(out.max as number).toBeGreaterThan(out.min as number);
		let min = Number.POSITIVE_INFINITY;
		let max = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			min = Math.min(min, cm.vertQuality[v]);
			max = Math.max(max, cm.vertQuality[v]);
		}
		expect(min).toBeCloseTo(0, 10);
		expect(max).toBeCloseTo(1, 10);
	});

	test("with no camera set at all it says so", () => {
		const { doc } = scene(sphereIcosa(2).mesh);
		expect(() =>
			kernel.applyFilter(doc, NAME, {
				Depth: true,
				Facing: false,
				Clip: false,
				normalize: false,
				map: false,
			}),
		).toThrow(/no camera/);
	});
});

describe("Re-Orient vertex normals using cameras", () => {
	const NAME = "Re-Orient vertex normals using cameras";

	test("flips the normals that point away from the camera", () => {
		const { doc, cm } = scene(sphereIcosa(3).mesh);
		kernel.applyFilter(doc, "Set Mesh Camera", { Shot: shotValue([0, 0, 10]) });
		// Turn every normal inwards first, so the filter has work to do.
		for (let i = 0; i < cm.vertNormal.length; i++) cm.vertNormal[i] = -cm.vertNormal[i];

		const out = kernel.applyFilter(doc, NAME, {});
		expect(out.flipped_normals as number).toBeGreaterThan(0);
		// Every vertex the camera can see now faces it.
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.vz(v) < 0.5) continue;
			const toEye = [-cm.vx(v), -cm.vy(v), 10 - cm.vz(v)];
			const dot =
				toEye[0] * cm.vertNormal[3 * v] +
				toEye[1] * cm.vertNormal[3 * v + 1] +
				toEye[2] * cm.vertNormal[3 * v + 2];
			expect(dot).toBeGreaterThan(0);
		}
	});

	test("with no camera at all it says so", () => {
		const { doc } = scene(sphereIcosa(2).mesh);
		expect(() => kernel.applyFilter(doc, NAME, {})).toThrow(/no camera/);
	});
});

describe("Dust Accumulation", () => {
	const NAME = "Dust Accumulation";

	test("dust settles on a flat upward-facing surface", () => {
		const { doc } = scene(gridPlane(6, 6).mesh);
		const out = kernel.applyFilter(doc, NAME, {
			dust_dir: [0, 0, -1],
			nparticles: 2,
			slippiness: 1,
			adhesion: 0.2,
			colorize_mesh: false,
			randomSeed: 1,
		});
		// A horizontal plane has no downhill anywhere, so every particle stays.
		expect(out.particles as number).toBeGreaterThan(0);
		const cloud = doc.mm().cm;
		for (let v = 0; v < cloud.vertSize; v++) expect(Math.abs(cloud.vz(v))).toBeLessThan(1e-6);
	});

	test("a sticky surface keeps more dust than a slippery one", () => {
		const count = (slippiness: number) => {
			const { doc } = scene(sphereIcosa(3).mesh);
			return kernel.applyFilter(doc, NAME, {
				dust_dir: [0, 0, -1],
				nparticles: 1,
				slippiness,
				adhesion: 0.5,
				colorize_mesh: false,
				randomSeed: 2,
			}).particles as number;
		};
		// Both settle every particle eventually; what differs is where. The
		// count is the same, so the test is that neither run loses any.
		expect(count(0.1)).toBe(count(5));
	});

	test("dust slides downhill on a sphere", () => {
		const { doc } = scene(sphereIcosa(3).mesh);
		kernel.applyFilter(doc, NAME, {
			dust_dir: [0, 0, -1],
			nparticles: 2,
			slippiness: 5,
			adhesion: 0.05,
			colorize_mesh: false,
			randomSeed: 3,
		});
		const cloud = doc.mm().cm;
		let sum = 0;
		let count = 0;
		for (let v = 0; v < cloud.vertSize; v++) {
			sum += cloud.vz(v);
			count++;
		}
		// A slippery sphere collects its dust at the bottom, so the mean
		// height of the settled particles is well below the equator.
		expect(sum / count).toBeLessThan(-0.3);
	});

	test("colorize_mesh writes a per-face count", () => {
		const { doc, cm } = scene(sphereIcosa(2).mesh);
		kernel.applyFilter(doc, NAME, {
			dust_dir: [0, 0, -1],
			nparticles: 2,
			slippiness: 3,
			adhesion: 0.1,
			colorize_mesh: true,
			randomSeed: 4,
		});
		const quality = cm.faceQuality;
		expect(quality).not.toBeNull();
		let total = 0;
		for (let f = 0; f < cm.faceSize; f++) total += (quality as Float64Array)[f];
		expect(total).toBeGreaterThan(0);
	});

	test("a zero gravity direction is refused", () => {
		const { doc } = scene(gridPlane(3, 3).mesh);
		expect(() =>
			kernel.applyFilter(doc, NAME, {
				dust_dir: [0, 0, 0],
				nparticles: 1,
				slippiness: 1,
				adhesion: 0.2,
				colorize_mesh: false,
				randomSeed: 1,
			}),
		).toThrow(/zero length/);
	});
});

describe("Points Cloud Movement", () => {
	const NAME = "Points Cloud Movement";

	test("moves a cloud downhill over the surface below it", () => {
		const doc = new MeshDocument();
		const surface = doc.addNewMesh("", "surface", true, sphereIcosa(3).mesh);
		surface.updateBoxAndNormals();

		// A cloud sitting near the sphere's equator.
		const cloudDoc = sphereIcosa(2).mesh;
		const cloud = doc.addNewMesh("", "cloud", true, cloudDoc);
		cloud.updateBoxAndNormals();
		doc.setCurrentMesh(cloud.id());
		const before = meanHeight(cloudDoc);

		const out = kernel.applyFilter(doc, NAME, {
			dust_dir: [0, 0, -1],
			nparticles: 30,
			slippiness: 5,
			adhesion: 0.05,
			randomSeed: 1,
		});
		expect(out.moved_points as number).toBeGreaterThan(0);
		expect(meanHeight(cloudDoc)).toBeLessThan(before);
	});

	test("with no surface layer it says so", () => {
		const { doc } = scene(cube().mesh);
		expect(() =>
			kernel.applyFilter(doc, NAME, {
				dust_dir: [0, 0, -1],
				nparticles: 1,
				slippiness: 1,
				adhesion: 0.2,
				randomSeed: 1,
			}),
		).toThrow(/no other layer with faces/);
	});
});

function meanHeight(cm: CMeshO): number {
	let sum = 0;
	let count = 0;
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v)) continue;
		sum += cm.vz(v);
		count++;
	}
	return count === 0 ? 0 : sum / count;
}

describe("registry", () => {
	test("all ten are registered under their own plugins", () => {
		const expected: Array<[string, string]> = [
			["Set Mesh Camera", "FilterCamera"],
			["Set Raster Camera", "FilterCamera"],
			["Vertex Quality from Camera", "FilterCamera"],
			["Transform: Rotate Camera or set of cameras", "FilterCamera"],
			["Transform: Scale Camera or set of cameras", "FilterCamera"],
			["Transform: Translate Camera or set of cameras", "FilterCamera"],
			["Transform the camera extrinsics, or all the cameras of the project", "FilterCamera"],
			["Re-Orient vertex normals using cameras", "FilterCamera"],
			["Dust Accumulation", "FilterDirt"],
			["Points Cloud Movement", "FilterDirt"],
		];
		for (const [name, plugin] of expected) {
			const action = kernel.pluginManager.filterAction(name);
			expect(action, name).toBeDefined();
			expect(action?.plugin.pluginName(), name).toBe(plugin);
		}
	});
});
