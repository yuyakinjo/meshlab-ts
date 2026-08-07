/**
 * `filter_colorproc` — colour adjustment, quality mapping, and the transfers
 * between the two element types.
 *
 * Colour operations have exact arithmetic, so most of these check the number
 * rather than a property: inverting twice is the identity, a white balance
 * against a colour turns that colour white, the quality ramp puts red at the
 * bottom and blue at the top. Where the operation is a blend or an average the
 * check is the bound it has to respect.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import { MLNotImplementedException } from "../../src/common/utilities/ml_exception.ts";
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { Platonic } from "../../src/vcg/complex/create/platonic.ts";
import { VertexFlag } from "../../src/vcg/complex/flags.ts";
import {
	blue,
	colorRamp,
	desaturate,
	fromHsv,
	green,
	invert,
	lerpColor,
	lightness,
	luminosity,
	red,
	rgba,
	whiteBalance,
} from "../../src/vcg/space/color4.ts";

const kernel = MeshLabKernel.default();

/** A document holding a sphere, with the named channels enabled. */
function scene(channels: number = MeshElement.MM_VERTCOLOR, subdiv = 2) {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", "m", true, Platonic.sphere(subdiv));
	m.updateDataMask(channels);
	m.updateBoxAndNormals();
	return { doc, m, cm: m.cm };
}

const paint = (cm: CMeshO, c: number) => {
	for (let v = 0; v < cm.vertSize; v++) cm.vertColor[v] = c;
};

describe("colour arithmetic", () => {
	test("packs and unpacks in vcg's byte order", () => {
		const c = rgba(10, 20, 30, 40);
		expect(red(c)).toBe(10);
		expect(green(c)).toBe(20);
		expect(blue(c)).toBe(30);
		// 0xAABBGGRR — red in the low byte, which is what a PLY reader produces.
		expect(c & 0xff).toBe(10);
	});

	test("clamps rather than wrapping", () => {
		// A channel that overflowed into the next one would show up as a
		// completely different colour, not merely a bright one.
		expect(red(rgba(300, 0, 0))).toBe(255);
		expect(red(rgba(-40, 0, 0))).toBe(0);
		expect(green(rgba(300, 400, 0))).toBe(255);
	});

	test("inverting twice is the identity", () => {
		for (const c of [rgba(0, 0, 0), rgba(255, 255, 255), rgba(13, 200, 47)]) {
			expect(invert(invert(c))).toBe(c);
		}
	});

	test("the three desaturations agree on grey and differ on colour", () => {
		const grey = rgba(90, 90, 90);
		for (const method of [0, 1, 2]) expect(desaturate(grey, method)).toBe(grey);
		// Pure green: lightness 127.5, luminosity 182, average 85. All different,
		// and the ordering is what makes luminosity the perceptual one.
		const g = rgba(0, 255, 0);
		expect(red(desaturate(g, 0))).toBe(128);
		expect(red(desaturate(g, 1))).toBe(182);
		expect(red(desaturate(g, 2))).toBe(85);
	});

	test("white balance turns the named colour white", () => {
		const cast = rgba(200, 220, 180);
		const fixed = whiteBalance(cast, cast);
		expect(red(fixed)).toBe(255);
		expect(green(fixed)).toBe(255);
		expect(blue(fixed)).toBe(255);
	});

	test("white balance survives a channel that recorded nothing", () => {
		// Dividing by a zero channel would give infinity; it is treated as one.
		expect(() => whiteBalance(rgba(10, 10, 10), rgba(0, 255, 255))).not.toThrow();
		expect(red(whiteBalance(rgba(10, 10, 10), rgba(0, 255, 255)))).toBe(255);
	});

	test("the quality ramp runs red to blue through green", () => {
		expect(colorRamp(0, 1, -1)).toBe(rgba(255, 0, 0));
		expect(colorRamp(0, 1, 0)).toBe(rgba(255, 0, 0));
		expect(colorRamp(0, 1, 2)).toBe(rgba(0, 0, 255));
		expect(colorRamp(0, 1, 1)).toBe(rgba(0, 0, 255));
		// The midpoint is green, which is the convention every MeshLab quality
		// visualisation relies on.
		expect(colorRamp(0, 1, 0.5)).toBe(rgba(0, 255, 0));
		expect(colorRamp(0, 1, 0.25)).toBe(rgba(255, 255, 0));
		expect(colorRamp(0, 1, 0.75)).toBe(rgba(0, 255, 255));
	});

	test("the ramp handles a reversed and a collapsed range", () => {
		// Reversed means the scale runs the other way, not that it breaks.
		expect(colorRamp(1, 0, 0)).toBe(colorRamp(0, 1, 1));
		expect(() => colorRamp(5, 5, 5)).not.toThrow();
	});

	test("HSV round-trips the primaries", () => {
		expect(fromHsv(0, 1, 1)).toBe(rgba(255, 0, 0));
		expect(fromHsv(120, 1, 1)).toBe(rgba(0, 255, 0));
		expect(fromHsv(240, 1, 1)).toBe(rgba(0, 0, 255));
		expect(fromHsv(0, 0, 1)).toBe(rgba(255, 255, 255));
		expect(fromHsv(200, 1, 0)).toBe(rgba(0, 0, 0));
		// Hue wraps rather than clamping.
		expect(fromHsv(360, 1, 1)).toBe(fromHsv(0, 1, 1));
		expect(fromHsv(-120, 1, 1)).toBe(fromHsv(240, 1, 1));
	});

	test("lerp stays inside the two ends", () => {
		const a = rgba(0, 0, 0);
		const b = rgba(255, 255, 255);
		expect(lerpColor(a, b, 0)).toBe(a);
		expect(lerpColor(a, b, 1)).toBe(b);
		expect(red(lerpColor(a, b, 0.5))).toBe(128);
		// Out of range is clamped, not extrapolated.
		expect(lerpColor(a, b, -1)).toBe(a);
		expect(lerpColor(a, b, 2)).toBe(b);
	});

	test("lightness and luminosity agree only on grey", () => {
		expect(lightness(rgba(70, 70, 70))).toBe(70);
		expect(luminosity(rgba(70, 70, 70))).toBeCloseTo(70, 9);
		expect(lightness(rgba(255, 0, 0))).not.toBeCloseTo(luminosity(rgba(255, 0, 0)), 1);
	});
});

describe("the per-vertex colour filters", () => {
	test("filling sets every vertex to one colour", () => {
		const { doc, cm } = scene();
		kernel.applyFilter(doc, "Vertex Color Filling", { color1: [10, 20, 30] });
		for (let v = 0; v < cm.vn; v++) expect(cm.vertColor[v]).toBe(rgba(10, 20, 30));
	});

	test("only-on-selection leaves the rest alone", () => {
		const { doc, cm } = scene();
		paint(cm, rgba(0, 0, 0));
		for (let v = 0; v < cm.vertSize; v += 2) cm.vertFlags[v] |= VertexFlag.SELECTED;
		const out = kernel.applyFilter(doc, "Vertex Color Filling", {
			color1: [255, 255, 255],
			onSelected: true,
		});
		expect(out.vertex_number).toBe(Math.ceil(cm.vn / 2));
		for (let v = 0; v < cm.vn; v++) {
			expect(cm.vertColor[v]).toBe(v % 2 === 0 ? rgba(255, 255, 255) : rgba(0, 0, 0));
		}
	});

	test("thresholding splits on lightness", () => {
		const { doc, cm } = scene();
		for (let v = 0; v < cm.vertSize; v++) cm.vertColor[v] = rgba(v % 256, v % 256, v % 256);
		kernel.applyFilter(doc, "Vertex Color Thresholding", {
			color1: [0, 0, 0],
			color2: [255, 255, 255],
			threshold: 100,
		});
		for (let v = 0; v < cm.vn; v++) {
			const expected = v % 256 <= 100 ? rgba(0, 0, 0) : rgba(255, 255, 255);
			expect(cm.vertColor[v], `v${v}`).toBe(expected);
		}
	});

	test("inversion is its own undo, through the filter too", () => {
		const { doc, cm } = scene();
		for (let v = 0; v < cm.vertSize; v++) cm.vertColor[v] = rgba(v * 7, v * 3, v * 11);
		const before = Uint32Array.from(cm.vertColor);
		kernel.applyFilter(doc, "Vertex Color Invert", {});
		kernel.applyFilter(doc, "Vertex Color Invert", {});
		expect(Array.from(cm.vertColor)).toEqual(Array.from(before));
	});

	test("brightness runs the mesh to white and to black", () => {
		for (const [brightness, expected] of [
			[255, 255],
			[-255, 0],
		] as const) {
			const { doc, cm } = scene();
			paint(cm, rgba(128, 128, 128));
			kernel.applyFilter(doc, "Vertex Color Brightness Contrast Gamma", { brightness });
			expect(red(cm.vertColor[0]), `brightness ${brightness}`).toBe(expected);
		}
	});

	test("the neutral settings change nothing", () => {
		const { doc, cm } = scene();
		paint(cm, rgba(77, 133, 201));
		kernel.applyFilter(doc, "Vertex Color Brightness Contrast Gamma", {
			brightness: 0,
			contrast: 0,
			gamma: 1,
		});
		// Rounding through 0..1 and back can move a channel by one step; more
		// than that would mean the identity is not the identity.
		expect(Math.abs(red(cm.vertColor[0]) - 77)).toBeLessThanOrEqual(1);
		expect(Math.abs(green(cm.vertColor[0]) - 133)).toBeLessThanOrEqual(1);
		expect(Math.abs(blue(cm.vertColor[0]) - 201)).toBeLessThanOrEqual(1);
	});

	test("levels can be restricted to one channel", () => {
		const { doc, cm } = scene();
		paint(cm, rgba(100, 100, 100));
		kernel.applyFilter(doc, "Vertex Color Levels Adjustment", {
			gamma: 0.5,
			rCh: true,
			gCh: false,
			bCh: false,
		});
		expect(red(cm.vertColor[0])).not.toBe(100);
		expect(green(cm.vertColor[0])).toBe(100);
		expect(blue(cm.vertColor[0])).toBe(100);
	});

	test("levels with an inverted input range does not divide by zero", () => {
		const { doc, cm } = scene();
		paint(cm, rgba(100, 100, 100));
		expect(() =>
			kernel.applyFilter(doc, "Vertex Color Levels Adjustment", { in_min: 255, in_max: 0 }),
		).not.toThrow();
		expect(Number.isFinite(red(cm.vertColor[0]))).toBe(true);
	});

	test("colourisation blends toward the chosen hue", () => {
		const { doc, cm } = scene();
		paint(cm, rgba(0, 0, 0));
		kernel.applyFilter(doc, "Vertex Color Colourisation", {
			hue: 0,
			saturation: 100,
			luminance: 100,
			intensity: 100,
		});
		expect(cm.vertColor[0]).toBe(rgba(255, 0, 0));

		// Half the blending gets half the way there.
		paint(cm, rgba(0, 0, 0));
		kernel.applyFilter(doc, "Vertex Color Colourisation", {
			hue: 0,
			saturation: 100,
			luminance: 100,
			intensity: 50,
		});
		expect(red(cm.vertColor[0])).toBeGreaterThan(120);
		expect(red(cm.vertColor[0])).toBeLessThan(136);
	});

	test("desaturation leaves every channel equal", () => {
		const { doc, cm } = scene();
		for (let v = 0; v < cm.vertSize; v++) cm.vertColor[v] = rgba(v * 5, v * 9, v * 2);
		kernel.applyFilter(doc, "Vertex Color Desaturation", { method: 1 });
		for (let v = 0; v < cm.vn; v++) {
			expect(red(cm.vertColor[v])).toBe(green(cm.vertColor[v]));
			expect(green(cm.vertColor[v])).toBe(blue(cm.vertColor[v]));
		}
	});

	test("white balance through the filter", () => {
		const { doc, cm } = scene();
		paint(cm, rgba(200, 180, 160));
		kernel.applyFilter(doc, "Vertex Color White Balance", { color: [200, 180, 160] });
		expect(cm.vertColor[0]).toBe(rgba(255, 255, 255));
	});

	test("noise stays inside the band its bit count promises", () => {
		const { doc, cm } = scene(MeshElement.MM_VERTCOLOR, 3);
		paint(cm, rgba(128, 128, 128));
		kernel.applyFilter(doc, "Color noise", { noiseBits: 3 });
		let moved = 0;
		for (let v = 0; v < cm.vn; v++) {
			// Three bits means offsets in [-8, +8].
			expect(Math.abs(red(cm.vertColor[v]) - 128)).toBeLessThanOrEqual(8);
			if (cm.vertColor[v] !== rgba(128, 128, 128)) moved++;
		}
		expect(moved).toBeGreaterThan(cm.vn / 2);
	});

	test("zero noise bits is close to a no-op, and negative is refused", () => {
		const { doc, cm } = scene();
		paint(cm, rgba(128, 128, 128));
		kernel.applyFilter(doc, "Color noise", { noiseBits: 0 });
		for (let v = 0; v < cm.vn; v++) {
			expect(Math.abs(red(cm.vertColor[v]) - 128)).toBeLessThanOrEqual(1);
		}
		expect(() => kernel.applyFilter(doc, "Color noise", { noiseBits: -2 })).toThrow();
	});
});

describe("quality and colour together", () => {
	test("vertex quality maps onto the ramp, low red and high blue", () => {
		const { doc, cm } = scene(MeshElement.MM_VERTCOLOR | MeshElement.MM_VERTQUALITY);
		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = v;
		const out = kernel.applyFilter(doc, "Colorize by vertex Quality", {
			minVal: 0,
			maxVal: cm.vn - 1,
		});
		expect(out.min_value).toBe(0);
		expect(cm.vertColor[0]).toBe(rgba(255, 0, 0));
		expect(cm.vertColor[cm.vn - 1]).toBe(rgba(0, 0, 255));
	});

	test("zero-symmetric puts green exactly at zero", () => {
		const { doc, cm } = scene(MeshElement.MM_VERTCOLOR | MeshElement.MM_VERTQUALITY);
		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = v === 0 ? 0 : v - 10;
		const out = kernel.applyFilter(doc, "Colorize by vertex Quality", {
			minVal: -10,
			maxVal: 100,
			zeroSym: true,
		});
		// Widened to the further side, so the middle of the ramp is zero.
		expect(out.min_value).toBe(-100);
		expect(out.max_value).toBe(100);
		expect(cm.vertColor[0]).toBe(rgba(0, 255, 0));
	});

	test("a percentile crop ignores the outlier", () => {
		const { doc, cm } = scene(MeshElement.MM_VERTCOLOR | MeshElement.MM_VERTQUALITY, 3);
		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = 1;
		cm.vertQuality[0] = 1e6;
		const out = kernel.applyFilter(doc, "Colorize by vertex Quality", { perc: 5 });
		// Without the crop the range would run to a million and everything but
		// one vertex would be the same red.
		expect(out.max_value).toBe(1);
	});

	test("clamping pins the values into the range and reports how many moved", () => {
		const { doc, cm } = scene(MeshElement.MM_VERTQUALITY);
		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = v - 50;
		const out = kernel.applyFilter(doc, "Clamp Vertex Quality", { minVal: -10, maxVal: 10 });
		expect(out.clamped as number).toBeGreaterThan(0);
		for (let v = 0; v < cm.vn; v++) {
			expect(cm.vertQuality[v]).toBeGreaterThanOrEqual(-10);
			expect(cm.vertQuality[v]).toBeLessThanOrEqual(10);
		}
	});

	test("face quality maps onto the ramp too", () => {
		const { doc, cm } = scene(MeshElement.MM_FACECOLOR | MeshElement.MM_FACEQUALITY);
		const quality = cm.faceQuality;
		expect(quality).not.toBeNull();
		if (quality === null) return;
		for (let f = 0; f < cm.faceSize; f++) quality[f] = f;
		kernel.applyFilter(doc, "Colorize by face Quality", { minVal: 0, maxVal: cm.fn - 1 });
		expect(cm.faceColor?.[0]).toBe(rgba(255, 0, 0));
		expect(cm.faceColor?.[cm.fn - 1]).toBe(rgba(0, 0, 255));
	});

	test("triangle shape scores 1 for an equilateral triangle", () => {
		// All three implemented metrics are normalised the same way, which is
		// what makes them comparable at a glance.
		const { doc, cm } = scene(MeshElement.MM_FACEQUALITY);
		// The icosahedron's faces are equilateral.
		const ico = new MeshDocument();
		const im = ico.addNewMesh("", "i", true, Platonic.icosahedron());
		im.updateDataMask(MeshElement.MM_FACEQUALITY);
		for (const metric of [1, 2]) {
			kernel.applyFilter(ico, "Per Face Quality according to Triangle shape and aspect ratio", {
				Metric: metric,
			});
			expect(im.cm.faceQuality?.[0], `metric ${metric}`).toBeCloseTo(1, 9);
		}
		void doc;
		void cm;
	});

	test("triangle shape refuses the metrics it cannot compute", () => {
		const { doc } = scene(MeshElement.MM_FACEQUALITY);
		for (const Metric of [4, 5, 6, 7]) {
			expect(() =>
				kernel.applyFilter(doc, "Per Face Quality according to Triangle shape and aspect ratio", {
					Metric,
				}),
			).toThrow(MLNotImplementedException);
		}
	});
});

describe("transfers between vertices and faces", () => {
	test("vertex colour to face is the average of the three corners", () => {
		const { doc, cm } = scene(MeshElement.MM_VERTCOLOR | MeshElement.MM_FACECOLOR);
		for (let v = 0; v < cm.vertSize; v++) cm.vertColor[v] = rgba(v % 256, 0, 0);
		kernel.applyFilter(doc, "Transfer Color: Vertex to Face", {});
		const expected = Math.round(
			((cm.fv(0, 0) % 256) + (cm.fv(0, 1) % 256) + (cm.fv(0, 2) % 256)) / 3,
		);
		expect(red(cm.faceColor?.[0] ?? 0)).toBe(expected);
	});

	test("a colour round trip through faces and back stays in range", () => {
		// Averaging twice loses detail — that is inherent — but must not drift
		// outside the colours it started between.
		const { doc, cm } = scene(MeshElement.MM_VERTCOLOR | MeshElement.MM_FACECOLOR);
		paint(cm, rgba(60, 120, 180));
		kernel.applyFilter(doc, "Transfer Color: Vertex to Face", {});
		kernel.applyFilter(doc, "Transfer Color: Face to Vertex", {});
		for (let v = 0; v < cm.vn; v++) {
			expect(Math.abs(red(cm.vertColor[v]) - 60)).toBeLessThanOrEqual(1);
			expect(Math.abs(green(cm.vertColor[v]) - 120)).toBeLessThanOrEqual(1);
			expect(Math.abs(blue(cm.vertColor[v]) - 180)).toBeLessThanOrEqual(1);
		}
	});

	test("quality transfers both ways", () => {
		const { doc, cm } = scene(MeshElement.MM_VERTQUALITY | MeshElement.MM_FACEQUALITY);
		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = 7;
		kernel.applyFilter(doc, "Transfer Quality: Vertex to Face", {});
		for (let f = 0; f < cm.fn; f++) expect(cm.faceQuality?.[f]).toBeCloseTo(7, 12);

		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = 0;
		kernel.applyFilter(doc, "Transfer Quality: Face to Vertexerror!", {});
		for (let v = 0; v < cm.vn; v++) expect(cm.vertQuality[v]).toBeCloseTo(7, 12);
	});
});

describe("labelling colours", () => {
	test("random face colour gives every face something", () => {
		const { doc, cm } = scene(MeshElement.MM_FACECOLOR);
		const out = kernel.applyFilter(doc, "Random Face Color", {});
		expect(out.face_number).toBe(cm.fn);
		const distinct = new Set<number>();
		for (let f = 0; f < cm.fn; f++) distinct.add(cm.faceColor?.[f] ?? 0);
		// Labels are meant to be told apart, so most of them must differ.
		expect(distinct.size).toBeGreaterThan(cm.fn / 2);
	});

	test("component colouring gives each piece one colour, and different pieces different ones", () => {
		const doc = new MeshDocument();
		// Two spheres, far apart.
		const combined = Platonic.sphere(1);
		const second = Platonic.sphere(1);
		const offset = combined.vertSize;
		const firstVert = Allocator.addVertices(combined, second.vertSize);
		for (let v = 0; v < second.vertSize; v++) {
			combined.setVert(firstVert + v, second.vx(v) + 10, second.vy(v), second.vz(v));
		}
		const firstFace = Allocator.addFaces(combined, second.faceSize);
		for (let f = 0; f < second.faceSize; f++) {
			combined.setFace(
				firstFace + f,
				second.fv(f, 0) + offset,
				second.fv(f, 1) + offset,
				second.fv(f, 2) + offset,
			);
		}
		const m = doc.addNewMesh("", "two", true, combined);
		m.updateDataMask(MeshElement.MM_FACECOLOR);

		const out = kernel.applyFilter(doc, "Random Component Color", {});
		expect(out.component_number).toBe(2);
		const colours = new Set<number>();
		for (let f = 0; f < combined.fn; f++) colours.add(combined.faceColor?.[f] ?? 0);
		expect(colours.size).toBe(2);
	});
});

describe("colour smoothing", () => {
	test("a uniform colour survives smoothing untouched", () => {
		const { doc, cm } = scene();
		paint(cm, rgba(50, 100, 150));
		kernel.applyFilter(doc, "Smooth: Laplacian Vertex Color", { iteration: 5 });
		for (let v = 0; v < cm.vn; v++) expect(cm.vertColor[v]).toBe(rgba(50, 100, 150));
	});

	test("smoothing pulls an outlier toward its neighbours", () => {
		const { doc, cm } = scene();
		paint(cm, rgba(0, 0, 0));
		cm.vertColor[0] = rgba(255, 255, 255);
		kernel.applyFilter(doc, "Smooth: Laplacian Vertex Color", { iteration: 1 });
		expect(red(cm.vertColor[0])).toBeLessThan(255);
		expect(red(cm.vertColor[0])).toBeGreaterThan(0);
	});

	test("more iterations spread it further", () => {
		const reach = (iteration: number) => {
			const { doc, cm } = scene(MeshElement.MM_VERTCOLOR, 3);
			paint(cm, rgba(0, 0, 0));
			cm.vertColor[0] = rgba(255, 255, 255);
			kernel.applyFilter(doc, "Smooth: Laplacian Vertex Color", { iteration });
			let touched = 0;
			for (let v = 0; v < cm.vn; v++) if (red(cm.vertColor[v]) > 0) touched++;
			return touched;
		};
		expect(reach(3)).toBeGreaterThan(reach(1));
	});

	test("face colour smooths too, and a uniform one is unchanged", () => {
		const { doc, cm } = scene(MeshElement.MM_FACECOLOR);
		const colors = cm.faceColor;
		expect(colors).not.toBeNull();
		if (colors === null) return;
		colors.fill(rgba(30, 60, 90));
		kernel.applyFilter(doc, "Smooth: Laplacian Face Color", { iteration: 3 });
		for (let f = 0; f < cm.fn; f++) expect(colors[f]).toBe(rgba(30, 60, 90));
	});
});

describe("registration", () => {
	test("every implemented filter matches the upstream catalogue", () => {
		for (const [name, pythonName] of [
			["Vertex Color Filling", "set_color_per_vertex"],
			["Vertex Color Thresholding", "apply_color_thresholding_per_vertex"],
			["Vertex Color Invert", "apply_color_inverse_per_vertex"],
			["Vertex Color Desaturation", "apply_color_desaturation_per_vertex"],
			["Vertex Color White Balance", "apply_color_white_balance_per_vertex"],
			["Colorize by vertex Quality", "compute_color_from_scalar_per_vertex"],
			["Colorize by face Quality", "compute_color_from_scalar_per_face"],
			["Clamp Vertex Quality", "apply_scalar_clamping_per_vertex"],
			["Transfer Color: Vertex to Face", "compute_color_transfer_vertex_to_face"],
			["Transfer Color: Face to Vertex", "compute_color_transfer_face_to_vertex"],
			["Random Face Color", "compute_color_random_per_face"],
			["Random Component Color", "compute_color_by_conntected_component_per_face"],
			["Smooth: Laplacian Vertex Color", "apply_color_laplacian_smoothing_per_vertex"],
			["Smooth: Laplacian Face Color", "apply_color_laplacian_smoothing_per_face"],
		] as const) {
			const action = kernel.pluginManager.filterAction(name);
			expect(action, name).toBeDefined();
			if (!action) continue;
			expect(action.pythonName, name).toBe(pythonName);
			expect(action.plugin.pluginName(), name).toBe("FilterColorProc");
		}
	});

	test("the ones that need more than geometry stay unimplemented", () => {
		// Textures are not loaded and histogram equalisation has not been
		// written — better to say so. (Discrete Curvatures used to be on this
		// list; it landed with the curvature module.)
		for (const name of [
			"Transfer Color: Texture to Vertex",
			"Equalize Vertex Color",
			"Saturate Vertex Quality",
			"Perlin color",
		]) {
			expect(kernel.filterAction(name).implemented, name).toBe(false);
		}
	});
});
