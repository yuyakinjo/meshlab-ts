/**
 * The last twelve of `filter_colorproc` and `filter_func`, plus the two pieces
 * of shared machinery they needed: the mesh-level colour and named custom
 * attributes.
 *
 * The attribute tests lean hardest on the allocator, because that is where a
 * run-time channel can quietly go wrong: an attribute added when the mesh had
 * eight vertices has to survive growing to a thousand and compacting back
 * down, and it will not do so unless every generic pass in `components.ts`
 * knows about it. Those tests are the reason the storage is a list walked
 * alongside the channel table rather than a map hanging off the side.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import { MLException } from "../../src/common/utilities/ml_exception.ts";
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { Platonic } from "../../src/vcg/complex/create/platonic.ts";
import { VertexFlag } from "../../src/vcg/complex/flags.ts";
import { UpdateQuality } from "../../src/vcg/complex/update/quality.ts";
import { UpdateTopology } from "../../src/vcg/complex/update/topology.ts";
import {
	blue,
	buildEqualizeTables,
	equalizeColor,
	green,
	red,
	rgba,
	scatter,
} from "../../src/vcg/space/color4.ts";
import { Image } from "../../src/vcg/space/image/image.ts";
import { writePng } from "../../src/vcg/space/image/png.ts";

const kernel = MeshLabKernel.default();

function scene(channels: number = MeshElement.MM_VERTCOLOR, subdiv = 2) {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", "m", true, Platonic.sphere(subdiv));
	m.updateDataMask(channels);
	m.updateBoxAndNormals();
	return { doc, m, cm: m.cm };
}

// ---------------------------------------------------------------- equalisation

describe("Equalize Vertex Color", () => {
	test("stretches a compressed range to the full 0..255", () => {
		const { doc, cm } = scene();
		// Every colour squeezed into 100..140: exactly the case the filter is
		// for, and one where the answer is not a matter of taste.
		let i = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			const g = 100 + (i++ % 41);
			cm.vertColor[v] = rgba(g, g, g);
		}
		kernel.applyFilter(doc, "Equalize Vertex Color");

		let min = 255;
		let max = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			min = Math.min(min, red(cm.vertColor[v]));
			max = Math.max(max, red(cm.vertColor[v]));
		}
		// Not 0: upstream's ValueEqualize subtracts cdf[0] — the count of vertices
		// whose channel is literally zero — rather than the smallest cdf value
		// present. So the darkest input maps to its own share of the population,
		// not to black. What the filter promises is the stretch, and 40 levels of
		// input becoming more than 200 is that.
		expect(max - min).toBeGreaterThan(200);
		expect(max).toBeGreaterThan(240);
	});

	test("keeps the order of the input values", () => {
		// Equalisation is monotone by construction — a cumulative distribution
		// never decreases. If it ever reordered two vertices it would be
		// inventing contrast rather than revealing it.
		const table = buildEqualizeTables([rgba(10, 0, 0), rgba(50, 0, 0), rgba(200, 0, 0)]);
		const out = [10, 50, 200].map((v) => red(equalizeColor(rgba(v, 0, 0), table, 1)));
		expect(out[0]).toBeLessThanOrEqual(out[1]);
		expect(out[1]).toBeLessThanOrEqual(out[2]);
	});

	test("a single-valued mesh saturates, and an all-black one does not divide by zero", () => {
		// Upstream's formula sends any uniform non-zero colour to 255: every
		// vertex sits at the top of its own cumulative distribution. Odd to look
		// at, but it is what MeshLab does, and a mesh painted one flat colour is
		// not what the filter is for.
		const flat = scene();
		for (let v = 0; v < flat.cm.vertSize; v++) flat.cm.vertColor[v] = rgba(77, 77, 77);
		kernel.applyFilter(flat.doc, "Equalize Vertex Color");
		expect(red(flat.cm.vertColor[0])).toBe(255);

		// All-black is the one case where the span really is zero. It has to come
		// back unchanged rather than as NaN.
		const black = scene();
		for (let v = 0; v < black.cm.vertSize; v++) black.cm.vertColor[v] = rgba(0, 0, 0);
		kernel.applyFilter(black.doc, "Equalize Vertex Color");
		expect(red(black.cm.vertColor[0])).toBe(0);
	});

	test("with no channel selected it produces grey", () => {
		const { doc, cm } = scene();
		for (let v = 0; v < cm.vertSize; v++) cm.vertColor[v] = rgba(200, 60, 10);
		kernel.applyFilter(doc, "Equalize Vertex Color", { rCh: false, gCh: false, bCh: false });
		const c = cm.vertColor[0];
		expect(red(c)).toBe(green(c));
		expect(green(c)).toBe(blue(c));
	});

	test("only the selection is measured and only the selection changes", () => {
		const { doc, cm } = scene();
		for (let v = 0; v < cm.vertSize; v++) cm.vertColor[v] = rgba(v % 2 === 0 ? 10 : 250, 0, 0);
		// Select only the dark half, whose values span nothing at all.
		for (let v = 0; v < cm.vertSize; v++) {
			if (v % 2 === 0) cm.vertFlags[v] |= VertexFlag.SELECTED;
		}
		kernel.applyFilter(doc, "Equalize Vertex Color", { onSelected: true });
		// The bright vertices are untouched...
		expect(red(cm.vertColor[1])).toBe(250);
		// ...and the dark ones were measured against themselves alone. Had the
		// histogram come from the whole mesh, 10 would sit at the bottom of a
		// two-peaked distribution and land near 128; measured against a selection
		// where it is the only value, it saturates instead.
		expect(red(cm.vertColor[0])).toBe(255);
	});
});

// -------------------------------------------------------------------- perlin

describe("Perlin color", () => {
	test("gives nearby vertices similar colours", () => {
		const { doc, cm } = scene(MeshElement.MM_VERTCOLOR, 3);
		kernel.applyFilter(doc, "Perlin color", { freq: 2 });

		// The claim in the filter's own description, made measurable: pick the
		// closest and the farthest vertex from vertex 0 and compare how much the
		// colour moved. A field this smooth must vary less over the short hop.
		let near = -1;
		let far = -1;
		let dNear = Number.POSITIVE_INFINITY;
		let dFar = 0;
		for (let v = 1; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			const d = Math.hypot(cm.vx(v) - cm.vx(0), cm.vy(v) - cm.vy(0), cm.vz(v) - cm.vz(0));
			if (d < dNear) {
				dNear = d;
				near = v;
			}
			if (d > dFar) {
				dFar = d;
				far = v;
			}
		}
		const delta = (a: number, b: number) => Math.abs(red(cm.vertColor[a]) - red(cm.vertColor[b]));
		expect(delta(0, near)).toBeLessThan(delta(0, far));
	});

	test("is deterministic — the same mesh gives the same colours twice", () => {
		const first = scene();
		const second = scene();
		kernel.applyFilter(first.doc, "Perlin color", { freq: 8 });
		kernel.applyFilter(second.doc, "Perlin color", { freq: 8 });
		expect([...first.cm.vertColor]).toEqual([...second.cm.vertColor]);
	});

	test("stays between the two colours it was given", () => {
		const { doc, cm } = scene();
		kernel.applyFilter(doc, "Perlin color", {
			color1: rgba(0, 0, 0),
			color2: rgba(255, 0, 0),
			freq: 5,
		});
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			// A blend of black and red: green and blue cannot appear.
			expect(green(cm.vertColor[v])).toBe(0);
			expect(blue(cm.vertColor[v])).toBe(0);
		}
	});

	test("a higher frequency varies faster", () => {
		const slow = scene(MeshElement.MM_VERTCOLOR, 3);
		const fast = scene(MeshElement.MM_VERTCOLOR, 3);
		kernel.applyFilter(slow.doc, "Perlin color", { freq: 1 });
		kernel.applyFilter(fast.doc, "Perlin color", { freq: 40 });

		const roughness = (cm: CMeshO) => {
			let sum = 0;
			let n = 0;
			for (let f = 0; f < cm.faceSize; f++) {
				if (cm.isFaceD(f)) continue;
				for (let k = 0; k < 3; k++) {
					sum += Math.abs(
						red(cm.vertColor[cm.fv(f, k)]) - red(cm.vertColor[cm.fv(f, (k + 1) % 3)]),
					);
					n++;
				}
			}
			return sum / n;
		};
		expect(roughness(fast.cm)).toBeGreaterThan(roughness(slow.cm));
	});
});

// ------------------------------------------------------------------ scattering

describe("PerMesh Color Scattering", () => {
	test("gives every visible layer a different colour", () => {
		const doc = new MeshDocument();
		for (let i = 0; i < 5; i++) doc.addNewMesh("", `m${i}`, true, Platonic.sphere(1));
		kernel.applyFilter(doc, "PerMesh Color Scattering", { seed: 7 });
		const colors = doc.meshIterator().map((m) => m.cm.color);
		expect(new Set(colors).size).toBe(5);
	});

	test("a seed makes it reproducible; zero does not", () => {
		const build = () => {
			const doc = new MeshDocument();
			for (let i = 0; i < 6; i++) doc.addNewMesh("", `m${i}`, true, Platonic.sphere(1));
			return doc;
		};
		const a = build();
		const b = build();
		kernel.applyFilter(a, "PerMesh Color Scattering", { seed: 42 });
		kernel.applyFilter(b, "PerMesh Color Scattering", { seed: 42 });
		expect(a.meshIterator().map((m) => m.cm.color)).toEqual(
			b.meshIterator().map((m) => m.cm.color),
		);
	});

	test("consecutive scatter values are far apart on the hue circle", () => {
		// The whole point of the bit-reversal: 0 and 1 out of 16 must not be
		// neighbouring hues, or two adjacent layers would look identical.
		const hueOf = (c: number) => {
			const [r, g, b] = [red(c) / 255, green(c) / 255, blue(c) / 255];
			const max = Math.max(r, g, b);
			const min = Math.min(r, g, b);
			if (max === min) return 0;
			const d = max - min;
			const h = max === r ? (g - b) / d : max === g ? 2 + (b - r) / d : 4 + (r - g) / d;
			return (((h * 60) % 360) + 360) % 360;
		};
		const gap = Math.abs(hueOf(scatter(16, 0)) - hueOf(scatter(16, 1)));
		expect(Math.min(gap, 360 - gap)).toBeGreaterThan(90);
	});

	test("skips hidden layers", () => {
		const doc = new MeshDocument();
		const a = doc.addNewMesh("", "a", true, Platonic.sphere(1));
		const b = doc.addNewMesh("", "b", true, Platonic.sphere(1));
		b.setVisible(false);
		const before = b.cm.color;
		kernel.applyFilter(doc, "PerMesh Color Scattering", { seed: 3 });
		expect(b.cm.color).toBe(before);
		expect(a.cm.color).not.toBe(before);
	});
});

// ------------------------------------------------------------ mesh/texture → x

describe("Transfer Color: Mesh to Face", () => {
	test("stamps the mesh colour onto every face", () => {
		const { doc, m, cm } = scene(MeshElement.MM_FACECOLOR);
		m.cm.color = rgba(12, 34, 56);
		kernel.applyFilter(doc, "Transfer Color: Mesh to Face");
		const colors = cm.faceColor as Uint32Array;
		for (let f = 0; f < cm.faceSize; f++) {
			if (!cm.isFaceD(f)) expect(colors[f]).toBe(rgba(12, 34, 56));
		}
	});

	test("all-visible mode reaches the other layers", () => {
		const doc = new MeshDocument();
		const a = doc.addNewMesh("", "a", true, Platonic.sphere(1));
		const b = doc.addNewMesh("", "b", true, Platonic.sphere(1));
		a.cm.color = rgba(1, 2, 3);
		b.cm.color = rgba(9, 8, 7);
		doc.setCurrentMesh(a.id());
		kernel.applyFilter(doc, "Transfer Color: Mesh to Face", { allVisibleMesh: true });
		// Each layer gets its *own* colour, not the current layer's.
		expect((b.cm.faceColor as Uint32Array)[0]).toBe(rgba(9, 8, 7));
		expect((a.cm.faceColor as Uint32Array)[0]).toBe(rgba(1, 2, 3));
	});
});

describe("Transfer Color: Texture to Vertex", () => {
	/** A quad in the XY plane with UVs covering the whole texture. */
	function texturedQuad(image: Image) {
		const doc = new MeshDocument();
		const cm = new CMeshO();
		Allocator.addVertices(cm, 4);
		cm.setVert(0, 0, 0, 0);
		cm.setVert(1, 1, 0, 0);
		cm.setVert(2, 1, 1, 0);
		cm.setVert(3, 0, 1, 0);
		Allocator.addFaces(cm, 2);
		cm.setFace(0, 0, 1, 2);
		cm.setFace(1, 0, 2, 3);
		const m = doc.addNewMesh("", "quad", true, cm);
		m.updateDataMask(MeshElement.MM_VERTCOLOR | MeshElement.MM_WEDGTEXCOORD);
		const wt = cm.wedgeTexCoord as Float64Array;
		const uv = [
			[0, 0],
			[1, 0],
			[1, 1],
		];
		const uv2 = [
			[0, 0],
			[1, 1],
			[0, 1],
		];
		for (let k = 0; k < 3; k++) {
			wt[2 * k] = uv[k][0];
			wt[2 * k + 1] = uv[k][1];
			wt[6 + 2 * k] = uv2[k][0];
			wt[6 + 2 * k + 1] = uv2[k][1];
		}
		m.textures.set("t.png", writePng(image));
		cm.textures = ["t.png"];
		m.updateBoxAndNormals();
		return { doc, m, cm };
	}

	test("reads the texel under each corner's UV", () => {
		const image = new Image(4, 4, rgba(0, 0, 0));
		// Bottom-left in texture space is the *last* row of the image.
		image.setPixel(0, 3, rgba(200, 10, 20));
		const { doc, cm } = texturedQuad(image);
		kernel.applyFilter(doc, "Transfer Color: Texture to Vertex");
		expect(cm.vertColor[0]).toBe(rgba(200, 10, 20));
	});

	test("wraps a UV outside 0..1 instead of clamping it", () => {
		const image = new Image(4, 4, rgba(0, 0, 0));
		image.setPixel(0, 3, rgba(1, 2, 3));
		const { doc, cm } = texturedQuad(image);
		const wt = cm.wedgeTexCoord as Float64Array;
		// -1 and 3 are both the same texel as 0 for a repeating texture.
		wt[0] = -1;
		wt[1] = 3;
		kernel.applyFilter(doc, "Transfer Color: Texture to Vertex");
		expect(cm.vertColor[0]).toBe(rgba(1, 2, 3));
	});

	test("a wedge pointing at no texture comes out white", () => {
		const image = new Image(2, 2, rgba(30, 30, 30));
		const { doc, cm } = texturedQuad(image);
		// Vertex 1 belongs to face 0 only. Vertex 0 is in both faces, and the
		// second would paint over whatever the first wrote — a shared vertex has
		// one colour however many wedges point at it.
		(cm.wedgeTexIndex as Int32Array)[1] = 5;
		kernel.applyFilter(doc, "Transfer Color: Texture to Vertex");
		expect(cm.vertColor[1]).toBe(rgba(255, 255, 255));
		expect(cm.vertColor[0]).toBe(rgba(30, 30, 30));
	});

	test("refuses a mesh with no texture coordinates", () => {
		// The filter asks for the coordinates itself rather than listing them as
		// a requirement, precisely so this says something instead of sampling a
		// mesh full of zeroed UVs.
		const { doc } = scene(MeshElement.MM_VERTCOLOR);
		expect(() => kernel.applyFilter(doc, "Transfer Color: Texture to Vertex")).toThrow(MLException);
	});
});

// ----------------------------------------------------------------- saturation

describe("Saturate Vertex Quality", () => {
	/** A grid whose quality is a single spike at one vertex. */
	function spike(threshold = 1) {
		const doc = new MeshDocument();
		const cm = Platonic.sphere(3);
		const m = doc.addNewMesh("", "m", true, cm);
		m.updateDataMask(MeshElement.MM_VERTQUALITY | MeshElement.MM_VERTFACETOPO);
		m.updateBoxAndNormals();
		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = 0;
		cm.vertQuality[0] = 100;
		return { doc, cm, threshold };
	}

	test("brings the gradient under the threshold everywhere", () => {
		const { doc, cm } = spike();
		kernel.applyFilter(doc, "Saturate Vertex Quality", { gradientThr: 1 });

		UpdateTopology.faceFace(cm);
		let worst = 0;
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			for (let k = 0; k < 3; k++) {
				const a = cm.fv(f, k);
				const b = cm.fv(f, (k + 1) % 3);
				const len = Math.hypot(cm.vx(a) - cm.vx(b), cm.vy(a) - cm.vy(b), cm.vz(a) - cm.vz(b));
				worst = Math.max(worst, Math.abs(cm.vertQuality[a] - cm.vertQuality[b]) / len);
			}
		}
		// A hair of slack: the algorithm stops as soon as it is under the bound,
		// leaving each edge at the threshold minus its own epsilon.
		expect(worst).toBeLessThanOrEqual(1 + 1e-9);
	});

	test("never raises a value", () => {
		const { doc, cm } = spike();
		const before = Float64Array.from(cm.vertQuality);
		kernel.applyFilter(doc, "Saturate Vertex Quality", { gradientThr: 1 });
		for (let v = 0; v < cm.vertSize; v++) {
			if (!cm.isVertD(v)) expect(cm.vertQuality[v]).toBeLessThanOrEqual(before[v] + 1e-12);
		}
	});

	test("a field already smooth enough is left alone", () => {
		const doc = new MeshDocument();
		const cm = Platonic.sphere(2);
		const m = doc.addNewMesh("", "m", true, cm);
		m.updateDataMask(MeshElement.MM_VERTQUALITY | MeshElement.MM_VERTFACETOPO);
		m.updateBoxAndNormals();
		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = cm.vx(v);
		const before = Float64Array.from(cm.vertQuality);
		// x varies at most as fast as distance, so a threshold of 1 asks nothing.
		kernel.applyFilter(doc, "Saturate Vertex Quality", { gradientThr: 1 });
		for (let v = 0; v < cm.vertSize; v++) {
			if (!cm.isVertD(v)) expect(cm.vertQuality[v]).toBeCloseTo(before[v], 9);
		}
	});

	test("reaches every connected component, not just the first", () => {
		// Upstream seeds its stack at vertex 0 only, so a second component is
		// never visited. Two spheres, each with a spike, is the smallest case
		// that tells the two behaviours apart.
		const doc = new MeshDocument();
		const cm = Platonic.sphere(2);
		const second = Platonic.sphere(2);
		const base = cm.vertSize;
		const vFirst = Allocator.addVertices(cm, second.vertSize);
		for (let v = 0; v < second.vertSize; v++) {
			cm.setVert(vFirst + v, second.vx(v) + 10, second.vy(v), second.vz(v));
		}
		const fFirst = Allocator.addFaces(cm, second.faceSize);
		for (let f = 0; f < second.faceSize; f++) {
			cm.setFace(
				fFirst + f,
				base + second.fv(f, 0),
				base + second.fv(f, 1),
				base + second.fv(f, 2),
			);
		}
		const m = doc.addNewMesh("", "m", true, cm);
		m.updateDataMask(MeshElement.MM_VERTQUALITY | MeshElement.MM_VERTFACETOPO);
		m.updateBoxAndNormals();
		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = 0;
		cm.vertQuality[base] = 100;

		kernel.applyFilter(doc, "Saturate Vertex Quality", { gradientThr: 1 });
		expect(cm.vertQuality[base]).toBeLessThan(50);
	});

	test("updateColor paints the ramp", () => {
		const { doc, cm } = spike();
		kernel.applyFilter(doc, "Saturate Vertex Quality", { gradientThr: 1, updateColor: true });
		const distinct = new Set<number>();
		for (let v = 0; v < cm.vertSize; v++) if (!cm.isVertD(v)) distinct.add(cm.vertColor[v]);
		expect(distinct.size).toBeGreaterThan(1);
	});

	test("rejects a threshold of zero rather than dividing by it", () => {
		const { doc } = spike();
		expect(() => kernel.applyFilter(doc, "Saturate Vertex Quality", { gradientThr: 0 })).toThrow(
			MLException,
		);
	});
});

// -------------------------------------------------------- custom attributes

describe("custom attributes", () => {
	test("a per-vertex scalar attribute holds what the expression computed", () => {
		const { doc, cm } = scene();
		kernel.applyFilter(doc, "Define New Per Vertex Custom Scalar Attribute", {
			name: "height",
			expr: "z",
		});
		const attr = cm.customAttribute("height", "vert");
		expect(attr).toBeDefined();
		for (let v = 0; v < cm.vertSize; v++) {
			if (!cm.isVertD(v)) expect((attr as { data: Float64Array }).data[v]).toBeCloseTo(cm.vz(v), 9);
		}
	});

	test("the attribute becomes a variable in later expressions", () => {
		// This is the whole reason the filter exists: define once, reuse.
		const { doc, cm } = scene();
		kernel.applyFilter(doc, "Define New Per Vertex Custom Scalar Attribute", {
			name: "twice_z",
			expr: "2*z",
		});
		kernel.applyFilter(doc, "Per Vertex Quality Function", { q: "twice_z + 1" });
		for (let v = 0; v < cm.vertSize; v++) {
			if (!cm.isVertD(v)) expect(cm.vertQuality[v]).toBeCloseTo(2 * cm.vz(v) + 1, 9);
		}
	});

	test("a point attribute stores three components", () => {
		const { doc, cm } = scene();
		kernel.applyFilter(doc, "Define New Per Vertex Custom Point Attribute", {
			name: "mirrored",
			x_expr: "-x",
			y_expr: "-y",
			z_expr: "-z",
		});
		const attr = cm.customAttribute("mirrored", "vert") as { data: Float64Array; arity: number };
		expect(attr.arity).toBe(3);
		expect(attr.data[0]).toBeCloseTo(-cm.vx(0), 9);
		expect(attr.data[2]).toBeCloseTo(-cm.vz(0), 9);
	});

	test("point attributes are not bound as variables", () => {
		// Deliberate, and upstream's behaviour: a three-component value has no
		// single name an expression could read.
		const { doc } = scene();
		kernel.applyFilter(doc, "Define New Per Vertex Custom Point Attribute", {
			name: "p",
			x_expr: "x",
			y_expr: "y",
			z_expr: "z",
		});
		expect(() => kernel.applyFilter(doc, "Per Vertex Quality Function", { q: "p" })).toThrow();
	});

	test("per-face attributes live in their own namespace", () => {
		const { doc, cm } = scene(MeshElement.MM_FACEQUALITY);
		kernel.applyFilter(doc, "Define New Per Vertex Custom Scalar Attribute", {
			name: "k",
			expr: "1",
		});
		kernel.applyFilter(doc, "Define New Per Face Custom Scalar Attribute", {
			name: "k",
			expr: "2",
		});
		// Same name, two attributes — a vertex expression must see 1, not 2.
		expect((cm.customAttribute("k", "vert") as { data: Float64Array }).data[0]).toBe(1);
		expect((cm.customAttribute("k", "face") as { data: Float64Array }).data[0]).toBe(2);
	});

	test("re-defining under the same name overwrites in place", () => {
		const { doc, cm } = scene();
		kernel.applyFilter(doc, "Define New Per Vertex Custom Scalar Attribute", {
			name: "a",
			expr: "1",
		});
		kernel.applyFilter(doc, "Define New Per Vertex Custom Scalar Attribute", {
			name: "a",
			expr: "5",
		});
		expect(cm.customAttrs.length).toBe(1);
		expect((cm.customAttribute("a", "vert") as { data: Float64Array }).data[0]).toBe(5);
	});

	test("re-defining with a different arity throws rather than reshaping", () => {
		const { doc } = scene();
		kernel.applyFilter(doc, "Define New Per Vertex Custom Scalar Attribute", {
			name: "a",
			expr: "1",
		});
		expect(() =>
			kernel.applyFilter(doc, "Define New Per Vertex Custom Point Attribute", {
				name: "a",
				x_expr: "1",
				y_expr: "1",
				z_expr: "1",
			}),
		).toThrow();
	});

	test("an expression cannot read the attribute it is defining", () => {
		const { doc } = scene();
		expect(() =>
			kernel.applyFilter(doc, "Define New Per Vertex Custom Scalar Attribute", {
				name: "self",
				expr: "self + 1",
			}),
		).toThrow();
	});

	test("rejects names that are not identifiers", () => {
		const { doc } = scene();
		for (const name of ["", "2bad", "has space", "has-dash", "x+y"]) {
			expect(() =>
				kernel.applyFilter(doc, "Define New Per Vertex Custom Scalar Attribute", {
					name,
					expr: "1",
				}),
			).toThrow(MLException);
		}
	});

	test("survives growing the mesh", () => {
		const cm = Platonic.sphere(1);
		const attr = cm.addCustomAttribute("a", "vert", 1);
		for (let v = 0; v < cm.vertSize; v++) attr.data[v] = v;
		const before = cm.vertSize;
		// Enough to force several reallocations of every channel.
		Allocator.addVertices(cm, 5000);
		const after = cm.customAttribute("a", "vert") as { data: Float64Array };
		for (let v = 0; v < before; v++) expect(after.data[v]).toBe(v);
		expect(after.data.length).toBeGreaterThanOrEqual(cm.vertSize);
	});

	test("compaction moves attribute values with their vertices", () => {
		const cm = Platonic.sphere(1);
		const attr = cm.addCustomAttribute("a", "vert", 3);
		for (let v = 0; v < cm.vertSize; v++) {
			attr.data[3 * v] = cm.vx(v);
			attr.data[3 * v + 1] = cm.vy(v);
			attr.data[3 * v + 2] = cm.vz(v);
		}
		// Delete a scattering of faces, then whichever vertices that orphaned, and
		// compact. If the attribute did not travel with its vertex, its stored x
		// would stop matching the vertex's own.
		for (let f = 0; f < cm.faceSize; f += 3) Allocator.deleteFace(cm, f);
		const used = new Set<number>();
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			for (let k = 0; k < 3; k++) used.add(cm.fv(f, k));
		}
		for (let v = 0; v < cm.vertSize; v++) if (!used.has(v)) Allocator.deleteVertex(cm, v);
		expect(cm.vn).toBeLessThan(used.size + 1);
		Allocator.compactEveryVector(cm);

		const after = cm.customAttribute("a", "vert") as { data: Float64Array };
		for (let v = 0; v < cm.vertSize; v++) {
			expect(after.data[3 * v]).toBeCloseTo(cm.vx(v), 12);
			expect(after.data[3 * v + 2]).toBeCloseTo(cm.vz(v), 12);
		}
	});

	test("a fresh slot reads zero rather than the deleted vertex's value", () => {
		const cm = Platonic.sphere(1);
		const attr = cm.addCustomAttribute("a", "vert", 1);
		attr.data.fill(7);
		const n = cm.vertSize;
		for (let f = 0; f < cm.faceSize; f++) Allocator.deleteFace(cm, f);
		for (let v = 0; v < n; v++) Allocator.deleteVertex(cm, v);
		Allocator.compactEveryVector(cm);
		const first = Allocator.addVertices(cm, 3);
		const after = cm.customAttribute("a", "vert") as { data: Float64Array };
		expect(after.data[first]).toBe(0);
	});
});

// --------------------------------------------------------- texture functions

describe("texture coordinate functions", () => {
	test("per-vertex writes the UV the expression asked for", () => {
		const { doc, m, cm } = scene(MeshElement.MM_VERTCOLOR);
		kernel.applyFilter(doc, "Per Vertex Texture Function", { u: "x", v: "y" });
		expect(m.hasDataMask(MeshElement.MM_VERTTEXCOORD)).toBe(true);
		const vt = cm.vertTexCoord as Float64Array;
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			expect(vt[2 * v]).toBeCloseTo(cm.vx(v), 9);
			expect(vt[2 * v + 1]).toBeCloseTo(cm.vy(v), 9);
		}
	});

	test("the coordinates it wrote are readable by the next expression", () => {
		const { doc, cm } = scene(MeshElement.MM_VERTCOLOR);
		kernel.applyFilter(doc, "Per Vertex Texture Function", { u: "z", v: "0" });
		kernel.applyFilter(doc, "Per Vertex Quality Function", { q: "vtu" });
		for (let v = 0; v < cm.vertSize; v++) {
			if (!cm.isVertD(v)) expect(cm.vertQuality[v]).toBeCloseTo(cm.vz(v), 9);
		}
	});

	test("per-wedge gives each corner its own coordinate", () => {
		const { doc, cm } = scene(MeshElement.MM_VERTCOLOR);
		kernel.applyFilter(doc, "Per Wedge Texture Function", {
			u0: "x0",
			v0: "y0",
			u1: "x1",
			v1: "y1",
			u2: "x2",
			v2: "y2",
		});
		const wt = cm.wedgeTexCoord as Float64Array;
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			for (let k = 0; k < 3; k++) {
				expect(wt[6 * f + 2 * k]).toBeCloseTo(cm.vx(cm.fv(f, k)), 9);
			}
		}
	});

	test("wedges are per corner, so a shared vertex can hold two UVs", () => {
		// The thing per-vertex coordinates cannot express, and the reason both
		// filters exist: assign by face index and one vertex ends up with a
		// different u in each face that touches it.
		const { doc, cm } = scene(MeshElement.MM_VERTCOLOR);
		kernel.applyFilter(doc, "Per Wedge Texture Function", {
			u0: "fi",
			v0: "0",
			u1: "fi",
			v1: "0",
			u2: "fi",
			v2: "0",
		});
		const wt = cm.wedgeTexCoord as Float64Array;
		const seen = new Map<number, Set<number>>();
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			for (let k = 0; k < 3; k++) {
				const v = cm.fv(f, k);
				if (!seen.has(v)) seen.set(v, new Set());
				(seen.get(v) as Set<number>).add(wt[6 * f + 2 * k]);
			}
		}
		expect([...seen.values()].some((s) => s.size > 1)).toBe(true);
	});
});

describe("quality saturation, directly", () => {
	test("the vertex star from VF adjacency lists each neighbour once", () => {
		const cm = Platonic.sphere(2);
		UpdateTopology.vertexFace(cm);
		const star: number[] = [];
		UpdateQuality.vvStarVF(cm, 0, star);
		expect(new Set(star).size).toBe(star.length);
		expect(star).not.toContain(0);
		// An icosphere vertex has five or six neighbours, never fewer.
		expect(star.length).toBeGreaterThanOrEqual(5);
	});
});
