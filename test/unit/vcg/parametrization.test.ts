/**
 * Disk parametrisation and distortion measures.
 *
 * A flat patch is the sharpest test available: it is developable, so a good
 * parametrisation of it is an isometry up to scale, and every distortion
 * measure must read zero. Anything that is not flat gets tested by its
 * guarantees instead — mean value coordinates must never fold a triangle, the
 * harmonic map must beat mean value on angles, the boundary must land exactly
 * on the shape it was pinned to.
 */
import { describe, expect, test } from "bun:test";
import { MeshDocument } from "../../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../../src/common/ml_document/mesh_element.ts";
import { Allocator } from "../../../src/vcg/complex/allocator.ts";
import { Clean } from "../../../src/vcg/complex/clean.ts";
import type { CMeshO } from "../../../src/vcg/complex/cmesho.ts";
import {
	angleDistortion,
	area3D,
	areaDistortion,
	areaUV,
	edgeDistortion,
	foldedNum,
	globallyUnfolded,
	isFolded,
	l2StretchEnergySquared,
	lInfStretchEnergy,
	meshAngleDistortion,
	meshL2Stretch,
	meshScalingFactor,
	signedAreaUV,
} from "../../../src/vcg/complex/parametrization/distortion.ts";
import {
	boundaryLoop,
	parametrizeDisk,
	type WeightScheme,
	writeWedgeUV,
} from "../../../src/vcg/complex/parametrization/harmonic.ts";
import { buildMesh, gridPlane, sphereIcosa, torus } from "../../helpers/mesh_builders.ts";

/** A grid with the wedge channel allocated, ready to receive UVs. */
function withWedges(cm: CMeshO): CMeshO {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", "p", true, cm);
	m.updateDataMask(MeshElement.MM_WEDGTEXCOORD);
	m.updateBoxAndNormals();
	return m.cm;
}

/** A hemisphere: a disk-topology patch that genuinely curves. */
function hemisphere(subdiv = 3): CMeshO {
	const full = sphereIcosa(subdiv).mesh;
	// Drop everything below the equator, then the vertices left unreferenced.
	for (let f = 0; f < full.faceSize; f++) {
		if (full.isFaceD(f)) continue;
		const z = (full.vz(full.fv(f, 0)) + full.vz(full.fv(f, 1)) + full.vz(full.fv(f, 2))) / 3;
		if (z < 0) Allocator.deleteFace(full, f);
	}
	// Only the vertices no surviving face uses: a face straddling the equator
	// keeps a corner just below it, and deleting that would leave a dangling
	// reference.
	Clean.removeUnreferencedVertex(full);
	Allocator.compactEveryVector(full);
	return full;
}

/** A cone: developable, so a perfect parametrisation exists but is not planar. */
function coneFan(sides = 24, radius = 1, height = 1): CMeshO {
	const coords: number[] = [0, 0, height];
	for (let i = 0; i < sides; i++) {
		const a = (2 * Math.PI * i) / sides;
		coords.push(radius * Math.cos(a), radius * Math.sin(a), 0);
	}
	const faces: number[] = [];
	for (let i = 0; i < sides; i++) faces.push(0, 1 + i, 1 + ((i + 1) % sides));
	return buildMesh(coords, faces);
}

describe("boundary loop", () => {
	test("a grid's boundary is its rectangle, walked once", () => {
		const cm = gridPlane(4, 3).mesh;
		const loop = boundaryLoop(cm);
		// (4+1) x (3+1) vertices: the boundary is the perimeter of the grid.
		expect(loop.length).toBe(2 * 4 + 2 * 3);
		expect(new Set(loop).size).toBe(loop.length);
	});

	test("a closed surface has no boundary, and says so", () => {
		expect(() => boundaryLoop(sphereIcosa(2).mesh)).toThrow(/no boundary/);
		expect(() => boundaryLoop(torus(2, 0.6, 12, 8).mesh)).toThrow(/no boundary/);
	});

	test("two holes are reported rather than silently half-walked", () => {
		const cm = gridPlane(5, 5).mesh;
		// Punch a hole in the middle, giving a second, inner boundary.
		Allocator.deleteFace(cm, 24);
		Allocator.deleteFace(cm, 25);
		expect(() => boundaryLoop(cm)).toThrow(/more than one boundary loop/);
	});
});

describe("parametrizeDisk", () => {
	test("the boundary lands exactly on the circle it was pinned to", () => {
		const cm = gridPlane(4, 4).mesh;
		const { uv, boundary } = parametrizeDisk(cm);
		for (const v of boundary) {
			const r = Math.hypot(uv[2 * v] - 0.5, uv[2 * v + 1] - 0.5);
			expect(r).toBeCloseTo(0.5, 12);
		}
	});

	test("a square boundary puts every boundary vertex on an edge of the square", () => {
		const cm = gridPlane(4, 4).mesh;
		const { uv, boundary } = parametrizeDisk(cm, { boundary: "square" });
		for (const v of boundary) {
			const u = uv[2 * v];
			const w = uv[2 * v + 1];
			const onEdge =
				Math.abs(u) < 1e-9 ||
				Math.abs(u - 1) < 1e-9 ||
				Math.abs(w) < 1e-9 ||
				Math.abs(w - 1) < 1e-9;
			expect(onEdge, `vertex ${v} at ${u},${w}`).toBe(true);
		}
	});

	test("the interior stays inside the boundary", () => {
		const cm = gridPlane(5, 5).mesh;
		const { uv, boundary } = parametrizeDisk(cm);
		const onBoundary = new Set(boundary);
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v) || onBoundary.has(v)) continue;
			// Strictly inside the pinned circle — that is what a convex
			// combination of the boundary means.
			expect(Math.hypot(uv[2 * v] - 0.5, uv[2 * v + 1] - 0.5)).toBeLessThan(0.5);
		}
	});

	test("mean value never folds a triangle, on flat or curved patches", () => {
		for (const [name, cm] of [
			["grid", gridPlane(6, 6).mesh],
			["hemisphere", hemisphere(3)],
			["cone", coneFan(24)],
		] as const) {
			const result = parametrizeDisk(cm, { weights: "mean-value" });
			expect(result.valid, name).toBe(true);
		}
	});

	test("harmonic beats mean value on angle distortion", () => {
		const build = (weights: WeightScheme) => {
			const cm = withWedges(hemisphere(3));
			const { uv } = parametrizeDisk(cm, { weights });
			writeWedgeUV(cm, uv);
			return meshAngleDistortion(cm);
		};
		// The cotangent weights minimise Dirichlet energy, which is exactly
		// the conformal (angle-preserving) criterion. If this ever inverts,
		// the two weight formulas have been swapped.
		expect(build("harmonic")).toBeLessThan(build("mean-value"));
	});

	test("uniform weights are the worst of the three, as Tutte's original is", () => {
		const measure = (weights: WeightScheme) => {
			const cm = withWedges(hemisphere(3));
			const { uv } = parametrizeDisk(cm, { weights });
			writeWedgeUV(cm, uv);
			return meshAngleDistortion(cm);
		};
		expect(measure("uniform")).toBeGreaterThan(measure("mean-value"));
	});

	test("chord spacing gives each boundary edge an arc matching its length", () => {
		// A patch whose boundary edges differ a lot in length, so the two
		// spacings cannot coincide by accident.
		const cm = gridPlane(6, 2).mesh;
		for (let v = 0; v < cm.vertSize; v++) cm.setVert(v, cm.vx(v) * 9, cm.vy(v), cm.vz(v));

		const { uv, boundary } = parametrizeDisk(cm, { boundarySpacing: "chord" });
		const angleAt = (v: number) => Math.atan2(uv[2 * v + 1] - 0.5, uv[2 * v] - 0.5);
		const arc = (i: number) => {
			let d = angleAt(boundary[(i + 1) % boundary.length]) - angleAt(boundary[i]);
			while (d <= 0) d += 2 * Math.PI;
			return d;
		};
		const length = (i: number) => {
			const a = boundary[i];
			const b = boundary[(i + 1) % boundary.length];
			return Math.hypot(cm.vx(a) - cm.vx(b), cm.vy(a) - cm.vy(b), cm.vz(a) - cm.vz(b));
		};

		let perimeter = 0;
		for (let i = 0; i < boundary.length; i++) perimeter += length(i);
		for (let i = 0; i < boundary.length; i++) {
			expect(arc(i) / (2 * Math.PI), `edge ${i}`).toBeCloseTo(length(i) / perimeter, 9);
		}
	});

	test("uniform spacing gives every boundary edge the same arc", () => {
		const cm = gridPlane(6, 2).mesh;
		for (let v = 0; v < cm.vertSize; v++) cm.setVert(v, cm.vx(v) * 9, cm.vy(v), cm.vz(v));
		const { uv, boundary } = parametrizeDisk(cm, { boundarySpacing: "uniform" });
		const angleAt = (v: number) => Math.atan2(uv[2 * v + 1] - 0.5, uv[2 * v] - 0.5);
		for (let i = 0; i < boundary.length; i++) {
			let d = angleAt(boundary[(i + 1) % boundary.length]) - angleAt(boundary[i]);
			while (d <= 0) d += 2 * Math.PI;
			expect(d).toBeCloseTo((2 * Math.PI) / boundary.length, 9);
		}
	});

	test("it converges well before the iteration cap", () => {
		const result = parametrizeDisk(gridPlane(6, 6).mesh, { iterations: 5000 });
		expect(result.iterations).toBeLessThan(5000);
		expect(result.iterations).toBeGreaterThan(0);
	});

	test("a closed mesh is refused rather than parametrised into nonsense", () => {
		expect(() => parametrizeDisk(sphereIcosa(2).mesh)).toThrow(/not a disk/);
	});
});

describe("distortion measures", () => {
	test("a flat patch mapped to itself has no distortion at all", () => {
		const cm = withWedges(gridPlane(4, 4).mesh);
		// The identity map, scaled into 0..1: an isometry up to scale, so
		// every measure below must be exactly zero.
		let minX = Number.POSITIVE_INFINITY;
		let maxX = Number.NEGATIVE_INFINITY;
		let minY = Number.POSITIVE_INFINITY;
		let maxY = Number.NEGATIVE_INFINITY;
		for (let v = 0; v < cm.vertSize; v++) {
			minX = Math.min(minX, cm.vx(v));
			maxX = Math.max(maxX, cm.vx(v));
			minY = Math.min(minY, cm.vy(v));
			maxY = Math.max(maxY, cm.vy(v));
		}
		const uv = new Float64Array(cm.vertSize * 2);
		for (let v = 0; v < cm.vertSize; v++) {
			uv[2 * v] = (cm.vx(v) - minX) / (maxX - minX);
			uv[2 * v + 1] = (cm.vy(v) - minY) / (maxY - minY);
		}
		writeWedgeUV(cm, uv);

		const { ratio } = meshScalingFactor(cm);
		expect(foldedNum(cm)).toBe(0);
		expect(globallyUnfolded(cm)).toBe(true);
		for (let f = 0; f < cm.faceSize; f++) {
			expect(angleDistortion(cm, f)).toBeCloseTo(0, 10);
			expect(areaDistortion(cm, f, ratio)).toBeCloseTo(0, 10);
			for (let e = 0; e < 3; e++) expect(edgeDistortion(cm, f, e, ratio)).toBeCloseTo(0, 10);
		}
		// An isometry has both singular values equal to one, so L2 is one.
		expect(meshL2Stretch(cm)).toBeCloseTo(1, 8);
	});

	test("a uniform scale is not a distortion", () => {
		const cm = withWedges(gridPlane(3, 3).mesh);
		const uv = new Float64Array(cm.vertSize * 2);
		for (let v = 0; v < cm.vertSize; v++) {
			// A tenth of the size: the scaling factor must absorb it.
			uv[2 * v] = cm.vx(v) * 0.1;
			uv[2 * v + 1] = cm.vy(v) * 0.1;
		}
		writeWedgeUV(cm, uv);
		const { ratio } = meshScalingFactor(cm);
		for (let f = 0; f < cm.faceSize; f++) {
			expect(areaDistortion(cm, f, ratio)).toBeCloseTo(0, 10);
		}
		expect(meshL2Stretch(cm)).toBeCloseTo(1, 8);
	});

	test("an anisotropic scale distorts angles but not areas", () => {
		const cm = withWedges(gridPlane(4, 4).mesh);
		const uv = new Float64Array(cm.vertSize * 2);
		for (let v = 0; v < cm.vertSize; v++) {
			// Twice as wide, half as tall: area is preserved exactly, and the
			// two singular values become 2 and 1/2.
			uv[2 * v] = cm.vx(v) * 2;
			uv[2 * v + 1] = cm.vy(v) * 0.5;
		}
		writeWedgeUV(cm, uv);
		const { ratio } = meshScalingFactor(cm);
		expect(ratio).toBeCloseTo(1, 10);
		for (let f = 0; f < cm.faceSize; f++) {
			expect(areaDistortion(cm, f, ratio)).toBeCloseTo(0, 10);
			// sqrt((1/4 + 4)/2) for the singular values 1/2 and 2.
			expect(Math.sqrt(l2StretchEnergySquared(cm, f, ratio))).toBeCloseTo(
				Math.sqrt((0.25 + 4) / 2),
				8,
			);
			expect(lInfStretchEnergy(cm, f, ratio)).toBeCloseTo(2, 8);
		}
		expect(meshAngleDistortion(cm)).toBeGreaterThan(0.1);
	});

	test("a mirrored map folds every face", () => {
		const cm = withWedges(gridPlane(3, 3).mesh);
		const uv = new Float64Array(cm.vertSize * 2);
		for (let v = 0; v < cm.vertSize; v++) {
			uv[2 * v] = -cm.vx(v);
			uv[2 * v + 1] = cm.vy(v);
		}
		writeWedgeUV(cm, uv);
		expect(foldedNum(cm)).toBe(cm.fn);
		// All of them the same way round, so it is a mirror image rather than
		// a broken parametrisation.
		expect(globallyUnfolded(cm)).toBe(true);
	});

	test("one flipped triangle is caught", () => {
		const cm = withWedges(gridPlane(3, 3).mesh);
		const { uv } = parametrizeDisk(cm);
		writeWedgeUV(cm, uv);
		expect(globallyUnfolded(cm)).toBe(true);

		// Swap two corners of one face in texture space.
		const wt = cm.wedgeTexCoord as Float64Array;
		const [u0, v0, u1, v1] = [wt[0], wt[1], wt[2], wt[3]];
		wt[0] = u1;
		wt[1] = v1;
		wt[2] = u0;
		wt[3] = v0;
		expect(isFolded(cm, 0)).toBe(true);
		expect(foldedNum(cm)).toBe(1);
		expect(globallyUnfolded(cm)).toBe(false);
	});

	test("the signed UV area matches the closed form for a known triangle", () => {
		const cm = withWedges(buildMesh([0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 2]));
		(cm.wedgeTexCoord as Float64Array).set([0, 0, 2, 0, 0, 3]);
		expect(signedAreaUV(cm, 0)).toBeCloseTo(3, 12);
		expect(areaUV(cm, 0)).toBeCloseTo(3, 12);
		expect(area3D(cm, 0)).toBeCloseTo(0.5, 12);
	});

	test("a cone flattens almost perfectly, because it is developable", () => {
		const cm = withWedges(coneFan(48, 1, 1));
		const { uv, valid } = parametrizeDisk(cm, { weights: "harmonic" });
		expect(valid).toBe(true);
		writeWedgeUV(cm, uv);
		// Not exactly zero — the boundary is pinned to a circle rather than
		// to the cone's own developed outline — but far better than a
		// hemisphere, which cannot be flattened at all.
		const cone = meshAngleDistortion(cm);

		const sphere = withWedges(hemisphere(3));
		const flat = parametrizeDisk(sphere, { weights: "harmonic" });
		writeWedgeUV(sphere, flat.uv);
		expect(cone).toBeLessThan(meshAngleDistortion(sphere));
	});
});
