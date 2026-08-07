import { describe, expect, test } from "bun:test";
import { UpdateBounding } from "../../../src/vcg/complex/update/bounding.ts";
import { UpdateNormal } from "../../../src/vcg/complex/update/normal.ts";
import { normalizeAt } from "../../../src/vcg/math/vec3.ts";
import {
	buildMesh,
	cube,
	gridPlane,
	singleTriangle,
	sphereIcosa,
} from "../../helpers/mesh_builders.ts";

describe("UpdateNormal", () => {
	test("a face in the z = 0 plane has a normal along z", () => {
		const { mesh } = singleTriangle();
		UpdateNormal.perFaceNormalized(mesh);
		expect([mesh.faceNormal[0], mesh.faceNormal[1], mesh.faceNormal[2]]).toEqual([0, 0, 1]);
	});

	test("cube face normals point outward along the axes", () => {
		const { mesh } = cube(2);
		UpdateNormal.perFaceNormalized(mesh);
		const seen = new Set<string>();
		for (let f = 0; f < mesh.fn; f++) {
			const n = [mesh.faceNormal[3 * f], mesh.faceNormal[3 * f + 1], mesh.faceNormal[3 * f + 2]];
			seen.add(n.join(","));
			// Every normal is an axis direction, and points away from the centre.
			const centreDot = n.reduce((acc, c, k) => {
				const a = mesh.fv(f, 0);
				const p = [mesh.vx(a), mesh.vy(a), mesh.vz(a)][k];
				return acc + c * p;
			}, 0);
			expect(centreDot).toBeGreaterThan(0);
		}
		expect(seen.size).toBe(6); // six distinct outward directions
	});

	test("the unnormalised face normal is twice the triangle's area", () => {
		const { mesh } = singleTriangle(); // area 1/2
		expect(UpdateNormal.faceDoubleArea(mesh, 0)).toBeCloseTo(1, 12);
	});

	test("vertex normals on a sphere point radially outward", () => {
		const { mesh } = sphereIcosa(2);
		UpdateNormal.perVertexNormalizedPerFaceNormalized(mesh);
		for (let v = 0; v < mesh.vn; v++) {
			// On the unit sphere the position is the outward normal.
			const dot =
				mesh.vertNormal[3 * v] * mesh.vx(v) +
				mesh.vertNormal[3 * v + 1] * mesh.vy(v) +
				mesh.vertNormal[3 * v + 2] * mesh.vz(v);
			expect(dot).toBeGreaterThan(0.99);
		}
	});

	test("area weighting and angle weighting agree on a symmetric mesh", () => {
		const { mesh } = sphereIcosa(1);
		UpdateNormal.perVertex(mesh);
		UpdateNormal.normalizePerVertex(mesh);
		const area = Array.from(mesh.vertNormal.subarray(0, mesh.vn * 3));
		UpdateNormal.perVertexAngleWeighted(mesh);
		UpdateNormal.normalizePerVertex(mesh);
		for (let i = 0; i < area.length; i++) {
			expect(mesh.vertNormal[i]).toBeCloseTo(area[i], 6);
		}
	});

	test("a degenerate face gets a zero normal rather than NaN", () => {
		// Three collinear points: zero area, no meaningful normal.
		const m = buildMesh([0, 0, 0, 1, 0, 0, 2, 0, 0], [0, 1, 2]);
		UpdateNormal.perFaceNormalized(m);
		expect([m.faceNormal[0], m.faceNormal[1], m.faceNormal[2]]).toEqual([0, 0, 0]);
	});

	test("normalizeAt stays accurate on subnormal input", () => {
		// Regression: a cross product of tiny coordinates lands in the
		// subnormal range, where computing the length before rescaling throws
		// away most of the precision and the result is not unit length.
		const arr = new Float64Array([6e-323, -1e-323, 3e-323]);
		normalizeAt(arr, 0);
		expect(Math.hypot(arr[0], arr[1], arr[2])).toBeCloseTo(1, 12);
	});

	test("normalizeAt stays accurate on huge input", () => {
		const arr = new Float64Array([1e200, -2e200, 2e200]);
		normalizeAt(arr, 0);
		expect(Math.hypot(arr[0], arr[1], arr[2])).toBeCloseTo(1, 12);
		expect(arr[0]).toBeCloseTo(1 / 3, 12);
	});
});

describe("UpdateBounding", () => {
	test("the box of a centred cube is symmetric about the origin", () => {
		const { mesh } = cube(4);
		UpdateBounding.box(mesh);
		expect(mesh.bbox.min).toEqual([-2, -2, -2]);
		expect(mesh.bbox.max).toEqual([2, 2, 2]);
		expect(mesh.bbox.diagonal).toBeCloseTo(Math.sqrt(48), 12);
	});

	test("a flat mesh gets a zero-thickness box, not an empty one", () => {
		const { mesh } = gridPlane();
		UpdateBounding.box(mesh);
		expect(mesh.bbox.isEmpty).toBe(false);
		expect(mesh.bbox.dimZ).toBe(0);
	});
});
