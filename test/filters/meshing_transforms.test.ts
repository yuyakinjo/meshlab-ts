/**
 * `filter_meshing`'s matrix family, plus crease selection and the
 * polygon-to-triangle conversion.
 *
 * Transform filters are pleasant to test because almost every one has an exact
 * expected answer: flipping twice is the identity, inverting after applying
 * gets back to where it started, a matrix set explicitly does exactly what the
 * matrix says. Where a test only checks a bound instead, it is because the
 * quantity really is a fit rather than a construction — the plane through a
 * selection, or the principal axes of a shape.
 *
 * One behaviour is worth flagging up front, because it looks like a bug and is
 * not: `Align to Principal Axis` puts the *longest* axis on Z in its default
 * (point-covariance) mode and on X in its inertia mode. Both take eigenvalues
 * ascending, but a covariance eigenvalue grows with spread while a moment of
 * inertia shrinks with it. Upstream has the same asymmetry, and the tests pin
 * both directions so neither can drift.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import { MLException } from "../../src/common/utilities/ml_exception.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { FaceFlag, VertexFlag } from "../../src/vcg/complex/flags.ts";
import { covariance, fitPlaneToPointSet, symmetricEigen3 } from "../../src/vcg/math/eigen3.ts";
import { Matrix44Ops } from "../../src/vcg/math/matrix44.ts";
import { cube, gridPlane, sphereIcosa } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

function docWith(cm: CMeshO) {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", "m", true, cm);
	m.updateBoxAndNormals();
	return { doc, m, cm };
}

/** Every live vertex position, in index order. */
function positions(cm: CMeshO): number[][] {
	const out: number[][] = [];
	for (let v = 0; v < cm.vertSize; v++) {
		if (!cm.isVertD(v)) out.push([cm.vx(v), cm.vy(v), cm.vz(v)]);
	}
	return out;
}

/** The extent of the mesh along each axis. */
function extents(cm: CMeshO): number[] {
	return [0, 1, 2].map((a) => {
		let lo = Number.POSITIVE_INFINITY;
		let hi = Number.NEGATIVE_INFINITY;
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			lo = Math.min(lo, cm.vertCoord[3 * v + a]);
			hi = Math.max(hi, cm.vertCoord[3 * v + a]);
		}
		return hi - lo;
	});
}

/** A sphere stretched to distinct radii, so its axes are unambiguous. */
function ellipsoid(rx: number, ry: number, rz: number): CMeshO {
	const cm = sphereIcosa(3).mesh;
	for (let v = 0; v < cm.vertSize; v++) {
		cm.setVert(v, cm.vx(v) * rx, cm.vy(v) * ry, cm.vz(v) * rz);
	}
	return cm;
}

// ------------------------------------------------------------- flip and swap

describe("Transform: Flip and/or swap axis", () => {
	test("flipping an axis negates that coordinate and nothing else", () => {
		const { doc, cm } = docWith(sphereIcosa(2).mesh);
		const before = positions(cm);
		kernel.applyFilter(doc, "Transform: Flip and/or swap axis", { flipY: true });
		positions(cm).forEach((p, i) => {
			expect(p[0]).toBeCloseTo(before[i][0], 12);
			expect(p[1]).toBeCloseTo(-before[i][1], 12);
			expect(p[2]).toBeCloseTo(before[i][2], 12);
		});
	});

	test("flipping twice is the identity", () => {
		const { doc, cm } = docWith(sphereIcosa(2).mesh);
		const before = positions(cm);
		kernel.applyFilter(doc, "Transform: Flip and/or swap axis", { flipX: true, flipZ: true });
		kernel.applyFilter(doc, "Transform: Flip and/or swap axis", { flipX: true, flipZ: true });
		positions(cm).forEach((p, i) => {
			for (let k = 0; k < 3; k++) expect(p[k]).toBeCloseTo(before[i][k], 12);
		});
	});

	test("swapping exchanges two coordinates", () => {
		const { doc, cm } = docWith(ellipsoid(3, 1, 1));
		const before = extents(cm);
		kernel.applyFilter(doc, "Transform: Flip and/or swap axis", { swapXZ: true });
		const after = extents(cm);
		expect(after[0]).toBeCloseTo(before[2], 9);
		expect(after[2]).toBeCloseTo(before[0], 9);
		expect(after[1]).toBeCloseTo(before[1], 9);
	});

	test("a mirroring transform has its winding corrected", () => {
		// A flip has determinant -1, so applying it to the coordinates alone would
		// leave every face wound backwards and the solid inside out. The transform
		// re-winds the faces to compensate, which is why the volume stays positive
		// and the same size rather than going negative.
		const { doc, cm } = docWith(sphereIcosa(2).mesh);
		const before = signedVolumeOf(cm);
		expect(before).toBeGreaterThan(0);
		kernel.applyFilter(doc, "Transform: Flip and/or swap axis", { flipX: true });
		expect(signedVolumeOf(cm)).toBeCloseTo(before, 9);

		// The geometry really did mirror, even though the orientation survived.
		expect(cm.vx(0)).toBeCloseTo(-sphereIcosa(2).mesh.vx(0), 12);
	});

	test("an even number of flips needs no correction", () => {
		// Two flips compose to a rotation, so there is nothing to re-wind and the
		// face indices come back exactly as they were.
		const { doc, cm } = docWith(sphereIcosa(2).mesh);
		const before = [...cm.faceVert.subarray(0, 3 * cm.faceSize)];
		kernel.applyFilter(doc, "Transform: Flip and/or swap axis", { flipX: true, flipY: true });
		expect([...cm.faceVert.subarray(0, 3 * cm.faceSize)]).toEqual(before);
	});

	test("no flags is a no-op", () => {
		const { doc, cm } = docWith(sphereIcosa(2).mesh);
		const before = positions(cm);
		kernel.applyFilter(doc, "Transform: Flip and/or swap axis", {});
		positions(cm).forEach((p, i) => {
			for (let k = 0; k < 3; k++) expect(p[k]).toBeCloseTo(before[i][k], 12);
		});
	});
});

function signedVolumeOf(cm: CMeshO): number {
	let total = 0;
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		const p = [0, 1, 2].map((k) => {
			const v = cm.fv(f, k);
			return [cm.vx(v), cm.vy(v), cm.vz(v)];
		});
		total +=
			(p[0][0] * (p[1][1] * p[2][2] - p[1][2] * p[2][1]) -
				p[0][1] * (p[1][0] * p[2][2] - p[1][2] * p[2][0]) +
				p[0][2] * (p[1][0] * p[2][1] - p[1][1] * p[2][0])) /
			6;
	}
	return total;
}

// ---------------------------------------------------------- set / invert

describe("Matrix: Set/Copy Transformation", () => {
	test("applies exactly the matrix it was given", () => {
		const { doc, cm } = docWith(cube(1).mesh);
		const before = positions(cm);
		// A scale of two on x, a translation of three on y.
		kernel.applyFilter(doc, "Matrix: Set/Copy Transformation", {
			TransformMatrix: [2, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1, 0, 0, 0, 0, 1],
		});
		positions(cm).forEach((p, i) => {
			expect(p[0]).toBeCloseTo(before[i][0] * 2, 12);
			expect(p[1]).toBeCloseTo(before[i][1] + 3, 12);
			expect(p[2]).toBeCloseTo(before[i][2], 12);
		});
	});

	test("without freezing, the layer keeps the matrix and the coordinates stand still", () => {
		const { doc, cm } = docWith(cube(1).mesh);
		const before = positions(cm);
		kernel.applyFilter(doc, "Matrix: Set/Copy Transformation", {
			TransformMatrix: [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1],
			Freeze: false,
		});
		positions(cm).forEach((p, i) => {
			for (let k = 0; k < 3; k++) expect(p[k]).toBeCloseTo(before[i][k], 12);
		});
		expect(cm.transformMatrix[0]).toBeCloseTo(2, 12);
	});

	test("compose multiplies onto what is already there", () => {
		const { doc, cm } = docWith(cube(1).mesh);
		const double = [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1];
		kernel.applyFilter(doc, "Matrix: Set/Copy Transformation", {
			TransformMatrix: double,
			Freeze: false,
			compose: false,
		});
		kernel.applyFilter(doc, "Matrix: Set/Copy Transformation", {
			TransformMatrix: double,
			Freeze: false,
			compose: true,
		});
		// Two doublings compose to a quadrupling; without compose it would
		// still read 2.
		expect(cm.transformMatrix[0]).toBeCloseTo(4, 12);
	});

	test("all-layers reaches every visible layer", () => {
		const doc = new MeshDocument();
		const a = doc.addNewMesh("", "a", true, cube(1).mesh);
		const b = doc.addNewMesh("", "b", true, cube(1).mesh);
		a.updateBoxAndNormals();
		b.updateBoxAndNormals();
		kernel.applyFilter(doc, "Matrix: Set/Copy Transformation", {
			TransformMatrix: [1, 0, 0, 7, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
			allLayers: true,
		});
		expect(a.cm.vx(0)).toBeCloseTo(b.cm.vx(0), 12);
		expect(a.cm.vx(0)).toBeGreaterThan(6);
	});
});

describe("Matrix: Invert Current Matrix", () => {
	test("applying then inverting returns the mesh to where it started", () => {
		const { doc, cm } = docWith(cube(1).mesh);
		const before = positions(cm);
		kernel.applyFilter(doc, "Matrix: Set/Copy Transformation", {
			TransformMatrix: [2, 0, 0, 1, 0, 3, 0, -2, 0, 0, 0.5, 4, 0, 0, 0, 1],
			Freeze: false,
		});
		kernel.applyFilter(doc, "Matrix: Invert Current Matrix", { Freeze: false });
		// The layer now carries the inverse, so freezing it takes the mesh
		// through the inverse of a transform it never actually underwent —
		// the coordinates end up at the inverse of where they began.
		kernel.applyFilter(doc, "Matrix: Invert Current Matrix", { Freeze: true });
		positions(cm).forEach((p, i) => {
			expect(p[0]).toBeCloseTo(before[i][0] * 2 + 1, 9);
			expect(p[1]).toBeCloseTo(before[i][1] * 3 - 2, 9);
			expect(p[2]).toBeCloseTo(before[i][2] * 0.5 + 4, 9);
		});
	});

	test("a singular matrix is refused rather than producing NaNs", () => {
		const { doc } = docWith(cube(1).mesh);
		kernel.applyFilter(doc, "Matrix: Set/Copy Transformation", {
			// A zero z scale collapses the mesh into a plane; there is no way back.
			TransformMatrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
			Freeze: false,
		});
		expect(() => kernel.applyFilter(doc, "Matrix: Invert Current Matrix")).toThrow(MLException);
	});

	test("the inverse really is the inverse", () => {
		const m = Matrix44Ops.fromArray([2, 1, 0, 3, 0, 3, 1, -2, 1, 0, 0.5, 4, 0, 0, 0, 1]);
		const inv = Matrix44Ops.invert(m) as Float64Array;
		const product = Matrix44Ops.multiply(m, inv);
		for (let r = 0; r < 4; r++) {
			for (let c = 0; c < 4; c++) {
				expect(product[4 * r + c]).toBeCloseTo(r === c ? 1 : 0, 9);
			}
		}
		expect(Matrix44Ops.invert(Matrix44Ops.scaling(1, 1, 0))).toBeNull();
	});
});

// -------------------------------------------------- translation/rotation/scale

describe("Matrix: Set from translation/rotation/scale", () => {
	test("translation and scale combine as scale-then-translate", () => {
		const { doc, cm } = docWith(cube(1).mesh);
		const before = positions(cm);
		kernel.applyFilter(doc, "Matrix: Set from translation/rotation/scale", {
			translationX: 5,
			scaleY: 2,
		});
		positions(cm).forEach((p, i) => {
			expect(p[0]).toBeCloseTo(before[i][0] + 5, 12);
			expect(p[1]).toBeCloseTo(before[i][1] * 2, 12);
		});
	});

	test("a 90 degree rotation about Z sends x to y", () => {
		const { doc, cm } = docWith(ellipsoid(3, 1, 1));
		kernel.applyFilter(doc, "Matrix: Set from translation/rotation/scale", { rotationZ: 90 });
		const after = extents(cm);
		expect(after[0]).toBeCloseTo(2, 6);
		expect(after[1]).toBeCloseTo(6, 6);
	});

	test("four 90 degree rotations are the identity", () => {
		const { doc, cm } = docWith(sphereIcosa(2).mesh);
		const before = positions(cm);
		for (let i = 0; i < 4; i++) {
			kernel.applyFilter(doc, "Matrix: Set from translation/rotation/scale", { rotationX: 90 });
		}
		positions(cm).forEach((p, i) => {
			for (let k = 0; k < 3; k++) expect(p[k]).toBeCloseTo(before[i][k], 9);
		});
	});

	test("the defaults are the identity", () => {
		const { doc, cm } = docWith(sphereIcosa(2).mesh);
		const before = positions(cm);
		kernel.applyFilter(doc, "Matrix: Set from translation/rotation/scale", {});
		positions(cm).forEach((p, i) => {
			for (let k = 0; k < 3; k++) expect(p[k]).toBeCloseTo(before[i][k], 12);
		});
	});
});

// ------------------------------------------------------------ principal axis

describe("Transform: Align to Principal Axis", () => {
	test("the covariance mode puts the longest axis on Z", () => {
		// Ascending eigenvalues, and a covariance eigenvalue grows with spread.
		const { doc, cm } = docWith(ellipsoid(4, 2, 1));
		kernel.applyFilter(doc, "Transform: Align to Principal Axis", { pointsFlag: true });
		const after = extents(cm);
		expect(after[0]).toBeLessThan(after[1]);
		expect(after[1]).toBeLessThan(after[2]);
		expect(after[2]).toBeCloseTo(8, 6);
	});

	test("the inertia mode puts the longest axis on X", () => {
		// The same ascending order applied to a matrix that runs the other way:
		// a long axis has the *smallest* moment of inertia about it.
		const { doc, cm } = docWith(ellipsoid(4, 2, 1));
		kernel.applyFilter(doc, "Transform: Align to Principal Axis", { pointsFlag: false });
		const after = extents(cm);
		expect(after[0]).toBeGreaterThan(after[1]);
		expect(after[1]).toBeGreaterThan(after[2]);
		expect(after[0]).toBeCloseTo(8, 6);
	});

	test("aligning an already-aligned mesh barely moves it", () => {
		const { doc, cm } = docWith(ellipsoid(1, 2, 4));
		const before = extents(cm);
		kernel.applyFilter(doc, "Transform: Align to Principal Axis", { pointsFlag: true });
		const after = extents(cm);
		for (let k = 0; k < 3; k++) expect(after[k]).toBeCloseTo(before[k], 6);
	});

	test("the transform is a rotation, not a mirroring", () => {
		// An eigenbasis is only defined up to sign, so a naive one can come out
		// left-handed and turn the mesh inside out.
		const { doc, cm } = docWith(ellipsoid(4, 2, 1));
		const before = signedVolumeOf(cm);
		kernel.applyFilter(doc, "Transform: Align to Principal Axis", { pointsFlag: true });
		expect(Math.sign(signedVolumeOf(cm))).toBe(Math.sign(before));
		expect(Math.abs(signedVolumeOf(cm))).toBeCloseTo(Math.abs(before), 6);
	});

	test("works on a point cloud, where the inertia tensor would not", () => {
		const src = ellipsoid(4, 1, 1);
		const cloud = src;
		// Drop the faces: only the covariance mode can say anything now.
		for (let f = 0; f < cloud.faceSize; f++) cloud.faceFlags[f] |= FaceFlag.DELETED;
		cloud.fn = 0;
		const { doc, cm } = docWith(cloud);
		kernel.applyFilter(doc, "Transform: Align to Principal Axis", { pointsFlag: true });
		expect(extents(cm)[2]).toBeCloseTo(8, 6);
	});

	test("refuses fewer than three points", () => {
		const cm = sphereIcosa(1).mesh;
		for (let v = 2; v < cm.vertSize; v++) cm.vertFlags[v] |= VertexFlag.DELETED;
		cm.vn = 2;
		const { doc } = docWith(cm);
		expect(() => kernel.applyFilter(doc, "Transform: Align to Principal Axis")).toThrow(
			MLException,
		);
	});
});

// ------------------------------------------------------------- rotate to fit

describe("Transform: Rotate to Fit to a plane", () => {
	/**
	 * A flat grid tilted 30 degrees about x, with every vertex selected.
	 *
	 * Genuinely planar, unlike a band of a sphere — so "did it land on the
	 * target plane" has an exact answer rather than one bounded by the band's
	 * own thickness.
	 */
	function tilted() {
		const cm = gridPlane(6, 6).mesh;
		const angle = (30 * Math.PI) / 180;
		const c = Math.cos(angle);
		const s = Math.sin(angle);
		for (let v = 0; v < cm.vertSize; v++) {
			const y = cm.vy(v);
			const z = cm.vz(v);
			cm.setVert(v, cm.vx(v), c * y - s * z, s * y + c * z);
			cm.vertFlags[v] |= VertexFlag.SELECTED;
		}
		return docWith(cm);
	}

	test("brings the fitted plane onto the target plane", () => {
		const { doc, cm } = tilted();
		kernel.applyFilter(doc, "Transform: Rotate to Fit to a plane", { targetPlane: 0 });
		// After fitting to the XY plane, the selected vertices should all sit at
		// nearly the same z.
		const zs: number[] = [];
		for (let v = 0; v < cm.vertSize; v++) {
			if (!cm.isVertD(v) && cm.isVertS(v)) zs.push(cm.vz(v));
		}
		const mean = zs.reduce((a, b) => a + b, 0) / zs.length;
		for (const z of zs) expect(Math.abs(z - mean)).toBeLessThan(1e-9);
	});

	test("reports the plane it fitted", () => {
		const { doc } = tilted();
		const out = kernel.applyFilter(doc, "Transform: Rotate to Fit to a plane", {
			targetPlane: 0,
		});
		const normal = out.fitting_plane_normal as number[];
		expect(normal.length).toBe(3);
		expect(Math.hypot(normal[0], normal[1], normal[2])).toBeCloseTo(1, 9);
		// A 30-degree tilt about x: the normal leans that far off z.
		expect(Math.abs(normal[2])).toBeCloseTo(Math.cos((30 * Math.PI) / 180), 2);
		expect(out.fitting_plane_avg_error as number).toBeLessThan(0.1);
	});

	test("refuses when nothing is selected", () => {
		const { doc } = docWith(sphereIcosa(2).mesh);
		expect(() => kernel.applyFilter(doc, "Transform: Rotate to Fit to a plane")).toThrow(
			MLException,
		);
	});

	test("a constrained axis only rotates about that axis", () => {
		const { doc, cm } = tilted();
		const before = positions(cm);
		kernel.applyFilter(doc, "Transform: Rotate to Fit to a plane", {
			targetPlane: 0,
			rotAxis: 1,
		});
		// Rotating about x cannot change any x coordinate.
		positions(cm).forEach((p, i) => {
			expect(p[0]).toBeCloseTo(before[i][0], 9);
		});
	});
});

// ------------------------------------------------------------- crease + tri

describe("Select Crease Edges", () => {
	test("finds every edge of a cube", () => {
		const { doc, cm } = docWith(cube(1).mesh);
		const out = kernel.applyFilter(doc, "Select Crease Edges", {
			AngleDegNeg: -30,
			AngleDegPos: 30,
		});
		// Twelve cube edges, each counted from both of its faces. The internal
		// diagonals are coplanar and must not be selected.
		expect(out.selected_edges).toBe(24);
		let marked = 0;
		for (let f = 0; f < cm.faceSize; f++) {
			for (let e = 0; e < 3; e++) {
				if ((cm.faceFlags[f] & (FaceFlag.FACEEDGESEL0 << e)) !== 0) marked++;
			}
		}
		expect(marked).toBe(24);
	});

	test("a smooth sphere has no creases at a wide threshold", () => {
		const { doc } = docWith(sphereIcosa(3).mesh);
		const out = kernel.applyFilter(doc, "Select Crease Edges", {
			AngleDegNeg: -60,
			AngleDegPos: 60,
		});
		expect(out.selected_edges).toBe(0);
	});

	test("the threshold is signed, so convex and concave can differ", () => {
		// A cube's edges are all convex. A threshold that admits convex angles
		// but not concave ones therefore still finds all of them, while the
		// mirror setting finds none.
		const convex = docWith(cube(1).mesh);
		const concave = docWith(cube(1).mesh);
		const a = kernel.applyFilter(convex.doc, "Select Crease Edges", {
			AngleDegNeg: -180,
			AngleDegPos: 30,
		});
		const b = kernel.applyFilter(concave.doc, "Select Crease Edges", {
			AngleDegNeg: -30,
			AngleDegPos: 180,
		});
		expect(a.selected_edges).toBe(24);
		expect(b.selected_edges).toBe(0);
	});

	test("a narrower threshold selects at least as much", () => {
		const wide = docWith(sphereIcosa(2).mesh);
		const narrow = docWith(sphereIcosa(2).mesh);
		const a = kernel.applyFilter(wide.doc, "Select Crease Edges", {
			AngleDegNeg: -40,
			AngleDegPos: 40,
		});
		const b = kernel.applyFilter(narrow.doc, "Select Crease Edges", {
			AngleDegNeg: -5,
			AngleDegPos: 5,
		});
		expect(b.selected_edges as number).toBeGreaterThanOrEqual(a.selected_edges as number);
	});
});

describe("Turn into a Pure-Triangular mesh", () => {
	test("clears the faux flags and moves nothing", () => {
		const { doc, cm } = docWith(cube(1).mesh);
		// Mark every second edge faux, as a quad-dominant mesh would have.
		for (let f = 0; f < cm.faceSize; f++) cm.faceFlags[f] |= FaceFlag.FAUX2;
		const before = positions(cm);
		const out = kernel.applyFilter(doc, "Turn into a Pure-Triangular mesh");
		expect(out.face_number).toBe(cm.fn);
		for (let f = 0; f < cm.faceSize; f++) {
			expect(cm.faceFlags[f] & FaceFlag.FAUX012).toBe(0);
		}
		// The geometry is untouched: this only forgets which edges were faux.
		positions(cm).forEach((p, i) => {
			for (let k = 0; k < 3; k++) expect(p[k]).toBe(before[i][k]);
		});
	});

	test("is idempotent, and a no-op on a mesh with no faux edges", () => {
		const { doc, cm } = docWith(sphereIcosa(2).mesh);
		const before = { vn: cm.vn, fn: cm.fn };
		kernel.applyFilter(doc, "Turn into a Pure-Triangular mesh");
		kernel.applyFilter(doc, "Turn into a Pure-Triangular mesh");
		expect(cm.vn).toBe(before.vn);
		expect(cm.fn).toBe(before.fn);
	});

	test("drops the polygonal datamask", () => {
		const { doc, m } = docWith(cube(1).mesh);
		m.updateDataMask(MeshElement.MM_POLYGONAL);
		kernel.applyFilter(doc, "Turn into a Pure-Triangular mesh");
		expect(m.hasDataMask(MeshElement.MM_POLYGONAL)).toBe(false);
	});
});

// ------------------------------------------------------------ eigen helpers

describe("the symmetric eigen solver", () => {
	test("recovers a known spectrum", () => {
		const e = symmetricEigen3([2, 1, 0, 1, 2, 0, 0, 0, 5]);
		expect(e.values[0]).toBeCloseTo(1, 9);
		expect(e.values[1]).toBeCloseTo(3, 9);
		expect(e.values[2]).toBeCloseTo(5, 9);
	});

	test("its vectors really are eigenvectors", () => {
		const a = [4, 1, 2, 1, 3, 0, 2, 0, 5];
		const e = symmetricEigen3(a);
		for (let i = 0; i < 3; i++) {
			const v = e.vectors[i];
			for (let r = 0; r < 3; r++) {
				const av = a[3 * r] * v[0] + a[3 * r + 1] * v[1] + a[3 * r + 2] * v[2];
				expect(av).toBeCloseTo(e.values[i] * v[r], 9);
			}
			expect(Math.hypot(v[0], v[1], v[2])).toBeCloseTo(1, 12);
		}
	});

	test("the vectors are mutually orthogonal", () => {
		const e = symmetricEigen3([4, 1, 2, 1, 3, 0, 2, 0, 5]);
		for (const [i, j] of [
			[0, 1],
			[0, 2],
			[1, 2],
		] as const) {
			const a = e.vectors[i];
			const b = e.vectors[j];
			expect(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).toBeCloseTo(0, 9);
		}
	});

	test("a plane fit finds the plane the points lie on", () => {
		const points = [
			[0, 0, 3],
			[1, 0, 3],
			[0, 1, 3],
			[2, 5, 3],
			[-3, 1, 3],
		];
		const plane = fitPlaneToPointSet(points) as NonNullable<ReturnType<typeof fitPlaneToPointSet>>;
		expect(Math.abs(plane.normal[2])).toBeCloseTo(1, 9);
		expect(Math.abs(plane.offset)).toBeCloseTo(3, 9);
	});

	test("the covariance of a spread-out set is largest along its longest axis", () => {
		const points = [
			[-4, 0, 0],
			[4, 0, 0],
			[0, -1, 0],
			[0, 1, 0],
		];
		const c = covariance(points, [0, 0, 0]);
		expect(c[0]).toBeGreaterThan(c[4]);
		expect(c[8]).toBe(0);
	});

	test("fewer than three points cannot define a plane", () => {
		expect(
			fitPlaneToPointSet([
				[0, 0, 0],
				[1, 1, 1],
			]),
		).toBeNull();
	});
});
