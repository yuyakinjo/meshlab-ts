/**
 * The polyline family, and the edge container underneath it.
 *
 * These are the first filters that produce an *edge mesh* — vertices and edges,
 * no faces — so about half of what is tested here is the container itself:
 * that edges grow, compact and weld like every other channel, and that a
 * routine which renumbers vertices renumbers the edges pointing at them too.
 * That last one is not hypothetical; it is what broke first when the container
 * landed.
 *
 * The filters have exact answers to check against. A cube's crease polyline is
 * its twelve edges over its eight corners. A unit sphere sliced at the equator
 * is a closed loop of circumference 2π, up to the polygonal error of whatever
 * tessellation it was given. A selection's perimeter is a closed loop, so every
 * vertex on it has exactly two incident edges.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MLException } from "../../src/common/utilities/ml_exception.ts";
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import { Clean } from "../../src/vcg/complex/clean.ts";
import { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { FaceFlag } from "../../src/vcg/complex/flags.ts";
import { Polyline } from "../../src/vcg/complex/polyline.ts";
import { cube, gridPlane, sphereIcosa } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

function docWith(cm: CMeshO) {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", "m", true, cm);
	m.updateBoxAndNormals();
	return { doc, m, cm };
}

/** Total length of every live edge. */
function polylineLength(cm: CMeshO): number {
	let total = 0;
	for (let e = 0; e < cm.edgeSize; e++) {
		if (cm.isEdgeD(e)) continue;
		const a = cm.ev(e, 0);
		const b = cm.ev(e, 1);
		total += Math.hypot(cm.vx(a) - cm.vx(b), cm.vy(a) - cm.vy(b), cm.vz(a) - cm.vz(b));
	}
	return total;
}

/** How many edges touch each vertex. */
function edgeValence(cm: CMeshO): number[] {
	const out = new Array<number>(cm.vertSize).fill(0);
	for (let e = 0; e < cm.edgeSize; e++) {
		if (cm.isEdgeD(e)) continue;
		out[cm.ev(e, 0)]++;
		out[cm.ev(e, 1)]++;
	}
	return out;
}

// -------------------------------------------------------- the edge container

describe("the edge container", () => {
	test("edges allocate, grow and count like any other channel", () => {
		const cm = new CMeshO();
		Allocator.addVertices(cm, 4);
		expect(cm.en).toBe(0);
		const first = Allocator.addEdges(cm, 3);
		expect(first).toBe(0);
		expect(cm.en).toBe(3);
		expect(cm.edgeSize).toBe(3);
		cm.setEdge(0, 0, 1);
		expect(cm.ev(0, 0)).toBe(0);
		expect(cm.ev(0, 1)).toBe(1);

		// Enough to force several reallocations.
		Allocator.addEdges(cm, 5000);
		expect(cm.en).toBe(5003);
		expect(cm.ev(0, 1)).toBe(1);
	});

	test("deletion is lazy and compaction reclaims the slots", () => {
		const cm = new CMeshO();
		Allocator.addVertices(cm, 4);
		for (let i = 0; i < 4; i++) Allocator.addEdge(cm, i % 4, (i + 1) % 4);
		Allocator.deleteEdge(cm, 1);
		expect(cm.en).toBe(3);
		expect(cm.edgeSize).toBe(4);
		Allocator.compactEdgeVector(cm);
		expect(cm.edgeSize).toBe(3);
		expect(cm.en).toBe(3);
	});

	test("compacting vertices renumbers the edges pointing at them", () => {
		// The bug the container shipped with: faces were remapped and edges were
		// not, so every edge pointed at whatever moved into its old slot.
		const cm = new CMeshO();
		Allocator.addVertices(cm, 5);
		for (let v = 0; v < 5; v++) cm.setVert(v, v, 0, 0);
		Allocator.addEdge(cm, 3, 4);
		// Delete two vertices the edge does not use, so 3 and 4 slide down to
		// 1 and 2.
		Allocator.deleteVertex(cm, 0);
		Allocator.deleteVertex(cm, 1);
		Allocator.compactVertexVector(cm);
		expect(cm.ev(0, 0)).toBe(1);
		expect(cm.ev(0, 1)).toBe(2);
		expect(cm.vx(cm.ev(0, 0))).toBe(3);
		expect(cm.vx(cm.ev(0, 1))).toBe(4);
	});

	test("welding duplicate vertices joins the segments and drops the collapsed ones", () => {
		const cm = new CMeshO();
		// Two segments sharing an endpoint, given as four separate vertices.
		Allocator.addVertices(cm, 4);
		cm.setVert(0, 0, 0, 0);
		cm.setVert(1, 1, 0, 0);
		cm.setVert(2, 1, 0, 0);
		cm.setVert(3, 2, 0, 0);
		Allocator.addEdge(cm, 0, 1);
		Allocator.addEdge(cm, 2, 3);
		// And one segment of zero length, which welding must remove entirely.
		Allocator.addVertices(cm, 2);
		cm.setVert(4, 5, 0, 0);
		cm.setVert(5, 5, 0, 0);
		Allocator.addEdge(cm, 4, 5);

		Clean.removeDuplicateVertex(cm);
		Allocator.compactEveryVector(cm);
		expect(cm.vn).toBe(4);
		expect(cm.en).toBe(2);
		// The two real segments now share their middle vertex.
		expect(Math.max(...edgeValence(cm))).toBe(2);
	});

	test("an ordinary mesh carries no edges and pays nothing for them", () => {
		const cm = sphereIcosa(2).mesh;
		expect(cm.en).toBe(0);
		expect(cm.edgeSize).toBe(0);
		expect(cm.edgeVert.length).toBe(0);
	});
});

// ------------------------------------------------------------ planar section

describe("Compute Planar Section", () => {
	test("slicing a unit sphere at the equator gives a loop of circumference 2 pi", () => {
		const { doc } = docWith(sphereIcosa(4).mesh);
		const out = kernel.applyFilter(doc, "Compute Planar Section", {
			planeAxis: 2,
			relativeTo: 2,
			planeOffset: 0,
		});
		const line = doc.mm().cm;
		expect(out.edge_number as number).toBeGreaterThan(0);
		// A polygon inscribed in the circle, so slightly short of 2 pi and
		// never over it.
		expect(polylineLength(line)).toBeLessThan(2 * Math.PI);
		expect(polylineLength(line)).toBeGreaterThan(2 * Math.PI * 0.999);
	});

	test("the section is a closed loop", () => {
		const { doc } = docWith(sphereIcosa(3).mesh);
		kernel.applyFilter(doc, "Compute Planar Section", {
			planeAxis: 2,
			relativeTo: 2,
			planeOffset: 0,
		});
		const line = doc.mm().cm;
		// Every vertex of a closed loop has exactly two edges.
		for (let v = 0; v < line.vertSize; v++) {
			if (!line.isVertD(v)) expect(edgeValence(line)[v]).toBe(2);
		}
	});

	test("a plane missing the mesh entirely gives nothing", () => {
		const { doc } = docWith(sphereIcosa(2).mesh);
		const out = kernel.applyFilter(doc, "Compute Planar Section", {
			planeAxis: 2,
			relativeTo: 2,
			planeOffset: 5,
		});
		expect(out.edge_number).toBe(0);
	});

	test("slicing off-centre gives a shorter loop", () => {
		const at = (offset: number) => {
			const { doc } = docWith(sphereIcosa(4).mesh);
			kernel.applyFilter(doc, "Compute Planar Section", {
				planeAxis: 2,
				relativeTo: 2,
				planeOffset: offset,
			});
			return polylineLength(doc.mm().cm);
		};
		// A circle of latitude at z = 0.6 has radius 0.8.
		expect(at(0.6)).toBeLessThan(at(0));
		expect(at(0.6) / at(0)).toBeCloseTo(0.8, 1);
	});

	test("a custom axis works and the result is perpendicular to it", () => {
		const { doc } = docWith(sphereIcosa(3).mesh);
		kernel.applyFilter(doc, "Compute Planar Section", {
			planeAxis: 3,
			customAxis: [1, 1, 0],
			relativeTo: 2,
			planeOffset: 0,
		});
		const line = doc.mm().cm;
		expect(line.en).toBeGreaterThan(0);
		// Every vertex lies on the plane x + y = 0.
		for (let v = 0; v < line.vertSize; v++) {
			if (!line.isVertD(v)) expect(line.vx(v) + line.vy(v)).toBeCloseTo(0, 9);
		}
	});

	test("refuses a zero axis", () => {
		const { doc } = docWith(sphereIcosa(2).mesh);
		expect(() =>
			kernel.applyFilter(doc, "Compute Planar Section", {
				planeAxis: 3,
				customAxis: [0, 0, 0],
			}),
		).toThrow(MLException);
	});
});

// --------------------------------------------------------------- perimeter

describe("Create Selection Perimeter Polyline", () => {
	/** A grid with a square block of faces selected. */
	function selectedBlock() {
		const { doc, cm } = docWith(gridPlane(4, 4).mesh);
		for (let f = 0; f < cm.faceSize; f++) {
			const inside = [0, 1, 2].every((k) => {
				const v = cm.fv(f, k);
				return cm.vx(v) <= 0.5 + 1e-9 && cm.vy(v) <= 0.5 + 1e-9;
			});
			if (inside) cm.faceFlags[f] |= FaceFlag.SELECTED;
		}
		return { doc, cm };
	}

	test("the perimeter is a closed loop around the selection", () => {
		const { doc } = selectedBlock();
		kernel.applyFilter(doc, "Create Selection Perimeter Polyline");
		const line = doc.mm().cm;
		expect(line.en).toBeGreaterThan(0);
		for (let v = 0; v < line.vertSize; v++) {
			if (!line.isVertD(v)) expect(edgeValence(line)[v]).toBe(2);
		}
	});

	test("its length is the perimeter of the selected region", () => {
		const { doc } = selectedBlock();
		kernel.applyFilter(doc, "Create Selection Perimeter Polyline");
		// Half the unit square on each side: a 0.5 x 0.5 block, perimeter 2.
		expect(polylineLength(doc.mm().cm)).toBeCloseTo(2, 9);
	});

	test("selecting everything gives the mesh's own boundary", () => {
		const { doc, cm } = docWith(gridPlane(4, 4).mesh);
		for (let f = 0; f < cm.faceSize; f++) cm.faceFlags[f] |= FaceFlag.SELECTED;
		kernel.applyFilter(doc, "Create Selection Perimeter Polyline");
		expect(polylineLength(doc.mm().cm)).toBeCloseTo(4, 9);
	});

	test("refuses an empty selection", () => {
		const { doc } = docWith(gridPlane(3, 3).mesh);
		expect(() => kernel.applyFilter(doc, "Create Selection Perimeter Polyline")).toThrow(
			MLException,
		);
	});
});

// ------------------------------------------------------- crease extraction

describe("Build a Polyline from Selected Edges", () => {
	test("a cube's creases are its twelve edges over its eight corners", () => {
		const { doc } = docWith(cube(1).mesh);
		kernel.applyFilter(doc, "Select Crease Edges", { AngleDegNeg: -30, AngleDegPos: 30 });
		const out = kernel.applyFilter(doc, "Build a Polyline from Selected Edges");
		expect(out.edge_number).toBe(12);
		expect(out.vertex_number).toBe(8);
		// Three edges meet at every corner of a cube.
		const line = doc.mm().cm;
		for (let v = 0; v < line.vertSize; v++) {
			if (!line.isVertD(v)) expect(edgeValence(line)[v]).toBe(3);
		}
	});

	test("each interior crease is emitted once, not once per face", () => {
		// Both faces of an interior edge carry the mark, so a naive pass would
		// give twenty-four segments for a cube instead of twelve.
		const { doc } = docWith(cube(1).mesh);
		kernel.applyFilter(doc, "Select Crease Edges", { AngleDegNeg: -30, AngleDegPos: 30 });
		kernel.applyFilter(doc, "Build a Polyline from Selected Edges");
		expect(polylineLength(doc.mm().cm)).toBeCloseTo(12, 9);
	});

	test("nothing selected gives an empty polyline", () => {
		const { doc } = docWith(sphereIcosa(2).mesh);
		const out = kernel.applyFilter(doc, "Build a Polyline from Selected Edges");
		expect(out.edge_number).toBe(0);
	});
});

// ------------------------------------------------------ cylindrical unwrap

describe("Geometric Cylindrical Unwrapping", () => {
	test("a cylinder unrolls into a flat sheet", () => {
		// Points on a cylinder of radius r about Y all have the same distance
		// from the axis, so their z after unrolling is constant.
		const cm = new CMeshO();
		const rings = 8;
		const steps = 24;
		Allocator.addVertices(cm, rings * steps);
		for (let i = 0; i < rings; i++) {
			for (let j = 0; j < steps; j++) {
				const t = (2 * Math.PI * j) / steps;
				cm.setVert(i * steps + j, 2 * Math.cos(t), i, 2 * Math.sin(t));
			}
		}
		for (let i = 0; i + 1 < rings; i++) {
			for (let j = 0; j < steps; j++) {
				const a = i * steps + j;
				const b = i * steps + ((j + 1) % steps);
				const f = Allocator.addFaces(cm, 2);
				cm.setFace(f, a, b, a + steps);
				cm.setFace(f + 1, b, b + steps, a + steps);
			}
		}
		const { doc } = docWith(cm);
		kernel.applyFilter(doc, "Geometric Cylindrical Unwrapping", {
			startAngle: 0,
			endAngle: 360,
		});
		const flat = doc.mm().cm;
		expect(flat.vn).toBeGreaterThan(0);
		for (let v = 0; v < flat.vertSize; v++) {
			if (!flat.isVertD(v)) expect(flat.vz(v)).toBeCloseTo(2, 9);
		}
	});

	test("the unrolled width is the circumference", () => {
		const cm = new CMeshO();
		const steps = 64;
		Allocator.addVertices(cm, 2 * steps);
		for (let i = 0; i < 2; i++) {
			for (let j = 0; j < steps; j++) {
				const t = (2 * Math.PI * j) / steps;
				cm.setVert(i * steps + j, 3 * Math.cos(t), i, 3 * Math.sin(t));
			}
		}
		for (let j = 0; j < steps; j++) {
			const a = j;
			const b = (j + 1) % steps;
			const f = Allocator.addFaces(cm, 2);
			cm.setFace(f, a, b, a + steps);
			cm.setFace(f + 1, b, b + steps, a + steps);
		}
		const { doc } = docWith(cm);
		kernel.applyFilter(doc, "Geometric Cylindrical Unwrapping", {
			startAngle: -180,
			endAngle: 180,
		});
		const flat = doc.mm().cm;
		let lo = Number.POSITIVE_INFINITY;
		let hi = Number.NEGATIVE_INFINITY;
		for (let v = 0; v < flat.vertSize; v++) {
			if (flat.isVertD(v)) continue;
			lo = Math.min(lo, flat.vx(v));
			hi = Math.max(hi, flat.vx(v));
		}
		// Just under the full circumference 2 pi r = 18.85. The range is
		// half-open, so the vertex at exactly +180 is excluded and the sheet is
		// short by a step at each end rather than closing on itself.
		const circumference = 2 * Math.PI * 3;
		const step = circumference / steps;
		expect(hi - lo).toBeGreaterThan(circumference - 2.5 * step);
		expect(hi - lo).toBeLessThanOrEqual(circumference + 1e-9);
	});

	test("an explicit radius overrides the measured one", () => {
		const width = (radius: number) => {
			const { doc } = docWith(sphereIcosa(3).mesh);
			kernel.applyFilter(doc, "Geometric Cylindrical Unwrapping", {
				startAngle: 0,
				endAngle: 360,
				radius,
			});
			const flat = doc.mm().cm;
			let lo = Number.POSITIVE_INFINITY;
			let hi = Number.NEGATIVE_INFINITY;
			for (let v = 0; v < flat.vertSize; v++) {
				if (flat.isVertD(v)) continue;
				lo = Math.min(lo, flat.vx(v));
				hi = Math.max(hi, flat.vx(v));
			}
			return hi - lo;
		};
		// The x axis is an angle scaled by the radius, so doubling it doubles
		// the width.
		expect(width(2) / width(1)).toBeCloseTo(2, 6);
	});

	test("keeps every vertex and drops only the faces that straddle the seam", () => {
		const { doc, cm } = docWith(sphereIcosa(3).mesh);
		const before = { vn: cm.vn, fn: cm.fn };
		const out = kernel.applyFilter(doc, "Geometric Cylindrical Unwrapping", {
			startAngle: 0,
			endAngle: 360,
		});
		expect(out.vertex_number).toBe(before.vn);
		// A face crossing the seam would be stretched across the whole sheet,
		// so it is dropped rather than drawn.
		expect(out.face_number as number).toBeLessThan(before.fn);
		expect(out.face_number as number).toBeGreaterThan(before.fn * 0.9);
	});

	test("a wider range unrolls more than once", () => {
		const { doc } = docWith(sphereIcosa(2).mesh);
		const one = kernel.applyFilter(doc, "Geometric Cylindrical Unwrapping", {
			startAngle: 0,
			endAngle: 360,
		});
		const two = docWith(sphereIcosa(2).mesh);
		const both = kernel.applyFilter(two.doc, "Geometric Cylindrical Unwrapping", {
			startAngle: 0,
			endAngle: 720,
		});
		expect(both.vertex_number as number).toBeGreaterThan(one.vertex_number as number);
	});
});

// ---------------------------------------------------------- the extractors

describe("the polyline extractors, directly", () => {
	test("a planar section touching a single vertex emits nothing", () => {
		// The plane grazes one corner. Counting "on the plane" as positive means
		// no edge straddles, so no zero-length segment comes out.
		const cm = new CMeshO();
		Allocator.addVertices(cm, 3);
		cm.setVert(0, 0, 0, 0);
		cm.setVert(1, 1, 0, 1);
		cm.setVert(2, 0, 1, 1);
		Allocator.addFaces(cm, 1);
		cm.setFace(0, 0, 1, 2);
		const line = Polyline.planarSection(cm, [0, 0, 1], 0);
		expect(line.en).toBe(0);
	});

	test("a planar section through a face gives exactly one segment", () => {
		const cm = new CMeshO();
		Allocator.addVertices(cm, 3);
		cm.setVert(0, 0, 0, -1);
		cm.setVert(1, 1, 0, 1);
		cm.setVert(2, 0, 1, 1);
		Allocator.addFaces(cm, 1);
		cm.setFace(0, 0, 1, 2);
		const line = Polyline.planarSection(cm, [0, 0, 1], 0);
		expect(line.en).toBe(1);
	});

	test("a degenerate normal is refused rather than dividing by zero", () => {
		const cm = sphereIcosa(1).mesh;
		expect(Polyline.planarSection(cm, [0, 0, 0], 0).en).toBe(0);
	});
});
