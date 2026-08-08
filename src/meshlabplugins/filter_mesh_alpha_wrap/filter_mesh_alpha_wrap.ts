/**
 * `filter_mesh_alpha_wrap` — a watertight shell around anything.
 *
 * The idea is a ball of radius alpha rolled over the outside of the input:
 * wherever it fits, it carves; wherever it does not, it bridges. The result
 * is watertight and outside the input by `offset` whatever the input was —
 * self-intersecting, non-manifold, a triangle soup, several disconnected
 * pieces. That robustness is the whole point; it is the standard way to get a
 * printable solid from a model nobody would call a solid.
 *
 * CGAL implements it by refining a Delaunay triangulation. This builds it as
 * a distance field instead: dilate the input by `alpha + offset`, then erode
 * by `alpha`. Dilating closes every gap narrower than the ball, and eroding
 * takes the padding back off without reopening them — the morphological
 * closing that "rolling a ball" describes. The surface is extracted by
 * marching tetrahedra, so it is watertight by construction.
 *
 * What this does not reproduce is CGAL's guarantee that the output is within
 * a bounded distance of the input everywhere; the grid resolution sets the
 * accuracy here instead.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import { RichInt, RichPercentage } from "../../common/parameters/rich_parameter.ts";
import { RichParameterList } from "../../common/parameters/rich_parameter_list.ts";
import { FilterArity, type FilterArityValue } from "../../common/plugins/filter_arity.ts";
import { FilterClass, type FilterClassMask } from "../../common/plugins/filter_class.ts";
import {
	type ActionIDType,
	type FilterOutput,
	FilterPlugin,
	type PostConditionBox,
} from "../../common/plugins/interfaces/filter_plugin.ts";
import type { CallBackPos } from "../../common/utilities/callback.ts";
import { MLException } from "../../common/utilities/ml_exception.ts";
import { marchingTetrahedra } from "../../vcg/complex/create/marching.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { SurfaceLookup } from "../../vcg/space/index/surface_lookup.ts";

export const FP = { MESH_ALPHA_WRAP: 0 } as const;

export class FilterMeshAlphaWrap extends FilterPlugin {
	pluginName(): string {
		return "FilterMeshAlphaWrap";
	}
	actions(): readonly ActionIDType[] {
		return Object.values(FP);
	}
	filterName(id: ActionIDType): string {
		if (id !== FP.MESH_ALPHA_WRAP) this.wrongActionCalled(id);
		return "Alpha Wrap";
	}
	pythonFilterName(id: ActionIDType): string {
		if (id !== FP.MESH_ALPHA_WRAP) this.wrongActionCalled(id);
		return "generate_alpha_wrap";
	}
	filterInfo(id: ActionIDType): string {
		if (id !== FP.MESH_ALPHA_WRAP) this.wrongActionCalled(id);
		return (
			"Build a watertight shell around the current mesh by rolling a ball of the given radius " +
			"over it. The input may be non-manifold, self-intersecting or in several pieces."
		);
	}
	override getClass(_id: ActionIDType): FilterClassMask {
		return FilterClass.Remeshing;
	}
	filterArity(_id: ActionIDType): FilterArityValue {
		return FilterArity.FIXED;
	}

	override initParameterList(_id: ActionIDType, m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		let diagonal = 1;
		if (m !== undefined) {
			UpdateBounding.box(m.cm);
			diagonal = m.cm.bbox.diagonal || 1;
		}
		list.add(
			new RichPercentage("Alpha", diagonal * 0.02, 0, diagonal, {
				description: "Alpha",
				tooltip:
					"The radius of the rolling ball. Any gap narrower than this is bridged; anything " +
					"wider is followed. A smaller radius costs more.",
			}),
		);
		list.add(
			new RichPercentage("Offset", diagonal * 0.001, 0, diagonal, {
				description: "Offset",
				tooltip: "How far outside the input the shell sits. It should be greater than zero.",
			}),
		);
		list.add(
			new RichInt("Resolution", 96, {
				description: "Grid resolution",
				tooltip:
					"The side of the grid the shell is extracted from. It sets the accuracy: the ball " +
					"radius should be several cells across or the closing has nothing to work with.",
			}),
		);
		return list;
	}

	applyFilter(
		id: ActionIDType,
		params: RichParameterList,
		doc: MeshDocument,
		post: PostConditionBox,
		cb: CallBackPos,
	): FilterOutput {
		if (id !== FP.MESH_ALPHA_WRAP) return this.wrongActionCalled(id);
		const m = doc.mm();
		const cm = m.cm;
		if (cm.fn === 0) throw new MLException("the mesh has no faces to wrap");

		const alpha = params.getAbsPerc("Alpha");
		const offset = params.getAbsPerc("Offset");
		const resolution = params.getInt("Resolution");
		if (alpha <= 0) throw new MLException(`the alpha radius must be positive, got ${alpha}`);
		if (offset <= 0) throw new MLException(`the offset must be positive, got ${offset}`);
		if (resolution < 8)
			throw new MLException(`the grid resolution must be at least 8, got ${resolution}`);

		UpdateBounding.box(cm);
		const box = cm.bbox;
		const size = [box.max[0] - box.min[0], box.max[1] - box.min[1], box.max[2] - box.min[2]];
		const step = Math.max(...size) / resolution;
		if (alpha < 2 * step) {
			throw new MLException(
				`an alpha of ${alpha} is smaller than two grid cells (${(2 * step).toFixed(6)}); raise ` +
					"the resolution or the radius, or the closing has nothing to work with",
			);
		}
		// Room for the dilation plus a cell of slack, so the shell never
		// touches the wall and gets cut open by the extraction.
		const pad = alpha + offset + 2 * step;
		const counts = size.map((s) => Math.max(2, Math.ceil((s + 2 * pad) / step) + 1));
		const min = box.min.map((c) => c - pad);
		const coord = (axis: number, i: number) => min[axis] + i * step;
		const index = (i: number, j: number, k: number) => (k * counts[1] + j) * counts[0] + i;

		const lookup = new SurfaceLookup(cm, box.diagonal + 2 * pad);
		const distance = new Float64Array(counts[0] * counts[1] * counts[2]);
		for (let k = 0; k < counts[2]; k++) {
			cb((50 * k) / counts[2], "Measuring the distance field");
			for (let j = 0; j < counts[1]; j++) {
				for (let i = 0; i < counts[0]; i++) {
					const hit = lookup.closest(coord(0, i), coord(1, j), coord(2, k));
					// Unsigned: the wrap surrounds the input, so which side of
					// it a point is on does not matter and a soup has no sides.
					distance[index(i, j, k)] =
						hit === null
							? Number.POSITIVE_INFINITY
							: distanceTo(cm, hit, coord(0, i), coord(1, j), coord(2, k));
				}
			}
		}

		// Dilate by alpha + offset: everything within that of the input.
		const dilated = new Float64Array(distance.length);
		for (let i = 0; i < distance.length; i++) dilated[i] = distance[i] - (alpha + offset);
		// The distance is unsigned, so dilating a closed surface gives a
		// *hollow shell* around it rather than a filled solid — and the wrap
		// would come back with an inner surface as well as an outer one.
		// Flooding from the grid's edge marks what is genuinely outside;
		// everything else is interior and is filled in.
		fillInterior(dilated, counts, index);
		// Erode by alpha: the distance *from the dilated solid*, which is what
		// undoes the padding without reopening the bridged gaps.
		cb(60, "Closing the shell");
		const eroded = erode(dilated, counts, step, alpha, index, coord);

		const shell = marchingTetrahedra(eroded, counts, coord, index);
		if (shell.fn === 0) {
			throw new MLException(
				"the wrap came out empty; the alpha radius is probably larger than the mesh",
			);
		}
		const target = doc.addNewMesh("", `${m.label()} wrapped`, true, shell);
		target.updateBoxAndNormals();
		post.mask = MeshElement.MM_NONE;
		doc.Log.log(
			`Wrapped "${m.label()}" with alpha ${alpha} and offset ${offset}: ` +
				`${shell.vn} vertices, ${shell.fn} faces`,
		);
		return { new_mesh_id: target.id(), vertex_number: shell.vn, face_number: shell.fn };
	}
}

/**
 * Turns an unsigned dilation into a filled solid.
 *
 * A cell is outside only if it can be reached from the grid's boundary
 * without passing through the dilated band. Anything positive that cannot be
 * reached is an enclosed void, and a wrap has no business leaving one.
 */
function fillInterior(
	field: Float64Array,
	counts: readonly number[],
	index: (i: number, j: number, k: number) => number,
): void {
	const outside = new Uint8Array(field.length);
	const stack: number[] = [];
	const push = (i: number, j: number, k: number) => {
		if (i < 0 || j < 0 || k < 0) return;
		if (i >= counts[0] || j >= counts[1] || k >= counts[2]) return;
		const at = index(i, j, k);
		if (outside[at] === 1 || field[at] <= 0) return;
		outside[at] = 1;
		stack.push(i, j, k);
	};
	// The grid was padded so its whole boundary is empty space.
	for (let k = 0; k < counts[2]; k++) {
		for (let j = 0; j < counts[1]; j++) {
			push(0, j, k);
			push(counts[0] - 1, j, k);
		}
	}
	for (let k = 0; k < counts[2]; k++) {
		for (let i = 0; i < counts[0]; i++) {
			push(i, 0, k);
			push(i, counts[1] - 1, k);
		}
	}
	for (let j = 0; j < counts[1]; j++) {
		for (let i = 0; i < counts[0]; i++) {
			push(i, j, 0);
			push(i, j, counts[2] - 1);
		}
	}

	while (stack.length > 0) {
		const k = stack.pop() as number;
		const j = stack.pop() as number;
		const i = stack.pop() as number;
		push(i + 1, j, k);
		push(i - 1, j, k);
		push(i, j + 1, k);
		push(i, j - 1, k);
		push(i, j, k + 1);
		push(i, j, k - 1);
	}

	for (let at = 0; at < field.length; at++) {
		if (field[at] > 0 && outside[at] === 0) field[at] = -field[at];
	}
}

function distanceTo(
	cm: ReturnType<() => MeshModel["cm"]>,
	hit: { face: number; bary: readonly number[] },
	x: number,
	y: number,
	z: number,
): number {
	const p = [0, 0, 0];
	for (let k = 0; k < 3; k++) {
		const v = cm.fv(hit.face, k);
		p[0] += cm.vx(v) * hit.bary[k];
		p[1] += cm.vy(v) * hit.bary[k];
		p[2] += cm.vz(v) * hit.bary[k];
	}
	return Math.hypot(p[0] - x, p[1] - y, p[2] - z);
}

/**
 * Erodes a solid given as a signed field, by `radius`.
 *
 * The eroded field at a point is its distance to the *outside* of the solid,
 * negated — so a point is kept only if the whole ball of that radius around
 * it was already inside. A brute-force scan of every cell against every other
 * would be hopeless; this uses a chamfer sweep, two passes over the grid
 * propagating distances from the boundary, which is linear in the cell count.
 */
function erode(
	field: Float64Array,
	counts: readonly number[],
	step: number,
	radius: number,
	index: (i: number, j: number, k: number) => number,
	_coord: (axis: number, i: number) => number,
): Float64Array {
	// Distance from every outside cell, so an inside cell learns how far it is
	// from the nearest outside one.
	const distance = new Float64Array(field.length).fill(Number.POSITIVE_INFINITY);
	for (let i = 0; i < field.length; i++) if (field[i] > 0) distance[i] = 0;

	const neighbours: Array<[number, number, number, number]> = [];
	for (let dk = -1; dk <= 1; dk++) {
		for (let dj = -1; dj <= 1; dj++) {
			for (let di = -1; di <= 1; di++) {
				if (di === 0 && dj === 0 && dk === 0) continue;
				neighbours.push([di, dj, dk, Math.hypot(di, dj, dk) * step]);
			}
		}
	}
	const forward = neighbours.filter(
		([di, dj, dk]) => dk < 0 || (dk === 0 && (dj < 0 || (dj === 0 && di < 0))),
	);
	const backward = neighbours.filter((n) => !forward.includes(n));

	const sweep = (
		order: ReadonlyArray<[number, number, number, number]>,
		reverse: boolean,
	): void => {
		const range = (n: number) => (reverse ? [...Array(n).keys()].reverse() : [...Array(n).keys()]);
		for (const k of range(counts[2])) {
			for (const j of range(counts[1])) {
				for (const i of range(counts[0])) {
					const at = index(i, j, k);
					let best = distance[at];
					for (const [di, dj, dk, cost] of order) {
						const ni = i + di;
						const nj = j + dj;
						const nk = k + dk;
						if (ni < 0 || nj < 0 || nk < 0) continue;
						if (ni >= counts[0] || nj >= counts[1] || nk >= counts[2]) continue;
						best = Math.min(best, distance[index(ni, nj, nk)] + cost);
					}
					distance[at] = best;
				}
			}
		}
	};
	sweep(forward, false);
	sweep(backward, true);

	// Inside by more than the radius survives the erosion.
	const out = new Float64Array(field.length);
	for (let i = 0; i < field.length; i++) {
		out[i] = field[i] > 0 ? Math.min(field[i], radius) : radius - distance[i];
	}
	return out;
}
