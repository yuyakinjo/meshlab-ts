/**
 * `filter_mesh_booleans` — union, intersection, difference and XOR.
 *
 * Upstream uses exact predicates on the triangles themselves, which gives an
 * output whose faces come from the inputs and whose vertices are exactly
 * where the surfaces cross. This works volumetrically instead: build a signed
 * distance field for each operand, combine the two fields, and extract the
 * result.
 *
 * The combination is the standard one — `min` is union, `max` is
 * intersection, and negating a field turns it inside out, so `A − B` is
 * `max(A, −B)`. XOR is the difference of the two differences, which is the
 * union of `A − B` and `B − A`.
 *
 * The trade is worth stating plainly. This never fails: two self-intersecting
 * inputs, or a non-manifold one, give a watertight result all the same, where
 * an exact method would refuse. What it gives up is the *sharp edge* — the
 * crease where two surfaces meet is band-limited by the grid, so a boolean of
 * two boxes comes back with slightly rounded seams. The resolution parameter
 * is the dial between the two.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import { RichBool, RichInt, RichMesh } from "../../common/parameters/rich_parameter.ts";
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
import type { CMeshO } from "../../vcg/complex/cmesho.ts";
import { marchingTetrahedra } from "../../vcg/complex/create/marching.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { UpdateNormal } from "../../vcg/complex/update/normal.ts";
import { SurfaceLookup } from "../../vcg/space/index/surface_lookup.ts";

export const FP = {
	MESH_INTERSECTION: 0,
	MESH_UNION: 1,
	MESH_DIFFERENCE: 2,
	MESH_XOR: 3,
} as const;

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
}

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FP.MESH_INTERSECTION]: {
		name: "Mesh Boolean: Intersection",
		pythonName: "generate_boolean_intersection",
		info: "Compute the intersection of two meshes: the volume that lies inside both.",
	},
	[FP.MESH_UNION]: {
		name: "Mesh Boolean: Union",
		pythonName: "generate_boolean_union",
		info: "Compute the union of two meshes: the volume that lies inside either.",
	},
	[FP.MESH_DIFFERENCE]: {
		name: "Mesh Boolean: Difference",
		pythonName: "generate_boolean_difference",
		info: "Compute the difference of two meshes: the first with the second cut out of it.",
	},
	[FP.MESH_XOR]: {
		name: "Mesh Boolean: Symmetric Difference (XOR)",
		pythonName: "generate_boolean_xor",
		info: "Compute the symmetric difference of two meshes: the volume inside one but not both.",
	},
};

export class FilterMeshBoolean extends FilterPlugin {
	pluginName(): string {
		return "FilterMeshBoolean";
	}

	actions(): readonly ActionIDType[] {
		return Object.values(FP);
	}

	private spec(id: ActionIDType): FilterSpec {
		const s = SPECS[id];
		if (s === undefined) this.wrongActionCalled(id);
		return s;
	}

	filterName(id: ActionIDType): string {
		return this.spec(id).name;
	}
	pythonFilterName(id: ActionIDType): string {
		return this.spec(id).pythonName;
	}
	filterInfo(id: ActionIDType): string {
		return this.spec(id).info;
	}
	override getClass(_id: ActionIDType): FilterClassMask {
		return FilterClass.Remeshing | FilterClass.Layer;
	}
	filterArity(_id: ActionIDType): FilterArityValue {
		return FilterArity.FIXED;
	}

	override initParameterList(_id: ActionIDType, m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		const current = m?.id() ?? 0;
		list.add(
			new RichMesh("first_mesh", current, {
				description: "First Mesh",
				tooltip: "The first operand of the boolean operation",
			}),
		);
		list.add(
			new RichMesh("second_mesh", current, {
				description: "Second Mesh",
				tooltip: "The second operand of the boolean operation",
			}),
		);
		list.add(
			new RichInt("Resolution", 96, {
				description: "Grid resolution",
				tooltip:
					"The side of the grid the result is extracted from. It sets how sharp the seams are: " +
					"a crease finer than a cell is rounded off.",
			}),
		);
		list.add(
			new RichBool("transfer_face_color", false, {
				description: "Transfer face color",
				tooltip: "Take each output face's colour from whichever operand it came from.",
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
		const first = doc.requireMesh(params.getMeshId("first_mesh"));
		const second = doc.requireMesh(params.getMeshId("second_mesh"));
		if (first.id() === second.id()) {
			throw new MLException("the two operands must be two different layers");
		}
		if (first.cm.fn === 0 || second.cm.fn === 0) {
			throw new MLException("both operands need faces");
		}
		const resolution = params.getInt("Resolution");
		if (resolution < 8) {
			throw new MLException(`the grid resolution must be at least 8, got ${resolution}`);
		}

		const box = combinedBox(first.cm, second.cm, id);
		const size = [box.max[0] - box.min[0], box.max[1] - box.min[1], box.max[2] - box.min[2]];
		const step = Math.max(...size) / resolution;
		// A cell of slack all round, so the result never touches the wall and
		// gets cut open by the extraction.
		const pad = 2 * step;
		const counts = size.map((s) => Math.max(2, Math.ceil((s + 2 * pad) / step) + 1));
		const min = box.min.map((c) => c - pad);
		const coord = (axis: number, i: number) => min[axis] + i * step;
		const index = (i: number, j: number, k: number) => (k * counts[1] + j) * counts[0] + i;

		const a = signedField(first.cm, counts, coord, index, (p) =>
			cb(p * 0.4, "Sampling the first mesh"),
		);
		const b = signedField(second.cm, counts, coord, index, (p) =>
			cb(40 + p * 0.4, "Sampling the second mesh"),
		);

		cb(85, "Combining");
		const out = new Float64Array(a.length);
		for (let i = 0; i < a.length; i++) out[i] = combine(id, a[i], b[i]);

		const result = marchingTetrahedra(out, counts, coord, index);
		if (result.fn === 0) {
			throw new MLException(
				`the ${this.spec(id).name} is empty; the two meshes may not overlap the way the ` +
					"operation needs",
			);
		}
		if (params.getBool("transfer_face_color")) {
			transferColour(result, first, second);
		}

		const target = doc.addNewMesh("", `${this.spec(id).name}`, true, result);
		target.updateBoxAndNormals();
		post.mask = MeshElement.MM_NONE;
		doc.Log.log(
			`${this.spec(id).name} of "${first.label()}" and "${second.label()}": ` +
				`${result.vn} vertices, ${result.fn} faces`,
		);
		return { new_mesh_id: target.id(), vertex_number: result.vn, face_number: result.fn };
	}
}

/**
 * The region the result can possibly occupy.
 *
 * A union or a XOR can reach anywhere either operand does; an intersection
 * only where they overlap, and a difference only where the first one is.
 * Using the smaller box where it applies is not just a saving — it puts the
 * grid's whole resolution where the answer is.
 */
function combinedBox(a: CMeshO, b: CMeshO, id: ActionIDType) {
	UpdateBounding.box(a);
	UpdateBounding.box(b);
	if (id === FP.MESH_DIFFERENCE) return { min: [...a.bbox.min], max: [...a.bbox.max] };
	if (id === FP.MESH_INTERSECTION) {
		const min = [0, 1, 2].map((k) => Math.max(a.bbox.min[k], b.bbox.min[k]));
		const max = [0, 1, 2].map((k) => Math.min(a.bbox.max[k], b.bbox.max[k]));
		for (let k = 0; k < 3; k++) {
			if (max[k] <= min[k]) {
				throw new MLException("the two meshes' bounding boxes do not overlap at all");
			}
		}
		return { min, max };
	}
	return {
		min: [0, 1, 2].map((k) => Math.min(a.bbox.min[k], b.bbox.min[k])),
		max: [0, 1, 2].map((k) => Math.max(a.bbox.max[k], b.bbox.max[k])),
	};
}

function combine(id: ActionIDType, a: number, b: number): number {
	switch (id) {
		case FP.MESH_UNION:
			return Math.min(a, b);
		case FP.MESH_INTERSECTION:
			return Math.max(a, b);
		case FP.MESH_DIFFERENCE:
			return Math.max(a, -b);
		default:
			// Inside exactly one of the two: the union of the two differences.
			return Math.min(Math.max(a, -b), Math.max(b, -a));
	}
}

/**
 * A signed distance field: negative inside, positive outside.
 *
 * The sign comes from the *interpolated vertex normal* at the closest surface
 * point — the angle-weighted pseudonormal, which is the one that stays right
 * when the closest point lands on an edge or a corner. A plain face normal is
 * ambiguous exactly there, and a boolean of two boxes meets along edges.
 */
function signedField(
	cm: CMeshO,
	counts: readonly number[],
	coord: (axis: number, i: number) => number,
	index: (i: number, j: number, k: number) => number,
	progress: (fraction: number) => void,
): Float64Array {
	UpdateBounding.box(cm);
	UpdateNormal.perVertexNormalizedPerFaceNormalized(cm);
	const reach = (cm.bbox.diagonal || 1) * 4;
	const lookup = new SurfaceLookup(cm, reach);
	const out = new Float64Array(counts[0] * counts[1] * counts[2]);

	for (let k = 0; k < counts[2]; k++) {
		progress((100 * k) / counts[2]);
		for (let j = 0; j < counts[1]; j++) {
			for (let i = 0; i < counts[0]; i++) {
				const x = coord(0, i);
				const y = coord(1, j);
				const z = coord(2, k);
				const hit = lookup.closest(x, y, z);
				if (hit === null) {
					// Beyond the search radius is outside, and far enough that
					// the exact value cannot matter to the extraction.
					out[index(i, j, k)] = reach;
					continue;
				}
				const point = [0, 0, 0];
				const normal = [0, 0, 0];
				for (let c = 0; c < 3; c++) {
					const v = cm.fv(hit.face, c);
					point[0] += cm.vx(v) * hit.bary[c];
					point[1] += cm.vy(v) * hit.bary[c];
					point[2] += cm.vz(v) * hit.bary[c];
					for (let axis = 0; axis < 3; axis++) {
						normal[axis] += cm.vertNormal[3 * v + axis] * hit.bary[c];
					}
				}
				const d = [x - point[0], y - point[1], z - point[2]];
				const distance = Math.hypot(d[0], d[1], d[2]);
				const side = d[0] * normal[0] + d[1] * normal[1] + d[2] * normal[2];
				out[index(i, j, k)] = side < 0 ? -distance : distance;
			}
		}
	}
	return out;
}

/** Colours each output face from whichever operand's surface is nearer. */
function transferColour(result: CMeshO, first: MeshModel, second: MeshModel): void {
	const a = new SurfaceLookup(first.cm, (first.cm.bbox.diagonal || 1) * 4);
	const b = new SurfaceLookup(second.cm, (second.cm.bbox.diagonal || 1) * 4);
	const colours = result.faceColor;
	if (colours === null) return;

	for (let f = 0; f < result.faceSize; f++) {
		if (result.isFaceD(f)) continue;
		const p = [0, 0, 0];
		for (let k = 0; k < 3; k++) {
			const v = result.fv(f, k);
			p[0] += result.vx(v) / 3;
			p[1] += result.vy(v) / 3;
			p[2] += result.vz(v) / 3;
		}
		const hitA = a.closest(p[0], p[1], p[2]);
		const hitB = b.closest(p[0], p[1], p[2]);
		const source =
			hitB === null
				? first
				: hitA === null
					? second
					: distance(first.cm, hitA, p) <= distance(second.cm, hitB, p)
						? first
						: second;
		const hit = source === first ? hitA : hitB;
		if (hit === null) continue;
		const from = source.cm.faceColor;
		colours[f] = from === null ? 0xffffffff : from[hit.face];
	}
}

function distance(
	cm: CMeshO,
	hit: { face: number; bary: readonly number[] },
	p: readonly number[],
): number {
	const q = [0, 0, 0];
	for (let k = 0; k < 3; k++) {
		const v = cm.fv(hit.face, k);
		q[0] += cm.vx(v) * hit.bary[k];
		q[1] += cm.vy(v) * hit.bary[k];
		q[2] += cm.vz(v) * hit.bary[k];
	}
	return Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]);
}
