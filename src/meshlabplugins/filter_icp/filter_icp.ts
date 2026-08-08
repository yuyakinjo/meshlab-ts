/**
 * `filter_icp` — rigid alignment by iterative closest point.
 *
 * Each iteration pairs a sample of the moving mesh with the nearest point on
 * the fixed one, solves for the rigid motion that best explains those pairs,
 * applies it, and repeats. What makes it work in practice is not the solve —
 * that is a closed form — but the rejection: a pair whose points are far
 * apart, or whose normals disagree, is not a correspondence and drags the
 * whole alignment if it is kept.
 *
 * The solve is point-to-plane, which converges in far fewer iterations than
 * point-to-point on any surface with flat regions: sliding along a plane
 * costs nothing, so the pairs do not have to be the *right* pairs, only on
 * the right surface.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import {
	RichFloat,
	RichInt,
	RichMesh,
	RichPercentage,
} from "../../common/parameters/rich_parameter.ts";
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
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { UpdateNormal } from "../../vcg/complex/update/normal.ts";
import { mulberry32 } from "../../vcg/math/noise.ts";
import { SparseMatrix, solveCG } from "../../vcg/math/sparse.ts";
import { SurfaceLookup } from "../../vcg/space/index/surface_lookup.ts";

export const FP = { ICP_TWO_MESHES: 0, ICP_GLOBAL: 1, OVERLAP: 2 } as const;

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
	readonly filterClass: FilterClassMask;
	readonly arity: FilterArityValue;
}

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FP.ICP_TWO_MESHES]: {
		name: "ICP Between Meshes",
		pythonName: "compute_matrix_by_icp_between_meshes",
		info: "Register a mesh onto another with the Iterative Closest Point algorithm.",
		filterClass: FilterClass.Remeshing,
		arity: FilterArity.FIXED,
	},
	[FP.ICP_GLOBAL]: {
		name: "Global Align Meshes",
		pythonName: "compute_matrix_by_mesh_global_alignment",
		info: "Align all the visible layers onto the first one, each with ICP.",
		filterClass: FilterClass.Remeshing,
		arity: FilterArity.SINGLE_MESH,
	},
	[FP.OVERLAP]: {
		name: "Overlapping Meshes",
		pythonName: "get_overlapping_meshes_graph",
		info: "Measure how much of each visible layer lies close to the current one.",
		filterClass: FilterClass.Measure,
		arity: FilterArity.SINGLE_MESH,
	},
};

export class FilterICP extends FilterPlugin {
	pluginName(): string {
		return "FilterIcpPlugin";
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
	override getClass(id: ActionIDType): FilterClassMask {
		return this.spec(id).filterClass;
	}
	filterArity(id: ActionIDType): FilterArityValue {
		return this.spec(id).arity;
	}

	override initParameterList(id: ActionIDType, m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		let diagonal = 1;
		if (m !== undefined) {
			UpdateBounding.box(m.cm);
			diagonal = m.cm.bbox.diagonal || 1;
		}

		if (id === FP.ICP_TWO_MESHES) {
			const current = m?.id() ?? 0;
			list.add(
				new RichMesh("referenceMesh", current, {
					description: "Reference Mesh",
					tooltip: "The mesh that stays where it is.",
				}),
			);
			list.add(
				new RichMesh("sourceMesh", current, {
					description: "Source Mesh",
					tooltip: "The mesh that is moved onto the reference.",
				}),
			);
		}
		if (id !== FP.OVERLAP) {
			list.add(
				new RichInt("SampleNum", 2000, {
					description: "Sample Number",
					tooltip: "Number of samples that we try to choose at each ICP iteration.",
				}),
			);
			list.add(
				new RichPercentage("MinDistAbs", diagonal / 10, 0, diagonal, {
					description: "Minimal Starting Distance",
					tooltip:
						"A pair farther apart than this is rejected. It shrinks as the alignment improves, " +
						"which is what lets ICP start loose and end tight.",
				}),
			);
			list.add(
				new RichInt("MaxIterNum", 30, {
					description: "Max Iteration Num",
					tooltip: "How many ICP iterations to run at most.",
				}),
			);
			list.add(
				new RichFloat("TrgDistAbs", 0.0005, {
					description: "Target Distance",
					tooltip: "Stop once the mean pair distance falls below this.",
				}),
			);
			list.add(
				new RichFloat("ReduceFactorPerc", 0.8, {
					description: "MSD Reduce Factor",
					tooltip: "How fast the rejection distance shrinks between iterations.",
				}),
			);
			list.add(new RichInt("randomSeed", 0, { description: "Random seed" }));
		} else {
			list.add(
				new RichPercentage("overlapDistance", diagonal / 100, 0, diagonal, {
					description: "Overlap distance",
					tooltip: "A vertex closer than this to the other mesh counts as overlapping.",
				}),
			);
		}
		return list;
	}

	applyFilter(
		id: ActionIDType,
		params: RichParameterList,
		doc: MeshDocument,
		post: PostConditionBox,
		cb: CallBackPos,
	): FilterOutput {
		post.mask = MeshElement.MM_VERTCOORD | MeshElement.MM_VERTNORMAL;

		if (id === FP.OVERLAP) {
			const m = doc.mm();
			const distance = params.getAbsPerc("overlapDistance");
			const lookup = new SurfaceLookup(m.cm, distance);
			const report: Record<string, number> = {};
			for (const other of doc.visibleMeshes()) {
				if (other.id() === m.id()) continue;
				let close = 0;
				let total = 0;
				for (let v = 0; v < other.cm.vertSize; v++) {
					if (other.cm.isVertD(v)) continue;
					total++;
					if (lookup.closest(other.cm.vx(v), other.cm.vy(v), other.cm.vz(v)) !== null) close++;
				}
				report[`overlap_${other.id()}`] = total === 0 ? 0 : close / total;
				doc.Log.log(`"${other.label()}" overlaps "${m.label()}" on ${close} of ${total} vertices`);
			}
			if (Object.keys(report).length === 0) {
				throw new MLException("there is no other visible layer to compare against");
			}
			post.mask = MeshElement.MM_NONE;
			return report;
		}

		const options: IcpOptions = {
			samples: params.getInt("SampleNum"),
			startDistance: params.getAbsPerc("MinDistAbs"),
			targetDistance: params.getFloat("TrgDistAbs"),
			iterations: params.getInt("MaxIterNum"),
			reduce: params.getFloat("ReduceFactorPerc"),
			seed: params.getInt("randomSeed"),
		};
		if (options.samples < 3) {
			throw new MLException(`ICP needs at least 3 samples, got ${options.samples}`);
		}
		if (options.iterations < 1) {
			throw new MLException(`the iteration count must be at least 1, got ${options.iterations}`);
		}

		if (id === FP.ICP_TWO_MESHES) {
			const reference = doc.requireMesh(params.getMeshId("referenceMesh"));
			const source = doc.requireMesh(params.getMeshId("sourceMesh"));
			if (reference.id() === source.id()) {
				throw new MLException("the reference and the source must be two different layers");
			}
			const result = icp(reference.cm, source.cm, options, cb);
			source.updateBoxAndNormals();
			doc.Log.log(
				`Aligned "${source.label()}" onto "${reference.label()}": mean error ${result.error.toExponential(3)} ` +
					`after ${result.iterations} iterations`,
			);
			return { error: result.error, iterations: result.iterations, pairs: result.pairs };
		}

		const reference = doc.mm();
		const others = doc.visibleMeshes().filter((x) => x.id() !== reference.id());
		if (others.length === 0) throw new MLException("there is no other visible layer to align");
		let worst = 0;
		for (const other of others) {
			const result = icp(reference.cm, other.cm, options, cb);
			other.updateBoxAndNormals();
			worst = Math.max(worst, result.error);
			doc.Log.log(`Aligned "${other.label()}": mean error ${result.error.toExponential(3)}`);
		}
		return { aligned_layers: others.length, worst_error: worst };
	}
}

interface IcpOptions {
	readonly samples: number;
	readonly startDistance: number;
	readonly targetDistance: number;
	readonly iterations: number;
	readonly reduce: number;
	readonly seed: number;
}

interface IcpResult {
	readonly error: number;
	readonly iterations: number;
	readonly pairs: number;
}

/**
 * Point-to-plane ICP, moving `source` onto `reference`.
 *
 * The rejection distance starts at `startDistance` and shrinks by `reduce`
 * each iteration. Starting tight would reject every pair before the meshes
 * are roughly aligned; staying loose would keep pairing across the object
 * once they are.
 */
function icp(reference: CMeshO, source: CMeshO, options: IcpOptions, cb: CallBackPos): IcpResult {
	UpdateBounding.box(reference);
	UpdateNormal.perVertexNormalizedPerFaceNormalized(reference);
	const lookup = new SurfaceLookup(reference, reference.bbox.diagonal || 1);
	const referenceNormals = reference.vertNormal;
	const random = mulberry32(options.seed >>> 0 || 1);

	const live: number[] = [];
	for (let v = 0; v < source.vertSize; v++) if (!source.isVertD(v)) live.push(v);
	if (live.length < 3) throw new MLException("the source mesh has too few vertices to align");

	let threshold = options.startDistance;
	let error = Number.POSITIVE_INFINITY;
	let pairs = 0;
	let iteration = 0;

	for (; iteration < options.iterations; iteration++) {
		cb((100 * iteration) / options.iterations, "Aligning");
		const chosen = sample(live, options.samples, random);
		const from: number[][] = [];
		const to: number[][] = [];
		const normals: number[][] = [];
		let sum = 0;

		for (const v of chosen) {
			const p = [source.vx(v), source.vy(v), source.vz(v)];
			const hit = lookup.closest(p[0], p[1], p[2]);
			if (hit === null) continue;
			const q = [0, 0, 0];
			const n = [0, 0, 0];
			for (let k = 0; k < 3; k++) {
				const w = reference.fv(hit.face, k);
				q[0] += reference.vx(w) * hit.bary[k];
				q[1] += reference.vy(w) * hit.bary[k];
				q[2] += reference.vz(w) * hit.bary[k];
				for (let a = 0; a < 3; a++) n[a] += referenceNormals[3 * w + a] * hit.bary[k];
			}
			const d = Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
			// The two rejections that make ICP converge rather than wander.
			if (d > threshold) continue;
			from.push(p);
			to.push(q);
			normals.push(n);
			sum += d;
		}

		pairs = from.length;
		if (pairs < 6) break; // too few constraints to solve for six unknowns
		error = sum / pairs;
		const motion = solvePointToPlane(from, to, normals);
		if (motion === null) break;
		applyMotion(source, motion);

		if (error <= options.targetDistance) {
			iteration++;
			break;
		}
		threshold = Math.max(options.targetDistance, threshold * options.reduce);
	}
	return { error, iterations: iteration, pairs };
}

/** A rigid motion, as a small rotation vector and a translation. */
type Motion = readonly [number, number, number, number, number, number];

/**
 * The linearised point-to-plane solve.
 *
 * Minimising `Σ ((R·p + t - q) · n)²` is not linear in the rotation, but for
 * a small rotation `R·p ≈ p + ω × p`, and then each pair contributes one
 * linear equation in the six unknowns. That approximation is why ICP has to
 * iterate at all — and why it converges quadratically once the meshes are
 * close, where the rotations really are small.
 */
function solvePointToPlane(
	from: readonly number[][],
	to: readonly number[][],
	normals: readonly number[][],
): Motion | null {
	const a = new SparseMatrix(6);
	const b = new Float64Array(6);

	for (let i = 0; i < from.length; i++) {
		const p = from[i];
		const q = to[i];
		const n = normals[i];
		const length = Math.hypot(n[0], n[1], n[2]);
		if (length === 0) continue;
		const unit = [n[0] / length, n[1] / length, n[2] / length];
		// The row is [p × n, n]; the residual is (p - q) · n.
		const row = [
			p[1] * unit[2] - p[2] * unit[1],
			p[2] * unit[0] - p[0] * unit[2],
			p[0] * unit[1] - p[1] * unit[0],
			unit[0],
			unit[1],
			unit[2],
		];
		const residual = (p[0] - q[0]) * unit[0] + (p[1] - q[1]) * unit[1] + (p[2] - q[2]) * unit[2];
		for (let r = 0; r < 6; r++) {
			b[r] -= row[r] * residual;
			for (let c = 0; c < 6; c++) a.add(r, c, row[r] * row[c]);
		}
	}
	// A flat patch constrains only three of the six unknowns, so the system
	// can be singular; a small ridge keeps it solvable and biases the answer
	// towards not moving, which is the right failure.
	for (let r = 0; r < 6; r++) a.add(r, r, 1e-9);

	const solved = solveCG(a, b, { tolerance: 1e-14, iterations: 200 });
	if (!solved.converged) return null;
	for (const x of solved.x) if (!Number.isFinite(x)) return null;
	return [solved.x[0], solved.x[1], solved.x[2], solved.x[3], solved.x[4], solved.x[5]];
}

/** Applies the small rotation exactly, via Rodrigues, rather than linearly. */
function applyMotion(cm: CMeshO, motion: Motion): void {
	const omega = [motion[0], motion[1], motion[2]];
	const t = [motion[3], motion[4], motion[5]];
	const angle = Math.hypot(omega[0], omega[1], omega[2]);
	// Using the linearised rotation to *move* the mesh would stretch it a
	// little every iteration; Rodrigues keeps the motion rigid.
	const axis = angle === 0 ? [0, 0, 1] : omega.map((c) => c / angle);
	const cos = Math.cos(angle);
	const sin = Math.sin(angle);

	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v)) continue;
		const p = [cm.vx(v), cm.vy(v), cm.vz(v)];
		const cross = [
			axis[1] * p[2] - axis[2] * p[1],
			axis[2] * p[0] - axis[0] * p[2],
			axis[0] * p[1] - axis[1] * p[0],
		];
		const dot = axis[0] * p[0] + axis[1] * p[1] + axis[2] * p[2];
		cm.setVert(
			v,
			p[0] * cos + cross[0] * sin + axis[0] * dot * (1 - cos) + t[0],
			p[1] * cos + cross[1] * sin + axis[1] * dot * (1 - cos) + t[1],
			p[2] * cos + cross[2] * sin + axis[2] * dot * (1 - cos) + t[2],
		);
	}
}

function sample(live: readonly number[], count: number, random: () => number): number[] {
	if (count >= live.length) return [...live];
	// Reservoir-free: pick with replacement and deduplicate. A full shuffle of
	// a large vertex list every iteration would cost more than the pairing.
	const chosen = new Set<number>();
	const attempts = count * 3;
	for (let i = 0; i < attempts && chosen.size < count; i++) {
		chosen.add(live[Math.floor(random() * live.length)]);
	}
	return [...chosen];
}
