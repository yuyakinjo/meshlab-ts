/**
 * `filter_developability` — one filter, and it edits the mesh in place.
 *
 * The algorithm lives in `vcg/complex/developability.ts`; this file is the
 * parameter list, the two-manifold precondition, and the log line.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import { RichBool, RichEnum, RichFloat, RichInt } from "../../common/parameters/rich_parameter.ts";
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
import { countNonManifoldEdgeFF, countNonManifoldVertexFF } from "../../vcg/complex/clean.ts";
import { makeDevelopable } from "../../vcg/complex/developability.ts";

export const FP = {
	FP_MAKE_DEVELOPABLE: 0,
} as const;

const NAME = "Make mesh developable";
const PYTHON_NAME = "apply_coord_developability_of_mesh";
const INFO =
	"The filter improves the developability of the current two-manifold triangular mesh by " +
	"applying an optimization process that encourages each vertex star to form an hinge or a " +
	"flat piece. The resulting mesh is similar to the initial, but it is comprised of one or " +
	"more developable pieces held toghether by highly regular seam curves, i.e. path of edges " +
	"which vertex stars did not form an hinge or a flat spot.<br>Since small interior angles " +
	"can have a negative impact on the outcome, an automatic remeshing that runs along the " +
	"optimization can be enabled.<br>When the obtained design is satisfactory, one may want to " +
	"refine the quality of the seams and the developability of the surfaces by alternating " +
	"between regular midpoint subdivisions and further optimization rounds.<br>For more details " +
	"see:<br><b>Oded Stein, Eitan Grinspun and Keenan Crane</b><br>" +
	"<a href=\"https://doi.org/10.1145/3197517.3201303\">'Developability of triangle meshes'</a>" +
	"<br>ACM Transactions on Graphics, Volume 37, Issue 4";

export class FilterDevelopability extends FilterPlugin {
	pluginName(): string {
		return "FilterDevelopability";
	}

	actions(): readonly ActionIDType[] {
		return Object.values(FP);
	}

	private check(id: ActionIDType): void {
		if (id !== FP.FP_MAKE_DEVELOPABLE) this.wrongActionCalled(id);
	}

	filterName(id: ActionIDType): string {
		this.check(id);
		return NAME;
	}
	pythonFilterName(id: ActionIDType): string {
		this.check(id);
		return PYTHON_NAME;
	}
	filterInfo(id: ActionIDType): string {
		this.check(id);
		return INFO;
	}
	override getClass(id: ActionIDType): FilterClassMask {
		this.check(id);
		return FilterClass.Remeshing;
	}
	filterArity(_id: ActionIDType): FilterArityValue {
		return FilterArity.SINGLE_MESH;
	}

	/** Rotation around a vertex needs FF adjacency, on every reset. */
	override getRequirements(_id: ActionIDType): number {
		return MeshElement.MM_FACEFACETOPO;
	}

	override postCondition(_id: ActionIDType): number {
		return MeshElement.MM_ALL;
	}

	override initParameterList(id: ActionIDType, _m: MeshModel | undefined): RichParameterList {
		this.check(id);
		const list = new RichParameterList();
		list.add(
			new RichEnum("OptMethod", 1, ["[F] Fixed stepsize", "[B] Backtracking line search"], {
				description: "Gradient method",
				tooltip: "The gradient method optimization algorithm to use",
			}),
		);
		list.add(
			new RichInt("MaxFunEvals", 400, {
				description: "Max function evaluations",
				tooltip: "The maximum number of function evaluation. Once reached, the optimization stops",
			}),
		);
		list.add(
			new RichFloat("Eps", 1e-5, {
				description: "Stop threshold",
				tooltip:
					"Optimization stops when the squared norm of the gradient is less than or equal to " +
					"the accuracy",
			}),
		);
		list.add(
			new RichFloat("StepSize", 0.01, {
				description: "Initial step size",
				tooltip: "The initial step size of the opt method, fixed when using [F] optimizer",
			}),
		);
		list.add(
			new RichFloat("MinStepSize", 1e-10, {
				description: "Min step size (B only)",
				tooltip: "The minimum step size for the backtracking line search opt method",
			}),
		);
		list.add(
			new RichFloat("Tau", 0.8, {
				description: "Tau (B only)",
				tooltip: "Scaling factor of the step size for the backtracking line search opt method",
			}),
		);
		list.add(
			new RichFloat("M1", 1e-4, {
				description: "Armijo constant (B only)",
				tooltip: "The constant of the Armijo condition of the backtracking line search opt method",
			}),
		);
		list.add(
			new RichBool("EdgeFlips", true, {
				description: "Apply edge flips",
				tooltip: "Whether or not to apply edge flips when necessary during optimization",
			}),
		);
		list.add(
			new RichBool("EdgeCollapses", true, {
				description: "Apply edge collapses",
				tooltip: "Whether or not to apply edge collapses when necessary during optimization",
			}),
		);
		list.add(
			new RichFloat("AngleThreshold", 18, {
				description: "Post-processing angle threshold (deg)",
				tooltip:
					"The maximum angle under which an edge flip or an edge collapse must be performed " +
					"during optimization",
			}),
		);
		return list;
	}

	applyFilter(
		id: ActionIDType,
		params: RichParameterList,
		doc: MeshDocument,
		_post: PostConditionBox,
		cb: CallBackPos,
	): FilterOutput {
		this.check(id);
		const model = doc.mm();
		if (model === undefined) throw new MLException("no current mesh");
		const m = model.cm;

		// The star rotation is only defined on a two-manifold. A fan of three
		// faces around an edge has no ring to walk, and a bowtie vertex has two.
		if (countNonManifoldEdgeFF(m) > 0) {
			throw new MLException(
				"non possible developability optimization because of non manifold edges",
			);
		}
		if (countNonManifoldVertexFF(m) > 0) {
			throw new MLException(
				"non possible developability optimization because of non manifold verties",
			);
		}

		const before = { vn: m.vn, fn: m.fn };
		const result = makeDevelopable(m, {
			method: params.getEnum("OptMethod") === 0 ? 0 : 1,
			maxFunEvals: params.getInt("MaxFunEvals"),
			eps: params.getFloat("Eps"),
			stepSize: params.getFloat("StepSize"),
			minStepSize: params.getFloat("MinStepSize"),
			tau: params.getFloat("Tau"),
			m1: params.getFloat("M1"),
			edgeFlips: params.getBool("EdgeFlips"),
			edgeCollapses: params.getBool("EdgeCollapses"),
			angleThreshold: (params.getFloat("AngleThreshold") * Math.PI) / 180,
			onProgress: (fraction) => {
				cb(Math.round(100 * fraction), "Optimizing developability energy...");
			},
		});

		doc.Log.log(
			`developability: ${result.functionEvaluations} function evaluations, ` +
				`energy ${result.energy.toFixed(6)}, |grad|^2 ${result.gradientSqNorm.toExponential(3)}` +
				(result.converged ? " (converged)" : " (budget exhausted)"),
		);
		if (result.remeshingRounds > 0) {
			doc.Log.log(
				`remeshing applied ${result.remeshingRounds} time(s): ` +
					`${before.vn} -> ${m.vn} vertices, ${before.fn} -> ${m.fn} faces`,
			);
		}
		model.updateBoxAndNormals();

		return {
			function_evaluations: result.functionEvaluations,
			energy: result.energy,
			gradient_sq_norm: result.gradientSqNorm,
			remeshing_rounds: result.remeshingRounds,
			converged: result.converged,
		};
	}
}
