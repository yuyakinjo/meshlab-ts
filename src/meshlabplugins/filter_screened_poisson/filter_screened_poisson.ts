/**
 * `filter_screened_poisson` — a watertight surface from an oriented point set.
 *
 * MeshLab exposes one filter here, and it does not edit the mesh it reads: the
 * reconstruction arrives as a new hidden layer named "Poisson mesh", with the
 * sample density in its per-vertex quality.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import { RichBool, RichFloat, RichInt } from "../../common/parameters/rich_parameter.ts";
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
import { reconstructScreenedPoisson } from "./poisson_recon.ts";

export const FP = {
	FP_SCREENED_POISSON: 0,
} as const;

const NAME = "Surface Reconstruction: Screened Poisson";
const PYTHON_NAME = "generate_surface_reconstruction_screened_poisson";
const INFO =
	"This surface reconstruction algorithm creates watertight surfaces from oriented point sets." +
	"<br>The filter uses the original code of Michael Kazhdan and Matthew Bolitho implementing the " +
	"algorithm described in the following paper:<br><i>Michael Kazhdan, Hugues Hoppe</i>,<br><b>" +
	'"Screened Poisson surface reconstruction"</b><br>Error!';

export class FilterScreenedPoisson extends FilterPlugin {
	pluginName(): string {
		return "FilterScreenedPoisson";
	}

	actions(): readonly ActionIDType[] {
		return Object.values(FP);
	}

	private check(id: ActionIDType): void {
		if (id !== FP.FP_SCREENED_POISSON) this.wrongActionCalled(id);
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
		return FilterArity.VARIABLE;
	}

	/** A new layer appears; the mesh that was read is untouched. */
	override postCondition(_id: ActionIDType): number {
		return MeshElement.MM_VERTNUMBER | MeshElement.MM_FACENUMBER;
	}

	/** Reads per-vertex normals, and per-vertex quality when confidence is on. */
	override getRequirements(_id: ActionIDType): number {
		return MeshElement.MM_VERTQUALITY;
	}

	override initParameterList(id: ActionIDType, _m: MeshModel | undefined): RichParameterList {
		this.check(id);
		const list = new RichParameterList();
		list.add(
			new RichBool("visibleLayer", false, {
				description: "Merge all visible layers",
				tooltip:
					"Enabling this flag means that all the visible layers will be used for providing the points.",
			}),
		);
		list.add(
			new RichInt("depth", 8, {
				description: "Reconstruction Depth",
				tooltip:
					"This integer is the maximum depth of the tree that will be used for surface " +
					"reconstruction. Running at depth d corresponds to solving on a voxel grid whose " +
					"resolution is no larger than 2^d x 2^d x 2^d. Note that since the reconstructor " +
					"adapts the octree to the sampling density, the specified reconstruction depth is " +
					"only an upper bound. The default value for this parameter is 8.",
			}),
		);
		list.add(
			new RichInt("fullDepth", 5, {
				description: "Adaptive Octree Depth",
				tooltip:
					"This integer specifies the depth beyond depth the octree will be adapted. At coarser " +
					"depths, the octree will be complete, containing all 2^d x 2^d x 2^d nodes. The " +
					"default value for this parameter is 5.",
				advanced: true,
			}),
		);
		list.add(
			new RichInt("cgDepth", 0, {
				description: "Conjugate Gradients Depth",
				tooltip:
					"This integer is the depth up to which a conjugate-gradients solver will be used to " +
					"solve the linear system. Beyond this depth Gauss-Seidel relaxation will be used. The " +
					"default value for this parameter is 0.",
				advanced: true,
			}),
		);
		list.add(
			new RichFloat("scale", 1.1, {
				description: "Scale Factor",
				tooltip:
					"This floating point value specifies the ratio between the diameter of the cube used " +
					"for reconstruction and the diameter of the samples' bounding cube. The default value " +
					"is 1.1.",
				advanced: true,
			}),
		);
		list.add(
			new RichFloat("samplesPerNode", 1.5, {
				description: "Minimum Number of Samples",
				tooltip:
					"This floating point value specifies the minimum number of sample points that should " +
					"fall within an octree node as the octree construction is adapted to sampling density. " +
					"For noise-free samples, small values in the range [1.0 - 5.0] can be used. For more " +
					"noisy samples, larger values in the range [15.0 - 20.0] may be needed to provide a " +
					"smoother, noise-reduced, reconstruction. The default value is 1.5.",
			}),
		);
		list.add(
			new RichFloat("pointWeight", 4, {
				description: "Interpolation Weight",
				tooltip:
					"This floating point value specifies the importants that interpolation of the point " +
					"samples is given in the formulation of the screened Poisson equation. The results of " +
					"the original (unscreened) Poisson Reconstruction can be obtained by setting this " +
					"value to 0. The default value for this parameter is 4.",
			}),
		);
		list.add(
			new RichInt("iters", 8, {
				description: "Gauss-Seidel Relaxations",
				tooltip:
					"This integer value specifies the number of Gauss-Seidel relaxations to be performed " +
					"at each level of the hierarchy. The default value for this parameter is 8.",
				advanced: true,
			}),
		);
		list.add(
			new RichBool("confidence", false, {
				description: "Confidence Flag",
				tooltip:
					"Enabling this flag tells the reconstructor to use the quality as confidence " +
					"information; this is done by scaling the unit normals with the quality values. When " +
					"the flag is not enabled, all normals are normalized to have unit-length prior to " +
					"reconstruction.",
			}),
		);
		list.add(
			new RichBool("preClean", false, {
				description: "Pre-Clean",
				tooltip:
					"Enabling this flag force a cleaning pre-pass on the data removing all unreferenced " +
					"vertices or vertices with null normals.",
			}),
		);
		list.add(
			new RichInt("threads", 1, {
				description: "Number Threads",
				tooltip:
					"Maximum number of threads that the reconstruction algorithm can use. Accepted for " +
					"compatibility only: the solve here is deterministic and single threaded.",
			}),
		);
		return list;
	}

	applyFilter(
		id: ActionIDType,
		params: RichParameterList,
		doc: MeshDocument,
		post: PostConditionBox,
		_cb: CallBackPos,
	): FilterOutput {
		this.check(id);
		post.mask = MeshElement.MM_NONE;

		const useVisible = params.getBool("visibleLayer");
		const sources: CMeshO[] = useVisible ? doc.visibleMeshes().map((m) => m.cm) : [doc.mm().cm];
		if (sources.length === 0) {
			throw new MLException("Screened Poisson reconstruction found no visible layer to read.");
		}

		const mesh = reconstructScreenedPoisson(sources, {
			depth: params.getInt("depth"),
			fullDepth: params.getInt("fullDepth"),
			cgDepth: params.getInt("cgDepth"),
			scale: params.getFloat("scale"),
			samplesPerNode: params.getFloat("samplesPerNode"),
			pointWeight: params.getFloat("pointWeight"),
			iters: params.getInt("iters"),
			confidence: params.getBool("confidence"),
			preClean: params.getBool("preClean"),
		});

		const pm = doc.addNewMesh("", "Poisson mesh", false, mesh);
		// MeshLab hides the new layer so the point cloud stays on screen; the
		// density it just computed lives in the quality channel.
		pm.setVisible(false);
		pm.updateDataMask(MeshElement.MM_VERTQUALITY);
		pm.updateBoxAndNormals();
		doc.Log.log(`Poisson reconstruction produced ${mesh.vn} vertices and ${mesh.fn} faces`);
		return { new_mesh_id: pm.id(), vertex_number: mesh.vn, face_number: mesh.fn };
	}
}
