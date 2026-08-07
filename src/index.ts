/** meshlab-ts — a TypeScript port of MeshLab. */

// Filter scripts (.mlx and JSON)
export { FilterScript, type ScriptStep } from "./common/filterscript.ts";
// Kernel and registry
export { MeshLabKernel } from "./common/meshlab_kernel.ts";
// Document model
export { type FilterScriptStep, MeshDocument } from "./common/ml_document/mesh_document.ts";
export {
	MeshElement,
	maskAnd,
	maskHas,
	maskIntersects,
	maskOf,
	maskOr,
	maskToString,
	maskWithout,
} from "./common/ml_document/mesh_element.ts";
export { MeshModel } from "./common/ml_document/mesh_model.ts";
// Parameters
export * from "./common/parameters/rich_parameter.ts";
export { RichParameterList } from "./common/parameters/rich_parameter_list.ts";
export type { ShotValue, Value } from "./common/parameters/value.ts";
export {
	FilterArity,
	type FilterArityValue,
	filterArityFromString,
	filterArityToString,
} from "./common/plugins/filter_arity.ts";
export {
	FilterClass,
	type FilterClassMask,
	filterClassFromString,
	filterClassToString,
} from "./common/plugins/filter_class.ts";
export {
	applyPostCondition,
	checkArity,
	type ExecuteOptions,
	executeFilter,
} from "./common/plugins/filter_executor.ts";
export {
	type FilterOutput,
	FilterPlugin,
	type OutputValue,
	type PostConditionBox,
} from "./common/plugins/interfaces/filter_plugin.ts";
export {
	type ExportCapability,
	IOPlugin,
	type OpenMaskBox,
} from "./common/plugins/interfaces/io_plugin.ts";
// Plugin interfaces
export { type ActionIDType, MeshLabPlugin } from "./common/plugins/meshlab_plugin.ts";
export { type FilterAction, PluginManager } from "./common/plugins/plugin_manager.ts";
export { type CallBackPos, noCallback } from "./common/utilities/callback.ts";
export { extensionOf, FileFormat } from "./common/utilities/file_format.ts";
export { Log, type LogEntry } from "./common/utilities/log.ts";
// Errors and utilities
export {
	InvalidParameterException,
	MissingPreconditionException,
	MLException,
	MLInternalException,
	MLIOException,
	MLNotImplementedException,
	UserCanceledException,
} from "./common/utilities/ml_exception.ts";
export { computePythonName } from "./common/utilities/python_name.ts";
// The upstream filter catalogue
export { FILTER_TABLE, type FilterTableRow } from "./meshlabplugins/_stub/filter_table.ts";
export {
	POISSON_DEFAULTS,
	type PoissonOptions,
	quantile,
	reconstructScreenedPoisson,
	trimByDensity,
} from "./meshlabplugins/filter_screened_poisson/poisson_recon.ts";
export { BaseMeshIOPlugin } from "./meshlabplugins/io_base/io_base.ts";
export {
	type ObjReadResult,
	type ObjSaveOptions,
	readObj,
	writeObj,
} from "./meshlabplugins/io_base/obj.ts";
export {
	type OffReadResult,
	type OffSaveOptions,
	readOff,
	writeOff,
} from "./meshlabplugins/io_base/off.ts";
export {
	type PlyHeader,
	type PlySaveOptions,
	parsePlyHeader,
	readPly,
	writePly,
} from "./meshlabplugins/io_base/ply.ts";
// Mesh I/O, for callers that hold bytes rather than a path
export {
	isBinaryStl,
	readStl,
	type StlSaveOptions,
	writeStl,
} from "./meshlabplugins/io_base/stl.ts";
export { Allocator } from "./vcg/complex/allocator.ts";
export { Clean } from "./vcg/complex/clean.ts";
// Mesh kernel
export { CMeshO, type Color4b } from "./vcg/complex/cmesho.ts";
export { Platonic } from "./vcg/complex/create/platonic.ts";
export {
	buildVertexFaces,
	collapseEdge,
	EdgeOps,
	type EdgePair,
	edgePairOf,
	flipEdge,
	linkCondition,
	sharedFaces,
	triQuality,
} from "./vcg/complex/edge_ops.ts";
export { borderBit, FaceFlag, fauxBit, VertexFlag } from "./vcg/complex/flags.ts";
// Algorithms, for callers that want them directly rather than through a filter
export { type EarStrategy, type FillHoleOptions, Hole } from "./vcg/complex/hole.ts";
export { Inertia, type MassProperties } from "./vcg/complex/inertia.ts";
export {
	clusteringDecimation,
	IsotropicRemeshing,
	isotropicRemeshing,
	REMESH_DEFAULTS,
	type RemeshOptions,
	type RemeshResult,
} from "./vcg/complex/isotropic_remeshing.ts";
export {
	type DecimateResult,
	defaultQuadricParameters,
	type QuadricParameters,
	quadricSimplification,
} from "./vcg/complex/local_optimization/tri_edge_collapse_quadric.ts";
export { Rng, SurfaceSampling } from "./vcg/complex/point_sampling.ts";
export { estimateNormals, type NormalOptions } from "./vcg/complex/pointcloud_normal.ts";
export { forEachBorderStep, Pos } from "./vcg/complex/pos.ts";
export {
	type EdgePredicate,
	type Interpolator,
	Refine,
	type RefineOptions,
} from "./vcg/complex/refine.ts";
export { Smooth, type SmoothOptions } from "./vcg/complex/smooth.ts";
export { UpdateBounding } from "./vcg/complex/update/bounding.ts";
export { UpdateFlags } from "./vcg/complex/update/flag.ts";
export { UpdateNormal } from "./vcg/complex/update/normal.ts";
export { UpdatePosition } from "./vcg/complex/update/position.ts";
export { UpdateTopology } from "./vcg/complex/update/topology.ts";
export type { Scalarm } from "./vcg/math/base.ts";
export { GenNormal, type SpherePoint } from "./vcg/math/gen_normal.ts";
export { type Matrix44, Matrix44Ops } from "./vcg/math/matrix44.ts";
export { Point3 } from "./vcg/math/vec3.ts";
export { Box3 } from "./vcg/space/box3.ts";
export { Color4 } from "./vcg/space/color4.ts";
export { KdTree, pointBounds } from "./vcg/space/index/kdtree.ts";
