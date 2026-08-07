/** meshlab-ts — a TypeScript port of MeshLab. */

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
export { Allocator } from "./vcg/complex/allocator.ts";
export { Clean } from "./vcg/complex/clean.ts";
// Mesh kernel
export { CMeshO, type Color4b } from "./vcg/complex/cmesho.ts";
export { borderBit, FaceFlag, fauxBit, VertexFlag } from "./vcg/complex/flags.ts";
export { forEachBorderStep, Pos } from "./vcg/complex/pos.ts";
export { UpdateBounding } from "./vcg/complex/update/bounding.ts";
export { UpdateFlags } from "./vcg/complex/update/flag.ts";
export { UpdateNormal } from "./vcg/complex/update/normal.ts";
export { UpdateTopology } from "./vcg/complex/update/topology.ts";
export type { Scalarm } from "./vcg/math/base.ts";
export { Point3 } from "./vcg/math/vec3.ts";
export { Box3 } from "./vcg/space/box3.ts";
