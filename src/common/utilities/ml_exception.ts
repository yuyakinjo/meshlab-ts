/**
 * MeshLab's exception hierarchy.
 *
 * Filters report failure by throwing, never by a return code — `applyFilter`
 * returns its output map or nothing at all. `MLException` is the base that
 * callers can catch to mean "the filter did not run".
 */

import { maskToString } from "../ml_document/mesh_element.ts";

export class MLException extends Error {
	override readonly name: string = "MLException";
}

/**
 * Thrown by a filter that exists in the registry but has no implementation yet.
 *
 * Every MeshLab filter is registered from day one so that `filterList()` is
 * complete and callers can discover names; the unimplemented ones fail loudly
 * here rather than quietly doing nothing.
 */
export class MLNotImplementedException extends MLException {
	override readonly name = "MLNotImplementedException";
	readonly filterName: string;
	readonly pluginName: string;

	constructor(filterName: string, pluginName: string) {
		super(`filter "${filterName}" is registered but not implemented yet (${pluginName})`);
		this.filterName = filterName;
		this.pluginName = pluginName;
	}
}

/** The current mesh lacks an attribute the filter declared it needs. */
export class MissingPreconditionException extends MLException {
	override readonly name = "MissingPreconditionException";
	readonly filterName: string;
	readonly missingMask: number;

	constructor(filterName: string, missingMask: number) {
		super(`filter "${filterName}" requires ${maskToString(missingMask)}, which the mesh lacks`);
		this.filterName = filterName;
		this.missingMask = missingMask;
	}
}

/** A parameter was unknown, of the wrong type, or out of range. */
export class InvalidParameterException extends MLException {
	override readonly name = "InvalidParameterException";
}

/** A file could not be read or written, or its contents were malformed. */
export class MLIOException extends MLException {
	override readonly name = "MLIOException";
	readonly fileName: string | undefined;

	constructor(message: string, fileName?: string) {
		super(fileName === undefined ? message : `${fileName}: ${message}`);
		this.fileName = fileName;
	}
}

/** A progress callback returned false. */
export class UserCanceledException extends MLException {
	override readonly name = "UserCanceledException";

	constructor(what = "operation") {
		super(`${what} was canceled`);
	}
}

/**
 * An internal invariant of the mesh kernel was violated — a bug in this
 * library, not in the caller's data. VCGLib's `assert`s map here.
 */
export class MLInternalException extends MLException {
	override readonly name = "MLInternalException";

	constructor(what: string) {
		super(`internal invariant violated: ${what}`);
	}
}
