/**
 * `RichParameter` and its fifteen concrete subclasses, mirroring
 * `src/common/parameters/rich_parameter/`.
 *
 * The class names follow MeshLab `main`, which renamed several of them, but
 * `stringType()` still returns the *old* string in four cases because that is
 * what `.mlx` files on disk contain. Both are reproduced: the class name is
 * what code reads, the string type is what files carry.
 */
import { InvalidParameterException } from "../utilities/ml_exception.ts";
import {
	boolValue,
	colorValue,
	floatValue,
	IDENTITY_MATRIX44,
	intValue,
	matrix44Value,
	point3Value,
	type ShotValue,
	shotValue,
	stringValue,
	type Value,
	valuesEqual,
} from "./value.ts";

export interface RichParameterOptions {
	/** Human-readable label. MeshLab calls this `fieldDesc`. */
	readonly description?: string;
	readonly tooltip?: string;
	/** Hidden behind "advanced" in the GUI; irrelevant headless, kept for fidelity. */
	readonly advanced?: boolean;
	readonly category?: string;
}

export abstract class RichParameter {
	readonly name: string;
	readonly fieldDescription: string;
	readonly toolTip: string;
	readonly isAdvanced: boolean;
	readonly category: string;
	readonly defaultValue: Value;
	private current: Value;

	protected constructor(name: string, defaultValue: Value, options: RichParameterOptions = {}) {
		this.name = name;
		this.defaultValue = defaultValue;
		this.current = defaultValue;
		this.fieldDescription = options.description ?? name;
		this.toolTip = options.tooltip ?? "";
		this.isAdvanced = options.advanced ?? false;
		this.category = options.category ?? "";
	}

	get value(): Value {
		return this.current;
	}

	get isValueDefault(): boolean {
		return valuesEqual(this.current, this.defaultValue);
	}

	/**
	 * Replaces the value, rejecting a wrong-typed one.
	 *
	 * Subclasses override to add range checks; a `RichPercentage` outside its
	 * bounds is as much a caller error as a string where a number belongs.
	 */
	setValue(v: Value): void {
		if (v.kind !== this.defaultValue.kind) {
			throw new InvalidParameterException(
				`parameter "${this.name}" expects a ${this.defaultValue.kind} value, got ${v.kind}`,
			);
		}
		this.current = v;
	}

	resetToDefault(): void {
		this.current = this.defaultValue;
	}

	/** The `stringType()` MeshLab writes into `.mlx`. */
	abstract stringType(): string;

	abstract clone(): RichParameter;

	/**
	 * Coerces a plain JS value from a caller's params object.
	 *
	 * This is the front door: `applyFilter(doc, name, { maxholesize: 30 })`
	 * arrives here, so the conversion has to be strict enough that a typo in
	 * the shape fails loudly rather than being silently reinterpreted.
	 */
	abstract fromPlain(raw: unknown): Value;

	protected typeError(raw: unknown, expected: string): never {
		throw typeErrorFor(this.name, raw, expected);
	}
}

function typeErrorFor(name: string, raw: unknown, expected: string): never {
	throw new InvalidParameterException(
		`parameter "${name}" expects ${expected}, got ${JSON.stringify(raw)}`,
	);
}

// ---------------------------------------------------------------------------

export class RichBool extends RichParameter {
	constructor(name: string, defval: boolean, options?: RichParameterOptions) {
		super(name, boolValue(defval), options);
	}
	stringType(): string {
		return "RichBool";
	}
	clone(): RichBool {
		return copyInto(this, new RichBool(this.name, (this.defaultValue as { value: boolean }).value));
	}
	fromPlain(raw: unknown): Value {
		if (typeof raw !== "boolean") this.typeError(raw, "a boolean");
		return boolValue(raw);
	}
}

export class RichInt extends RichParameter {
	constructor(name: string, defval: number, options?: RichParameterOptions) {
		super(name, intValue(defval), options);
	}
	stringType(): string {
		return "RichInt";
	}
	clone(): RichInt {
		return copyInto(this, new RichInt(this.name, (this.defaultValue as { value: number }).value));
	}
	fromPlain(raw: unknown): Value {
		if (typeof raw !== "number" || !Number.isFinite(raw)) this.typeError(raw, "a finite number");
		if (!Number.isInteger(raw)) this.typeError(raw, "an integer");
		return intValue(raw);
	}
}

export class RichFloat extends RichParameter {
	constructor(name: string, defval: number, options?: RichParameterOptions) {
		super(name, floatValue(defval), options);
	}
	stringType(): string {
		return "RichFloat";
	}
	clone(): RichFloat {
		return copyInto(this, new RichFloat(this.name, (this.defaultValue as { value: number }).value));
	}
	fromPlain(raw: unknown): Value {
		if (typeof raw !== "number" || !Number.isFinite(raw)) this.typeError(raw, "a finite number");
		return floatValue(raw);
	}
}

export class RichString extends RichParameter {
	constructor(name: string, defval: string, options?: RichParameterOptions) {
		super(name, stringValue(defval), options);
	}
	stringType(): string {
		return "RichString";
	}
	clone(): RichString {
		return copyInto(
			this,
			new RichString(this.name, (this.defaultValue as { value: string }).value),
		);
	}
	fromPlain(raw: unknown): Value {
		if (typeof raw !== "string") this.typeError(raw, "a string");
		return stringValue(raw);
	}
}

/** A choice among named alternatives, stored as the chosen index. */
export class RichEnum extends RichParameter {
	readonly enumValues: readonly string[];

	constructor(
		name: string,
		defval: number,
		enumValues: readonly string[],
		options?: RichParameterOptions,
	) {
		super(name, intValue(defval), options);
		this.enumValues = [...enumValues];
	}
	stringType(): string {
		return "RichEnum";
	}
	clone(): RichEnum {
		return copyInto(
			this,
			new RichEnum(this.name, (this.defaultValue as { value: number }).value, this.enumValues),
		);
	}
	override setValue(v: Value): void {
		super.setValue(v);
		const i = (v as { value: number }).value;
		if (i < 0 || i >= this.enumValues.length) {
			throw new InvalidParameterException(
				`parameter "${this.name}" index ${i} is outside 0..${this.enumValues.length - 1}`,
			);
		}
	}
	/** Accepts either the index or the option's name, as PyMeshLab does. */
	fromPlain(raw: unknown): Value {
		if (typeof raw === "number") return intValue(raw);
		if (typeof raw === "string") {
			const i = this.enumValues.indexOf(raw);
			if (i < 0) {
				throw new InvalidParameterException(
					`parameter "${this.name}" has no option "${raw}"; expected one of ${this.enumValues
						.map((s) => `"${s}"`)
						.join(", ")}`,
				);
			}
			return intValue(i);
		}
		return this.typeError(raw, "an enum index or option name");
	}
}

/** A bounded scalar; `stringType()` keeps the pre-rename `RichAbsPerc`. */
export class RichPercentage extends RichParameter {
	readonly min: number;
	readonly max: number;

	constructor(
		name: string,
		defval: number,
		min: number,
		max: number,
		options?: RichParameterOptions,
	) {
		super(name, floatValue(defval), options);
		this.min = min;
		this.max = max;
	}
	stringType(): string {
		return "RichAbsPerc";
	}
	clone(): RichPercentage {
		return copyInto(
			this,
			new RichPercentage(
				this.name,
				(this.defaultValue as { value: number }).value,
				this.min,
				this.max,
			),
		);
	}
	override setValue(v: Value): void {
		super.setValue(v);
		const x = (v as { value: number }).value;
		if (x < this.min || x > this.max) {
			throw new InvalidParameterException(
				`parameter "${this.name}" value ${x} is outside ${this.min}..${this.max}`,
			);
		}
	}
	fromPlain(raw: unknown): Value {
		if (typeof raw !== "number" || !Number.isFinite(raw)) this.typeError(raw, "a finite number");
		return floatValue(raw);
	}
}

/** A bounded scalar the GUI shows as a slider. */
export class RichDynamicFloat extends RichParameter {
	readonly min: number;
	readonly max: number;

	constructor(
		name: string,
		defval: number,
		min: number,
		max: number,
		options?: RichParameterOptions,
	) {
		super(name, floatValue(defval), options);
		this.min = min;
		this.max = max;
	}
	stringType(): string {
		return "RichDynamicFloat";
	}
	clone(): RichDynamicFloat {
		return copyInto(
			this,
			new RichDynamicFloat(
				this.name,
				(this.defaultValue as { value: number }).value,
				this.min,
				this.max,
			),
		);
	}
	override setValue(v: Value): void {
		super.setValue(v);
		const x = (v as { value: number }).value;
		if (x < this.min || x > this.max) {
			throw new InvalidParameterException(
				`parameter "${this.name}" value ${x} is outside ${this.min}..${this.max}`,
			);
		}
	}
	fromPlain(raw: unknown): Value {
		if (typeof raw !== "number" || !Number.isFinite(raw)) this.typeError(raw, "a finite number");
		return floatValue(raw);
	}
}

function readTriple(self: RichParameter, raw: unknown): [number, number, number] {
	if (!Array.isArray(raw) || raw.length !== 3 || raw.some((x) => typeof x !== "number")) {
		typeErrorFor(self.name, raw, "an array of three numbers");
	}
	return [raw[0] as number, raw[1] as number, raw[2] as number];
}

/** A point in space. Formerly `RichPoint3f`. */
export class RichPosition extends RichParameter {
	constructor(
		name: string,
		defval: readonly [number, number, number],
		options?: RichParameterOptions,
	) {
		super(name, point3Value(defval), options);
	}
	stringType(): string {
		return "RichPosition";
	}
	clone(): RichPosition {
		return copyInto(
			this,
			new RichPosition(
				this.name,
				(this.defaultValue as { value: readonly [number, number, number] }).value,
			),
		);
	}
	fromPlain(raw: unknown): Value {
		return point3Value(readTriple(this, raw));
	}
}

/** A direction. Distinct from {@link RichPosition} so the GUI can normalise it. */
export class RichDirection extends RichParameter {
	constructor(
		name: string,
		defval: readonly [number, number, number],
		options?: RichParameterOptions,
	) {
		super(name, point3Value(defval), options);
	}
	stringType(): string {
		return "RichDirection";
	}
	clone(): RichDirection {
		return copyInto(
			this,
			new RichDirection(
				this.name,
				(this.defaultValue as { value: readonly [number, number, number] }).value,
			),
		);
	}
	fromPlain(raw: unknown): Value {
		return point3Value(readTriple(this, raw));
	}
}

/** A 4×4 transform; `stringType()` keeps the pre-rename `RichMatrix44f`. */
export class RichMatrix44 extends RichParameter {
	constructor(
		name: string,
		defval: readonly number[] = IDENTITY_MATRIX44,
		options?: RichParameterOptions,
	) {
		super(name, matrix44Value(defval), options);
	}
	stringType(): string {
		return "RichMatrix44f";
	}
	clone(): RichMatrix44 {
		return copyInto(
			this,
			new RichMatrix44(this.name, (this.defaultValue as { value: readonly number[] }).value),
		);
	}
	fromPlain(raw: unknown): Value {
		if (!Array.isArray(raw) || raw.length !== 16 || raw.some((x) => typeof x !== "number")) {
			this.typeError(raw, "an array of sixteen numbers");
		}
		return matrix44Value(raw as number[]);
	}
}

export class RichColor extends RichParameter {
	constructor(name: string, defval: number, options?: RichParameterOptions) {
		super(name, colorValue(defval), options);
	}
	stringType(): string {
		return "RichColor";
	}
	clone(): RichColor {
		return copyInto(this, new RichColor(this.name, (this.defaultValue as { value: number }).value));
	}
	/** Accepts a packed number or `[r, g, b]` / `[r, g, b, a]` in 0..255. */
	fromPlain(raw: unknown): Value {
		if (typeof raw === "number") return colorValue(raw);
		if (Array.isArray(raw) && (raw.length === 3 || raw.length === 4)) {
			const [r, g, b, a = 255] = raw as number[];
			if ([r, g, b, a].some((c) => typeof c !== "number" || c < 0 || c > 255)) {
				this.typeError(raw, "colour components in 0..255");
			}
			return colorValue(((a << 24) | (b << 16) | (g << 8) | r) >>> 0);
		}
		return this.typeError(raw, "a packed colour or [r, g, b(, a)]");
	}
}

/** A reference to another mesh in the document, by id. */
export class RichMesh extends RichParameter {
	constructor(name: string, meshIndex: number, options?: RichParameterOptions) {
		super(name, intValue(meshIndex), options);
	}
	stringType(): string {
		return "RichMesh";
	}
	clone(): RichMesh {
		return copyInto(this, new RichMesh(this.name, (this.defaultValue as { value: number }).value));
	}
	fromPlain(raw: unknown): Value {
		if (typeof raw !== "number" || !Number.isInteger(raw)) this.typeError(raw, "a mesh id");
		return intValue(raw);
	}
}

/** A file to read; `stringType()` is `RichOpenFile`. */
export class RichFileOpen extends RichParameter {
	readonly extensions: readonly string[];

	constructor(
		name: string,
		defval: string,
		extensions: readonly string[] = [],
		options?: RichParameterOptions,
	) {
		super(name, stringValue(defval), options);
		this.extensions = [...extensions];
	}
	stringType(): string {
		return "RichOpenFile";
	}
	clone(): RichFileOpen {
		return copyInto(
			this,
			new RichFileOpen(this.name, (this.defaultValue as { value: string }).value, this.extensions),
		);
	}
	fromPlain(raw: unknown): Value {
		if (typeof raw !== "string") this.typeError(raw, "a file path");
		return stringValue(raw);
	}
}

/** A file to write; `stringType()` is `RichSaveFile`. */
export class RichFileSave extends RichParameter {
	readonly extension: string;

	constructor(name: string, defval: string, extension = "", options?: RichParameterOptions) {
		super(name, stringValue(defval), options);
		this.extension = extension;
	}
	stringType(): string {
		return "RichSaveFile";
	}
	clone(): RichFileSave {
		return copyInto(
			this,
			new RichFileSave(this.name, (this.defaultValue as { value: string }).value, this.extension),
		);
	}
	fromPlain(raw: unknown): Value {
		if (typeof raw !== "string") this.typeError(raw, "a file path");
		return stringValue(raw);
	}
}

/** A camera shot; `stringType()` keeps the pre-rename `RichShotf`. */
export class RichShot extends RichParameter {
	constructor(name: string, defval: ShotValue, options?: RichParameterOptions) {
		super(name, shotValue(defval), options);
	}
	stringType(): string {
		return "RichShotf";
	}
	clone(): RichShot {
		return copyInto(
			this,
			new RichShot(this.name, (this.defaultValue as { value: ShotValue }).value),
		);
	}
	fromPlain(raw: unknown): Value {
		if (typeof raw !== "object" || raw === null) this.typeError(raw, "a shot object");
		return shotValue(raw as ShotValue);
	}
}

/**
 * Carries the description metadata and the mutable current value across a
 * clone.
 *
 * Each subclass's `clone()` reconstructs only the type-specific parts (bounds,
 * enum options, extensions); the shared metadata is copied here so that
 * fifteen constructors do not each have to thread four optional fields.
 */
function copyInto<T extends RichParameter>(from: RichParameter, to: T): T {
	const meta = to as unknown as {
		fieldDescription: string;
		toolTip: string;
		isAdvanced: boolean;
		category: string;
	};
	meta.fieldDescription = from.fieldDescription;
	meta.toolTip = from.toolTip;
	meta.isAdvanced = from.isAdvanced;
	meta.category = from.category;
	if (!from.isValueDefault) to.setValue(from.value);
	return to;
}
