/**
 * `RichParameterList` — an ordered, name-keyed bag of parameters with the
 * typed accessors filters use inside `applyFilter`.
 */
import { InvalidParameterException } from "../utilities/ml_exception.ts";
import type { RichParameter } from "./rich_parameter.ts";
import type { ShotValue, Value } from "./value.ts";
import { valueToPlain } from "./value.ts";

export class RichParameterList {
	private readonly params: RichParameter[] = [];
	private readonly byName = new Map<string, RichParameter>();

	constructor(params: readonly RichParameter[] = []) {
		for (const p of params) this.add(p);
	}

	add(p: RichParameter): this {
		if (this.byName.has(p.name)) {
			throw new InvalidParameterException(`duplicate parameter name "${p.name}"`);
		}
		this.params.push(p);
		this.byName.set(p.name, p);
		return this;
	}

	get size(): number {
		return this.params.length;
	}

	get isEmpty(): boolean {
		return this.params.length === 0;
	}

	hasParameter(name: string): boolean {
		return this.byName.has(name);
	}

	getParameterByName(name: string): RichParameter {
		const p = this.byName.get(name);
		if (p === undefined) {
			const known = this.params.map((q) => q.name).join(", ");
			throw new InvalidParameterException(
				`no parameter named "${name}"${known === "" ? "" : `; known parameters: ${known}`}`,
			);
		}
		return p;
	}

	[Symbol.iterator](): Iterator<RichParameter> {
		return this.params[Symbol.iterator]();
	}

	toArray(): readonly RichParameter[] {
		return [...this.params];
	}

	clone(): RichParameterList {
		return new RichParameterList(this.params.map((p) => p.clone()));
	}

	// ---- typed accessors ---------------------------------------------------

	private expect(name: string, kind: Value["kind"]): Value {
		const v = this.getParameterByName(name).value;
		if (v.kind !== kind) {
			throw new InvalidParameterException(`parameter "${name}" holds a ${v.kind}, not a ${kind}`);
		}
		return v;
	}

	getBool(name: string): boolean {
		return (this.expect(name, "bool") as { value: boolean }).value;
	}
	getInt(name: string): number {
		return (this.expect(name, "int") as { value: number }).value;
	}
	getFloat(name: string): number {
		return (this.expect(name, "float") as { value: number }).value;
	}
	getString(name: string): string {
		return (this.expect(name, "string") as { value: string }).value;
	}
	getPoint3m(name: string): readonly [number, number, number] {
		return (this.expect(name, "point3") as { value: readonly [number, number, number] }).value;
	}
	getMatrix44(name: string): readonly number[] {
		return (this.expect(name, "matrix44") as { value: readonly number[] }).value;
	}
	getShotf(name: string): ShotValue {
		return (this.expect(name, "shot") as { value: ShotValue }).value;
	}
	/** The packed 0xAABBGGRR colour. */
	getColor4b(name: string): number {
		return (this.expect(name, "color") as { value: number }).value;
	}
	/** Alias of {@link getColor4b}; MeshLab exposes both spellings. */
	getColor(name: string): number {
		return this.getColor4b(name);
	}
	/** A `RichPercentage`, which is stored as a float. */
	getAbsPerc(name: string): number {
		return this.getFloat(name);
	}
	/** A `RichDynamicFloat`, which is stored as a float. */
	getDynamicFloat(name: string): number {
		return this.getFloat(name);
	}
	/** A `RichEnum`'s selected index, which is stored as an int. */
	getEnum(name: string): number {
		return this.getInt(name);
	}
	/** A `RichMesh`'s target mesh id, which is stored as an int. */
	getMeshId(name: string): number {
		return this.getInt(name);
	}
	getOpenFileName(name: string): string {
		return this.getString(name);
	}
	getSaveFileName(name: string): string {
		return this.getString(name);
	}

	// ---- bulk assignment ---------------------------------------------------

	/**
	 * Applies a caller's plain object onto these parameters.
	 *
	 * An unknown key is an error rather than something to ignore. Silently
	 * dropping `{ maxholesizze: 30 }` would run the filter with its default and
	 * report success, which is the worst possible outcome for a typo.
	 */
	applyPlain(values: Readonly<Record<string, unknown>>): this {
		for (const [key, raw] of Object.entries(values)) {
			const p = this.byName.get(key);
			if (p === undefined) {
				const known = this.params
					.map((q) => q.name)
					.sort()
					.join(", ");
				throw new InvalidParameterException(
					`unknown parameter "${key}"${known === "" ? "" : `; expected one of: ${known}`}`,
				);
			}
			p.setValue(p.fromPlain(raw));
		}
		return this;
	}

	/** A JSON-safe snapshot, used by the filter history and `.mlx` writing. */
	toPlain(): Record<string, unknown> {
		const out: Record<string, unknown> = {};
		for (const p of this.params) out[p.name] = valueToPlain(p.value);
		return out;
	}
}
