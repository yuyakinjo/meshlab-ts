/**
 * `FilterScript` — a recorded sequence of filters, loadable and replayable.
 *
 * MeshLab's format is `.mlx`, an XML document of `<filter>` elements each
 * holding `<Param>` children. That is what a user exports from the GUI and
 * what a pipeline is usually shipped as, so it is read and written verbatim.
 * JSON is offered alongside because it is far easier to hand-write.
 */

import type { MeshLabKernel } from "./meshlab_kernel.ts";
import type { MeshDocument } from "./ml_document/mesh_document.ts";
import type { FilterOutput } from "./plugins/interfaces/filter_plugin.ts";
import type { CallBackPos } from "./utilities/callback.ts";
import { MLIOException } from "./utilities/ml_exception.ts";

export interface ScriptStep {
	readonly filterName: string;
	readonly params: Record<string, unknown>;
}

export interface RunScriptResult {
	/** One entry per step, in order. */
	readonly outputs: FilterOutput[];
}

export class FilterScript {
	readonly steps: ScriptStep[];

	constructor(steps: readonly ScriptStep[] = []) {
		this.steps = [...steps];
	}

	add(filterName: string, params: Record<string, unknown> = {}): this {
		this.steps.push({ filterName, params });
		return this;
	}

	get length(): number {
		return this.steps.length;
	}

	// ---- JSON -------------------------------------------------------------

	static fromJSON(text: string, fileName?: string): FilterScript {
		let parsed: unknown;
		try {
			parsed = JSON.parse(text);
		} catch (cause) {
			throw new MLIOException(`not valid JSON (${(cause as Error).message})`, fileName);
		}
		const raw = Array.isArray(parsed) ? parsed : (parsed as { filters?: unknown }).filters;
		if (!Array.isArray(raw)) {
			throw new MLIOException(
				"expected an array of steps, or an object with a `filters` array",
				fileName,
			);
		}
		return new FilterScript(
			raw.map((entry, i) => {
				const e = entry as { filterName?: unknown; name?: unknown; params?: unknown };
				const name = e.filterName ?? e.name;
				if (typeof name !== "string") {
					throw new MLIOException(`step ${i} has no filter name`, fileName);
				}
				return { filterName: name, params: (e.params ?? {}) as Record<string, unknown> };
			}),
		);
	}

	toJSON(): string {
		return `${JSON.stringify({ filters: this.steps }, null, 2)}\n`;
	}

	// ---- MLX ---------------------------------------------------------------

	/**
	 * Parses MeshLab's `.mlx`.
	 *
	 * Deliberately lenient about the schema and strict about the names: a
	 * script written by a different MeshLab version may carry attributes we do
	 * not model, and ignoring those is right, whereas guessing at a filter
	 * name we do not recognise is not.
	 */
	static fromMLX(text: string, fileName?: string): FilterScript {
		const steps: ScriptStep[] = [];
		// The self-closing form has to be tried first. Matching the paired form
		// first lets `[^>]*` swallow the trailing slash of `<filter name="X"/>`,
		// treating it as an opening tag and scanning on to the *next*
		// `</filter>` — quietly eating every filter in between.
		const filterRe = /<filter\b([^>]*?)\/>|<filter\b([^>]*?)>([\s\S]*?)<\/filter>/g;
		let hit = filterRe.exec(text);
		while (hit !== null) {
			const attrs = hit[1] ?? hit[2] ?? "";
			const body = hit[3] ?? "";
			const name = attributeOf(attrs, "name");
			if (name === undefined) {
				throw new MLIOException("a <filter> element has no name attribute", fileName);
			}
			steps.push({ filterName: decodeXml(name), params: parseParams(body) });
			hit = filterRe.exec(text);
		}
		if (steps.length === 0 && !text.includes("<FilterScript")) {
			throw new MLIOException("no <filter> elements found; is this an .mlx script?", fileName);
		}
		return new FilterScript(steps);
	}

	toMLX(): string {
		const lines = ["<!DOCTYPE FilterScript>", "<FilterScript>"];
		for (const step of this.steps) {
			const params = Object.entries(step.params);
			if (params.length === 0) {
				lines.push(` <filter name="${encodeXml(step.filterName)}"/>`);
				continue;
			}
			lines.push(` <filter name="${encodeXml(step.filterName)}">`);
			for (const [key, value] of params) {
				lines.push(
					`  <Param name="${encodeXml(key)}" type="${mlxTypeOf(value)}" ` +
						`value="${encodeXml(mlxValueOf(value))}"/>`,
				);
			}
			lines.push(" </filter>");
		}
		lines.push("</FilterScript>", "");
		return lines.join("\n");
	}

	/** Chooses the format from the file's extension, defaulting to `.mlx`. */
	static parse(text: string, fileName = ""): FilterScript {
		return fileName.toLowerCase().endsWith(".json")
			? FilterScript.fromJSON(text, fileName)
			: FilterScript.fromMLX(text, fileName);
	}

	// ---- running ------------------------------------------------------------

	/**
	 * Runs every step against `doc`.
	 *
	 * Stops at the first failure and lets the exception through, naming the
	 * step. A pipeline that half-ran and reported success would be worse than
	 * one that stopped.
	 */
	run(kernel: MeshLabKernel, doc: MeshDocument, cb?: CallBackPos): RunScriptResult {
		const outputs: FilterOutput[] = [];
		for (const [i, step] of this.steps.entries()) {
			try {
				outputs.push(
					kernel.applyFilter(doc, step.filterName, step.params, cb === undefined ? {} : { cb }),
				);
			} catch (cause) {
				const err = cause as Error;
				err.message = `step ${i + 1} of ${this.steps.length}, "${step.filterName}": ${err.message}`;
				throw err;
			}
		}
		return { outputs };
	}
}

function attributeOf(attrs: string, name: string): string | undefined {
	const re = new RegExp(`\\b${name}\\s*=\\s*"([^"]*)"`);
	const hit = re.exec(attrs);
	return hit === null ? undefined : hit[1];
}

/**
 * Reads the `<Param>` children of one filter.
 *
 * MeshLab writes the declared type as an attribute, and it is honoured rather
 * than inferred: a `RichString` whose value happens to look like a number must
 * stay a string, or a filename like "123" would arrive as an integer.
 */
function parseParams(body: string): Record<string, unknown> {
	const out: Record<string, unknown> = {};
	const paramRe = /<Param\b([^>]*?)\/?>/g;
	let hit = paramRe.exec(body);
	while (hit !== null) {
		const attrs = hit[1];
		const name = attributeOf(attrs, "name");
		const raw = attributeOf(attrs, "value");
		const type = attributeOf(attrs, "type") ?? "";
		if (name !== undefined && raw !== undefined) {
			out[decodeXml(name)] = decodeParamValue(decodeXml(raw), type, attrs);
		}
		hit = paramRe.exec(body);
	}
	return out;
}

function decodeParamValue(raw: string, type: string, attrs: string): unknown {
	switch (type) {
		case "RichBool":
			return raw === "true" || raw === "1";
		case "RichInt":
		case "RichEnum":
		case "RichMesh":
			return Number.parseInt(raw, 10);
		case "RichFloat":
		case "RichAbsPerc":
		case "RichDynamicFloat":
			return Number.parseFloat(raw);
		case "RichPosition":
		case "RichDirection": {
			// MeshLab writes these as x/y/z attributes rather than one value.
			const x = attributeOf(attrs, "x");
			const y = attributeOf(attrs, "y");
			const z = attributeOf(attrs, "z");
			if (x !== undefined && y !== undefined && z !== undefined) {
				return [Number.parseFloat(x), Number.parseFloat(y), Number.parseFloat(z)];
			}
			return raw
				.trim()
				.split(/[\s,]+/)
				.map(Number);
		}
		case "RichString":
		case "RichOpenFile":
		case "RichSaveFile":
			return raw;
		default:
			// An unknown type: leave the text as it is and let the parameter's
			// own coercion decide, which is the only honest thing to do.
			return raw;
	}
}

function mlxTypeOf(value: unknown): string {
	if (typeof value === "boolean") return "RichBool";
	if (typeof value === "number") return Number.isInteger(value) ? "RichInt" : "RichFloat";
	if (Array.isArray(value)) return "RichPosition";
	return "RichString";
}

function mlxValueOf(value: unknown): string {
	if (Array.isArray(value)) return value.join(" ");
	return String(value);
}

function encodeXml(s: string): string {
	return s
		.replaceAll("&", "&amp;")
		.replaceAll("<", "&lt;")
		.replaceAll(">", "&gt;")
		.replaceAll('"', "&quot;");
}

function decodeXml(s: string): string {
	return s
		.replaceAll("&lt;", "<")
		.replaceAll("&gt;", ">")
		.replaceAll("&quot;", '"')
		.replaceAll("&apos;", "'")
		.replaceAll("&amp;", "&");
}
