/**
 * A small expression evaluator, compatible with the muParser dialect MeshLab's
 * `filter_func` exposes to users.
 *
 * MeshLab links muParser and hands it a table of per-element variables — `x`,
 * `nz`, `q`, `vi` and so on — then evaluates one expression once per vertex or
 * face. Every published MeshLab recipe that uses those filters is written in
 * that dialect, so matching it is the whole point: the operator set, the
 * precedence table, the function names and the two constants are muParser's,
 * not JavaScript's.
 *
 * Three places where the dialect differs from what a JavaScript reader would
 * assume, all of them load-bearing:
 *
 *  - `^` is exponentiation, not xor, and it is right-associative.
 *  - A unary sign binds *tighter* than `+`/`-` but *looser* than `^`, so
 *    `-2^2` is -4 while `-2*3` is -6.
 *  - There are no booleans. A comparison yields 1 or 0, and `&&`/`||` treat
 *    any non-zero as true and return 1 or 0 rather than an operand.
 *
 * Compilation walks the source once into a closure tree; evaluation then costs
 * no parsing and no property lookup, because variables resolve to a slot in a
 * `Float64Array` the caller refills per element.
 */

import { MLException } from "../../common/utilities/ml_exception.ts";

/** A compiled expression, ready to run against a variable buffer. */
export interface CompiledExpression {
	/** The variable names, in the slot order {@link evaluate} expects. */
	readonly names: readonly string[];
	/** Which of those names the expression actually reads. */
	readonly used: ReadonlySet<string>;
	evaluate(values: Float64Array): number;
}

type Node = (v: Float64Array) => number;

const CONSTANTS: Readonly<Record<string, number>> = {
	_pi: Math.PI,
	_e: Math.E,
};

/** muParser's built-in functions, by arity. Variadic ones take at least one argument. */
const FUNCTIONS_1: Readonly<Record<string, (a: number) => number>> = {
	sin: Math.sin,
	cos: Math.cos,
	tan: Math.tan,
	cot: (a) => 1 / Math.tan(a),
	asin: Math.asin,
	acos: Math.acos,
	atan: Math.atan,
	sinh: Math.sinh,
	cosh: Math.cosh,
	tanh: Math.tanh,
	coth: (a) => 1 / Math.tanh(a),
	asinh: Math.asinh,
	acosh: Math.acosh,
	atanh: Math.atanh,
	log2: Math.log2,
	log10: Math.log10,
	// muParser's `log` is the natural logarithm, not the base-10 one. Reading it
	// as log10 would quietly rescale every expression that uses it.
	log: Math.log,
	ln: Math.log,
	exp: Math.exp,
	sqrt: Math.sqrt,
	sign: (a) => (a < 0 ? -1 : a > 0 ? 1 : 0),
	rint: (a) => {
		// Round half to even, as C's rint does under the default rounding mode.
		const r = Math.round(a);
		return Math.abs(a % 1) === 0.5 && r % 2 !== 0 ? r - Math.sign(a) : r;
	},
	abs: Math.abs,
};

const FUNCTIONS_2: Readonly<Record<string, (a: number, b: number) => number>> = {
	atan2: Math.atan2,
};

const FUNCTIONS_VARIADIC: Readonly<Record<string, (args: number[]) => number>> = {
	min: (args) => Math.min(...args),
	max: (args) => Math.max(...args),
	sum: (args) => args.reduce((a, b) => a + b, 0),
	avg: (args) => args.reduce((a, b) => a + b, 0) / args.length,
};

/** Precedence, straight from muParser's `EOprtPrecedence`. */
const BINARY_PRECEDENCE: Readonly<Record<string, number>> = {
	"||": 1,
	"&&": 2,
	"|": 3,
	"&": 4,
	"<": 5,
	">": 5,
	"<=": 5,
	">=": 5,
	"==": 5,
	"!=": 5,
	"+": 6,
	"-": 6,
	"*": 7,
	"/": 7,
	"^": 8,
};

/**
 * A unary sign sits at 7 — above `+`/`-`, below `^`.
 *
 * That is muParser's `prINFIX`, and it is why `-2^2` is -4 rather than 4.
 */
const UNARY_PRECEDENCE = 7;

const bool = (b: boolean): number => (b ? 1 : 0);

const BINARY_OPS: Readonly<Record<string, (a: number, b: number) => number>> = {
	"||": (a, b) => bool(a !== 0 || b !== 0),
	"&&": (a, b) => bool(a !== 0 && b !== 0),
	"|": (a, b) => (a | b) >>> 0,
	"&": (a, b) => (a & b) >>> 0,
	"<": (a, b) => bool(a < b),
	">": (a, b) => bool(a > b),
	"<=": (a, b) => bool(a <= b),
	">=": (a, b) => bool(a >= b),
	"==": (a, b) => bool(a === b),
	"!=": (a, b) => bool(a !== b),
	"+": (a, b) => a + b,
	"-": (a, b) => a - b,
	"*": (a, b) => a * b,
	"/": (a, b) => a / b,
	"^": (a, b) => a ** b,
};

interface Token {
	readonly kind: "number" | "name" | "op" | "(" | ")" | "," | "?" | ":";
	readonly text: string;
	readonly at: number;
}

/** Longest first, so `<=` is never read as `<` followed by `=`. */
const OPERATORS = ["||", "&&", "<=", ">=", "==", "!=", "<", ">", "+", "-", "*", "/", "^", "|", "&"];

function tokenize(source: string): Token[] {
	const out: Token[] = [];
	let i = 0;
	while (i < source.length) {
		const c = source[i];
		if (/\s/.test(c)) {
			i++;
			continue;
		}
		if (/[0-9.]/.test(c)) {
			// A number, possibly in exponent form. `1e-3` has to survive the
			// minus that would otherwise be read as an operator.
			const m = /^[0-9]*\.?[0-9]*(?:[eE][+-]?[0-9]+)?/.exec(source.slice(i));
			const text = m === null ? "" : m[0];
			if (text === "" || text === ".") {
				throw new MLException(`expression: stray "${c}" at position ${i}`);
			}
			out.push({ kind: "number", text, at: i });
			i += text.length;
			continue;
		}
		if (/[A-Za-z_]/.test(c)) {
			const m = /^[A-Za-z_][A-Za-z0-9_]*/.exec(source.slice(i));
			const text = m === null ? c : m[0];
			out.push({ kind: "name", text, at: i });
			i += text.length;
			continue;
		}
		if (c === "(" || c === ")" || c === "," || c === "?" || c === ":") {
			out.push({ kind: c, text: c, at: i });
			i++;
			continue;
		}
		const op = OPERATORS.find((o) => source.startsWith(o, i));
		if (op === undefined) {
			throw new MLException(`expression: unexpected "${c}" at position ${i}`);
		}
		out.push({ kind: "op", text: op, at: i });
		i += op.length;
	}
	return out;
}

/**
 * Compiles `source` against a fixed list of variable names.
 *
 * An identifier that is neither a variable, a constant nor a function is an
 * error rather than a silent zero — a typo in a per-vertex expression would
 * otherwise run over the whole mesh and produce a plausible-looking result.
 */
export function compileExpression(source: string, names: readonly string[]): CompiledExpression {
	const slots = new Map<string, number>();
	for (const [index, name] of names.entries()) slots.set(name, index);
	const used = new Set<string>();
	const tokens = tokenize(source);
	let at = 0;

	const peek = (): Token | undefined => tokens[at];
	const done = () => at >= tokens.length;
	const expect = (kind: Token["kind"], what: string): Token => {
		const token = tokens[at];
		if (token === undefined || token.kind !== kind) {
			throw new MLException(
				`expression: expected ${what}${token === undefined ? " but the expression ended" : ` but found "${token.text}"`}`,
			);
		}
		at++;
		return token;
	};

	function parsePrimary(): Node {
		const token = peek();
		if (token === undefined)
			throw new MLException("expression: it ends where a value was expected");

		if (token.kind === "op" && (token.text === "-" || token.text === "+")) {
			at++;
			const operand = parseBinary(UNARY_PRECEDENCE);
			return token.text === "-" ? (v) => -operand(v) : operand;
		}
		if (token.kind === "(") {
			at++;
			const inner = parseTernary();
			expect(")", 'a closing ")"');
			return inner;
		}
		if (token.kind === "number") {
			at++;
			const value = Number(token.text);
			return () => value;
		}
		if (token.kind === "name") {
			at++;
			if (peek()?.kind === "(") return parseCall(token.text);
			const constant = CONSTANTS[token.text];
			if (constant !== undefined) return () => constant;
			const slot = slots.get(token.text);
			if (slot === undefined) {
				throw new MLException(
					`expression: unknown name "${token.text}"; available here: ${[...names, ...Object.keys(CONSTANTS)].join(", ")}`,
				);
			}
			used.add(token.text);
			return (v) => v[slot];
		}
		throw new MLException(`expression: unexpected "${token.text}" where a value was expected`);
	}

	function parseCall(name: string): Node {
		expect("(", 'an opening "(" after a function name');
		const args: Node[] = [];
		if (peek()?.kind !== ")") {
			args.push(parseTernary());
			while (peek()?.kind === ",") {
				at++;
				args.push(parseTernary());
			}
		}
		expect(")", 'a closing ")" after the arguments');

		const one = FUNCTIONS_1[name];
		if (one !== undefined) {
			if (args.length !== 1) throw new MLException(`expression: ${name} takes one argument`);
			const a = args[0];
			return (v) => one(a(v));
		}
		const two = FUNCTIONS_2[name];
		if (two !== undefined) {
			if (args.length !== 2) throw new MLException(`expression: ${name} takes two arguments`);
			const [a, b] = args;
			return (v) => two(a(v), b(v));
		}
		const many = FUNCTIONS_VARIADIC[name];
		if (many !== undefined) {
			if (args.length === 0)
				throw new MLException(`expression: ${name} takes at least one argument`);
			return (v) => many(args.map((arg) => arg(v)));
		}
		// Deliberately not supported: muParser's `rnd`. A filter that produced a
		// different mesh on every run would make every downstream hash useless.
		if (name === "rnd") {
			throw new MLException(
				'expression: "rnd" is not supported, because a filter whose output changes between runs cannot be checked',
			);
		}
		throw new MLException(`expression: unknown function "${name}"`);
	}

	function parseBinary(minPrecedence: number): Node {
		let left = parsePrimary();
		for (;;) {
			const token = peek();
			if (token === undefined || token.kind !== "op") break;
			const precedence = BINARY_PRECEDENCE[token.text];
			if (precedence === undefined || precedence < minPrecedence) break;
			at++;
			// `^` is the only right-associative operator, so it recurses at its
			// own precedence rather than one above it.
			const right = parseBinary(token.text === "^" ? precedence : precedence + 1);
			const apply = BINARY_OPS[token.text];
			const l = left;
			left = (v) => apply(l(v), right(v));
		}
		return left;
	}

	function parseTernary(): Node {
		const condition = parseBinary(1);
		if (peek()?.kind !== "?") return condition;
		at++;
		const whenTrue = parseTernary();
		expect(":", 'a ":" to close the conditional');
		const whenFalse = parseTernary();
		return (v) => (condition(v) !== 0 ? whenTrue(v) : whenFalse(v));
	}

	if (tokens.length === 0) throw new MLException("expression: it is empty");
	const root = parseTernary();
	if (!done()) {
		const token = tokens[at];
		throw new MLException(`expression: trailing "${token.text}" at position ${token.at}`);
	}
	return { names, used, evaluate: root };
}

export const Expression = { compileExpression } as const;
