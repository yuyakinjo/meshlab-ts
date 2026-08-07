/**
 * The muParser-dialect expression evaluator.
 *
 * The dialect is the specification here, not JavaScript: MeshLab hands these
 * expressions to muParser, so every published recipe is written against
 * muParser's operators, precedence and function names. The tests that matter
 * most are the three places the two dialects disagree — `^`, the precedence of
 * a unary sign, and the absence of booleans.
 */
import { describe, expect, test } from "bun:test";
import { MLException } from "../../../src/common/utilities/ml_exception.ts";
import { compileExpression } from "../../../src/vcg/math/expression.ts";

const NAMES = ["x", "y", "q"] as const;
const values = (x: number, y = 0, q = 0) => Float64Array.from([x, y, q]);

/** Compiles and runs in one go, against `x = 2, y = 3, q = 0.5` unless given. */
const run = (source: string, v = values(2, 3, 0.5)): number =>
	compileExpression(source, NAMES).evaluate(v);

describe("arithmetic", () => {
	test("respects the usual precedence", () => {
		expect(run("1+2*3")).toBe(7);
		expect(run("(1+2)*3")).toBe(9);
		expect(run("10-2-3")).toBe(5);
		expect(run("100/10/2")).toBe(5);
	});

	test("^ is exponentiation and binds to the right", () => {
		// Not xor, and not left-associative: `2^3^2` is 2^9, not 8^2.
		expect(run("2^3^2")).toBe(512);
		expect(run("2^10")).toBe(1024);
	});

	test("a unary sign binds tighter than +- but looser than ^", () => {
		// muParser puts the infix sign at the same precedence as `*`, which is
		// the one rule that makes these two disagree.
		expect(run("-2^2")).toBe(-4);
		expect(run("-2*3")).toBe(-6);
		expect(run("3 - -2")).toBe(5);
		expect(run("+2^2")).toBe(4);
	});

	test("reads numbers in exponent form", () => {
		// The minus in `1e-3` must not be taken for an operator.
		expect(run("1e-3")).toBeCloseTo(0.001, 12);
		expect(run("1e3")).toBe(1000);
		expect(run("2.5E2")).toBe(250);
		expect(run(".5")).toBe(0.5);
		expect(run("1+1e-3")).toBeCloseTo(1.001, 12);
	});
});

describe("comparison and logic", () => {
	test("a comparison yields 1 or 0, not a boolean", () => {
		// Which is what lets `255*(z>0)` work as a colour expression.
		expect(run("3 > 2")).toBe(1);
		expect(run("3 < 2")).toBe(0);
		expect(run("3 >= 3")).toBe(1);
		expect(run("3 <= 2")).toBe(0);
		expect(run("3 == 3")).toBe(1);
		expect(run("3 != 3")).toBe(0);
		expect(run("255*(x>1)")).toBe(255);
	});

	test("&& and || return 1 or 0 rather than an operand", () => {
		// JavaScript would give 5 for `0 || 5`; muParser gives 1.
		expect(run("0 || 5")).toBe(1);
		expect(run("7 && 5")).toBe(1);
		expect(run("0 && 5")).toBe(0);
		expect(run("0 || 0")).toBe(0);
	});

	test("|| binds looser than &&", () => {
		// `a || b && c` is `a || (b && c)`. Reading it the other way would flip
		// this case.
		expect(run("1 || 0 && 0")).toBe(1);
		expect(run("(1 || 0) && 0")).toBe(0);
	});

	test("comparison binds looser than arithmetic and tighter than logic", () => {
		expect(run("1 + 1 == 2")).toBe(1);
		expect(run("x > 1 && y > 2")).toBe(1);
		expect(run("x > 1 && y > 5")).toBe(0);
	});

	test("the conditional picks a branch", () => {
		expect(run("x > 1 ? 10 : 20")).toBe(10);
		expect(run("x > 5 ? 10 : 20")).toBe(20);
		// It nests to the right, so a chain reads as a series of else-ifs.
		expect(run("x > 5 ? 1 : x > 1 ? 2 : 3")).toBe(2);
	});
});

describe("functions and constants", () => {
	test("the trigonometric and exponential set", () => {
		expect(run("sin(0)")).toBe(0);
		expect(run("cos(0)")).toBe(1);
		expect(run("atan2(1,1)")).toBeCloseTo(Math.PI / 4, 12);
		expect(run("sqrt(16)")).toBe(4);
		expect(run("exp(0)")).toBe(1);
		expect(run("cot(_pi/4)")).toBeCloseTo(1, 12);
	});

	test("log is the natural logarithm, not the base-10 one", () => {
		// muParser's `log` is `ln`. Reading it as log10 would quietly rescale
		// every expression that uses it.
		expect(run("log(_e)")).toBeCloseTo(1, 12);
		expect(run("ln(_e)")).toBeCloseTo(1, 12);
		expect(run("log10(1000)")).toBeCloseTo(3, 12);
		expect(run("log2(8)")).toBeCloseTo(3, 12);
	});

	test("sign and rint", () => {
		expect(run("sign(-3)")).toBe(-1);
		expect(run("sign(0)")).toBe(0);
		expect(run("sign(3)")).toBe(1);
		// rint rounds half to even, as C does under the default mode.
		expect(run("rint(2.5)")).toBe(2);
		expect(run("rint(3.5)")).toBe(4);
		expect(run("rint(-2.5)")).toBe(-2);
		expect(run("rint(2.4)")).toBe(2);
	});

	test("the variadic ones take any number of arguments", () => {
		expect(run("min(3,1,2)")).toBe(1);
		expect(run("max(3,1,2)")).toBe(3);
		expect(run("sum(1,2,3,4)")).toBe(10);
		expect(run("avg(1,2,3,4)")).toBe(2.5);
		expect(run("min(5)")).toBe(5);
	});

	test("the two constants", () => {
		expect(run("_pi")).toBe(Math.PI);
		expect(run("_e")).toBe(Math.E);
	});
});

describe("variables", () => {
	test("resolve to their slot", () => {
		expect(run("x")).toBe(2);
		expect(run("y")).toBe(3);
		expect(run("x*y-q")).toBe(5.5);
	});

	test("re-reading the buffer gives a new answer without recompiling", () => {
		// Which is the whole point: one compile, one evaluation per element.
		const compiled = compileExpression("x+y", NAMES);
		expect(compiled.evaluate(values(1, 2))).toBe(3);
		expect(compiled.evaluate(values(10, 20))).toBe(30);
	});

	test("reports which names the expression actually reads", () => {
		expect([...compileExpression("x + q", NAMES).used].sort()).toEqual(["q", "x"]);
		expect([...compileExpression("1 + 2", NAMES).used]).toEqual([]);
	});

	test("a name that is not a variable is an error, not a zero", () => {
		// A typo in a per-vertex expression would otherwise run over the whole
		// mesh and produce a plausible-looking result.
		expect(() => compileExpression("nope", NAMES)).toThrow(MLException);
		expect(() => compileExpression("x + zz", NAMES)).toThrow(MLException);
		try {
			compileExpression("nope", NAMES);
		} catch (err) {
			// The message has to name what *is* available, or the user is stuck.
			expect((err as Error).message).toContain("x, y, q");
		}
	});
});

describe("rejecting bad input", () => {
	test("an empty or truncated expression", () => {
		for (const bad of ["", "   ", "x +", "1 +* 2", "(1", "1)", "x ? 1", "min(", ","]) {
			expect(() => compileExpression(bad, NAMES), JSON.stringify(bad)).toThrow(MLException);
		}
	});

	test("a stray character says where it is", () => {
		try {
			compileExpression("x @ y", NAMES);
			throw new Error("should have thrown");
		} catch (err) {
			expect((err as Error).message).toContain("position 2");
		}
	});

	test("the wrong number of arguments", () => {
		expect(() => compileExpression("sin(1,2)", NAMES)).toThrow(MLException);
		expect(() => compileExpression("atan2(1)", NAMES)).toThrow(MLException);
		expect(() => compileExpression("min()", NAMES)).toThrow(MLException);
	});

	test("an unknown function", () => {
		expect(() => compileExpression("frobnicate(1)", NAMES)).toThrow(MLException);
	});

	test("rnd, on purpose", () => {
		// muParser has it; a filter whose output changes between runs would make
		// every downstream hash and every golden comparison useless.
		expect(() => compileExpression("rnd()", NAMES)).toThrow(MLException);
		expect(() => compileExpression("rnd(5)", NAMES)).toThrow(/cannot be checked/);
	});
});

describe("arithmetic edge cases", () => {
	test("division by zero follows IEEE, as muParser's doubles do", () => {
		expect(run("1/0")).toBe(Number.POSITIVE_INFINITY);
		expect(Number.isNaN(run("0/0"))).toBe(true);
	});

	test("whitespace anywhere is fine", () => {
		expect(run("  x   *   (  y  +  1 )  ")).toBe(8);
		expect(run("x*(y+1)")).toBe(8);
	});

	test("deep nesting does not fall over", () => {
		const deep = `${"(".repeat(60)}1${")".repeat(60)}`;
		expect(run(deep)).toBe(1);
	});
});
