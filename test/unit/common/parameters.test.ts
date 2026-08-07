import { describe, expect, test } from "bun:test";
import {
	RichBool,
	RichColor,
	RichDirection,
	RichDynamicFloat,
	RichEnum,
	RichFileOpen,
	RichFileSave,
	RichFloat,
	RichInt,
	RichMatrix44,
	RichMesh,
	RichPercentage,
	RichPosition,
	RichShot,
	RichString,
} from "../../../src/common/parameters/rich_parameter.ts";
import { RichParameterList } from "../../../src/common/parameters/rich_parameter_list.ts";
import { IDENTITY_MATRIX44, type ShotValue } from "../../../src/common/parameters/value.ts";
import { InvalidParameterException } from "../../../src/common/utilities/ml_exception.ts";

const SHOT: ShotValue = {
	rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
	translation: [0, 0, 0],
	focalMm: 30,
	pixelSizeMm: [0.01, 0.01],
	centerPx: [512, 384],
	viewportPx: [1024, 768],
};

/** One of every parameter type, with the `stringType()` MeshLab writes. */
const ALL_TYPES = [
	[new RichBool("b", true), "RichBool"],
	[new RichInt("i", 3), "RichInt"],
	[new RichFloat("f", 1.5), "RichFloat"],
	[new RichString("s", "hi"), "RichString"],
	[new RichEnum("e", 1, ["a", "b", "c"]), "RichEnum"],
	[new RichPercentage("p", 5, 0, 10), "RichAbsPerc"],
	[new RichDynamicFloat("d", 0.5, 0, 1), "RichDynamicFloat"],
	[new RichPosition("pos", [1, 2, 3]), "RichPosition"],
	[new RichDirection("dir", [0, 0, 1]), "RichDirection"],
	[new RichMatrix44("mat"), "RichMatrix44f"],
	[new RichColor("c", 0xff0000ff), "RichColor"],
	[new RichMesh("m", 0), "RichMesh"],
	[new RichFileOpen("in", "", ["ply"]), "RichOpenFile"],
	[new RichFileSave("out", "", "ply"), "RichSaveFile"],
	[new RichShot("shot", SHOT), "RichShotf"],
] as const;

describe("RichParameter types", () => {
	test("all fifteen exist with the stringType MeshLab writes", () => {
		expect(ALL_TYPES).toHaveLength(15);
		for (const [param, stringType] of ALL_TYPES) {
			expect(param.stringType(), param.name).toBe(stringType);
		}
	});

	test("the four renamed classes keep their legacy stringType", () => {
		// The classes were renamed upstream but .mlx files on disk still carry
		// the old strings, so both have to be reproduced.
		expect(new RichPercentage("p", 0, 0, 1).stringType()).toBe("RichAbsPerc");
		expect(new RichMatrix44("m").stringType()).toBe("RichMatrix44f");
		expect(new RichFileOpen("f", "").stringType()).toBe("RichOpenFile");
		expect(new RichFileSave("f", "").stringType()).toBe("RichSaveFile");
		expect(new RichShot("s", SHOT).stringType()).toBe("RichShotf");
	});

	test("a fresh parameter reports its value as the default", () => {
		for (const [param] of ALL_TYPES) {
			expect(param.isValueDefault, param.name).toBe(true);
		}
	});

	test("cloning carries a modified value across, for every type", () => {
		// A non-default value for each type, so the clone has something to
		// lose if it only copies the default.
		const samples: ReadonlyArray<[(typeof ALL_TYPES)[number][0], unknown]> = [
			[new RichBool("b", true), false],
			[new RichInt("i", 3), 7],
			[new RichFloat("f", 1.5), -2.25],
			[new RichString("s", "hi"), "bye"],
			[new RichEnum("e", 1, ["a", "b", "c"]), 2],
			[new RichPercentage("p", 5, 0, 10), 9],
			[new RichDynamicFloat("d", 0.5, 0, 1), 0.25],
			[new RichPosition("pos", [1, 2, 3]), [4, 5, 6]],
			[new RichDirection("dir", [0, 0, 1]), [0, 1, 0]],
			[new RichMatrix44("mat"), Array.from({ length: 16 }, (_, i) => i)],
			[new RichColor("c", 0xff0000ff), [0, 255, 0]],
			[new RichMesh("m", 0), 4],
			[new RichFileOpen("in", "", ["ply"]), "a.ply"],
			[new RichFileSave("out", "", "ply"), "b.ply"],
			[new RichShot("shot", SHOT), { ...SHOT, focalMm: 55 }],
		];
		expect(samples).toHaveLength(ALL_TYPES.length);

		for (const [param, sample] of samples) {
			param.setValue(param.fromPlain(sample));
			expect(param.isValueDefault, param.name).toBe(false);
			const copy = param.clone();
			expect(copy.stringType()).toBe(param.stringType());
			expect(copy.value, param.name).toEqual(param.value);
			expect(copy.isValueDefault, param.name).toBe(false);
		}
	});

	test("metadata survives a clone", () => {
		const p = new RichInt("n", 1, {
			description: "Count",
			tooltip: "how many",
			advanced: true,
			category: "Basic",
		});
		const c = p.clone();
		expect(c.fieldDescription).toBe("Count");
		expect(c.toolTip).toBe("how many");
		expect(c.isAdvanced).toBe(true);
		expect(c.category).toBe("Basic");
	});
});

describe("parameter coercion", () => {
	test("an int rejects a fractional value", () => {
		const p = new RichInt("n", 0);
		expect(() => p.fromPlain(1.5)).toThrow(InvalidParameterException);
		expect(() => p.fromPlain("3")).toThrow(InvalidParameterException);
		expect(p.fromPlain(3)).toEqual({ kind: "int", value: 3 });
	});

	test("a float rejects NaN and infinity", () => {
		const p = new RichFloat("x", 0);
		expect(() => p.fromPlain(Number.NaN)).toThrow(InvalidParameterException);
		expect(() => p.fromPlain(Number.POSITIVE_INFINITY)).toThrow(InvalidParameterException);
	});

	test("an enum accepts an index or an option name", () => {
		const p = new RichEnum("mode", 0, ["fast", "accurate"]);
		expect(p.fromPlain(1)).toEqual({ kind: "int", value: 1 });
		expect(p.fromPlain("accurate")).toEqual({ kind: "int", value: 1 });
		expect(() => p.fromPlain("nonsense")).toThrow(/no option "nonsense"/);
	});

	test("an enum rejects an out-of-range index", () => {
		const p = new RichEnum("mode", 0, ["a", "b"]);
		expect(() => p.setValue(p.fromPlain(2))).toThrow(/outside 0\.\.1/);
	});

	test("a bounded parameter rejects a value outside its range", () => {
		const p = new RichPercentage("size", 5, 0, 10);
		expect(() => p.setValue(p.fromPlain(11))).toThrow(/outside 0\.\.10/);
		expect(() => p.setValue(p.fromPlain(-1))).toThrow(/outside 0\.\.10/);
		p.setValue(p.fromPlain(10));
		expect(p.value.value).toBe(10);
	});

	test("a colour accepts a packed number or components", () => {
		const p = new RichColor("c", 0);
		expect(p.fromPlain([255, 0, 0])).toEqual({ kind: "color", value: 0xff0000ff });
		expect(p.fromPlain([255, 0, 0, 128])).toEqual({ kind: "color", value: 0x800000ff });
		expect(() => p.fromPlain([300, 0, 0])).toThrow(InvalidParameterException);
	});

	test("a position wants exactly three numbers", () => {
		const p = new RichPosition("p", [0, 0, 0]);
		expect(p.fromPlain([1, 2, 3])).toEqual({ kind: "point3", value: [1, 2, 3] });
		expect(() => p.fromPlain([1, 2])).toThrow(InvalidParameterException);
		expect(() => p.fromPlain([1, 2, "3"])).toThrow(InvalidParameterException);
	});

	test("a matrix wants exactly sixteen numbers", () => {
		const p = new RichMatrix44("m");
		expect(p.fromPlain([...IDENTITY_MATRIX44])).toBeDefined();
		expect(() => p.fromPlain([1, 2, 3])).toThrow(InvalidParameterException);
	});

	test("assigning the wrong kind is rejected", () => {
		const p = new RichInt("n", 0);
		expect(() => p.setValue({ kind: "string", value: "x" })).toThrow(/expects a int value/);
	});
});

describe("RichParameterList", () => {
	function list(): RichParameterList {
		return new RichParameterList([
			new RichInt("maxholesize", 30, { description: "Max size to be closed" }),
			new RichBool("selfintersection", true),
			new RichPercentage("threshold", 1, 0, 100),
			new RichEnum("mode", 0, ["fast", "accurate"]),
		]);
	}

	test("typed accessors return the right values", () => {
		const l = list();
		expect(l.getInt("maxholesize")).toBe(30);
		expect(l.getBool("selfintersection")).toBe(true);
		expect(l.getAbsPerc("threshold")).toBe(1);
		expect(l.getEnum("mode")).toBe(0);
		expect(l.size).toBe(4);
	});

	test("reading a parameter as the wrong type is an error", () => {
		expect(() => list().getBool("maxholesize")).toThrow(/holds a int, not a bool/);
	});

	test("an unknown parameter name lists the known ones", () => {
		try {
			list().getInt("maxholesizze");
			throw new Error("should have thrown");
		} catch (err) {
			expect((err as Error).message).toContain("known parameters");
			expect((err as Error).message).toContain("maxholesize");
		}
	});

	test("applyPlain merges onto the defaults", () => {
		const l = list();
		l.applyPlain({ maxholesize: 100, mode: "accurate" });
		expect(l.getInt("maxholesize")).toBe(100);
		expect(l.getEnum("mode")).toBe(1);
		expect(l.getBool("selfintersection")).toBe(true); // untouched
	});

	test("an unknown key is rejected rather than ignored", () => {
		// The whole point: a typo must not run the filter with a default and
		// report success.
		const l = list();
		expect(() => l.applyPlain({ maxholesizze: 100 })).toThrow(InvalidParameterException);
		try {
			l.applyPlain({ nonsense: 1 });
		} catch (err) {
			expect((err as Error).message).toContain("expected one of");
		}
		expect(l.getInt("maxholesize")).toBe(30);
	});

	test("duplicate parameter names are rejected", () => {
		const l = new RichParameterList([new RichInt("n", 1)]);
		expect(() => l.add(new RichInt("n", 2))).toThrow(/duplicate parameter name/);
	});

	test("clone is deep", () => {
		const a = list();
		const b = a.clone();
		b.applyPlain({ maxholesize: 999 });
		expect(a.getInt("maxholesize")).toBe(30);
		expect(b.getInt("maxholesize")).toBe(999);
	});

	test("toPlain round-trips through applyPlain", () => {
		const a = list();
		a.applyPlain({ maxholesize: 42, selfintersection: false, threshold: 7, mode: 1 });
		const b = list();
		b.applyPlain(a.toPlain());
		expect(b.toPlain()).toEqual(a.toPlain());
	});

	test("iterating yields the parameters in declaration order", () => {
		expect([...list()].map((p) => p.name)).toEqual([
			"maxholesize",
			"selfintersection",
			"threshold",
			"mode",
		]);
	});
});
