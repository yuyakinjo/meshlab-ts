import fc from "fast-check";

/**
 * How many cases each property runs. CI keeps this small enough to stay in the
 * fast feedback loop; `bun run test:prop:deep` raises it for nightly runs.
 */
export const NUM_RUNS: number = Number(process.env.FAST_CHECK_NUM_RUNS ?? 100);

export const propertyOptions: fc.Parameters<unknown> = { numRuns: NUM_RUNS };

/** A coordinate that is finite and well away from the limits of a double. */
export const arbCoord = (): fc.Arbitrary<number> =>
	fc.double({ min: -1e3, max: 1e3, noNaN: true, noDefaultInfinity: true });

/** One step of a random allocator workload. */
export type AllocatorOp =
	| { kind: "addVerts"; n: number }
	| { kind: "addFaces"; n: number }
	| { kind: "deleteVert"; pick: number }
	| { kind: "deleteFace"; pick: number }
	| { kind: "compactFaces" }
	| { kind: "compactVerts" }
	| { kind: "compactAll" };

/**
 * A random sequence of allocator operations.
 *
 * `pick` is a fraction in [0, 1); the runner turns it into an index into
 * whatever is live at that moment, so a shrunk counterexample stays valid.
 */
export const arbAllocatorOps = (maxLen = 40): fc.Arbitrary<AllocatorOp[]> =>
	fc.array(
		fc.oneof(
			{
				arbitrary: fc.integer({ min: 1, max: 8 }).map((n) => ({ kind: "addVerts", n }) as const),
				weight: 4,
			},
			{
				arbitrary: fc.integer({ min: 1, max: 6 }).map((n) => ({ kind: "addFaces", n }) as const),
				weight: 4,
			},
			{
				arbitrary: fc
					.double({ min: 0, max: 0.999, noNaN: true })
					.map((pick) => ({ kind: "deleteVert", pick }) as const),
				weight: 2,
			},
			{
				arbitrary: fc
					.double({ min: 0, max: 0.999, noNaN: true })
					.map((pick) => ({ kind: "deleteFace", pick }) as const),
				weight: 3,
			},
			{ arbitrary: fc.constant({ kind: "compactFaces" } as const), weight: 1 },
			{ arbitrary: fc.constant({ kind: "compactVerts" } as const), weight: 1 },
			{ arbitrary: fc.constant({ kind: "compactAll" } as const), weight: 1 },
		),
		{ maxLength: maxLen },
	);

/**
 * A random triangle soup: `nv` vertices and `nf` triangles drawn from them.
 *
 * Deliberately unconstrained — the result may be non-manifold, degenerate,
 * disconnected or all three. That is the point: kernel invariants must hold on
 * inputs no well-behaved modelling tool would produce.
 */
export const arbTriSoup = (
	maxVerts = 12,
	maxFaces = 16,
): fc.Arbitrary<{ coords: number[]; faces: number[] }> =>
	fc
		.integer({ min: 3, max: maxVerts })
		.chain((nv) =>
			fc.record({
				coords: fc.array(arbCoord(), { minLength: nv * 3, maxLength: nv * 3 }),
				faces: fc.array(fc.integer({ min: 0, max: nv - 1 }), {
					minLength: 0,
					maxLength: maxFaces * 3,
				}),
			}),
		)
		.map(({ coords, faces }) => ({
			coords,
			// Trim to a whole number of triangles.
			faces: faces.slice(0, faces.length - (faces.length % 3)),
		}));
