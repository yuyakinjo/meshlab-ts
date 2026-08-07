/**
 * `FileFormat` — a description plus the extensions that select it.
 *
 * One format can own several extensions, which is why this is a list and not a
 * string: TIFF is `TIF` and `TIFF`, JPEG is `JPG` and `JPEG`.
 */
export class FileFormat {
	readonly description: string;
	/** Uppercase, without a leading dot. */
	readonly extensions: readonly string[];

	constructor(description: string, extensions: string | readonly string[]) {
		this.description = description;
		const list = typeof extensions === "string" ? [extensions] : extensions;
		this.extensions = list.map((e) => e.replace(/^\./, "").toUpperCase());
	}

	matches(extension: string): boolean {
		return this.extensions.includes(extension.replace(/^\./, "").toUpperCase());
	}
}

/** The extension of `path`, uppercase and without the dot. */
export function extensionOf(path: string): string {
	const base = path.split(/[\\/]/).pop() ?? path;
	const dot = base.lastIndexOf(".");
	return dot < 0 ? "" : base.slice(dot + 1).toUpperCase();
}
