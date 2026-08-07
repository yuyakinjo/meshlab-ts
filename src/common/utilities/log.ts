/**
 * `GLLogStream` — where filters write the messages MeshLab shows in its log
 * panel. Headless, they are simply collected on the document.
 */

export interface LogEntry {
	readonly level: "info" | "warning" | "error" | "debug";
	readonly message: string;
}

export class Log {
	private readonly entries: LogEntry[] = [];
	/** Set to mirror everything to the console as it happens. */
	echo = false;

	log(message: string): void {
		this.push("info", message);
	}
	warning(message: string): void {
		this.push("warning", message);
	}
	error(message: string): void {
		this.push("error", message);
	}
	debug(message: string): void {
		this.push("debug", message);
	}

	private push(level: LogEntry["level"], message: string): void {
		this.entries.push({ level, message });
		if (this.echo) console.error(`[${level}] ${message}`);
	}

	all(): readonly LogEntry[] {
		return this.entries;
	}

	messages(): readonly string[] {
		return this.entries.map((e) => e.message);
	}

	clear(): void {
		this.entries.length = 0;
	}
}
