/**
 * `MeshLabPlugin` — the base every plugin kind shares.
 *
 * `ActionIDType` is a plain int, plugin-local: two plugins may both use 0.
 * What has to be globally unique is the *filter name*, which is the invariant
 * `PluginManager` enforces at registration.
 */
export type ActionIDType = number;

export type MeshLabPluginType = "filter" | "io" | "decorate";

export abstract class MeshLabPlugin {
	/** Discriminant, standing in for C++ RTTI over the plugin interfaces. */
	abstract readonly pluginType: MeshLabPluginType;

	abstract pluginName(): string;

	vendor(): string {
		return "CNR-ISTI VCLab";
	}

	private _enabled = true;

	isEnabled(): boolean {
		return this._enabled;
	}

	setEnabled(enabled: boolean): void {
		this._enabled = enabled;
	}
}
