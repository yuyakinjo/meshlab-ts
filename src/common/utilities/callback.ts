/**
 * `vcg::CallBackPos` — the progress hook every long-running filter accepts.
 *
 * Return `false` to ask the filter to stop; it must then throw
 * {@link UserCanceledException} rather than return a half-finished mesh.
 */
export type CallBackPos = (pos: number, message: string) => boolean;

/** The do-nothing callback, so filters never have to null-check. */
export const noCallback: CallBackPos = () => true;
