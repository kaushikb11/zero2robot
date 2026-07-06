// Shared top-down world<->pixel transform. The renderer draws with it and the
// drag-to-push controller inverts it, so a pointer at a canvas pixel maps to the
// exact world point the pusher is steered toward. Single constant keeps them
// consistent.
//
// PushT workspace is +-0.45 m (the table plane half-size); walls sit at +-0.41.
export const WORLD_HALF_EXTENT = 0.45;

/** World (x, y) metres -> canvas pixel (world +y is up). */
export function worldToPx(
  canvas: { width: number; height: number },
  x: number,
  y: number,
): [number, number] {
  const scale = canvas.width / (2 * WORLD_HALF_EXTENT);
  return [canvas.width / 2 + x * scale, canvas.height / 2 - y * scale];
}

/** A pointer event's clientX/clientY -> world (x, y) metres. */
export function eventToWorld(
  canvas: HTMLCanvasElement,
  clientX: number,
  clientY: number,
): [number, number] {
  const rect = canvas.getBoundingClientRect();
  // CSS pixels within the element, scaled to the canvas backing-store resolution.
  const cx = ((clientX - rect.left) / rect.width) * canvas.width;
  const cy = ((clientY - rect.top) / rect.height) * canvas.height;
  const scale = canvas.width / (2 * WORLD_HALF_EXTENT);
  return [(cx - canvas.width / 2) / scale, -(cy - canvas.height / 2) / scale];
}
