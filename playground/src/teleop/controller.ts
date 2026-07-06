// Drag-to-push teleop: the pointer is the learner's hand on the robot. While
// dragging, the pusher is steered toward the pointer — the action is the
// displacement (pointer - pusher) mapped to a target velocity and clipped to the
// env's +-1 action box. Drag further = push faster; release = stop. Faithful to
// the action space (float32[2] target velocity, produced once per 10 Hz control
// step by the main loop).
import { eventToWorld } from './viewport';

// Displacement (m) -> velocity gain. At 1/GAIN metres of drag the action
// saturates to +-1, so ~0.125 m of pointer lead already commands full speed —
// direct and responsive without feeling twitchy.
const GAIN = 8;

export class DragController {
  private dragging = false;
  private pointerWorld: [number, number] = [0, 0];

  constructor(private canvas: HTMLCanvasElement) {
    canvas.addEventListener('pointerdown', this.onDown);
    canvas.addEventListener('pointermove', this.onMove);
    // Listen on window for up/cancel so a release outside the canvas still stops.
    window.addEventListener('pointerup', this.onUp);
    window.addEventListener('pointercancel', this.onUp);
  }

  get isDragging(): boolean {
    return this.dragging;
  }

  /** World-space point the pointer is currently over (for the drag cue). */
  get target(): [number, number] {
    return this.pointerWorld;
  }

  private onDown = (e: PointerEvent): void => {
    this.dragging = true;
    this.pointerWorld = eventToWorld(this.canvas, e.clientX, e.clientY);
    this.canvas.setPointerCapture(e.pointerId);
    e.preventDefault();
  };

  private onMove = (e: PointerEvent): void => {
    if (!this.dragging) return;
    this.pointerWorld = eventToWorld(this.canvas, e.clientX, e.clientY);
  };

  private onUp = (): void => {
    this.dragging = false;
  };

  /**
   * The action for this control step, given the pusher's current world xy.
   * Zero when not dragging (hands off = stop).
   */
  action(pusherX: number, pusherY: number): [number, number] {
    if (!this.dragging) return [0, 0];
    const clamp = (v: number): number => Math.max(-1, Math.min(1, v));
    return [
      clamp(GAIN * (this.pointerWorld[0] - pusherX)),
      clamp(GAIN * (this.pointerWorld[1] - pusherY)),
    ];
  }

  dispose(): void {
    this.canvas.removeEventListener('pointerdown', this.onDown);
    this.canvas.removeEventListener('pointermove', this.onMove);
    window.removeEventListener('pointerup', this.onUp);
    window.removeEventListener('pointercancel', this.onUp);
  }
}
