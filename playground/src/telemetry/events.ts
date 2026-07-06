// WHITELIST of learner-progress telemetry events the playground may emit — the
// contract referenced by playground/CLAUDE.md ("the explicit learner-progress
// events listed in src/telemetry/events.ts"). This file IS that whitelist.
//
// Privacy is enforced by construction, not by convention:
//   - Only the event TYPES named in EVENT_TYPES may be emitted; anything else
//     throws (fail-closed).
//   - Only the FIELD names listed per type in EVENT_FIELDS survive; every other
//     key on a payload is DROPPED before the event reaches a sink, so no
//     free-form string or PII can ride along on a progress event.
//   - NO network by default. The default persistent sink is localStorage (a
//     small capped ring, on-device), falling back to a no-op if storage is
//     unavailable/blocked. Telemetry is opt-in and never leaves the device
//     unless a host explicitly installs a network sink — none ships here.
//
// This module is pure TypeScript with no DOM access at import time (localStorage
// is touched lazily, guarded), so it transpiles + runs under Node for tests
// exactly like contracts.ts / lerobot_writer.ts.

/** The only event types the playground may ever emit. */
export const EVENT_TYPES = [
  'sim-booted', // MuJoCo WASM + PushT scene became interactive
  'policy-loaded', // an ONNX policy passed the fail-closed contract gate
  'episode-recorded', // a teleop episode was closed into the interchange buffer
  'policy-drove', // the loaded policy drove an episode to done (success or timeout)
] as const;

export type EventType = (typeof EVENT_TYPES)[number];

/** Allowed field names per event. The emitter copies ONLY these keys — the
 *  whitelist that keeps free-form/PII fields off a progress event. */
export const EVENT_FIELDS = {
  'sim-booted': ['loadMs'],
  'policy-loaded': ['obsDim', 'actDim', 'contractVersion'],
  'episode-recorded': ['steps', 'withImages'],
  'policy-drove': ['episodes', 'successes'],
} as const satisfies Record<EventType, readonly string[]>;

/** Typed payloads callers pass to emit(). Numbers/booleans plus one controlled
 *  contract-version string — deliberately no free-form text. */
export interface EventPayloads {
  'sim-booted': { loadMs: number };
  'policy-loaded': { obsDim: number; actDim: number; contractVersion: string };
  'episode-recorded': { steps: number; withImages: boolean };
  'policy-drove': { episodes: number; successes: number };
}

/** A sanitized event as it reaches a sink: the type, a timestamp, and only the
 *  whitelisted primitive fields for that type. */
export interface EmittedEvent {
  type: EventType;
  /** ms since epoch when emitted (Date.now) — the only implicit field. */
  t: number;
  [field: string]: number | boolean | string;
}

/** Build the on-the-wire event: rejects unknown types, and copies ONLY the
 *  whitelisted keys whose values are primitive (number/boolean/string). Any
 *  extra or non-primitive field is silently dropped. Pure + DOM-free. */
export function sanitizeEvent<T extends EventType>(
  type: T,
  payload: EventPayloads[T],
  now: number = Date.now(),
): EmittedEvent {
  if (!(EVENT_TYPES as readonly string[]).includes(type)) {
    throw new Error(`telemetry: "${type}" is not a whitelisted event type`);
  }
  const allowed = EVENT_FIELDS[type];
  const out: EmittedEvent = { type, t: now };
  for (const key of allowed) {
    const v = (payload as Record<string, unknown>)[key];
    if (typeof v === 'number' || typeof v === 'boolean' || typeof v === 'string') {
      out[key] = v;
    }
  }
  return out;
}

/** Where sanitized events go. Sinks must never throw (telemetry is best-effort)
 *  and must never see anything sanitizeEvent did not produce. */
export interface TelemetrySink {
  record(event: EmittedEvent): void;
}

/** Drops everything — the safe fallback when no storage/network is available. */
export const noopSink: TelemetrySink = { record() {} };

/** In-memory, capped, observable sink. Used for the always-on buffer behind
 *  telemetry.events() (and mirrored to window.__telemetry) and by tests. */
export class MemorySink implements TelemetrySink {
  readonly events: EmittedEvent[] = [];
  constructor(private cap = 200) {}
  record(event: EmittedEvent): void {
    this.events.push(event);
    while (this.events.length > this.cap) this.events.shift();
  }
}

const STORAGE_KEY = 'z2r:telemetry';

/** On-device localStorage sink (a capped JSON ring). Guarded: if localStorage
 *  is absent (Node/tests) or blocked (private mode / quota), returns noopSink so
 *  emit() is a no-op rather than an error. Never throws. */
export function localStorageSink(key = STORAGE_KEY, cap = 200): TelemetrySink {
  let store: Storage | null = null;
  try {
    store = typeof localStorage !== 'undefined' ? localStorage : null;
  } catch {
    store = null; // access itself can throw when cookies/storage are disabled
  }
  if (!store) return noopSink;
  const s = store;
  return {
    record(event) {
      try {
        const raw = s.getItem(key);
        const arr: EmittedEvent[] = raw ? (JSON.parse(raw) as EmittedEvent[]) : [];
        arr.push(event);
        while (arr.length > cap) arr.shift();
        s.setItem(key, JSON.stringify(arr));
      } catch {
        /* storage full/blocked — best-effort, swallow */
      }
    },
  };
}

/** The playground's telemetry emitter. Always records to an in-memory buffer
 *  (observability) and fans out to the configured persistent sinks (default:
 *  localStorage only). enabled=false makes emit() a no-op. */
export class Telemetry {
  readonly memory = new MemorySink();
  private sinks: TelemetrySink[];
  private enabled: boolean;

  constructor(opts: { sinks?: TelemetrySink[]; enabled?: boolean } = {}) {
    this.enabled = opts.enabled ?? true;
    // Default: on-device localStorage only. NO network sink ships.
    this.sinks = opts.sinks ?? [localStorageSink()];
  }

  setEnabled(on: boolean): void {
    this.enabled = on;
  }

  /** Emit a whitelisted progress event. Returns the sanitized event (for
   *  observability/tests) or null when disabled. Never throws for a valid type;
   *  an unknown type throws via sanitizeEvent (a programming error, not input). */
  emit<T extends EventType>(type: T, payload: EventPayloads[T]): EmittedEvent | null {
    if (!this.enabled) return null;
    const ev = sanitizeEvent(type, payload);
    this.memory.record(ev);
    for (const sink of this.sinks) sink.record(ev);
    return ev;
  }

  /** The most recent emitted events (capped, in-memory). */
  events(): readonly EmittedEvent[] {
    return this.memory.events;
  }
}

/** Process-wide default emitter. Constructing it touches no DOM (localStorage is
 *  only read/written inside record()), so importing this module is side-effect
 *  free enough to run under Node. */
export const telemetry = new Telemetry();
