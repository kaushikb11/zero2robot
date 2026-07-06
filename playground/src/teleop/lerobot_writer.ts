// z2r-teleop-1 interchange writer — the FROZEN browser output format (decision
// 008 §"Interchange contract"). This is NOT a LeRobot v3 dataset: it is a
// compact, library-agnostic bundle (interchange.json manifest + one PNG per
// frame) that chapter 0.4's record.py / the Colab cell convert to canonical v3
// via the pinned `lerobot` library (the same LeRobotDataset.create path as
// gen_demos.py), so schema parity with the reference datasets is automatic.
//
// The manifest MUST match scripts/ref_interchange_writer.mjs byte-for-byte in
// SCHEMA (keys, feature dtypes/shapes/names, per-episode array layout). The
// golden test (scripts/golden_interchange_test.mjs) pins this.
import {
  ACT_DIM,
  ACTION_NAMES,
  CONTROL_HZ,
  IMG_HW,
  OBS_DIM,
  REPO_ID,
  ROBOT_TYPE,
  STATE_NAMES,
  TASK,
} from './pusht_obs';
import { zipStore, type ZipEntry } from './zip';

const INTERCHANGE_VERSION = 'z2r-teleop-1';

export interface FeatureSpec {
  dtype: string;
  shape: number[];
  names: string[];
}

export interface InterchangeEpisode {
  length: number;
  'observation.state': number[][]; // length x OBS_DIM
  action: number[][]; // length x ACT_DIM
  timestamp: number[]; // seconds, advisory (converter re-derives from fps)
  'observation.image'?: string[]; // one PNG path per frame, when images on
}

export interface InterchangeManifest {
  interchange_version: string;
  repo_id: string;
  robot_type: string;
  fps: number;
  task: string;
  features: Record<string, FeatureSpec>;
  episodes: InterchangeEpisode[];
}

export interface InterchangeBundle {
  manifest: InterchangeManifest;
  files: ZipEntry[]; // interchange.json + frames/**.png, ready for zipStore
}

/** Feature block — mirrors gen_demos.build_features exactly (the browser declares
 * the spec so the Python converter stays env-agnostic). */
export function buildFeatures(withImages: boolean): Record<string, FeatureSpec> {
  const features: Record<string, FeatureSpec> = {
    'observation.state': {
      dtype: 'float32',
      shape: [OBS_DIM],
      names: [...STATE_NAMES],
    },
    action: {
      dtype: 'float32',
      shape: [ACT_DIM],
      names: [...ACTION_NAMES],
    },
  };
  if (withImages) {
    features['observation.image'] = {
      dtype: 'video',
      shape: [IMG_HW, IMG_HW, 3],
      names: ['height', 'width', 'channel'],
    };
  }
  return features;
}

/** A single recorded episode with images already resolved to bytes. */
export interface ResolvedEpisode {
  state: number[][];
  action: number[][];
  timestamp: number[];
  images?: Uint8Array[]; // present iff images were recorded
}

export interface SerializeOptions {
  repoId?: string;
  task?: string;
  fps?: number;
}

/**
 * Pure serializer: resolved episodes -> { manifest, files }. No browser APIs, so
 * the golden test can call it directly. `frames/ep{e}/f{i}.png` paths are emitted
 * only when an episode carries images.
 */
export function serializeInterchange(
  episodes: ResolvedEpisode[],
  opts: SerializeOptions = {},
): InterchangeBundle {
  const withImages = episodes.some((e) => e.images && e.images.length > 0);
  // Image recording is all-or-nothing per z2r-teleop-1: if the bundle declares
  // observation.image, every episode must carry one PNG per frame, else the
  // Python reader (record.py) hits a missing key on an image-less episode.
  if (withImages && !episodes.every((e) => e.images && e.images.length === e.state.length)) {
    throw new Error(
      'z2r-teleop-1: image recording must be all-or-nothing — every episode needs one PNG per frame',
    );
  }
  const files: ZipEntry[] = [];
  const manifestEpisodes: InterchangeEpisode[] = [];

  episodes.forEach((ep, e) => {
    const length = ep.state.length;
    const record: InterchangeEpisode = {
      length,
      'observation.state': ep.state,
      action: ep.action,
      timestamp: ep.timestamp,
    };
    if (ep.images && ep.images.length > 0) {
      const paths: string[] = [];
      ep.images.forEach((bytes, i) => {
        const rel = `frames/ep${e}/f${i}.png`;
        paths.push(rel);
        files.push({ path: rel, bytes });
      });
      record['observation.image'] = paths;
    }
    manifestEpisodes.push(record);
  });

  const manifest: InterchangeManifest = {
    interchange_version: INTERCHANGE_VERSION,
    repo_id: opts.repoId ?? REPO_ID,
    robot_type: ROBOT_TYPE,
    fps: opts.fps ?? CONTROL_HZ,
    task: opts.task ?? TASK,
    features: buildFeatures(withImages),
    episodes: manifestEpisodes,
  };

  // interchange.json first, then frame PNGs.
  files.unshift({
    path: 'interchange.json',
    bytes: new TextEncoder().encode(JSON.stringify(manifest, null, 2)),
  });

  return { manifest, files };
}

/**
 * Records (obs, action, timestamp[, image]) per control step and serializes to a
 * z2r-teleop-1 bundle. Image frames are captured as PNG Blobs asynchronously
 * (canvas.toBlob) and awaited at build time, so recording never blocks the
 * control loop.
 */
export class InterchangeRecorder {
  private episodes: RecordingEpisode[] = [];
  private current: RecordingEpisode | null = null;

  /** Begin a new episode (call at env.reset). */
  startEpisode(): void {
    this.current = { state: [], action: [], timestamp: [], images: [] };
    this.episodes.push(this.current);
  }

  get isRecording(): boolean {
    return this.current !== null;
  }
  get episodeCount(): number {
    return this.episodes.length;
  }
  get currentLength(): number {
    return this.current ? this.current.state.length : 0;
  }
  get totalFrames(): number {
    return this.episodes.reduce((a, e) => a + e.state.length, 0);
  }

  /**
   * Record one control step. `obs` and `action` are the (observation-before,
   * action-applied) pair — the same convention as gen_demos.add_frame. `image`
   * is an optional PNG (bytes or a pending Blob promise) for observation.image.
   */
  recordStep(
    obs: ArrayLike<number>,
    action: ArrayLike<number>,
    timestamp: number,
    image?: Promise<Uint8Array> | Uint8Array,
  ): void {
    if (!this.current) throw new Error('recordStep called before startEpisode');
    const state: number[] = [];
    for (let i = 0; i < OBS_DIM; i++) state.push(obs[i]);
    const act: number[] = [];
    for (let i = 0; i < ACT_DIM; i++) act.push(action[i]);
    this.current.state.push(state);
    this.current.action.push(act);
    this.current.timestamp.push(timestamp);
    if (image !== undefined) this.current.images.push(image);
  }

  /** Close the current episode (does not clear prior episodes). */
  finishEpisode(): void {
    this.current = null;
  }

  /** Drop the in-progress episode entirely (e.g. reset without saving). */
  discardCurrent(): void {
    if (this.current) {
      const idx = this.episodes.indexOf(this.current);
      if (idx >= 0) this.episodes.splice(idx, 1);
      this.current = null;
    }
  }

  /** Clear everything (after a successful download, or to start over). */
  clear(): void {
    this.episodes = [];
    this.current = null;
  }

  /** Await any pending frame Blobs and serialize to a z2r-teleop-1 bundle. */
  async buildBundle(opts: SerializeOptions = {}): Promise<InterchangeBundle> {
    const resolved: ResolvedEpisode[] = [];
    for (const ep of this.episodes) {
      const images = ep.images.length > 0 ? await Promise.all(ep.images) : undefined;
      resolved.push({
        state: ep.state,
        action: ep.action,
        timestamp: ep.timestamp,
        images,
      });
    }
    return serializeInterchange(resolved, opts);
  }
}

interface RecordingEpisode {
  state: number[][];
  action: number[][];
  timestamp: number[];
  images: (Promise<Uint8Array> | Uint8Array)[];
}

/** Zip a bundle and trigger a browser download of `<name>.zip`. */
export async function downloadInterchange(
  bundle: InterchangeBundle,
  filename = 'pusht_teleop_interchange.zip',
): Promise<void> {
  const zipped = zipStore(bundle.files);
  // Copy into a fresh ArrayBuffer so Blob gets a plain (non-shared) buffer.
  const blob = new Blob([zipped.slice()], { type: 'application/zip' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Revoke on the next tick so the download has a chance to start.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Encode a canvas as PNG bytes (observation.image frame). The caller renders the
 * current sim state into a 96x96 canvas (matching the observation.image shape)
 * and passes it here; toBlob keeps encoding off the control loop.
 */
export function pngFromCanvas(canvas: HTMLCanvasElement): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) return reject(new Error('canvas.toBlob returned null'));
      blob.arrayBuffer().then((buf) => resolve(new Uint8Array(buf)), reject);
    }, 'image/png');
  });
}
