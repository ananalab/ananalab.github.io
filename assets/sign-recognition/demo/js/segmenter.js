// Real-time sign segmentation from hand presence and motion energy.
//
// States: idle -> signing once hands are visible for `startFrames` frames; signing ends when the
// hands are gone for `endNoHandFrames`, still for `stillFrames` or after `maxFrames`. After a
// "still" end the segmenter waits in cooldown until the hands move again or go down.
//
// Frames are unified-format Float64Array(L*3). The segmenter only buffers them; the caller
// classifies the emitted segment with preprocess() and the ONNX model.

import { N_LANDMARKS, POSE_POS, SLICES } from "./landmarks.js";

const DEFAULT_SEGMENTER = {
  startFrames: 3,
  preRoll: 4,
  endNoHandFrames: 6,
  stillFrames: 14,
  stillThreshold: 0.012, // mean hand displacement per frame, in shoulder widths
  minFrames: 8,
  maxFrames: 120,
};

const L = N_LANDMARKS;

export function handsVisible(frame) {
  for (const part of ["left_hand", "right_hand"]) {
    const [a, b] = SLICES[part];
    for (let l = a; l < b; l++) if (!Number.isNaN(frame[l * 3])) return true;
  }
  return false;
}

function shoulderWidth(frame) {
  const a = POSE_POS.l_shoulder * 3;
  const b = POSE_POS.r_shoulder * 3;
  const w = Math.hypot(frame[a] - frame[b], frame[a + 1] - frame[b + 1]);
  return Number.isFinite(w) && w > 0.02 ? w : 0.25;
}

/** Mean displacement of the visible hand landmarks between two frames (shoulder widths). */
export function motionEnergy(prev, cur) {
  if (!prev) return 0;
  let s = 0;
  let n = 0;
  for (let l = 0; l < 42; l++) {
    const dx = cur[l * 3] - prev[l * 3];
    const dy = cur[l * 3 + 1] - prev[l * 3 + 1];
    if (Number.isFinite(dx) && Number.isFinite(dy)) {
      s += Math.hypot(dx, dy);
      n++;
    }
  }
  return n ? s / n / shoulderWidth(cur) : 0;
}

export class Segmenter {
  constructor(cfg = {}) {
    this.cfg = { ...DEFAULT_SEGMENTER, ...cfg };
    this.reset();
  }

  reset() {
    this.state = "idle";
    this.buffer = [];
    this.history = [];
    this.handRun = 0;
    this.noHand = 0;
    this.still = 0;
    this.prev = null;
  }

  /** Push one frame; returns {segment: Float64Array[] | null, reason} when a sign ends. */
  push(frame) {
    const c = this.cfg;
    const visible = handsVisible(frame);
    const energy = motionEnergy(this.prev, frame);
    this.prev = frame;
    this.history.push(frame);
    if (this.history.length > c.preRoll + c.startFrames) this.history.shift();

    if (this.state === "cooldown") {
      if (!visible || energy > 2 * c.stillThreshold) this.state = "idle";
      this.handRun = visible ? this.handRun : 0;
      return { segment: null };
    }
    if (this.state === "idle") {
      this.handRun = visible ? this.handRun + 1 : 0;
      if (this.handRun >= c.startFrames) {
        this.state = "signing";
        this.buffer = this.history.slice();
        this.noHand = 0;
        this.still = 0;
      }
      return { segment: null };
    }
    // signing
    this.buffer.push(frame);
    this.noHand = visible ? 0 : this.noHand + 1;
    this.still = visible && energy < c.stillThreshold ? this.still + 1 : 0;
    let reason = null;
    if (this.noHand >= c.endNoHandFrames) reason = "hands-down";
    else if (this.still >= c.stillFrames && this.buffer.length >= c.minFrames) reason = "still";
    else if (this.buffer.length >= c.maxFrames) reason = "max-length";
    if (!reason) return { segment: null };

    const segment = this.buffer;
    this.buffer = [];
    this.handRun = 0;
    // only a "still" end waits for new motion; after a cut the signer is still moving
    this.state = reason === "still" ? "cooldown" : "idle";
    if (segment.length < c.minFrames) return { segment: null };
    return { segment, reason };
  }

  /** Progress of the current recording in [0, 1] (for the UI). */
  progress() {
    return this.state === "signing" ? Math.min(1, this.buffer.length / this.cfg.maxFrames) : 0;
  }
}

/** Stack frames into the {T, data} sequence format used by preprocess(). */
export function toSequence(frames) {
  const data = new Float64Array(frames.length * L * 3);
  frames.forEach((f, t) => data.set(f, t * L * 3));
  return { T: frames.length, data };
}
