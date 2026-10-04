// Line-by-line port of src/signrec/preprocess.py (parity checked by tests/test_js_parity.py).
// A sequence is {T, data: Float64Array(T * L * 3)} with NaN for missing landmarks.

import { MIRROR, N_LANDMARKS, POSE_POS, SLICES } from "./landmarks.js";

const L = N_LANDMARKS;
const EPS = 1e-6;
const [LH0, LH1] = SLICES.left_hand;
const [RH0, RH1] = SLICES.right_hand;

const at = (t, l, c) => (t * L + l) * 3 + c;

export function featureCount(cfg) {
  const d = cfg.use_z ? 3 : 2;
  let n = landmarkIds(cfg).length * d;
  if (cfg.hand_local) n += 21 * d;
  return n * (cfg.velocity ? 2 : 1);
}

export function landmarkIds(cfg) {
  const ids = [];
  for (const p of cfg.parts) for (let i = SLICES[p][0]; i < SLICES[p][1]; i++) ids.push(i);
  return ids;
}

function handPresence(seq) {
  const { T, data } = seq;
  const lh = new Array(T).fill(false);
  const rh = new Array(T).fill(false);
  for (let t = 0; t < T; t++) {
    for (let l = LH0; l < LH1; l++) if (!Number.isNaN(data[(t * L + l) * 3])) { lh[t] = true; break; }
    for (let l = RH0; l < RH1; l++) if (!Number.isNaN(data[(t * L + l) * 3])) { rh[t] = true; break; }
  }
  return { lh, rh };
}

function slice(seq, t0, t1) {
  return { T: t1 - t0, data: seq.data.slice(t0 * L * 3, t1 * L * 3) };
}

export function trimHandless(seq) {
  const { lh, rh } = handPresence(seq);
  let first = -1;
  let last = -1;
  for (let t = 0; t < seq.T; t++) if (lh[t] || rh[t]) { if (first < 0) first = t; last = t; }
  return first < 0 ? seq : slice(seq, first, last + 1);
}

export function mirror(seq) {
  const { T, data } = seq;
  const out = new Float64Array(data.length);
  for (let t = 0; t < T; t++) {
    for (let l = 0; l < L; l++) {
      const src = (t * L + MIRROR[l]) * 3;
      const dst = (t * L + l) * 3;
      out[dst] = 1.0 - data[src];
      out[dst + 1] = data[src + 1];
      out[dst + 2] = data[src + 2];
    }
  }
  return { T, data: out };
}

export function canonicalizeHand(seq) {
  const { lh, rh } = handPresence(seq);
  const nl = lh.filter(Boolean).length;
  const nr = rh.filter(Boolean).length;
  return nl > nr ? mirror(seq) : seq;
}

export function normalize(seq, minShoulderWidth) {
  const { T, data } = seq;
  const ls = POSE_POS.l_shoulder;
  const rs = POSE_POS.r_shoulder;
  let wSum = 0, cx = 0, cy = 0, nOk = 0;
  for (let t = 0; t < T; t++) {
    const a = [data[at(t, ls, 0)], data[at(t, ls, 1)]];
    const b = [data[at(t, rs, 0)], data[at(t, rs, 1)]];
    if ([...a, ...b].some(Number.isNaN)) continue;
    wSum += Math.hypot(a[0] - b[0], a[1] - b[1]);
    cx += (a[0] + b[0]) / 2;
    cy += (a[1] + b[1]) / 2;
    nOk++;
  }
  const width = nOk ? wSum / nOk : 0;
  let center, scale;
  if (width >= minShoulderWidth) {
    center = [cx / nOk, cy / nOk];
    scale = width;
  } else {
    let n = 0, sx = 0, sy = 0;
    for (let i = 0; i < T * L; i++) {
      const x = data[i * 3], y = data[i * 3 + 1];
      if (Number.isNaN(x) || Number.isNaN(y)) continue;
      sx += x; sy += y; n++;
    }
    if (n === 0) return { T, data: data.slice() };
    center = [sx / n, sy / n];
    let vx = 0, vy = 0;
    for (let i = 0; i < T * L; i++) {
      const x = data[i * 3], y = data[i * 3 + 1];
      if (Number.isNaN(x) || Number.isNaN(y)) continue;
      vx += (x - center[0]) ** 2; vy += (y - center[1]) ** 2;
    }
    const std = (Math.sqrt(vx / n) + Math.sqrt(vy / n)) / 2;
    scale = Math.max(std * 4.0, EPS);
  }
  const out = new Float64Array(data.length);
  for (let i = 0; i < T * L; i++) {
    out[i * 3] = (data[i * 3] - center[0]) / scale;
    out[i * 3 + 1] = (data[i * 3 + 1] - center[1]) / scale;
    out[i * 3 + 2] = data[i * 3 + 2] / scale;
  }
  return { T, data: out };
}

function linspace(stop, num) {
  // numpy.linspace(0, stop, num): i * step, with the last element set exactly to stop
  const step = stop / (num - 1);
  const out = new Float64Array(num);
  for (let i = 0; i < num; i++) out[i] = i * step;
  if (num > 1) out[num - 1] = stop;
  return out;
}

export function resample(seq, T) {
  const n = seq.T;
  const data = seq.data;
  if (n === T) return { T, data: data.slice() };
  const out = new Float64Array(T * L * 3);
  if (n === 1) {
    for (let t = 0; t < T; t++) out.set(data.subarray(0, L * 3), t * L * 3);
    return { T, data: out };
  }
  const pos = linspace(n - 1, T);
  for (let t = 0; t < T; t++) {
    const lo = Math.floor(pos[t]);
    const hi = Math.min(lo + 1, n - 1);
    const w = pos[t] - lo;
    for (let k = 0; k < L * 3; k++) {
      const a = data[lo * L * 3 + k];
      const b = data[hi * L * 3 + k];
      let v = a * (1 - w) + b * w;
      if (Number.isNaN(v)) v = w < 0.5 ? a : b;
      out[t * L * 3 + k] = v;
    }
  }
  return { T, data: out };
}

export function fixLength(seq, T, mode) {
  if (mode === "resample" || (mode === "pad" && seq.T > T)) {
    return { seq: resample(seq, T), mask: new Array(T).fill(true) };
  }
  if (mode !== "pad") throw new Error(`unknown length_mode ${mode}`);
  const out = new Float64Array(T * L * 3).fill(NaN);
  out.set(seq.data.subarray(0, seq.T * L * 3));
  const mask = new Array(T).fill(false);
  for (let t = 0; t < seq.T; t++) mask[t] = true;
  return { seq: { T, data: out }, mask };
}

function handLocal(seq) {
  const { T, data } = seq;
  const out = new Float64Array(T * 21 * 3);
  for (let t = 0; t < T; t++) {
    const w = (t * L + RH0) * 3;
    const m = (t * L + RH0 + 9) * 3;
    const size = Math.max(Math.hypot(data[m] - data[w], data[m + 1] - data[w + 1]), EPS);
    for (let j = 0; j < 21; j++) {
      const s = (t * L + RH0 + j) * 3;
      for (let c = 0; c < 3; c++) out[(t * 21 + j) * 3 + c] = (data[s + c] - data[w + c]) / size;
    }
  }
  return out;
}

/**
 * Raw unified sequence to {features: Float32Array(T*F), mask: Float32Array(T)}.
 * yScale = frame height / width makes MediaPipe's normalised coordinates isotropic.
 */
export function preprocess(input, cfg, yScale = 1.0) {
  if (input.T === 0) throw new Error("empty sequence");
  let seq = { T: input.T, data: Float64Array.from(input.data) };
  if (yScale !== 1.0) for (let i = 1; i < seq.data.length; i += 3) seq.data[i] *= Math.fround(yScale);
  if (cfg.trim) seq = trimHandless(seq);
  if (cfg.canonical_hand) seq = canonicalizeHand(seq);
  seq = normalize(seq, cfg.min_shoulder_width);
  const fixed = fixLength(seq, cfg.T, cfg.length_mode);
  seq = fixed.seq;
  const mask = fixed.mask;

  const d = cfg.use_z ? 3 : 2;
  const ids = landmarkIds(cfg);
  const hl = cfg.hand_local ? handLocal(seq) : null;
  const P = (ids.length + (hl ? 21 : 0)) * d;
  const F = P * (cfg.velocity ? 2 : 1);
  const T = cfg.T;
  const pos = new Float64Array(T * P);
  for (let t = 0; t < T; t++) {
    let k = 0;
    for (const l of ids) for (let c = 0; c < d; c++) pos[t * P + k++] = seq.data[(t * L + l) * 3 + c];
    if (hl) for (let j = 0; j < 21; j++) for (let c = 0; c < d; c++) pos[t * P + k++] = hl[(t * 21 + j) * 3 + c];
  }
  const feats = new Float32Array(T * F);
  const clean = (v) => (Number.isFinite(v) ? v : 0);
  for (let t = 0; t < T; t++) {
    for (let k = 0; k < P; k++) {
      feats[t * F + k] = mask[t] ? clean(pos[t * P + k]) : 0;
      if (cfg.velocity) {
        let v = 0;
        if (t > 0 && mask[t] && mask[t - 1]) v = pos[t * P + k] - pos[(t - 1) * P + k];
        feats[t * F + P + k] = mask[t] ? clean(v) : 0;
      }
    }
  }
  return { features: feats, mask: Float32Array.from(mask, (m) => (m ? 1 : 0)) };
}
