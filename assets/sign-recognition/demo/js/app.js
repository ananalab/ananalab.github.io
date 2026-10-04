// Browser demo: MediaPipe Holistic landmarks, then preprocess.js, then ONNX Runtime Web.
import { FilesetResolver, HolisticLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.min.mjs";
import { EDGES, GROUPS, N_LANDMARKS, SLICES } from "./landmarks.js";
import { featureCount, preprocess } from "./preprocess.js";
import { Segmenter, handsVisible, toSequence } from "./segmenter.js";

const MP_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const MP_MODEL =
  "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task";
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

const L = N_LANDMARKS;
const $ = (id) => document.getElementById(id);
const ui = {
  status: $("status"), statusText: $("status-text"), stage: $("stage"), video: $("video"), overlay: $("overlay"),
  empty: $("stage-empty"), fps: $("hud-fps"), lat: $("hud-lat"), badge: $("state-badge"), recFill: $("rec-fill"),
  bars: $("bars"), gloss: $("top1-gloss"), hint: $("top1-hint"), conf: $("top1-conf"),
  pred: $("prediction"), predMeta: $("pred-meta"), history: $("history"), easyGrid: $("easy-grid"), easyNote: $("easy-note"),
  vocabGrid: $("vocab-grid"), vocabSearch: $("vocab-search"), vocabCount: $("vocab-count"), vocabNote: $("vocab-note"),
  modelPicker: $("model-picker"), modePicker: $("mode-picker"), fileInput: $("file-input"),
  optSkeleton: $("opt-skeleton"), optMirror: $("opt-mirror"),
};

const app = {
  holistic: null, models: [], model: null, mode: "auto", running: false, stream: null,
  segmenter: new Segmenter(), holdFrames: null, fileFrames: null, lastFrame: null,
  frameTimes: [], mpMs: 0, modelMs: 0, history: [], lastVideoTime: -1, tsCounter: 0,
  examples: null, exampleOrder: null, exampleYScale: 1.0, fit: null, replayId: 0,
};

function setStatus(text, state = "loading") {
  ui.statusText.textContent = text;
  ui.status.dataset.state = state;
}
function setBadge(text, state) {
  ui.badge.textContent = text;
  ui.badge.dataset.state = state;
}
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

async function loadModelIndex() {
  const res = await fetch("models/index.json");
  if (!res.ok) throw new Error("models/index.json not found, run scripts/export_web_assets.py");
  app.models = (await res.json()).models;
  ui.modelPicker.innerHTML = "";
  for (const m of app.models) {
    const b = document.createElement("button");
    b.setAttribute("role", "radio");
    b.dataset.id = m.id;
    b.innerHTML = `${m.short} <span class="sub">${m.n_classes} signs</span>`;
    b.onclick = () => selectModel(m.id);
    ui.modelPicker.appendChild(b);
  }
  $("footer-models").textContent = app.models.map((m) => `${m.name} (${m.citation})`).join("; ");
  const wanted = new URLSearchParams(location.search).get("model");
  await selectModel(app.models.some((m) => m.id === wanted) ? wanted : app.models[0].id);
}

async function selectModel(id) {
  const meta = app.models.find((m) => m.id === id);
  setStatus(`Loading ${meta.short} model…`);
  const base = `models/${id}/`;
  const [labels, cfg, session] = await Promise.all([
    fetch(base + "labels.json").then((r) => r.json()),
    fetch(base + "preprocess.json").then((r) => r.json()),
    ort.InferenceSession.create(base + meta.file, { executionProviders: ["wasm"], graphOptimizationLevel: "all" }),
  ]);
  app.model = { ...meta, labels, cfg, session };
  // warm-up run so that the first displayed latency is representative
  const F = featureCount(cfg);
  await session.run({
    x: new ort.Tensor("float32", new Float32Array(cfg.T * F), [1, cfg.T, F]),
    mask: new ort.Tensor("float32", new Float32Array(cfg.T).fill(1), [1, cfg.T]),
  });
  for (const b of ui.modelPicker.children) b.setAttribute("aria-checked", String(b.dataset.id === id));
  renderVocab();
  renderEasy();
  resetPrediction();
  await loadExamples(meta);
  const u = new URL(location);
  u.searchParams.set("model", id);
  history.replaceState(null, "", u);
  setStatus(app.holistic ? "Ready" : "Loading landmarker…", app.holistic ? "ready" : "loading");
}

async function loadHolistic() {
  const vision = await FilesetResolver.forVisionTasks(MP_WASM);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MP_MODEL, delegate },
    runningMode: "VIDEO",
    minHandLandmarksConfidence: 0.5,
  });
  try {
    app.holistic = await HolisticLandmarker.createFromOptions(vision, opts("GPU"));
  } catch {
    app.holistic = await HolisticLandmarker.createFromOptions(vision, opts("CPU"));
  }
}

function toUnified(res) {
  const frame = new Float64Array(L * 3).fill(NaN);
  const src = {
    left_hand: res.leftHandLandmarks?.[0],
    right_hand: res.rightHandLandmarks?.[0],
    pose: res.poseLandmarks?.[0],
    face: res.faceLandmarks?.[0],
  };
  let k = 0;
  for (const g of GROUPS) {
    const pts = src[g.source];
    for (const i of g.indices) {
      const p = pts?.[i];
      if (p) frame.set([p.x, p.y, p.z], k * 3);
      k++;
    }
  }
  return frame;
}

function currentYScale() {
  // MediaPipe normalises x by the width and y by the height of the frame
  if (app.fit) return app.exampleYScale ?? 1.0;
  const v = ui.video;
  return v.videoWidth ? v.videoHeight / v.videoWidth : 0.75;
}

async function classify(frames, source) {
  if (!app.model || frames.length < 2) return;
  const m = app.model;
  const t0 = performance.now();
  const { features, mask } = preprocess(toSequence(frames), m.cfg, currentYScale());
  const F = features.length / m.cfg.T;
  const out = await m.session.run({
    x: new ort.Tensor("float32", features, [1, m.cfg.T, F]),
    mask: new ort.Tensor("float32", mask, [1, m.cfg.T]),
  });
  app.modelMs = performance.now() - t0;
  ui.lat.textContent = app.fit
    ? `recorded landmarks · model ${app.modelMs.toFixed(1)} ms`
    : `landmarks ${app.mpMs.toFixed(0)} ms · model ${app.modelMs.toFixed(1)} ms`;
  const probs = out.probs.data;
  const order = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]).slice(0, 5);
  showPrediction(order.map((i) => ({ label: m.labels[i], p: probs[i] })), frames.length, source);
}

function resetPrediction() {
  ui.gloss.textContent = "—";
  ui.hint.textContent = "Sign something to see the model's guess.";
  ui.conf.textContent = "";
  ui.bars.innerHTML = "";
  ui.predMeta.textContent = "";
}

function showPrediction(top, nFrames, source) {
  const best = top[0];
  const sure = best.p >= 0.35;
  ui.pred.classList.toggle("uncertain", !sure);
  ui.gloss.textContent = best.label;
  ui.hint.textContent = sure ? `${source} · ${nFrames} frames` : "Low confidence, try again a bit slower";
  ui.conf.textContent = `${Math.round(best.p * 100)}%`;
  ui.predMeta.textContent = `${app.modelMs.toFixed(1)} ms`;
  ui.bars.innerHTML = top
    .map((t) => `<li><span class="name">${t.label}</span><span class="val">${(t.p * 100).toFixed(1)}%</span>
      <div class="track"><div class="fill" style="width:${(t.p * 100).toFixed(1)}%"></div></div></li>`)
    .join("");
  app.history.push({ label: best.label, p: best.p });
  renderHistory();
  highlightVocab(best.label);
}

function renderHistory() {
  ui.history.innerHTML = app.history.length
    ? app.history.slice(-24).map((h) => `<span class="${h.p < 0.35 ? "low" : ""}">${h.label}<small>${Math.round(h.p * 100)}%</small></span>`).join("")
    : '<span class="muted">Nothing yet.</span>';
}

function signLink(m, l, extra = "") {
  return m.dictionary
    ? `<a href="${m.dictionary.replace("{gloss}", encodeURIComponent(l.toLowerCase()))}" target="_blank" rel="noopener" data-l="${l}">${l}${extra}</a>`
    : `<span data-l="${l}">${l}${extra}</span>`;
}
function renderEasy() {
  const m = app.model;
  const easy = m.easy || [];
  ui.easyGrid.innerHTML = easy.map((e) => signLink(m, e.label, `<small>${Math.round(e.acc * 100)}%</small>`)).join("");
  ui.easyNote.textContent = easy.length
    ? `Best-recognised signs on unseen test signers (per-class accuracy).${m.dictionary ? " Click one to see how it is performed." : ""}`
    : "";
}
function renderVocab(filter = "") {
  const m = app.model;
  const q = filter.trim().toLowerCase();
  const items = m.labels.filter((l) => l.toLowerCase().includes(q));
  ui.vocabCount.textContent = `· ${m.labels.length}`;
  ui.vocabGrid.innerHTML = items.map((l) => signLink(m, l)).join("");
  ui.vocabNote.textContent = m.dictionary ? "Click a sign to see how it is performed (external dictionary)." : m.note || "";
}
function highlightVocab(label) {
  for (const el of [...ui.vocabGrid.children, ...ui.easyGrid.children]) el.classList.toggle("hit", el.dataset.l === label);
}

function drawOverlay(frame) {
  const c = ui.overlay;
  const v = ui.video;
  const W = (c.width = c.clientWidth * devicePixelRatio);
  const H = (c.height = c.clientHeight * devicePixelRatio);
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  if (!frame || (!ui.optSkeleton.checked && !app.fit)) return;
  let px;
  if (app.fit) {
    // replayed clip: fit its bounding box into the canvas
    const f = app.fit;
    const ys = app.exampleYScale ?? 1.0; // isotropic coordinates
    const s = 0.9 * Math.min(W / f.w, H / (f.h * ys));
    const ox = (W - f.w * s) / 2;
    const oy = (H - f.h * ys * s) / 2;
    px = (l) => [ox + (frame[l * 3] - f.x0) * s, oy + (frame[l * 3 + 1] - f.y0) * ys * s];
  } else {
    if (!v.videoWidth) return;
    // match object-fit: contain
    const s = Math.min(W / v.videoWidth, H / v.videoHeight);
    const ox = (W - v.videoWidth * s) / 2;
    const oy = (H - v.videoHeight * s) / 2;
    px = (l) => [ox + frame[l * 3] * v.videoWidth * s, oy + frame[l * 3 + 1] * v.videoHeight * s];
  }
  const colorOf = (l) =>
    l < SLICES.left_hand[1] ? cssVar("--left") : l < SLICES.right_hand[1] ? cssVar("--right") : l < SLICES.pose[1] ? "#c3c2b7" : "#e87ba4";
  ctx.lineCap = "round";
  ctx.lineWidth = 2.5 * devicePixelRatio;
  for (const [a, b] of EDGES) {
    if (Number.isNaN(frame[a * 3]) || Number.isNaN(frame[b * 3])) continue;
    const [x1, y1] = px(a);
    const [x2, y2] = px(b);
    ctx.strokeStyle = colorOf(a);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }
  ctx.fillStyle = "#ffffff";
  for (let l = 0; l < SLICES.pose[1]; l++) {
    if (Number.isNaN(frame[l * 3])) continue;
    const [x, y] = px(l);
    ctx.beginPath();
    ctx.arc(x, y, 2.2 * devicePixelRatio, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function onFrame(frame, now) {
  const visible = handsVisible(frame);
  app.frameTimes.push(now);
  while (app.frameTimes.length && now - app.frameTimes[0] > 1000) app.frameTimes.shift();

  if (app.fit) {
    // replayed example: classified as a whole at the end
  } else if (app.mode === "auto") {
    const r = app.segmenter.push(frame);
    ui.recFill.style.width = `${app.segmenter.progress() * 100}%`;
    if (app.segmenter.state === "signing") setBadge("Signing…", "signing");
    else if (!r.segment) setBadge(visible ? (app.segmenter.state === "cooldown" ? "Lower your hands or move on" : "Hands detected") : "Waiting for hands", "idle");
    if (r.segment) { setBadge("Recognised", "done"); classify(r.segment, `auto (${r.reason})`); }
  } else if (app.mode === "hold" && app.holdFrames) {
    app.holdFrames.push(frame);
    ui.recFill.style.width = `${Math.min(1, app.holdFrames.length / 120) * 100}%`;
  } else if (app.mode === "file" && app.fileFrames) {
    app.fileFrames.push(frame);
  }
  app.lastFrame = frame;
  drawOverlay(frame);
  if (app.fit) {
    ui.fps.textContent = "replay";
    ui.lat.textContent = `recorded landmarks · model ${app.modelMs.toFixed(1)} ms`;
  } else {
    ui.fps.textContent = `${app.frameTimes.length} fps`;
    ui.lat.textContent = `landmarks ${app.mpMs.toFixed(0)} ms · model ${app.modelMs.toFixed(1)} ms`;
  }
}

function detect(now) {
  const v = ui.video;
  const t0 = performance.now();
  // VIDEO mode needs strictly increasing timestamps
  app.tsCounter = Math.max(app.tsCounter + 1, Math.round(now));
  const res = app.holistic.detectForVideo(v, app.tsCounter);
  app.mpMs = performance.now() - t0;
  return toUnified(res);
}

function cameraLoop() {
  if (!app.running) return;
  const v = ui.video;
  if (v.readyState >= 2 && v.currentTime !== app.lastVideoTime) {
    app.lastVideoTime = v.currentTime;
    onFrame(detect(performance.now()), performance.now());
  }
  requestAnimationFrame(cameraLoop);
}

async function startCamera() {
  if (!app.holistic) return;
  stopFile();
  try {
    app.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false });
  } catch {
    setStatus("Camera permission denied", "error");
    return;
  }
  app.fit = null;
  app.replayId++;
  ui.video.srcObject = app.stream;
  await ui.video.play();
  ui.empty.hidden = true;
  ui.stage.classList.toggle("mirrored", ui.optMirror.checked);
  $("btn-camera").textContent = "Stop camera";
  app.running = true;
  app.segmenter.reset();
  if (app.mode === "file") setMode("auto");
  requestAnimationFrame(cameraLoop);
}

function stopCamera() {
  app.running = false;
  app.stream?.getTracks().forEach((t) => t.stop());
  app.stream = null;
  ui.video.srcObject = null;
  $("btn-camera").textContent = "Start camera";
  ui.empty.hidden = false;
  drawOverlay(null);
}

function stopFile() {
  app.fileFrames = null;
  if (ui.video.src) { ui.video.pause(); URL.revokeObjectURL(ui.video.src); ui.video.removeAttribute("src"); }
}

async function analyseFile(file) {
  if (!file || !app.holistic) return;
  stopCamera();
  setMode("file", false);
  ui.empty.hidden = true;
  ui.stage.classList.remove("mirrored");
  app.fit = null;
  app.replayId++;
  ui.video.src = URL.createObjectURL(file);
  ui.video.playbackRate = 1;
  app.fileFrames = [];
  setBadge("Analysing video…", "signing");
  const step = (now, meta) => {
    if (!app.fileFrames) return;
    onFrame(detect(meta.mediaTime * 1000), now);
    if (!ui.video.ended) ui.video.requestVideoFrameCallback(step);
  };
  ui.video.onended = async () => {
    const frames = app.fileFrames || [];
    setBadge("Recognised", "done");
    await classify(frames, `video file`);
    app.fileFrames = null;
  };
  ui.video.requestVideoFrameCallback(step);
  await ui.video.play();
}

async function loadExamples(meta) {
  app.examples = null;
  app.exampleOrder = null;
  if (meta.examples) {
    try {
      const ex = await (await fetch(meta.examples)).json();
      app.examples = ex.clips;
      app.exampleYScale = ex.y_scale ?? 1.0;
    } catch {
      app.examples = null;
    }
  }
  for (const id of ["btn-example", "btn-example-big"]) $(id).hidden = !app.examples?.length;
}

function clipBounds(clip) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < clip.T * L; i++) {
    const x = clip.data[i * 3], y = clip.data[i * 3 + 1];
    if (x === null || y === null) continue;
    x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  }
  return { x0, y0, w: Math.max(x1 - x0, 1e-3), h: Math.max(y1 - y0, 1e-3) };
}

async function playExample() {
  if (!app.examples?.length || !app.model) return;
  stopCamera();
  stopFile();
  // cycle through the examples in a random order, without repeating one before all were shown
  if (!app.exampleOrder?.length) {
    app.exampleOrder = app.examples.map((_, i) => i).sort(() => Math.random() - 0.5);
  }
  const clip = app.examples[app.exampleOrder.pop()];
  const id = ++app.replayId;
  resetPrediction();
  ui.hint.textContent = "Replaying…";
  ui.empty.hidden = true;
  ui.stage.classList.remove("mirrored");
  app.fit = clipBounds(clip);
  setBadge(`Replaying a clip of signer ${clip.signer}`, "signing");
  const frames = [];
  for (let t = 0; t < clip.T; t++) {
    if (id !== app.replayId) return;
    const f = Float64Array.from(clip.data.slice(t * L * 3, (t + 1) * L * 3), (v) => (v === null ? NaN : v));
    frames.push(f);
    onFrame(f, performance.now());
    await new Promise((r) => setTimeout(r, 1000 / clip.fps));
  }
  setBadge("Recognised", "done");
  await classify(frames, "example");
  const ok = ui.gloss.textContent === clip.label;
  ui.hint.textContent = `True sign: ${clip.label} ${ok ? "✓" : "✗"} · LSFB-ISOL test signer ${clip.signer}`;
}

function setMode(mode, sideEffects = true) {
  app.mode = mode;
  for (const b of ui.modePicker.children) b.setAttribute("aria-checked", String(b.dataset.mode === mode));
  ui.recFill.style.width = "0%";
  if (!sideEffects) return;
  if (mode === "file") ui.fileInput.click();
  else if (!app.running) startCamera();
  if (mode === "hold") setBadge("Hold Space while signing", "idle");
}

function holdStart() {
  if (app.mode !== "hold" || app.holdFrames || !app.running) return;
  app.holdFrames = [];
  setBadge("Recording…", "signing");
}
function holdEnd() {
  if (!app.holdFrames) return;
  const frames = app.holdFrames;
  app.holdFrames = null;
  ui.recFill.style.width = "0%";
  setBadge("Recognised", "done");
  classify(frames, "hold");
}

function bindEvents() {
  for (const b of ui.modePicker.children) b.onclick = () => setMode(b.dataset.mode);
  $("btn-camera").onclick = () => (app.running ? stopCamera() : startCamera());
  $("btn-camera-big").onclick = startCamera;
  $("btn-file-big").onclick = () => ui.fileInput.click();
  $("btn-example").onclick = playExample;
  $("btn-example-big").onclick = playExample;
  ui.fileInput.onchange = (e) => analyseFile(e.target.files[0]);
  $("btn-clear").onclick = () => { app.history = []; renderHistory(); };
  ui.optMirror.onchange = () => ui.stage.classList.toggle("mirrored", ui.optMirror.checked && app.running);
  ui.optSkeleton.onchange = () => drawOverlay(app.lastFrame);
  ui.vocabSearch.oninput = (e) => renderVocab(e.target.value);
  window.addEventListener("keydown", (e) => { if (e.code === "Space" && e.target === document.body) { e.preventDefault(); holdStart(); } });
  window.addEventListener("keyup", (e) => { if (e.code === "Space") holdEnd(); });
  ui.stage.addEventListener("pointerdown", holdStart);
  window.addEventListener("pointerup", holdEnd);
}

async function main() {
  bindEvents();
  try {
    await loadModelIndex();
    setStatus("Loading landmarker…");
    await loadHolistic();
    setStatus("Ready", "ready");
  } catch (e) {
    console.error(e);
    setStatus(e.message.slice(0, 80), "error");
  }
}

// Test hook: class probabilities of example clip i, computed exactly as in the live demo.
window.signrecExampleProbs = async (i) => {
  const clip = app.examples[i];
  const data = Float64Array.from(clip.data, (v) => (v === null ? NaN : v));
  const { features, mask } = preprocess({ T: clip.T, data }, app.model.cfg, app.exampleYScale ?? 1.0);
  const F = features.length / app.model.cfg.T;
  const out = await app.model.session.run({
    x: new ort.Tensor("float32", features, [1, app.model.cfg.T, F]),
    mask: new ort.Tensor("float32", mask, [1, app.model.cfg.T]),
  });
  return { label: clip.label, probs: Array.from(out.probs.data) };
};

main();
