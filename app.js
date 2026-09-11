/* LayerMagic — app.js v2
   Simplified: no modes. Click any object → extract it. Click a layer → select+drag. */

import {
  SamModel,
  AutoProcessor,
  RawImage,
  Tensor,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5";

/* ================= i18n ================= */
const STR = {
  ar: {
    tagline: "فصل الصور إلى طبقات بالذكاء الاصطناعي",
    engine: "المحرك",
    "status-idle": "النموذج غير محمّل بعد",
    "status-loading": "جارٍ تحميل النموذج...",
    "status-ready": "جاهز — ابدأ بالنقر على الصورة",
    "status-error": "تعذر تحميل النموذج",
    "engine-note": "يعمل داخل متصفحك بالكامل — لا يُرفع ملفك إلى أي سيرفر.",
    layers: "الطبقات",
    "layers-empty": "انقر على أي عنصر في الصورة لاستخراجه كطبقة.",
    export: "التصدير",
    "export-zip": "تنزيل كل الطبقات ZIP",
    "export-composite": "تنزيل الصورة المركبة",
    open: "فتح صورة",
    demo: "صورة تجريبية",
    clear: "مسح الطبقات",
    "dz-title": "اسحب صورتك هنا",
    "dz-sub": "أو اضغط الزر بالأسفل — JPG / PNG / WebP — كل شيء يعمل محلياً",
    "choose-image": "اختر صورة",
    "try-sample": "جرّب الصورة التجريبية",
    "hint-1": "💡 انقر على أي عنصر في الصورة لاستخراجه كطبقة",
    "hint-2": "💡 اسحب الطبقات لتحريكها — عجلة الفأرة للتكبير",
    "busy-model": "تحميل نموذج الذكاء الاصطناعي (مرة واحدة فقط)...",
    "busy-embed": "تحليل الصورة...",
    "busy-click": "استخراج العنصر...",
    "layer-full": "الصورة كاملة",
    layer: "عنصر",
    "confirm-clear": "مسح كل الطبقات؟",
    "mask-empty": "لم يُعثر على عنصر هنا — جرّب نقطة أخرى",
    transform: "تحكم بالعنصر",
    "tf-scale": "التكبير",
    "tf-rot": "التدوير",
    "tf-opacity": "الشفافية",
    "tf-up": "للأمام",
    "tf-down": "للخلف",
    "tf-reset": "إعادة",
  },
  en: {
    tagline: "Split images into AI layers, in your browser",
    engine: "Engine",
    "status-idle": "Model not loaded yet",
    "status-loading": "Loading model...",
    "status-ready": "Ready — start clicking the image",
    "status-error": "Failed to load model",
    "engine-note": "Runs 100% in your browser — your file never leaves your device.",
    layers: "Layers",
    "layers-empty": "Click any object in the image to extract it as a layer.",
    export: "Export",
    "export-zip": "Download all layers (ZIP)",
    "export-composite": "Download composite",
    open: "Open image",
    demo: "Sample image",
    clear: "Clear layers",
    "dz-title": "Drop your image here",
    "dz-sub": "or pick below — JPG / PNG / WebP — everything stays local",
    "choose-image": "Choose image",
    "try-sample": "Try the sample image",
    "hint-1": "💡 Click any object in the image to extract it as a layer",
    "hint-2": "💡 Drag layers to move them — mouse wheel to scale",
    "busy-model": "Loading AI model (one time only)...",
    "busy-embed": "Analyzing image...",
    "busy-click": "Extracting element...",
    "layer-full": "Full image",
    layer: "Object",
    "confirm-clear": "Clear all layers?",
    "mask-empty": "No object found there — try another spot",
    transform: "Element control",
    "tf-scale": "Scale",
    "tf-rot": "Rotate",
    "tf-opacity": "Opacity",
    "tf-up": "Forward",
    "tf-down": "Backward",
    "tf-reset": "Reset",
  },
};
let lang = "ar";
const t = (k) => STR[lang][k] ?? k;

function applyStaticLang() {
  document.documentElement.lang = lang;
  document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const v = STR[lang][el.dataset.i18n];
    if (v != null) el.textContent = v;
  });
  document.getElementById("btn-lang").textContent = lang === "ar" ? "EN" : "ع";
  renderLayers();
  renderTransformPanel();
}

/* ================= state ================= */
const $ = (id) => document.getElementById(id);
const view = $("view");
const vctx = view.getContext("2d");

const state = {
  model: null,
  processor: null,
  ready: false,
  srcCanvas: null,
  srcImageData: null,
  rawImage: null,
  imageInputs: null,
  imageEmb: null,
  layers: [], // {id,name,color,visible,canvas,area, dx,dy,scale,rot,opacity}
  nextId: 1,
  selected: null,
  drag: null, // {mx,my,lx,ly} when dragging
  busy: false,
};
window.LM = state;

const PALETTE = ["#10b981", "#0ea5e9", "#f59e0b", "#ef4444", "#8b5cf6", "#14b8a6", "#f97316", "#ec4899", "#84cc16", "#6366f1"];

/* ================= helpers ================= */
function setStatus(kind, textKey) {
  const dot = $("status-dot");
  dot.className = "dot " + (kind || "");
  $("status-text").textContent = t(textKey);
}

function showBusy(textKey, detail = "") {
  state.busy = true;
  $("busy").classList.remove("hidden");
  $("busy-text").textContent = t(textKey) + (detail ? " " + detail : "");
}
function setBusyText(txt) { $("busy-text").textContent = txt; }
function hideBusy() {
  state.busy = false;
  $("busy").classList.add("hidden");
}

function setButtons() {
  const hasImg = !!state.srcCanvas;
  const hasLayers = state.layers.length > 0;
  $("btn-clear").disabled = !hasLayers || state.busy;
  $("btn-zip").disabled = !hasLayers;
  $("btn-composite").disabled = !hasLayers;
  view.classList.toggle("on-canvas", hasImg);
}

/* ================= model ================= */
async function ensureModel() {
  if (state.ready) return;
  if (!state.model) {
    setStatus("loading", "status-loading");
    showBusy("busy-model");
    const files = {};
    const progress_callback = (p) => {
      if (p.status === "progress" && p.file) {
        files[p.file] = p.progress || 0;
        const vals = Object.values(files);
        const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
        $("progress-wrap").classList.remove("hidden");
        $("progress-label").textContent = `${p.file} — ${Math.round(avg)}%`;
        $("progress-fill").style.width = avg + "%";
      }
    };
    try {
      state.model = await SamModel.from_pretrained("Xenova/slimsam-77-uniform", { progress_callback });
      state.processor = await AutoProcessor.from_pretrained("Xenova/slimsam-77-uniform", { progress_callback });
      state.ready = true;
      setStatus("ready", "status-ready");
      $("progress-wrap").classList.add("hidden");
    } catch (e) {
      console.error(e);
      setStatus("error", "status-error");
      state.model = null;
      hideBusy();
      throw e;
    }
    hideBusy();
  }
}

/* ================= image load ================= */
async function loadFromSource(src) {
  try {
    await ensureModel();
    showBusy("busy-embed");

    const img = new Image();
    img.crossOrigin = "anonymous";
    await new Promise((res, rej) => {
      img.onload = res;
      img.onerror = rej;
      img.src = src;
    });

    const MAX = 1024;
    const scale = Math.min(1, MAX / Math.max(img.naturalWidth, img.naturalHeight));
    const w = Math.max(2, Math.round(img.naturalWidth * scale));
    const h = Math.max(2, Math.round(img.naturalHeight * scale));
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    c.getContext("2d").drawImage(img, 0, 0, w, h);
    state.srcCanvas = c;
    state.srcImageData = c.getContext("2d").getImageData(0, 0, w, h);

    const pixelData = new Uint8ClampedArray(state.srcImageData.data);
    state.rawImage = new RawImage(pixelData, w, h, 4);
    state.imageInputs = await state.processor(state.rawImage);
    state.imageEmb = null;
    state.layers = [];
    state.selected = null;

    view.width = w; view.height = h;
    view.classList.add("on");
    $("dropzone").classList.add("hidden");
    redraw();
    hideBusy();
    setButtons();
    renderLayers();
    renderTransformPanel();
  } catch (e) {
    console.error(e);
    hideBusy();
  }
}

function fileToDataURL(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.onerror = rej;
    r.readAsDataURL(file);
  });
}

/* ================= SAM core ================= */
async function getImageEmb() {
  if (!state.imageEmb) {
    state.imageEmb = await state.model.get_image_embeddings(state.imageInputs);
  }
  return state.imageEmb;
}

async function segmentAt(points) {
  const emb = await getImageEmb();
  const [oh, ow] = state.imageInputs.original_sizes[0];
  const [rh, rw] = state.imageInputs.reshaped_input_sizes[0];
  const scaled = points.map(([x, y]) => [(x * rw) / ow, (y * rh) / oh]);
  // SAM expects rank-4 input_points: [batch, point_batch, points_per_batch, 2]
  const input_points = new Tensor("float32", Float32Array.from(scaled.flat()), [1, 1, points.length, 2]);
  const input_labels = new Tensor("int64", BigInt64Array.from(points.map(() => 1n)), [1, 1, points.length]);

  const outputs = await state.model({
    ...state.imageInputs,
    ...emb,
    input_points,
    input_labels,
  });

  const scores = outputs.iou_scores.data;
  let best = 0;
  for (let i = 1; i < 3; i++) if (scores[i] > scores[best]) best = i;

  const masks = await state.processor.post_process_masks(
    outputs.pred_masks,
    state.imageInputs.original_sizes,
    state.imageInputs.reshaped_input_sizes
  );
  const m = masks[0];
  const [,, maskH, maskW] = m.dims;
  const plane = best * maskW * maskH;
  const data = m.data;
  const mask = new Uint8Array(maskW * maskH);
  let area = 0;
  for (let i = 0; i < mask.length; i++) {
    if (data[plane + i]) { mask[i] = 1; area++; }
  }
  return { mask, w: maskW, h: maskH, area, score: scores[best] };
}

function layerCanvasFromMask(seg) {
  const w = seg.w, h = seg.h;
  const src = state.srcImageData.data;
  const out = new ImageData(w, h);
  const o = out.data;
  for (let i = 0, p = 0; i < seg.mask.length; i++, p += 4) {
    if (seg.mask[i]) {
      o[p] = src[p]; o[p + 1] = src[p + 1]; o[p + 2] = src[p + 2]; o[p + 3] = 255;
    }
  }
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  c.getContext("2d").putImageData(out, 0, 0);
  return c;
}

/* ================= layers ================= */
function addLayer(canvas, name, area) {
  const idx = state.layers.length;
  state.layers.push({
    id: state.nextId++,
    name,
    color: PALETTE[idx % PALETTE.length],
    visible: true,
    canvas,
    area: area ?? canvas.width * canvas.height,
    dx: 0, dy: 0, scale: 1, rot: 0, opacity: 1,
  });
}

function renderLayers() {
  const list = $("layers-list");
  list.innerHTML = "";
  $("layers-count").textContent = state.layers.length;
  $("layers-empty").style.display = state.layers.length ? "none" : "block";
  state.layers.forEach((L, i) => {
    const li = document.createElement("li");
    li.className = "layer-row" + (L.visible ? "" : " off") + (state.selected === i ? " selected" : "");
    li.innerHTML = `
      <span class="layer-chip" style="background:${L.color}"></span>
      <span class="layer-name"></span>
      <span class="layer-meta">${Math.round((L.area / (view.width * view.height)) * 100)}%</span>
      <button class="icon-btn" data-act="eye" title="show/hide">${L.visible ? "◉" : "○"}</button>
      <button class="icon-btn" data-act="dl" title="download PNG">⭳</button>
      <button class="icon-btn del" data-act="del" title="delete">✕</button>`;
    li.querySelector(".layer-name").textContent = L.name;
    li.querySelector('[data-act="eye"]').onclick = () => { L.visible = !L.visible; renderLayers(); redraw(); };
    li.querySelector('[data-act="dl"]').onclick = () => downloadCanvas(L.canvas, `layermagic_${i + 1}.png`);
    li.querySelector('[data-act="del"]').onclick = () => {
      state.layers.splice(i, 1);
      state.layers.forEach((l2, j) => { l2.color = PALETTE[j % PALETTE.length]; });
      if (state.selected === i) state.selected = null;
      else if (state.selected !== null && state.selected > i) state.selected--;
      renderLayers(); renderTransformPanel(); redraw(); setButtons();
    };
    li.onclick = (ev) => {
      if (ev.target.closest("button")) return;
      selectLayer(i);
    };
    list.appendChild(li);
  });
}

function selectLayer(i) {
  state.selected = i;
  renderLayers();
  renderTransformPanel();
  redraw();
}

function downloadCanvas(canvas, filename) {
  canvas.toBlob((b) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b);
    a.download = filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }, "image/png");
}

/* ================= render ================= */
function redraw() {
  if (!state.srcCanvas) return;
  const w = view.width, h = view.height;
  vctx.clearRect(0, 0, w, h);

  if (!state.layers.length) {
    vctx.drawImage(state.srcCanvas, 0, 0);
    return;
  }

  state.layers.forEach((L) => {
    if (!L.visible) return;
    vctx.save();
    vctx.globalAlpha = L.opacity;
    const cx = L.dx + L.canvas.width / 2;
    const cy = L.dy + L.canvas.height / 2;
    vctx.translate(cx, cy);
    vctx.rotate((L.rot * Math.PI) / 180);
    vctx.scale(L.scale, L.scale);
    vctx.drawImage(L.canvas, -L.canvas.width / 2, -L.canvas.height / 2);
    vctx.restore();
  });

  // selection outline
  if (state.selected !== null && state.layers[state.selected]) {
    const L = state.layers[state.selected];
    if (L.visible) {
      vctx.save();
      vctx.strokeStyle = "#10b981";
      vctx.lineWidth = 2;
      vctx.setLineDash([8, 5]);
      const cx = L.dx + L.canvas.width / 2;
      const cy = L.dy + L.canvas.height / 2;
      vctx.translate(cx, cy);
      vctx.rotate((L.rot * Math.PI) / 180);
      vctx.scale(L.scale, L.scale);
      vctx.strokeRect(-L.canvas.width / 2, -L.canvas.height / 2, L.canvas.width, L.canvas.height);
      vctx.fillStyle = "#10b981";
      vctx.setLineDash([]);
      const hs = 6 / L.scale;
      [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(([sx, sy]) => {
        vctx.fillRect(sx * L.canvas.width / 2 - hs, sy * L.canvas.height / 2 - hs, hs * 2, hs * 2);
      });
      vctx.restore();
    }
  }
}

/* ================= transform panel ================= */
function renderTransformPanel() {
  const card = $("transform-card");
  if (state.selected === null || !state.layers[state.selected]) {
    card.classList.add("hidden");
    return;
  }
  card.classList.remove("hidden");
  const L = state.layers[state.selected];
  $("transform-idx").textContent = "#" + (state.selected + 1);
  $("tf-scale").value = L.scale;
  $("tf-scale-val").textContent = L.scale.toFixed(2) + "×";
  $("tf-rot").value = L.rot;
  $("tf-rot-val").textContent = Math.round(L.rot) + "°";
  $("tf-opacity").value = L.opacity;
  $("tf-opacity-val").textContent = Math.round(L.opacity * 100) + "%";
}

function wireTransformPanel() {
  $("tf-scale").addEventListener("input", (e) => {
    if (state.selected === null) return;
    state.layers[state.selected].scale = parseFloat(e.target.value);
    $("tf-scale-val").textContent = state.layers[state.selected].scale.toFixed(2) + "×";
    redraw();
  });
  $("tf-rot").addEventListener("input", (e) => {
    if (state.selected === null) return;
    state.layers[state.selected].rot = parseFloat(e.target.value);
    $("tf-rot-val").textContent = Math.round(state.layers[state.selected].rot) + "°";
    redraw();
  });
  $("tf-opacity").addEventListener("input", (e) => {
    if (state.selected === null) return;
    state.layers[state.selected].opacity = parseFloat(e.target.value);
    $("tf-opacity-val").textContent = Math.round(state.layers[state.selected].opacity * 100) + "%";
    redraw();
  });
  $("tf-up").addEventListener("click", () => moveLayer(1));
  $("tf-down").addEventListener("click", () => moveLayer(-1));
  $("tf-reset").addEventListener("click", () => {
    if (state.selected === null) return;
    const L = state.layers[state.selected];
    L.dx = 0; L.dy = 0; L.scale = 1; L.rot = 0; L.opacity = 1;
    renderTransformPanel();
    redraw();
  });
}

function moveLayer(dir) {
  if (state.selected === null) return;
  const i = state.selected;
  const j = i + dir;
  if (j < 0 || j >= state.layers.length) return;
  const tmp = state.layers[i];
  state.layers[i] = state.layers[j];
  state.layers[j] = tmp;
  state.layers.forEach((l2, k) => { l2.color = PALETTE[k % PALETTE.length]; });
  state.selected = j;
  renderLayers();
  renderTransformPanel();
  redraw();
}

/* ================= unified click + drag ================= */
function canvasCoords(ev) {
  const r = view.getBoundingClientRect();
  const x = ((ev.clientX - r.left) * view.width) / r.width;
  const y = ((ev.clientY - r.top) * view.height) / r.height;
  return { x, y };
}

const _alphaCache = new WeakMap();
function getAlphaMask(L) {
  if (_alphaCache.has(L.canvas)) return _alphaCache.get(L.canvas);
  const ctx = L.canvas.getContext("2d");
  const d = ctx.getImageData(0, 0, L.canvas.width, L.canvas.height).data;
  const mask = new Uint8Array(L.canvas.width * L.canvas.height);
  for (let i = 0, p = 3; i < mask.length; i++, p += 4) mask[i] = d[p];
  _alphaCache.set(L.canvas, mask);
  return mask;
}

function hitTest(x, y) {
  for (let i = state.layers.length - 1; i >= 0; i--) {
    const L = state.layers[i];
    if (!L.visible || L.opacity < 0.05) continue;
    const cx = L.dx + L.canvas.width / 2;
    const cy = L.dy + L.canvas.height / 2;
    let px = x - cx;
    let py = y - cy;
    const rad = (-L.rot * Math.PI) / 180;
    const rx = px * Math.cos(rad) - py * Math.sin(rad);
    const ry = px * Math.sin(rad) + py * Math.cos(rad);
    const lx = rx / L.scale + L.canvas.width / 2;
    const ly = ry / L.scale + L.canvas.height / 2;
    if (lx < 0 || ly < 0 || lx >= L.canvas.width || ly >= L.canvas.height) continue;
    const mask = getAlphaMask(L);
    if (mask[Math.floor(ly) * L.canvas.width + Math.floor(lx)] > 20) return i;
  }
  return -1;
}

// The core unified handler: click anywhere on the image.
// 1. If hit a layer → select + start drag
// 2. If missed → run SAM to extract a new layer at that point
view.addEventListener("mousedown", async (ev) => {
  if (!state.srcCanvas || state.busy) return;
  const { x, y } = canvasCoords(ev);
  if (x < 0 || y < 0 || x >= view.width || y >= view.height) return;

  const idx = hitTest(x, y);
  if (idx >= 0) {
    // SELECT + DRAG existing layer
    state.selected = idx;
    const L = state.layers[idx];
    state.drag = { mx: x, my: y, lx: L.dx, ly: L.dy };
    view.classList.add("dragging");
    renderLayers();
    renderTransformPanel();
    redraw();
  } else {
    // EXTRACT new layer with SAM
    state.selected = null;
    renderLayers();
    renderTransformPanel();
    redraw();
    try {
      showBusy("busy-click");
      await ensureModel();
      const seg = await segmentAt([[x, y]]);
      const ratio = seg.area / (view.width * view.height);
      if (ratio < 0.0005 || ratio > 0.98) {
        setBusyText(t("mask-empty"));
        setTimeout(hideBusy, 900);
        return;
      }
      addLayer(layerCanvasFromMask(seg), `${t("layer")} ${state.layers.length}`, seg.area);
      // auto-select the new layer so the user can immediately drag/scale it
      state.selected = state.layers.length - 1;
      renderLayers(); renderTransformPanel(); redraw();
      hideBusy(); setButtons();
    } catch (e) {
      console.error(e);
      hideBusy();
    }
  }
});

window.addEventListener("mousemove", (ev) => {
  if (!state.drag || state.selected === null) return;
  const { x, y } = canvasCoords(ev);
  const L = state.layers[state.selected];
  L.dx = state.drag.lx + (x - state.drag.mx);
  L.dy = state.drag.ly + (y - state.drag.my);
  redraw();
});

window.addEventListener("mouseup", () => {
  if (state.drag) {
    state.drag = null;
    view.classList.remove("dragging");
  }
});

// touch support
view.addEventListener("touchstart", async (ev) => {
  if (!state.srcCanvas || state.busy) return;
  if (!ev.touches[0]) return;
  ev.preventDefault();
  const tch = ev.touches[0];
  const { x, y } = canvasCoords({ clientX: tch.clientX, clientY: tch.clientY });
  if (x < 0 || y < 0 || x >= view.width || y >= view.height) return;

  const idx = hitTest(x, y);
  if (idx >= 0) {
    state.selected = idx;
    const L = state.layers[idx];
    state.drag = { mx: x, my: y, lx: L.dx, ly: L.dy };
    renderLayers();
    renderTransformPanel();
    redraw();
  } else {
    state.selected = null;
    renderLayers();
    renderTransformPanel();
    redraw();
    try {
      showBusy("busy-click");
      await ensureModel();
      const seg = await segmentAt([[x, y]]);
      const ratio = seg.area / (view.width * view.height);
      if (ratio < 0.0005 || ratio > 0.98) {
        setBusyText(t("mask-empty"));
        setTimeout(hideBusy, 900);
        return;
      }
      addLayer(layerCanvasFromMask(seg), `${t("layer")} ${state.layers.length}`, seg.area);
      state.selected = state.layers.length - 1;
      renderLayers(); renderTransformPanel(); redraw();
      hideBusy(); setButtons();
    } catch (e) {
      console.error(e);
      hideBusy();
    }
  }
}, { passive: false });

view.addEventListener("touchmove", (ev) => {
  if (!state.drag || state.selected === null) return;
  if (!ev.touches[0]) return;
  ev.preventDefault();
  const tch = ev.touches[0];
  const { x, y } = canvasCoords({ clientX: tch.clientX, clientY: tch.clientY });
  const L = state.layers[state.selected];
  L.dx = state.drag.lx + (x - state.drag.mx);
  L.dy = state.drag.ly + (y - state.drag.my);
  redraw();
}, { passive: false });

view.addEventListener("touchend", () => { state.drag = null; });

// wheel to scale selected layer
view.addEventListener("wheel", (ev) => {
  if (state.selected === null) return;
  ev.preventDefault();
  const L = state.layers[state.selected];
  const delta = ev.deltaY > 0 ? 0.95 : 1.05;
  L.scale = Math.max(0.1, Math.min(3, L.scale * delta));
  renderTransformPanel();
  redraw();
}, { passive: false });

// click empty area to deselect
view.addEventListener("contextmenu", (ev) => {
  ev.preventDefault();
  state.selected = null;
  renderLayers();
  renderTransformPanel();
  redraw();
});

/* ================= exports ================= */
function renderComposite(targetCanvas) {
  targetCanvas.width = view.width;
  targetCanvas.height = view.height;
  const cc = targetCanvas.getContext("2d");
  cc.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
  state.layers.forEach((L) => {
    if (!L.visible) return;
    cc.save();
    cc.globalAlpha = L.opacity;
    const cx = L.dx + L.canvas.width / 2;
    const cy = L.dy + L.canvas.height / 2;
    cc.translate(cx, cy);
    cc.rotate((L.rot * Math.PI) / 180);
    cc.scale(L.scale, L.scale);
    cc.drawImage(L.canvas, -L.canvas.width / 2, -L.canvas.height / 2);
    cc.restore();
  });
  return targetCanvas;
}

$("btn-zip").addEventListener("click", async () => {
  if (!state.layers.length) return;
  if (typeof JSZip === "undefined") { alert("JSZip not loaded"); return; }
  const zip = new JSZip();
  const folder = zip.folder("layers");
  state.layers.forEach((L, i) => {
    const b64 = L.canvas.toDataURL("image/png").split(",")[1];
    folder.file(`${String(i + 1).padStart(2, "0")}_${L.name.replace(/\s+/g, "_")}.png`, b64, { base64: true });
  });
  const comp = renderComposite(document.createElement("canvas"));
  zip.file("composite.png", comp.toDataURL("image/png").split(",")[1], { base64: true });
  const blob = await zip.generateAsync({ type: "blob" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "layermagic_layers.zip";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
});

$("btn-composite").addEventListener("click", () => {
  const comp = renderComposite(document.createElement("canvas"));
  downloadCanvas(comp, "layermagic_composite.png");
});

/* ================= toolbar wiring ================= */
const openFile = () => $("file-input").click();
$("btn-open").addEventListener("click", openFile);
$("btn-open-2").addEventListener("click", openFile);
$("file-input").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (f) await loadFromSource(await fileToDataURL(f));
  e.target.value = "";
});

const openDemo = () => loadFromSource("sample.jpg");
$("btn-demo").addEventListener("click", openDemo);
$("btn-demo-2").addEventListener("click", openDemo);

$("btn-clear").addEventListener("click", () => {
  if (!state.layers.length) return;
  if (!confirm(t("confirm-clear"))) return;
  state.layers = [];
  state.selected = null;
  renderLayers(); renderTransformPanel(); redraw(); setButtons();
});

$("btn-lang").addEventListener("click", () => {
  lang = lang === "ar" ? "en" : "ar";
  applyStaticLang();
});

/* ================= drag & drop ================= */
const dz = $("dropzone");
["dragenter", "dragover"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); })
);
["dragleave", "drop"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); })
);
dz.addEventListener("drop", async (e) => {
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) await loadFromSource(await fileToDataURL(f));
});

/* ================= init ================= */
applyStaticLang();
wireTransformPanel();
setButtons();
setStatus("idle", "status-idle");
