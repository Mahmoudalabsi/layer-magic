/* LayerMagic — app.js v1
   AI image → layers, fully in-browser (Transformers.js + SlimSAM). */

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
    "status-ready": "جاهز — SAM داخل المتصفح",
    "status-error": "تعذر تحميل النموذج",
    "engine-note": "يعمل داخل متصفحك بالكامل — لا يُرفع ملفك إلى أي سيرفر.",
    layers: "الطبقات",
    "layers-empty": "لا طبقات بعد — حمّل صورة ثم اضغط «فصل تلقائي» أو انقر على أي عنصر.",
    export: "التصدير",
    "export-zip": "تنزيل كل الطبقات ZIP",
    "export-composite": "تنزيل الصورة المركبة",
    open: "فتح صورة",
    demo: "صورة تجريبية",
    auto: "فصل تلقائي",
    "click-mode": "وضع النقر",
    "edit-mode": "تحرير العناصر",
    parallax: "بارالاكس",
    clear: "مسح الطبقات",
    transform: "تحكم بالعنصر",
    "tf-scale": "التكبير",
    "tf-rot": "التدوير",
    "tf-opacity": "الشفافية",
    "tf-up": "للأمام",
    "tf-down": "للخلف",
    "tf-reset": "إعادة",
    "dz-title": "اسحب صورتك هنا",
    "dz-sub": "أو اضغط الزر بالأسفل — JPG / PNG / WebP — كل شيء يعمل محلياً",
    "choose-image": "اختر صورة",
    "try-sample": "جرّب الصورة التجريبية",
    "hint-auto": "«فصل تلقائي»: يقسم الصورة كاملة إلى طبقات كائنات تلقائياً",
    "hint-click": "«وضع النقر»: انقر أي عنصر لاستخراجه كطبقة مستقلة",
    "hint-edit": "«تحرير العناصر»: اسحب/كبّر/أدر الطبقات بالفأرة — عجلة الفأرة للتكبير",
    "busy-model": "تحميل نموذج الذكاء الاصطناعي (مرة واحدة فقط)...",
    "busy-embed": "تحليل الصورة...",
    "busy-auto": "فصل تلقائي جارٍ — نقطة",
    "busy-click": "استخراج الطبقة...",
    "layer-full": "الصورة كاملة",
    layer: "كائن",
    "done-auto": "تم إنشاء الطبقات",
    "done-click": "أُضيفت طبقة جديدة",
    "need-image": "افتح صورة أولاً",
    "no-layers": "لا توجد طبقات للتصدير",
    "confirm-clear": "مسح كل الطبقات؟",
    "export-done": "تم التنزيل",
    "mask-empty": "لم يُعثر على عنصر هنا — جرّب نقطة أخرى",
  },
  en: {
    tagline: "Split images into AI layers, in your browser",
    engine: "Engine",
    "status-idle": "Model not loaded yet",
    "status-loading": "Loading model...",
    "status-ready": "Ready — SAM in-browser",
    "status-error": "Failed to load model",
    "engine-note": "Runs 100% in your browser — your file never leaves your device.",
    layers: "Layers",
    "layers-empty": "No layers yet — load an image then press Auto-split, or click any object.",
    export: "Export",
    "export-zip": "Download all layers (ZIP)",
    "export-composite": "Download composite",
    open: "Open image",
    demo: "Sample image",
    auto: "Auto-split",
    "click-mode": "Click mode",
    "edit-mode": "Edit elements",
    parallax: "Parallax",
    clear: "Clear layers",
    transform: "Element control",
    "tf-scale": "Scale",
    "tf-rot": "Rotate",
    "tf-opacity": "Opacity",
    "tf-up": "Forward",
    "tf-down": "Backward",
    "tf-reset": "Reset",
    "dz-title": "Drop your image here",
    "dz-sub": "or pick below — JPG / PNG / WebP — everything stays local",
    "choose-image": "Choose image",
    "try-sample": "Try the sample image",
    "hint-auto": "Auto-split: divides the whole image into object layers automatically",
    "hint-click": "Click mode: click any object to extract it as its own layer",
    "hint-edit": "Edit elements: drag/scale/rotate layers with the mouse — wheel to zoom",
    "busy-model": "Loading AI model (one time only)...",
    "busy-embed": "Analyzing image...",
    "busy-auto": "Auto-splitting — point",
    "busy-click": "Extracting layer...",
    "layer-full": "Full image",
    layer: "Object",
    "done-auto": "Layers created",
    "done-click": "Layer added",
    "need-image": "Open an image first",
    "no-layers": "No layers to export",
    "confirm-clear": "Clear all layers?",
    "export-done": "Downloaded",
    "mask-empty": "No object found there — try another spot",
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
  srcCanvas: null, // working-resolution source
  srcImageData: null,
  rawImage: null,
  imageInputs: null, // processor outputs (pixel_values, original_sizes, reshaped_input_sizes)
  imageEmb: null, // cached embeddings
  layers: [], // {id,name,color,visible,canvas,area, dx,dy,scale,rot,opacity}
  nextId: 1,
  clickMode: false,
  editMode: false,
  selected: null, // index into state.layers, or null
  drag: null, // {mx,my,lx,ly} when dragging
  parallax: false,
  mouse: null, // {x,y} in canvas coords
  busy: false,
};
window.LM = state; // debug/testing handle

/* palette for layer chips */
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
function setBusyText(txt) {
  $("busy-text").textContent = txt;
}
function hideBusy() {
  state.busy = false;
  $("busy").classList.add("hidden");
}

function setButtons() {
  const hasImg = !!state.srcCanvas;
  const hasLayers = state.layers.length > 0;
  $("btn-auto").disabled = !hasImg || state.busy;
  $("btn-click").disabled = !hasImg || state.busy;
  $("btn-edit").disabled = !hasLayers || state.busy;
  $("btn-parallax").disabled = !hasLayers;
  $("btn-clear").disabled = !hasLayers;
  $("btn-zip").disabled = !hasLayers;
  $("btn-composite").disabled = !hasLayers;
  $("btn-click").classList.toggle("active", state.clickMode);
  $("btn-edit").classList.toggle("active", state.editMode);
  $("btn-parallax").classList.toggle("active", state.parallax);
  view.classList.toggle("click-mode", state.clickMode && hasImg);
  view.classList.toggle("edit-mode", state.editMode && hasLayers);
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
async function loadFromSource(src, name) {
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
    state.imageEmb = null; // computed lazily on first use

    view.width = w; view.height = h;
    view.classList.add("on");
    $("dropzone").classList.add("hidden");
    redraw();
    hideBusy();
    setButtons();
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

/* points: array of [x,y] in SOURCE pixel coords; returns {mask: Uint8Array, w, h, area, score} or null */
async function segmentAt(points) {
  const emb = await getImageEmb();
  const [oh, ow] = state.imageInputs.original_sizes[0];
  const [rh, rw] = state.imageInputs.reshaped_input_sizes[0];
  const scaled = points.map(([x, y]) => [(x * rw) / ow, (y * rh) / oh]);
  // SAM expects rank-4 input_points: [batch, point_batch, points_per_batch, 2]
  const input_points = new Tensor("float32", Float32Array.from(scaled.flat()), [1, 1, points.length, 2]);
  // and rank-3 input_labels: [batch, point_batch, points_per_batch]
  const input_labels = new Tensor("int64", BigInt64Array.from(points.map(() => 1n)), [1, 1, points.length]);

  const outputs = await state.model({
    ...state.imageInputs,
    ...emb,
    input_points,
    input_labels,
  });

  const scores = outputs.iou_scores.data; // Float32Array(3)
  let best = 0;
  for (let i = 1; i < 3; i++) if (scores[i] > scores[best]) best = i;

  const masks = await state.processor.post_process_masks(
    outputs.pred_masks,
    state.imageInputs.original_sizes,
    state.imageInputs.reshaped_input_sizes
  );
  const m = masks[0]; // Tensor [1,3,h,w]  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width  → dims[2]=height, dims[3]=width
  const [,, maskH, maskW] = m.dims; // upscaled to ORIGINAL image size (h, w)
  const plane = best * maskW * maskH;
  const data = m.data; // 0/1
  const mask = new Uint8Array(maskW * maskH);
  let area = 0;
  for (let i = 0; i < mask.length; i++) {
    if (data[plane + i]) { mask[i] = 1; area++; }
  }
  return { mask, w: maskW, h: maskH, area, score: scores[best] };
}

/* build a layer canvas from a mask */
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

function fullImageLayer() {
  const c = document.createElement("canvas");
  c.width = state.srcCanvas.width; c.height = state.srcCanvas.height;
  c.getContext("2d").putImageData(state.srcImageData, 0, 0);
  return c;
}

/* IoU on coarse 64-grid */
function coarseBits(seg, G = 64) {
  const bits = new Uint8Array(G * G);
  const gw = Math.ceil(seg.w / G), gh = Math.ceil(seg.h / G);
  for (let gy = 0; gy < G; gy++) {
    for (let gx = 0; gx < G; gx++) {
      const x0 = gx * gw, y0 = gy * gh;
      let hit = 0;
      for (let y = y0; y < Math.min(y0 + gh, seg.h) && !hit; y += Math.max(1, gh >> 2)) {
        for (let x = x0; x < Math.min(x0 + gw, seg.w); x += Math.max(1, gw >> 2)) {
          if (seg.mask[y * seg.w + x]) { hit = 1; break; }
        }
      }
      bits[gy * G + gx] = hit;
    }
  }
  return bits;
}
function iouBits(a, b) {
  let inter = 0, uni = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] & b[i]) inter++;
    if (a[i] | b[i]) uni++;
  }
  return uni ? inter / uni : 0;
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
    // interactive transform — default = identity (no change from extracted position)
    dx: 0,
    dy: 0,
    scale: 1,
    rot: 0,
    opacity: 1,
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
    li.querySelector('[data-act="dl"]').onclick = () => downloadCanvas(L.canvas, `${baseName()}_${i + 1}.png`);
    li.querySelector('[data-act="del"]').onclick = () => {
      state.layers.splice(i, 1);
      state.layers.forEach((l2, j) => { l2.color = PALETTE[j % PALETTE.length]; });
      if (state.selected === i) state.selected = null;
      else if (state.selected !== null && state.selected > i) state.selected--;
      renderLayers(); renderTransformPanel(); redraw(); setButtons();
    };
    // click on row → select layer (when in edit mode)
    li.onclick = (ev) => {
      if (ev.target.closest("button")) return;
      if (!state.editMode) {
        // auto-enable edit mode when picking from list
        state.editMode = true;
        setButtons();
      }
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

function baseName() { return "layermagic"; }

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
  const n = state.layers.length;
  let layers = state.layers;
  if (!n) {
    vctx.drawImage(state.srcCanvas, 0, 0);
    return;
  }
  const m = state.mouse;
  layers.forEach((L, i) => {
    if (!L.visible) return;
    // parallax offset (only when not actively dragging)
    let pdx = 0, pdy = 0;
    if (state.parallax && m && n > 1 && !state.drag) {
      const depth = i / (n - 1);
      pdx = (m.x - w / 2) * 0.05 * depth;
      pdy = (m.y - h / 2) * 0.05 * depth;
    }
    vctx.save();
    vctx.globalAlpha = L.opacity;
    const cx = L.dx + L.canvas.width / 2 + pdx;
    const cy = L.dy + L.canvas.height / 2 + pdy;
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
      // corner handles
      vctx.fillStyle = "#10b981";
      vctx.setLineDash([]);
      const hs = 6 / L.scale;
      [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(([sx, sy]) => {
        vctx.fillRect(
          sx * L.canvas.width / 2 - hs,
          sy * L.canvas.height / 2 - hs,
          hs * 2,
          hs * 2
        );
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
  // swap
  const tmp = state.layers[i];
  state.layers[i] = state.layers[j];
  state.layers[j] = tmp;
  // recolor
  state.layers.forEach((l2, k) => { l2.color = PALETTE[k % PALETTE.length]; });
  state.selected = j;
  renderLayers();
  renderTransformPanel();
  redraw();
}

/* ================= hit-test + drag ================= */
function canvasCoords(ev) {
  const r = view.getBoundingClientRect();
  const x = ((ev.clientX - r.left) * view.width) / r.width;
  const y = ((ev.clientY - r.top) * view.height) / r.height;
  return { x, y };
}

// cache of alpha masks per layer for fast hit-testing
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
    // inverse-transform the point to layer-local coords
    const cx = L.dx + L.canvas.width / 2;
    const cy = L.dy + L.canvas.height / 2;
    let px = x - cx;
    let py = y - cy;
    // unrotate
    const rad = (-L.rot * Math.PI) / 180;
    const rx = px * Math.cos(rad) - py * Math.sin(rad);
    const ry = px * Math.sin(rad) + py * Math.cos(rad);
    // unscale
    const lx = rx / L.scale + L.canvas.width / 2;
    const ly = ry / L.scale + L.canvas.height / 2;
    if (lx < 0 || ly < 0 || lx >= L.canvas.width || ly >= L.canvas.height) continue;
    // check alpha
    const mask = getAlphaMask(L);
    if (mask[Math.floor(ly) * L.canvas.width + Math.floor(lx)] > 20) return i;
  }
  return -1;
}

function wireEditDrag() {
  view.addEventListener("mousedown", (ev) => {
    if (!state.editMode || state.clickMode || !state.srcCanvas) return;
    const { x, y } = canvasCoords(ev);
    const idx = hitTest(x, y);
    if (idx >= 0) {
      state.selected = idx;
      const L = state.layers[idx];
      state.drag = { mx: x, my: y, lx: L.dx, ly: L.dy };
      view.classList.add("dragging");
      renderLayers();
      renderTransformPanel();
      redraw();
    } else {
      state.selected = null;
      renderLayers();
      renderTransformPanel();
      redraw();
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
  view.addEventListener("touchstart", (ev) => {
    if (!state.editMode || state.clickMode || !state.srcCanvas) return;
    if (!ev.touches[0]) return;
    ev.preventDefault();
    const t = ev.touches[0];
    const { x, y } = canvasCoords({ clientX: t.clientX, clientY: t.clientY });
    const idx = hitTest(x, y);
    if (idx >= 0) {
      state.selected = idx;
      const L = state.layers[idx];
      state.drag = { mx: x, my: y, lx: L.dx, ly: L.dy };
      renderLayers();
      renderTransformPanel();
      redraw();
    }
  }, { passive: false });

  view.addEventListener("touchmove", (ev) => {
    if (!state.drag || state.selected === null) return;
    if (!ev.touches[0]) return;
    ev.preventDefault();
    const t = ev.touches[0];
    const { x, y } = canvasCoords({ clientX: t.clientX, clientY: t.clientY });
    const L = state.layers[state.selected];
    L.dx = state.drag.lx + (x - state.drag.mx);
    L.dy = state.drag.ly + (y - state.drag.my);
    redraw();
  }, { passive: false });

  view.addEventListener("touchend", () => { state.drag = null; });

  // wheel to scale selected layer
  view.addEventListener("wheel", (ev) => {
    if (!state.editMode || state.selected === null) return;
    ev.preventDefault();
    const L = state.layers[state.selected];
    const delta = ev.deltaY > 0 ? 0.95 : 1.05;
    L.scale = Math.max(0.1, Math.min(3, L.scale * delta));
    renderTransformPanel();
    redraw();
  }, { passive: false });
}

/* ================= auto split ================= */
async function autoSplit() {
  if (!state.srcCanvas) return;
  try {
    showBusy("busy-auto");
    await ensureModel();
    const W = state.srcCanvas.width, H = state.srcCanvas.height;
    const total = W * H;

    let step = Math.max(56, Math.round(Math.min(W, H) / 10));
    let pts = [];
    const buildPts = () => {
      const arr = [];
      for (let y = Math.round(step / 2); y < H; y += step)
        for (let x = Math.round(step / 2); x < W; x += step) arr.push([x, y]);
      return arr;
    };
    pts = buildPts();

    const accepted = []; // {seg, bits}
    let t0 = performance.now();
    let done = 0;

    for (let pi = 0; pi < pts.length; pi++) {
      const [x, y] = pts[pi];
      const seg = await segmentAt([[x, y]]);
      done++;
      if (done % 4 === 0) {
        setBusyText(`${t("busy-auto")} ${done}/${pts.length}`);
        await new Promise((r) => setTimeout(r, 0));
        // adaptive: if too slow, coarsen remaining grid
        const el = performance.now() - t0;
        if (el / done > 350 && step < 160) {
          step = Math.round(step * 1.7);
          pts = buildPts();
        }
      }
      const ratio = seg.area / total;
      if (ratio < 0.001 || ratio > 0.93) continue; // junk / whole-frame
      const bits = coarseBits(seg);
      let dup = false;
      for (const a of accepted) {
        if (iouBits(a.bits, bits) > 0.82) { dup = true; break; }
      }
      if (dup) continue;
      accepted.push({ seg, bits });
      if (accepted.length >= 18) break;
    }

    // rebuild layer stack: full image at bottom, objects by area desc
    state.layers = [];
    state.nextId = 1;
    addLayer(fullImageLayer(), t("layer-full"), total);
    accepted
      .sort((a, b) => b.seg.area - a.seg.area)
      .forEach((a, i) => addLayer(layerCanvasFromMask(a.seg), `${t("layer")} ${i + 1}`, a.seg.area));

    renderLayers();
    redraw();
    hideBusy();
    setButtons();
  } catch (e) {
    console.error(e);
    hideBusy();
  }
}

/* ================= click mode ================= */
view.addEventListener("click", async (ev) => {
  if (!state.clickMode || !state.srcCanvas || state.busy) return;
  const r = view.getBoundingClientRect();
  const x = Math.round(((ev.clientX - r.left) * view.width) / r.width);
  const y = Math.round(((ev.clientY - r.top) * view.height) / r.height);
  if (x < 0 || y < 0 || x >= view.width || y >= view.height) return;
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
    renderLayers(); redraw(); hideBusy(); setButtons();
  } catch (e) {
    console.error(e);
    hideBusy();
  }
});

/* ================= parallax mouse ================= */
$("canvas-wrap").addEventListener("mousemove", (ev) => {
  if (!state.parallax) return;
  const r = view.getBoundingClientRect();
  state.mouse = { x: ev.clientX - r.left, y: ev.clientY - r.top };
  if (!redraw.pending) {
    redraw.pending = true;
    requestAnimationFrame(() => { redraw.pending = false; redraw(); });
  }
});
$("canvas-wrap").addEventListener("mouseleave", () => { state.mouse = null; redraw(); });

/* ================= exports ================= */
// Render the current visible layers (with transforms applied) onto a fresh canvas
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

$("btn-auto").addEventListener("click", autoSplit);

$("btn-click").addEventListener("click", () => {
  state.clickMode = !state.clickMode;
  if (state.clickMode && state.editMode) state.editMode = false;
  setButtons();
});

$("btn-edit").addEventListener("click", () => {
  state.editMode = !state.editMode;
  if (state.editMode && state.clickMode) state.clickMode = false;
  if (!state.editMode) state.selected = null;
  renderLayers();
  renderTransformPanel();
  setButtons();
  redraw();
});

$("btn-parallax").addEventListener("click", () => {
  state.parallax = !state.parallax;
  setButtons(); redraw();
});

$("btn-clear").addEventListener("click", () => {
  if (!state.layers.length) return;
  if (!confirm(t("confirm-clear"))) return;
  state.layers = [];
  renderLayers(); redraw(); setButtons();
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
wireEditDrag();
setButtons();
setStatus("idle", "status-idle");
