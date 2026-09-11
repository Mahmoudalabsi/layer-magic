# LayerMagic — فصل الصور إلى طبقات بالذكاء الاصطناعي

**Split images into AI layers — fully in your browser.**

LayerMagic is a local-first web tool that divides any image into separate
transparent PNG object layers using Meta's SAM (Segment Anything) model,
running entirely inside your browser via Transformers.js. No server, no API
key, no uploads — your images never leave your device.

Based on the same idea as commercial tools like Canva "Magic Layers" and
open models like Qwen-Image-Layered — but free, private, and deployable on
any static host.

## Features

- **Auto-split (فصل تلقائي)**: scans the whole image with a smart point grid,
  generates object masks, deduplicates them, and stacks them as layers
  (bottom layer = the full image, so the composite always matches the original).
- **Click mode (وضع النقر)**: click any object to extract it instantly as its
  own layer.
- **Layers panel**: show/hide, per-layer PNG download, delete, area stats.
- **Parallax preview**: move the mouse over the canvas to see a pseudo-3D
  depth effect built from the layer stack.
- **Export**: every layer as PNG, or all layers + composite in one ZIP.
- **Arabic / English** interface with RTL/LTR support.
- **Private by design**: model weights are downloaded once from the Hugging Face
  CDN (~40 MB), then everything runs locally (WASM, with WebGPU when available).

## Run

It is a pure static site — serve the folder with any static server:

```sh
python3 -m http.server 8080
# or
npx serve .
```

Then open http://localhost:8080.

First model load downloads SlimSAM-77 weights once and caches them in the
browser. After that, all segmentation runs offline.

## Deploy

The folder is static — deploy it to Cloudflare Pages / Netlify / GitHub Pages
as-is (no build step):

```sh
npx wrangler pages deploy . --project-name layer-magic
```

## How it works

1. The image is downscaled to ≤1024 px and encoded once by the SAM image
   encoder (the expensive step, done a single time per image).
2. Auto-split prompts the mask decoder with a grid of points; each decode is
   cheap, so the grid adapts if the device is slow.
3. Masks are deduplicated with a coarse-grid IoU, filtered by area, sorted,
   and rendered as RGBA layers.
4. Because the full image is kept as the bottom layer, hiding/showing or
   exporting layers always recomposes to the original image.

## Credits

- [SAM / Segment Anything](https://segment-anything.com/) — Meta AI
- [Transformers.js](https://huggingface.co/docs/transformers.js) — Hugging Face
- Inspired by Canva Magic Layers and [Qwen-Image-Layered](https://github.com/QwenLM/Qwen-Image-Layered)

## License

MIT — free to use, modify, and redistribute.
