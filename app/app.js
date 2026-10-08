// Leaf & Gill: identify plants, trees and mushrooms on-device with BioCLIP 2.
import * as ort from './vendor/ort/ort.webgpu.min.mjs';

const params = new URLSearchParams(location.search);
const MODEL_URL = params.get('model') || 'https://huggingface.co/mohamedaminehamdi/leaf-and-gill-bioclip2/resolve/main/vision.fp16.onnx';
const MODEL_CACHE = 'leaf-and-gill-model-v1';
const SIZE = 224;
const MEAN = [0.48145466, 0.4578275, 0.40821073];
const STD = [0.26862954, 0.26130258, 0.27577711];

ort.env.wasm.wasmPaths = new URL('./vendor/ort/', import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const status = (html) => { $('status').innerHTML = html; };

// ---------- data pack ----------

function halfToFloat(u16) {
  // IEEE 754 half -> float32, via one 64k lookup table (fast for millions of values).
  const table = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    table[h] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
  }
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = table[u16[i]];
  return out;
}

async function loadPack() {
  const [species, meta, danger, emb] = await Promise.all([
    fetch('data/species.json').then((r) => r.json()),
    fetch('data/meta.json').then((r) => r.json()),
    fetch('data/danger.json').then((r) => r.json()),
    fetch('data/text_emb.f16').then((r) => r.arrayBuffer()),
  ]);
  const optional = (f) => fetch(f).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  const [about, lookalikes, trust] = await Promise.all([optional('data/about.json'), optional('data/lookalikes.json'), optional('data/trust.json')]);
  return { species, meta, danger, about, lookalikes, trust, text: halfToFloat(new Uint16Array(emb)) };
}

// ---------- model (downloaded once, then offline) ----------

async function download(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`model download failed (${res.status})`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  // One preallocated buffer: a 600 MB model must not be held twice on a phone.
  let bytes = new Uint8Array(total || 64e6), seen = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (seen + value.length > bytes.length) {
      const grown = new Uint8Array(Math.max(bytes.length * 2, seen + value.length));
      grown.set(bytes.subarray(0, seen));
      bytes = grown;
    }
    bytes.set(value, seen);
    seen += value.length;
    onProgress(seen, total);
  }
  return seen === bytes.length ? bytes : bytes.slice(0, seen);
}

/** The model, from the offline cache when we have it. Returns {bytes, offline}. */
async function modelBytes(onProgress) {
  const cache = await caches.open(MODEL_CACHE);
  const cached = await cache.match(MODEL_URL);
  if (cached) return { bytes: new Uint8Array(await cached.arrayBuffer()), offline: true };
  const bytes = await download(MODEL_URL, onProgress);
  try {
    await cache.put(MODEL_URL, new Response(bytes, { headers: { 'content-type': 'application/octet-stream' } }));
    return { bytes, offline: true };
  } catch (err) {
    console.warn('could not save the model for offline use', err);  // e.g. storage quota
    return { bytes, offline: false };
  }
}

async function createSession(bytes) {
  const tries = navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'];
  for (const ep of tries) {
    try {
      return { session: await ort.InferenceSession.create(bytes, { executionProviders: [ep], graphOptimizationLevel: 'all' }), ep };
    } catch (err) {
      console.warn(`${ep} failed, trying next`, err);
    }
  }
  throw new Error('This browser cannot run the model.');
}

// ---------- identify ----------

function pixels(bitmap) {
  const side = Math.min(bitmap.width, bitmap.height);
  const canvas = new OffscreenCanvas(SIZE, SIZE);
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, (bitmap.width - side) / 2, (bitmap.height - side) / 2, side, side, 0, 0, SIZE, SIZE);
  const rgba = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const x = new Float32Array(3 * SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    for (let c = 0; c < 3; c++) x[c * SIZE * SIZE + i] = (rgba[i * 4 + c] / 255 - MEAN[c]) / STD[c];
  }
  return x;
}

function topMatches(embedding, pack, k = 5) {
  const { text, meta } = pack;
  const dim = meta.dim, n = text.length / dim;
  let norm = 0;
  for (let j = 0; j < dim; j++) norm += embedding[j] * embedding[j];
  norm = Math.sqrt(norm);
  const logits = new Float32Array(n);
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    let dot = 0;
    for (let j = 0; j < dim; j++) dot += embedding[j] * text[i * dim + j];
    logits[i] = (meta.logit_scale * dot) / norm;
    if (logits[i] > max) max = logits[i];
  }
  let sum = 0;
  for (let i = 0; i < n; i++) sum += Math.exp(logits[i] - max);
  const order = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, k);
  return order.map((i) => ({ i, p: Math.exp(logits[i] - max) / sum, cos: logits[i] / meta.logit_scale }));
}

function dangerFor(matches, pack) {
  const hits = new Map();
  for (const m of matches) {
    const genus = pack.species[m.i].genus;
    if (pack.danger.genera[genus]) hits.set(genus, pack.danger.genera[genus]);
  }
  return [...hits];
}

function render(photoUrl, matches, pack, ms, ep) {
  const sp = (m) => pack.species[m.i];
  const danger = dangerFor(matches, pack);
  const fungi = sp(matches[0]).kingdom === 'Fungi';
  const rows = matches.map((m) => {
    const s = sp(m), pct = Math.round(m.p * 100);
    const deadly = pack.danger.genera[s.genus] ? '<em class="tag">TOXIC GENUS</em>' : '';
    return `<li class="match"><div><b>${esc(s.common || s.name)}</b>${deadly}<div class="sci">${esc(s.name)} · ${esc(s.family)}</div></div>
      <span class="pct">${pct}%</span><div class="bar"><i style="width:${Math.max(2, pct)}%"></i></div></li>`;
  }).join('');
  $('result').innerHTML = `
    ${danger.length ? `<div class="banner danger" role="alert">Dangerous look-alikes in these matches
      ${danger.map(([g, why]) => `<p><b>${esc(g)}</b>: ${esc(why)}</p>`).join('')}</div>` : ''}
    ${fungi ? `<div class="banner warn">Never eat a wild mushroom based on an app.<p>If someone has eaten one and feels unwell, call your local poison centre or emergency number now.</p></div>` : ''}
    <div class="card"><img class="photo" src="${photoUrl}" alt="Your photo"><ol class="matches">${rows}</ol>
      <div class="meta">Identified on this device in ${ms} ms (${ep === 'webgpu' ? 'GPU' : 'CPU'}) · ${pack.species.length.toLocaleString()} species</div></div>`;
}

// ---------- boot ----------

async function main() {
  let pack, session, ep, offline;
  try {
    status('Loading species list…');
    pack = await loadPack();
    status('Preparing the model (one-time download, then it works offline)…<br><progress id="dl" max="1" value="0"></progress>');
    await navigator.storage?.persist?.();
    let bytes;
    ({ bytes, offline } = await modelBytes((seen, total) => {
      const bar = $('dl');
      if (bar && total) bar.value = seen / total;
      $('mode').textContent = total ? `Downloading ${Math.round((100 * seen) / total)}%` : `Downloading ${Math.round(seen / 1e6)} MB`;
    }));
    status('Starting the model…');
    ({ session, ep } = await createSession(bytes));
  } catch (err) {
    status(`<b>Couldn't start:</b> ${esc(err.message)}`);
    $('mode').textContent = 'Error';
    return;
  }
  $('mode').textContent = offline ? 'Ready offline' : 'Ready (online only)';
  $('mode').classList.toggle('ok', offline);
  $('snap').setAttribute('aria-disabled', 'false');
  status(offline ? '' : 'Not enough free storage to save the model for offline use. It works now, but needs signal next time.');

  $('camera').addEventListener('change', async (ev) => {
    const file = ev.target.files[0];
    if (!file) return;
    status('Looking closely…');
    const bitmap = await createImageBitmap(file);
    const t0 = performance.now();
    const input = new ort.Tensor('float32', pixels(bitmap), [1, 3, SIZE, SIZE]);
    const out = await session.run({ [session.inputNames[0]]: input });
    const embedding = out[session.outputNames[0]].data;
    const matches = topMatches(embedding, pack);
    const ms = Math.round(performance.now() - t0);
    status('');
    render(URL.createObjectURL(file), matches, pack, ms, ep);
    ev.target.value = '';
  });
}

if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js');
main();
