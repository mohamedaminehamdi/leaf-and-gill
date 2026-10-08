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
  const [about, lookalikes, trust, confused] = await Promise.all(['about', 'lookalikes', 'trust', 'confused_with'].map((f) => optional(`data/${f}.json`)));
  return { species, meta, danger, about, lookalikes, trust, confused, text: halfToFloat(new Uint16Array(emb)) };
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
  // A toxic genus anywhere in the top 5, or among the known look-alikes of the top 3, raises the alarm:
  // "coriander" from a photo of poison hemlock must still warn about hemlock.
  const hits = new Map();
  const flag = (i, via) => {
    const genus = pack.species[i].genus;
    if (pack.danger.genera[genus] && !hits.has(genus)) hits.set(genus, { why: pack.danger.genera[genus], via });
  };
  matches.forEach((m) => flag(m.i, null));
  matches.slice(0, 3).forEach((m) => confusedWith(pack, m.i).forEach((j) => flag(j, pack.species[m.i])));
  return [...hits];
}

// ---------- trust, cards, life list ----------

function axisPos(axis, v) {
  // Position of v on an evenly spaced axis: [lower grid index, fraction towards the next one].
  const n = axis.length, x = Math.min(n - 1, Math.max(0, ((v - axis[0]) / (axis[n - 1] - axis[0])) * (n - 1)));
  const i = Math.min(n - 2, Math.floor(x));
  return [i, x - i];
}

function trustFor(matches, pack) {
  // TabPFN learned P(top-1 is right) from (top-1 probability, margin, top-1 similarity, kingdom) on fresh
  // iNaturalist photos. Its predictions ship as a grid; we interpolate between grid points offline.
  const t = pack.trust;
  if (!t) return null;
  const top = matches[0];
  const grid = pack.species[top.i].kingdom === 'Fungi' ? t.fungi : t.plants;
  const [a, ta] = axisPos(t.p, top.p), [b, tb] = axisPos(t.p, top.p - matches[1].p), [c, tc] = axisPos(t.cos, top.cos);
  let p = 0;
  for (const [da, wa] of [[0, 1 - ta], [1, ta]])
    for (const [db, wb] of [[0, 1 - tb], [1, tb]])
      for (const [dc, wc] of [[0, 1 - tc], [1, tc]]) p += wa * wb * wc * grid[a + da][b + db][c + dc];
  const level = p >= t.confident ? 'confident' : p >= t.likely ? 'likely' : 'unsure';
  return { p, level };
}

const TRUST_TEXT = {
  confident: ['Very likely', 'Right about 9 times in 10 in testing. Still check the features below.'],
  likely: ['Likely', 'Often right, but compare it with the other matches.'],
  unsure: ['Not sure', 'Compare these possibilities. Look closer or try another angle.'],
};

function nameOf(pack, i) {
  const s = pack.species[i];
  return `${esc(s.common || s.name)}${pack.danger.genera[s.genus] ? ' <em class="tag">TOXIC</em>' : ''}`;
}

function confusedWith(pack, i) {
  const real = pack.confused?.[i] || [];
  const near = pack.lookalikes?.[i] || [];
  return [...new Set([...real, ...near])].filter((j) => j !== i).slice(0, 3);
}

function card(pack, m, open) {
  const s = pack.species[m.i], about = pack.about?.[m.i], pct = Math.round(m.p * 100);
  const deadly = pack.danger.genera[s.genus] ? '<em class="tag">TOXIC GENUS</em>' : '';
  const lookalikes = confusedWith(pack, m.i).map((j) => nameOf(pack, j)).join(', ');
  return `<li class="match"><details ${open ? 'open' : ''}>
    <summary><span><b>${esc(s.common || s.name)}</b>${deadly}<span class="sci">${esc(s.name)} · ${esc(s.family)}</span></span>
      <span class="pct">${pct}%</span><span class="bar"><i style="width:${Math.max(2, pct)}%"></i></span></summary>
    ${about ? `<p class="about">${esc(about.text)} <a href="${esc(about.url)}" target="_blank" rel="noopener">From Wikipedia</a>, CC BY-SA 4.0.</p>` : ''}
    ${lookalikes ? `<p class="look"><b>Often confused with:</b> ${lookalikes}</p>` : ''}
  </details></li>`;
}

const LIST_KEY = 'leaf-and-gill-finds';
const finds = () => { try { return JSON.parse(localStorage.getItem(LIST_KEY)) || []; } catch { return []; } };

async function thumbnail(file) {
  const bitmap = await createImageBitmap(file);
  const side = Math.min(bitmap.width, bitmap.height), canvas = document.createElement('canvas');
  canvas.width = canvas.height = 96;
  canvas.getContext('2d').drawImage(bitmap, (bitmap.width - side) / 2, (bitmap.height - side) / 2, side, side, 0, 0, 96, 96);
  return canvas.toDataURL('image/jpeg', 0.7);
}

function renderFinds() {
  const list = finds();
  $('finds').innerHTML = list.length ? `<details class="card"><summary><b>My finds</b> · ${list.length} saved on this phone</summary>
    <ul class="finds">${list.map((f) => `<li><img src="${f.thumb}" alt="" width="48" height="48"><span><b>${esc(f.common || f.name)}</b>
      <span class="sci">${esc(f.name)} · ${esc(new Date(f.at).toLocaleDateString())}</span></span></li>`).join('')}</ul>
    <button type="button" class="ghost" id="clear-finds">Clear list</button></details>` : '';
  $('clear-finds')?.addEventListener('click', () => {
    if (confirm('Delete all saved finds from this phone?')) { localStorage.removeItem(LIST_KEY); renderFinds(); }
  });
}

function render(file, matches, pack, ms, ep) {
  const top = pack.species[matches[0].i];
  const danger = dangerFor(matches, pack);
  const trust = trustFor(matches, pack);
  const [label, hint] = trust ? TRUST_TEXT[trust.level] : ['', ''];
  $('result').innerHTML = `
    ${danger.length ? `<div class="banner danger" role="alert">Dangerous look-alikes in these matches
      ${danger.map(([g, { why, via }]) => `<p><b>${esc(g)}</b>${via ? ` (often confused with ${esc(via.common || via.name)})` : ''}: ${esc(why)}</p>`).join('')}</div>` : ''}
    ${top.kingdom === 'Fungi' ? `<div class="banner warn">Never eat a wild mushroom based on an app.<p>If someone has eaten one and feels unwell, call your local poison centre or emergency number now.</p></div>` : ''}
    <div class="card"><img class="photo" src="${URL.createObjectURL(file)}" alt="Your photo">
      ${trust ? `<div class="trust ${trust.level}"><b>${label}</b> · ${Math.round(trust.p * 100)}% likely right<span>${hint}</span></div>` : ''}
      <ol class="matches">${matches.map((m, k) => card(pack, m, k === 0)).join('')}</ol>
      <button type="button" class="ghost" id="save">Save “${esc(top.common || top.name)}” to my finds</button>
      <div class="meta">Identified on this device in ${ms} ms (${ep === 'webgpu' ? 'GPU' : 'CPU'}) · ${pack.species.length.toLocaleString()} species</div></div>`;
  $('save').addEventListener('click', async (ev) => {
    const list = finds();
    list.unshift({ name: top.name, common: top.common, at: Date.now(), thumb: await thumbnail(file) });
    try { localStorage.setItem(LIST_KEY, JSON.stringify(list.slice(0, 300))); } catch { /* storage full: keep going */ }
    ev.target.textContent = 'Saved ✓';
    ev.target.disabled = true;
    renderFinds();
  });
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
  renderFinds();
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
    render(file, matches, pack, ms, ep);
    ev.target.value = '';
  });
}

if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js');
main();
