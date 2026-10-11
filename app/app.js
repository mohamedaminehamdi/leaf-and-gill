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
  // meta.json names the pack version; every file indexed by species position is fetched with it,
  // so a cached old file can never be paired with a new species list.
  const meta = await fetch('data/meta.json', { cache: 'no-cache' }).then((r) => r.json());
  const v = `?v=${meta.pack}`;
  const [species, danger, emb] = await Promise.all([
    fetch(`data/species.json${v}`).then((r) => r.json()),
    fetch('data/danger.json').then((r) => r.json()),
    fetch(`data/text_emb.f16${v}`).then((r) => r.arrayBuffer()),
  ]);
  const text = halfToFloat(new Uint16Array(emb));
  if (species.length !== meta.count || text.length !== meta.count * meta.dim) {
    throw new Error('The species pack is out of date. Reload the page while online.');
  }
  const optional = (f) => fetch(f).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  const [about, lookalikes, confused, trust] = await Promise.all([
    optional(`data/about.json${v}`), optional(`data/lookalikes.json${v}`), optional(`data/confused_with.json${v}`), optional('data/trust.json'),
  ]);
  return { species, meta, danger, about, lookalikes, trust, confused, text };
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
  // Up to three warnings, most reliable first:
  // 1. a toxic genus among the top 5 matches,
  // 2. a classic field-guide pair for the top 3 (chanterelle -> jack-o'-lantern),
  // 3. a toxic species among the model's own look-alikes of the top 3 (coriander -> poison hemlock).
  const hits = new Map();
  const add = (genus, via) => {
    if (pack.danger.genera[genus] && !hits.has(genus)) hits.set(genus, { why: pack.danger.genera[genus], via });
  };
  const top3 = matches.slice(0, 3).map((m) => pack.species[m.i]);
  matches.forEach((m) => add(pack.species[m.i].genus, null));
  top3.forEach((s) => (pack.danger.known_lookalikes?.[s.genus] || []).forEach((g) => add(g, s)));
  matches.slice(0, 3).forEach((m) => confusedWith(pack, m.i, Infinity).forEach((j) => add(pack.species[j].genus, pack.species[m.i])));
  return [...hits].slice(0, 3);  // more than three warnings stop being read
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

function confusedWith(pack, i, limit = 3) {
  // Real mix-ups measured on test photos first, then the model's nearest species from other genera.
  const real = pack.confused?.[i] || [];
  const near = pack.lookalikes?.[i] || [];
  return [...new Set([...real, ...near])].filter((j) => j !== i).slice(0, limit);
}

function details(pack, i) {
  const about = pack.about?.[i];
  const lookalikes = confusedWith(pack, i).map((j) => nameOf(pack, j)).join(', ');
  return `${about ? `<p class="about">${esc(about.text)} <a href="${esc(about.url)}" target="_blank" rel="noopener">Wikipedia</a> · CC BY-SA</p>` : ''}
    ${lookalikes ? `<p class="look"><span>Often confused with</span> ${lookalikes}</p>` : ''}`;
}

function row(pack, m) {
  const s = pack.species[m.i], pct = Math.round(m.p * 100);
  const deadly = pack.danger.genera[s.genus] ? ' <em class="tag">TOXIC</em>' : '';
  return `<li><details><summary>
      <span class="name"><b>${esc(s.common || s.name)}</b>${deadly}<span class="sci">${esc(s.name)}</span></span>
      <span class="pct">${pct}%</span><span class="bar"><i style="width:${Math.max(2, pct)}%"></i></span>
    </summary>${details(pack, m.i)}</details></li>`;
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
  $('finds').innerHTML = list.length ? `<details class="panel"><summary><b>My finds</b><span>${list.length} saved on this phone</span></summary>
    <ul class="finds">${list.map((f) => `<li><img src="${f.thumb}" alt="" width="44" height="44"><span><b>${esc(f.common || f.name)}</b>
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
  const deadly = pack.danger.genera[top.genus] ? '<em class="tag">TOXIC GENUS</em>' : '';
  $('result').innerHTML = `
    ${danger.length ? `<div class="banner danger" role="alert"><b>Dangerous look-alikes</b>
      ${danger.map(([g, { why, via }]) => `<p><strong>${esc(g)}</strong>${via ? ` <span>(looks like ${esc(via.common || via.name)})</span>` : ''}: ${esc(why)}</p>`).join('')}</div>` : ''}
    ${top.kingdom === 'Fungi' ? `<div class="banner warn"><b>Never eat a wild mushroom based on an app.</b><p>If someone has eaten one and feels unwell, call your local poison centre or emergency number now.</p></div>` : ''}
    <article class="hero-card">
      <img class="photo" src="${URL.createObjectURL(file)}" alt="Your photo">
      <div class="hero-body">
        <div class="eyebrow">Top match · ${Math.round(matches[0].p * 100)}% model score</div>
        <h2>${esc(top.common || top.name)}</h2>
        <p class="sci">${esc(top.name)} · ${esc(top.family)} ${deadly}</p>
        ${trust ? `<div class="trust ${trust.level}"><i></i><span><b>${label}</b> · ${Math.round(trust.p * 100)}% likely right<small>${hint}</small></span></div>` : ''}
        ${details(pack, matches[0].i)}
      </div>
    </article>
    <section class="panel"><h3>Other possibilities</h3><ol class="others">${matches.slice(1).map((m) => row(pack, m)).join('')}</ol></section>
    <button type="button" class="ghost" id="save">Save “${esc(top.common || top.name)}” to my finds</button>
    <p class="meta">Identified on this device in ${ms} ms (${ep === 'webgpu' ? 'GPU' : 'CPU'}) · ${pack.species.length.toLocaleString()} species · no photo uploaded</p>`;
  $('save').addEventListener('click', async (ev) => {
    const list = finds();
    list.unshift({ name: top.name, common: top.common, at: Date.now(), thumb: await thumbnail(file) });
    try { localStorage.setItem(LIST_KEY, JSON.stringify(list.slice(0, 300))); } catch { /* storage full: keep going */ }
    ev.target.textContent = 'Saved ✓';
    ev.target.disabled = true;
    renderFinds();
  });
  $('result').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
}

// ---------- boot ----------

function setReady(ready) {
  document.querySelectorAll('[data-needs-model]').forEach((el) => {
    el.setAttribute('aria-disabled', String(!ready));
    if (el.tagName === 'BUTTON') el.disabled = !ready;
  });
}

function setup(seen, total) {
  const pct = total ? Math.round((100 * seen) / total) : 0;
  $('setup-bar').style.width = `${pct}%`;
  $('setup-text').textContent = total ? `${Math.round(seen / 1e6)} of ${Math.round(total / 1e6)} MB · ${pct}%` : `${Math.round(seen / 1e6)} MB`;
  $('mode').textContent = `Setting up ${pct}%`;
}

async function main() {
  let pack, session, ep, offline;
  setReady(false);
  try {
    $('setup-text').textContent = 'Loading the species list…';
    pack = await loadPack();
    await navigator.storage?.persist?.();
    let bytes;
    ({ bytes, offline } = await modelBytes(setup));
    $('setup-text').textContent = 'Starting the model…';
    ({ session, ep } = await createSession(bytes));
  } catch (err) {
    $('setup').innerHTML = `<b>Couldn't start</b><p>${esc(err.message)}</p>`;
    $('mode').textContent = 'Error';
    return;
  }
  $('setup').hidden = true;
  $('mode').textContent = offline ? 'Ready offline' : 'Ready (online only)';
  $('mode').classList.toggle('ok', offline);
  setReady(true);
  renderFinds();
  status(offline ? '' : 'Not enough free storage to save the model for offline use. It works now, but needs signal next time.');

  async function identify(file) {
    status('Looking closely…');
    const bitmap = await createImageBitmap(file);
    const t0 = performance.now();
    const input = new ort.Tensor('float32', pixels(bitmap), [1, 3, SIZE, SIZE]);
    const out = await session.run({ [session.inputNames[0]]: input });
    const matches = topMatches(out[session.outputNames[0]].data, pack);
    const ms = Math.round(performance.now() - t0);
    status('');
    render(file, matches, pack, ms, ep);
  }

  $('camera').addEventListener('change', async (ev) => {
    if (ev.target.files[0]) await identify(ev.target.files[0]);
    ev.target.value = '';
  });
  document.querySelectorAll('[data-example]').forEach((btn) => btn.addEventListener('click', async () => {
    const blob = await (await fetch(btn.dataset.example)).blob();
    await identify(new File([blob], btn.dataset.example.split('/').pop(), { type: blob.type }));
  }));
}

if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js');
main();
