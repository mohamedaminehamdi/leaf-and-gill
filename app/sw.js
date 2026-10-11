// Offline shell: everything the app needs except the model (app.js caches that itself).
const SHELL = 'leaf-and-gill-shell-v6';
const FILES = [
  './', 'index.html', 'app.js', 'manifest.webmanifest',
  'icons/icon.svg', 'icons/icon-180.png', 'icons/icon-192.png', 'icons/icon-512.png',
  'vendor/ort/ort.webgpu.min.mjs', 'vendor/ort/ort-wasm-simd-threaded.asyncify.mjs', 'vendor/ort/ort-wasm-simd-threaded.asyncify.wasm',
  'data/meta.json', 'data/danger.json', 'data/trust.json',
];
// Files indexed by species position: cached under their ?v=<pack> URL, never mixed across versions.
const PACK_FILES = ['species.json', 'text_emb.f16', 'about.json', 'lookalikes.json', 'confused_with.json'];

async function cacheAll(cache) {
  const meta = await fetch('data/meta.json', { cache: 'no-cache' }).then((r) => r.json());
  const urls = [...FILES, ...PACK_FILES.map((f) => `data/${f}?v=${meta.pack}`)];
  await Promise.all(urls.map((u) => cache.add(u).catch(() => {})));  // optional files may not exist
}

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(SHELL).then(cacheAll).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k.startsWith('leaf-and-gill-shell-') && k !== SHELL).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

// Same-origin requests: answer from cache instantly, refresh it in the background when online.
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== 'GET' || url.origin !== location.origin) return;
  if (url.pathname.endsWith('/data/meta.json')) {
    // Network first: the pack version only changes while online, when its files can be fetched too.
    event.respondWith(caches.open(SHELL).then((cache) => fetch(event.request).then((res) => {
      if (res.ok) cache.put('data/meta.json', res.clone());
      return res;
    }).catch(() => cache.match('data/meta.json'))));
    return;
  }
  event.respondWith(caches.open(SHELL).then(async (cache) => {
    // Versioned pack files must match exactly; everything else ignores ?query noise.
    const cached = await cache.match(event.request, { ignoreSearch: !url.search.startsWith('?v=') });
    const fresh = fetch(event.request).then((res) => {
      if (res.ok) cache.put(event.request, res.clone());
      return res;
    }).catch(() => cached);
    return cached || fresh;
  }));
});
