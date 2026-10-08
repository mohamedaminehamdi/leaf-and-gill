// Offline shell: everything the app needs except the model (app.js caches that itself).
const SHELL = 'leaf-and-gill-shell-v2';
const FILES = [
  './', 'index.html', 'app.js', 'manifest.webmanifest',
  'icons/icon.svg', 'icons/icon-180.png', 'icons/icon-192.png', 'icons/icon-512.png',
  'vendor/ort/ort.webgpu.min.mjs', 'vendor/ort/ort-wasm-simd-threaded.asyncify.mjs', 'vendor/ort/ort-wasm-simd-threaded.asyncify.wasm',
  'data/species.json', 'data/meta.json', 'data/danger.json', 'data/text_emb.f16',
  'data/lookalikes.json', 'data/about.json', 'data/trust.json', 'data/confused_with.json',
];

self.addEventListener('install', (event) => {
  // Optional data files may not exist yet; cache what is there instead of failing the install.
  event.waitUntil(caches.open(SHELL).then((c) => Promise.all(FILES.map((f) => c.add(f).catch(() => {})))).then(() => self.skipWaiting()));
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
  event.respondWith(caches.open(SHELL).then(async (cache) => {
    const cached = await cache.match(event.request, { ignoreSearch: true });
    const fresh = fetch(event.request).then((res) => {
      if (res.ok) cache.put(event.request, res.clone());
      return res;
    }).catch(() => cached);
    return cached || fresh;
  }));
});
