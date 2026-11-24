'use strict';

(() => {
  const statusNode = document.querySelector('#map-status');
  if (!window.L) { statusNode.textContent = 'Карта кітапханасы жүктелмеді. Бетті қайта ашыңыз.'; return; }
  const map = L.map('map', {scrollWheelZoom: false}).setView([45.0156, 78.3739], 13);
  map.zoomControl.setPosition('topright');
  const zoomIn = document.querySelector('.leaflet-control-zoom-in');
  const zoomOut = document.querySelector('.leaflet-control-zoom-out');
  zoomIn.title = 'Жақындату'; zoomIn.setAttribute('aria-label', 'Жақындату');
  zoomOut.title = 'Алыстату'; zoomOut.setAttribute('aria-label', 'Алыстату');
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 19, attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap қатысушылары</a>'}).addTo(map).on('tileerror', () => {
    document.querySelector('#tile-status').textContent = 'Карта қабаты қолжетімсіз. Интернетті тексеріңіз; төмендегі тізім жұмыс істейді.';
  });
  const markers = L.layerGroup().addTo(map);
  const statuses = {predicted: 'Модель болжамы', confirmed: 'Расталған', rejected: 'Қабылданбаған'};
  const colors = {predicted: '#c99544', confirmed: '#168e85', rejected: '#8494a5'};
  const category = document.querySelector('#category-filter');
  const review = document.querySelector('#status-filter');
  let serial = 0;
  function text(tag, content, className) {
    const element = document.createElement(tag);
    element.textContent = content;
    if (className) element.className = className;
    return element;
  }
  function syncUrl() {
    const url = new URL(window.location.href);
    for (const [key, value] of [['category', category.value], ['status', review.value]]) {
      if (value) url.searchParams.set(key, value); else url.searchParams.delete(key);
    }
    window.history.pushState(null, '', url);
  }
  async function loadEvents() {
    const run = ++serial;
    const params = new URLSearchParams({category: category.value, status: review.value});
    statusNode.textContent = 'Бақылаулар жүктелуде…';
    const table = document.querySelector('#map-records');
    table.setAttribute('aria-busy', 'true');
    try {
      const response = await fetch(`/api/events?${params}`);
      if (!response.ok) throw new Error('Деректі алу мүмкін болмады. Сүзгіні тазалаңыз немесе жүйеге қайта кіріңіз.');
      const events = await response.json();
      if (run !== serial) return;
      markers.clearLayers();
      const fragment = document.createDocumentFragment();
      let located = 0;
      for (const event of events) {
        const link = text('a', event.is_demo ? 'ДЕМО · жазба ↗' : 'Жазбаны қарау ↗', 'text-link');
        link.href = event.detail;
        link.setAttribute('aria-label', `#${event.id} ${event.label} оқиғасының дәлелін қарау`);
        const row = document.createElement('tr');
        const state = text('span', statuses[event.review_status], `badge ${event.review_status}`);
        const stateCell = document.createElement('td'); stateCell.append(state);
        const date = new Intl.DateTimeFormat('kk-KZ', {dateStyle: 'short', timeStyle: 'short', timeZone: 'Asia/Almaty'}).format(new Date(event.observed_at));
        row.append(text('td', `#${event.id} · ${event.label}`), text('td', `${event.segment || 'Учаске жоқ'} · ${event.latitude === null ? 'Координат жоқ' : 'Орын көрсетілген'}`), text('td', `${date} · ${formatNumber(event.seconds, 1)} с`), stateCell);
        const cell = document.createElement('td'); cell.append(link); row.append(cell); fragment.append(row);
        if (event.latitude === null || event.longitude === null) continue;
        located++;
        const popup = document.createElement('div'); popup.className = 'map-popup';
        popup.append(text('strong', event.label), text('p', `${statuses[event.review_status]} · ${formatNumber(event.confidence * 100, 1)}%`), text('p', event.location_source));
        const image = document.createElement('img');
        image.src = event.evidence; image.alt = 'Белгіленген бейне дәлелі';
        image.width = event.width; image.height = event.height; image.loading = 'lazy';
        popup.append(image, link.cloneNode(true));
        L.circleMarker([event.latitude, event.longitude], {radius: 8, color: colors[event.review_status], weight: 2, fillOpacity: .8}).bindPopup(popup).addTo(markers);
      }
      if (!events.length) { const row = document.createElement('tr'); const cell = text('td', 'Бұл сүзгі бойынша жазба жоқ. Басқа санатты таңдаңыз немесе сүзгіні тазалаңыз.'); cell.colSpan = 5; row.append(cell); fragment.append(row); }
      table.replaceChildren(fragment);
      table.setAttribute('aria-busy', 'false');
      statusNode.textContent = `Сүзгі нәтижесі: ${formatNumber(events.length)} жазба · картада ${formatNumber(located)} · координатсыз ${formatNumber(events.length - located)}`;
    } catch (error) {
      if (run !== serial) return;
      table.replaceChildren(); markers.clearLayers();
      table.setAttribute('aria-busy', 'false');
      statusNode.textContent = error instanceof TypeError ? 'Байланыс үзілді. Интернетті тексеріп, «Сүзгіні тазалау» батырмасын басыңыз.' : error.message;
    }
  }
  function changeFilters() { syncUrl(); loadEvents(); }
  category.addEventListener('change', changeFilters);
  review.addEventListener('change', changeFilters);
  document.querySelector('#map-filters').addEventListener('submit', event => { event.preventDefault(); changeFilters(); });
  document.querySelector('#reset-filters').addEventListener('click', () => { category.value = ''; review.value = ''; changeFilters(); });
  window.addEventListener('popstate', () => {
    const params = new URLSearchParams(window.location.search);
    category.value = params.get('category') || '';
    review.value = params.get('status') || '';
    loadEvents();
  });
  loadEvents();
})();
