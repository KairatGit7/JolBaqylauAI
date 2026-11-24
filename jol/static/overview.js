'use strict';

(() => {
  const status = document.querySelector('#overview-map-status');
  if (!window.L) { status.className = 'help'; status.textContent = 'Карта кітапханасы жүктелмеді. Толық карта бетін қайта ашыңыз.'; return; }
  const map = L.map('overview-map', {scrollWheelZoom: false, dragging: false, touchZoom: false, doubleClickZoom: false, boxZoom: false, keyboard: false, zoomControl: false}).setView([45.0156, 78.3739], 12);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 19, attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap қатысушылары</a>'}).addTo(map).on('tileerror', () => {
    status.className = 'help';
    status.textContent = 'Карта қабаты қолжетімсіз. Интернетті тексеріңіз; бақылаулар тізімі толық карта бетінде сақталады.';
  });
  fetch('/api/events').then(response => { if (!response.ok) throw new Error(); return response.json(); }).then(events => {
    let located = 0;
    for (const event of events) {
      if (event.latitude === null || event.longitude === null) continue;
      located++;
      L.circleMarker([event.latitude, event.longitude], {radius: 5, color: event.review_status === 'confirmed' ? '#168e85' : '#c99544', weight: 2, fillOpacity: .8, interactive: false}).addTo(map);
    }
    if (status.className === 'sr-only') status.textContent = `Координаты бар ${located} оқиға картаға қосылды. Бұл — қаланың бастапқы көрінісі.`;
  }).catch(() => { status.className = 'help'; status.textContent = 'Карта жазбалары жүктелмеді. Толық карта бетін қайта ашыңыз.'; });
})();
