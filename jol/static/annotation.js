'use strict';
(() => {
  const svg = document.querySelector('#gt-canvas');
  if (!svg) return;
  const width = Number(svg.dataset.width), height = Number(svg.dataset.height);
  const field = document.querySelector('#gt-boxes');
  const list = document.querySelector('#gt-list');
  const labels = JSON.parse(list.dataset.labels);
  const shapes = document.querySelector('#gt-shapes');
  const draft = document.querySelector('#gt-draft');
  const status = document.querySelector('#gt-status');
  const negative = document.querySelector('#gt-negative');
  const category = document.querySelector('#gt-category');
  let boxes = JSON.parse(field.value), history = [], start = null;
  const clone = value => JSON.parse(JSON.stringify(value));
  function remember() { history.push(clone(boxes)); if (history.length > 30) history.shift(); }
  function render(dirty = false) {
    if (boxes.length) negative.checked = false;
    field.value = JSON.stringify(boxes);
    const rows = document.createDocumentFragment(), drawings = document.createDocumentFragment();
    boxes.forEach((item, index) => {
      const [x1, y1, x2, y2] = item.bbox;
      const shape = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      for (const [key, value] of Object.entries({x: x1, y: y1, width: x2-x1, height: y2-y1, fill: '#087f8c22', stroke: '#36ffe2', 'stroke-width': 2})) shape.setAttribute(key, value);
      drawings.append(shape);
      const row = document.createElement('tr');
      for (const value of [`${index+1}. ${labels[item.category]}`, item.bbox.map(n => formatNumber(n, 1)).join(' / ')]) {
        const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
      }
      const cell = document.createElement('td'), remove = document.createElement('button');
      remove.type = 'button'; remove.className = 'button secondary small'; remove.textContent = 'Жою';
      remove.setAttribute('aria-label', `${index+1}-қорапты жою`);
      remove.addEventListener('click', () => { remember(); boxes.splice(index, 1); render(true); });
      cell.append(remove); row.append(cell); rows.append(row);
    });
    list.replaceChildren(rows); shapes.replaceChildren(drawings);
    document.querySelector('#gt-undo').disabled = !history.length;
    status.textContent = `Эталон қораптары: ${formatNumber(boxes.length)}.`;
    if (dirty) {
      document.querySelector('[name="complete"]').checked = false;
      field.dispatchEvent(new Event('input', {bubbles: true}));
    }
  }
  function add(box) {
    if (negative.checked) { status.textContent = 'Алдымен «Ақау жоқ» белгісін алып тастаңыз.'; return; }
    if (boxes.length >= 200 || box.some(n => !Number.isFinite(n)) ||
        !(0 <= box[0] && box[0] < box[2] && box[2] <= width && 0 <= box[1] && box[1] < box[3] && box[3] <= height)) {
      status.textContent = 'Шекарадағы дұрыс координаттарды енгізіңіз (ең көбі 200 қорап).'; return;
    }
    remember(); boxes.push({category: category.value, bbox: box}); render(true);
  }
  document.querySelector('#gt-add').addEventListener('click', () => add(['x1','y1','x2','y2'].map(key => {
    const input = document.querySelector(`#gt-${key}`);
    return input.value === '' ? NaN : Number(input.value);
  })));
  document.querySelector('#gt-undo').addEventListener('click', () => { if (history.length) { boxes = history.pop(); render(true); } });
  negative.addEventListener('change', () => {
    document.querySelector('[name="complete"]').checked = false;
    if (negative.checked && boxes.length) {
      negative.checked = false;
      status.textContent = 'Теріс кадрда қорап болмауы тиіс. Қораптарды алдымен жойыңыз.';
    }
  });
  function point(event) {
    const p = new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse());
    return [Math.max(0, Math.min(width, p.x)), Math.max(0, Math.min(height, p.y))];
  }
  function rectangle(a, b) { return [Math.min(a[0],b[0]), Math.min(a[1],b[1]), Math.max(a[0],b[0]), Math.max(a[1],b[1])].map(n => Math.round(n*100)/100); }
  function cancel() { start = null; draft.setAttribute('hidden', ''); }
  svg.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    if (negative.checked) { status.textContent = 'Алдымен «Ақау жоқ» белгісін алып тастаңыз.'; return; }
    event.preventDefault(); start = point(event); svg.setPointerCapture(event.pointerId);
  });
  svg.addEventListener('pointermove', event => {
    if (!start) return;
    const [x1,y1,x2,y2] = rectangle(start, point(event));
    draft.removeAttribute('hidden');
    for (const [key,value] of Object.entries({x:x1, y:y1, width:x2-x1, height:y2-y1})) draft.setAttribute(key,value);
  });
  svg.addEventListener('pointerup', event => {
    if (!start) return;
    const box = rectangle(start, point(event)); cancel();
    if (box[2]-box[0] >= 1 && box[3]-box[1] >= 1) add(box);
  });
  svg.addEventListener('pointercancel', cancel);
  document.addEventListener('keydown', event => { if (event.key === 'Escape') cancel(); });
  render();
})();
