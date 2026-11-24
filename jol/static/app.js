'use strict';

// Shared locale formatting keeps dates and numeric columns consistent.
const locale = 'kk-KZ';
function formatNumber(value, digits = 0) {
  return new Intl.NumberFormat(locale, {minimumFractionDigits: digits, maximumFractionDigits: digits}).format(value);
}
document.querySelectorAll('[data-number]').forEach(node => {
  node.textContent = formatNumber(Number(node.dataset.number), Number(node.dataset.digits || 0));
});
document.querySelectorAll('time[data-date]').forEach(node => {
  const value = new Date(node.dateTime);
  if (Number.isNaN(value.getTime())) return;
  const options = {dateStyle: node.dataset.date === 'long' ? 'long' : 'short', timeZone: 'Asia/Almaty'};
  if (node.dataset.date === 'time') options.timeStyle = 'short';
  node.textContent = new Intl.DateTimeFormat(locale, options).format(value);
});

// Native dialog provides focus containment and Escape-to-close on mobile.
const navigation = document.querySelector('#navigation-dialog');
const menuButton = document.querySelector('#menu-toggle');
menuButton?.addEventListener('click', () => {
  navigation.showModal();
  menuButton.setAttribute('aria-expanded', 'true');
});
document.querySelector('[data-close-menu]')?.addEventListener('click', () => navigation.close());
navigation?.addEventListener('close', () => menuButton.setAttribute('aria-expanded', 'false'));
navigation?.addEventListener('click', event => { if (event.target === navigation) navigation.close(); });
navigation?.addEventListener('keydown', event => {
  if (event.key !== 'Tab') return;
  const controls = navigation.querySelectorAll('a[href], button:not(:disabled)');
  const first = controls[0];
  const last = controls[controls.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault(); last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault(); first.focus();
  }
});

// Inline validation uses Kazakh messages and keeps the first invalid control focused.
let fieldSequence = 0;
function fieldError(control, message) {
  if (!control.id) control.id = `field-${++fieldSequence}`;
  const id = `${control.id}-error`;
  let error = document.getElementById(id);
  if (!error) {
    error = document.createElement('p');
    error.id = id;
    error.className = 'form-error';
    error.setAttribute('aria-live', 'polite');
    control.insertAdjacentElement('afterend', error);
    const describedBy = control.getAttribute('aria-describedby') || '';
    control.setAttribute('aria-describedby', `${describedBy} ${id}`.trim());
  }
  error.textContent = message;
  control.setAttribute('aria-invalid', message ? 'true' : 'false');
}
function validationMessage(control) {
  if (control.validity.valueMissing) return control.type === 'checkbox' ? 'Жалғастыру үшін бұл өрісті растаңыз.' : 'Бұл өрісті толтырыңыз.';
  if (control.validity.typeMismatch) return 'Көрсетілген форматқа сәйкес мән енгізіңіз.';
  if (control.validity.rangeUnderflow || control.validity.rangeOverflow) return `Рұқсат етілген аралық: ${control.min || '—'} – ${control.max || '—'}.`;
  return 'Мәнді тексеріп, дұрыс форматта енгізіңіз.';
}
const dirtyForms = new Set();
document.querySelectorAll('form').forEach(form => {
  if (!form.hasAttribute('autocomplete')) form.autocomplete = 'off';
  form.querySelectorAll('input[type="url"], input[name="identifier"]').forEach(input => { input.spellcheck = false; });
  form.querySelectorAll('input[type="number"]').forEach(input => { input.inputMode = 'decimal'; });
  form.addEventListener('invalid', event => {
    event.preventDefault();
    fieldError(event.target, validationMessage(event.target));
    form.querySelector(':invalid')?.focus();
  }, true);
  form.addEventListener('input', event => {
    if (event.target.matches('input,select,textarea') && event.target.getAttribute('aria-invalid') === 'true') fieldError(event.target, '');
    if (form.hasAttribute('data-dirty-warning')) dirtyForms.add(form);
  });
  form.addEventListener('submit', event => {
    if (event.defaultPrevented || form.method.toLowerCase() !== 'post') return;
    // Upload-specific validation runs before marking a request as in flight.
    if (form.id === 'upload-form' && !validateVideoFile()) { event.preventDefault(); return; }
    dirtyForms.clear();
    const button = event.submitter;
    if (button) {
      // Disabled submit buttons are omitted from HTML form data; retain their verdict.
      form.querySelectorAll('[data-submit-value]').forEach(input => input.remove());
      if (button.name) {
        const value = document.createElement('input');
        value.type = 'hidden'; value.name = button.name; value.value = button.value;
        value.dataset.submitValue = 'true'; form.append(value);
      }
      button.disabled = true;
      button.setAttribute('aria-busy', 'true');
      const spinner = document.createElement('span');
      spinner.className = 'spinner';
      spinner.setAttribute('aria-hidden', 'true');
      button.prepend(spinner);
    }
    if (form.id === 'upload-form') document.querySelector('#upload-status').textContent = 'Бейне жүктелуде. Жүктеу аяқталғанша бетті жаппаңыз…';
  });
});
window.addEventListener('beforeunload', event => {
  if (dirtyForms.size) { event.preventDefault(); event.returnValue = ''; }
});
// Back/forward cache must not leave a formerly submitted button disabled.
window.addEventListener('pageshow', () => {
  document.querySelectorAll('[data-submit-value]').forEach(input => input.remove());
  document.querySelectorAll('button[aria-busy="true"]').forEach(button => {
    button.disabled = false;
    button.removeAttribute('aria-busy');
    button.querySelector('.spinner')?.remove();
  });
});
document.querySelector('.flash-message.error')?.focus();

// Presets are authoritative on the server; reflect their fixed confidence in the form.
document.querySelectorAll('[data-preset-select]').forEach(select => {
  const confidence = select.form.querySelector('[name="confidence"]');
  if (!confidence) return;
  let manual = confidence.value;
  let previous = 'default';
  function updatePreset() {
    if (previous === 'default') manual = confidence.value;
    confidence.readOnly = select.value !== 'default';
    confidence.value = confidence.readOnly ? select.selectedOptions[0].dataset.confidence : manual;
    previous = select.value;
  }
  select.addEventListener('change', updatePreset);
  updatePreset();
});

const file = document.querySelector('#video-file');
function validateVideoFile() {
  if (!file) return true;
  const selected = file.files[0];
  if (!selected) return false;
  let message = '';
  if (!/\.(mp4|mov|avi)$/i.test(selected.name)) message = 'MP4, MOV немесе AVI бейнесін таңдаңыз.';
  else if (selected.size > Number(file.dataset.maxBytes)) message = `Файл тым үлкен. Ең көбі ${formatNumber(Number(file.dataset.maxBytes) / 1048576)} МБ бейне таңдаңыз.`;
  if (message) {
    fieldError(file, message);
    file.focus();
    return false;
  }
  fieldError(file, '');
  return true;
}
file?.addEventListener('change', () => {
  const selected = file.files[0];
  const info = document.querySelector('#file-info');
  info.textContent = selected ? `${selected.name} · ${formatNumber(selected.size / 1048576, 1)} МБ` : 'Бейне файлын таңдаңыз.';
  info.className = selected ? 'file-selection' : 'help';
  if (selected) validateVideoFile();
  dirtyForms.add(file.form);
});
const dropZone = document.querySelector('#upload-zone');
['dragenter', 'dragover'].forEach(name => dropZone?.addEventListener(name, event => {
  event.preventDefault();
  dropZone.classList.add('drag-over');
}));
dropZone?.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone?.addEventListener('drop', event => {
  event.preventDefault();
  dropZone.classList.remove('drag-over');
  if (event.dataTransfer.files.length !== 1) {
    document.querySelector('#file-info').textContent = 'Бір бейне файлын таңдаңыз.';
    return;
  }
  file.files = event.dataTransfer.files;
  file.dispatchEvent(new Event('change', {bubbles: true}));
});

// Evidence panels can be opened via a URL without losing map filter parameters.
const panelIds = new Set(new URL(window.location.href).searchParams.getAll('panel'));
document.querySelectorAll('details[data-url-panel]').forEach(panel => {
  if (panelIds.has(panel.id)) panel.open = true;
  panel.addEventListener('toggle', () => {
    const url = new URL(window.location.href);
    const ids = new Set(url.searchParams.getAll('panel'));
    if (panel.open) ids.add(panel.id); else ids.delete(panel.id);
    url.searchParams.delete('panel');
    ids.forEach(id => url.searchParams.append('panel', id));
    window.history.replaceState(null, '', url);
  });
});

const job = document.querySelector('#job-status');
if (job && ['queued', 'processing'].includes(job.dataset.state)) {
  const names = {queued: 'Кезекте', processing: 'Талдануда', completed: 'Аяқталды', failed: 'Қате'};
  const retry = document.querySelector('#retry-status');
  async function update() {
    retry.hidden = true;
    try {
      const response = await fetch(`/api/videos/${job.dataset.video}/status`);
      if (!response.ok) throw new Error('Күйді алу мүмкін болмады. Байланысты тексеріп, қайта жүктеңіз немесе жүйеге қайта кіріңіз.');
      const state = await response.json();
      document.querySelector('#job-error').hidden = true;
      document.querySelector('#progress').value = state.progress;
      document.querySelector('#progress-label').textContent = `${formatNumber(state.progress)}%`;
      document.querySelector('#status-label').textContent = names[state.status];
      document.querySelector('#frame-count').textContent = `Талданған кадр: ${formatNumber(state.frames)}`;
      if (['completed', 'failed'].includes(state.status)) { window.location.reload(); return; }
      window.setTimeout(update, 2000);
    } catch (error) {
      const message = document.querySelector('#job-error');
      message.textContent = error instanceof TypeError ? 'Байланыс үзілді. Интернетті тексеріп, күйді қайта жүктеңіз.' : error.message;
      message.hidden = false;
      retry.hidden = false;
    }
  }
  retry.addEventListener('click', update);
  update();
}
document.querySelectorAll('[data-copy]').forEach(button => button.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(document.querySelector(button.dataset.copy).textContent);
    document.querySelector('#announcements').textContent = 'Өтініш мәтіні көшірілді.';
    button.textContent = 'Көшірілді ✓';
  } catch (_) { document.querySelector('#announcements').textContent = 'Мәтінді қолмен белгілеу арқылы көшіріңіз.'; }
}));
