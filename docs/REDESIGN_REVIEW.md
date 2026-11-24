# Website redesign verification · 2026-11-20

## Installed skill

- Vercel source: <https://github.com/vercel-labs/agent-skills/tree/main/skills/web-design-guidelines>
- Global installation: `~/.config/opencode/skills/web-design-guidelines/SKILL.md`
- Guidelines reviewed: <https://raw.githubusercontent.com/vercel-labs/web-interface-guidelines/main/command.md>

## Resolved review findings

### jol/static/app.css

- `jol/static/app.css:58` — ✓ secondary text contrast corrected across navigation, tables, forms and badges.
- `jol/static/app.css:279` — ✓ table scroll container contains offscreen accessible labels; no page overflow at 320/390/768 px.

### jol/static/app.js

- `jol/static/app.js:29` — ✓ mobile dialog wraps Tab/Shift+Tab; Escape restores opener focus.
- `jol/static/app.js:51` — ✓ Kazakh inline errors announced; first invalid input receives focus.
- `jol/static/app.js:5` — ✓ shared locale-aware numeric formatting; date formatting uses Asia/Almaty.

### jol/templates/video.html

- `jol/templates/video.html:10` — ✓ labelled progress and live heading updates preserve heading semantics.
- Evidence images include intrinsic dimensions, descriptive alternatives and lazy loading.

### jol/static/map.js

- ✓ Filters survive reload and browser history; reset updates the URL.
- ✓ Request failure gives a Kazakh recovery instruction; reset retries successfully.

## Recorded checks

- `.venv/bin/python -m pytest -q`: **25 passed in 6.24s**, including 3 real-model integration tests.
- `.venv/bin/python scripts/browser_smoke.py`: **passed**, Chromium.
- axe-core **4.10.3**: **0 automated violations in 20 page states** for WCAG 2 A/AA, WCAG 2.1 AA and best-practice rules. Complete results, including checks requiring manual review, are saved in `artifacts/redesign/accessibility.json`.
- Desktop screenshots: 1440 px. Responsive overflow checks: 320, 390 and 768 px.
- Verified login, file validation, actual file drag/drop, unsaved-input protection, keyboard skip link, evidence deep links, rejection of an unjustified review, map filters/history/reset, network-failure recovery, PDF text, complaint download and clipboard copying.
- Verified anonymous raw/evidence requests redirect to login and the event API rejects anonymous access.
- Real demo: 12 analyzed frames and 7 prediction events. The browser run reused the existing SHA-256-matched analysis; this was not new inference. No coordinates or contracts were added.
- JavaScript page errors: **0**.

## Artifacts

- `artifacts/redesign/browser-check.json`
- `artifacts/redesign/accessibility.json`
- `artifacts/redesign/screenshots/` — actual Chromium captures of the redesigned application.
- `artifacts/redesign/demo-monitoring-report.pdf`

For an accessibility-enabled rerun, place axe-core 4.10.3 at `.bootstrap/axe.min.js` before running the browser script. The source used for this run was <https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.10.3/axe.min.js>. Without that optional local file, the script still runs workflow checks and explicitly records zero accessibility pages.
