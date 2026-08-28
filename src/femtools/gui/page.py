"""Embedded single-page front end for the femtools GUI.

Kept as a Python string so the wheel needs no static data files; both
GUI backends serve it at ``/``.
"""

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>femtools</title>
<style>
  :root {
    --bg: #0f1420; --panel: #182032; --line: #2a3550;
    --text: #dce3f2; --muted: #8b98b8; --accent: #5aa9ff; --ok: #4fd08c; --err: #ff6b6b;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  header {
    display: flex; align-items: baseline; gap: 12px;
    padding: 14px 22px; border-bottom: 1px solid var(--line); background: var(--panel);
  }
  header h1 { margin: 0; font-size: 20px; letter-spacing: .04em; }
  header .sub { color: var(--muted); font-size: 13px; }
  main {
    display: grid; grid-template-columns: minmax(340px, 1fr) minmax(380px, 1.2fr);
    gap: 18px; padding: 18px 22px; max-width: 1280px; margin: 0 auto;
  }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  section {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 10px; padding: 16px 18px;
  }
  h2 { margin: 0 0 10px; font-size: 14px; text-transform: uppercase;
       letter-spacing: .08em; color: var(--muted); }
  textarea {
    width: 100%; min-height: 220px; resize: vertical; border-radius: 8px;
    background: #0b0f19; color: var(--text); border: 1px solid var(--line);
    font: 13px/1.5 ui-monospace, "SF Mono", Menlo, Consolas, monospace; padding: 10px;
  }
  form.loader { display: flex; gap: 8px; align-items: center; }
  form.loader input {
    flex: 1; border-radius: 8px; background: #0b0f19; color: var(--text);
    border: 1px solid var(--line); padding: 8px 10px;
    font: 13px/1.4 ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  }
  form.loader button { margin-top: 0; }
  .hint { color: var(--muted); font-size: 12px; margin: 6px 0 0; }
  button {
    background: var(--accent); color: #071120; font-weight: 600; border: 0;
    border-radius: 8px; padding: 8px 18px; margin-top: 10px; cursor: pointer;
  }
  button.secondary { background: transparent; color: var(--accent);
                     border: 1px solid var(--accent); }
  button:hover { filter: brightness(1.15); }
  pre {
    background: #0b0f19; border: 1px solid var(--line); border-radius: 8px;
    padding: 10px; font-size: 12.5px; overflow: auto; max-height: 220px;
    white-space: pre-wrap;
  }
  .ok { color: var(--ok); } .err { color: var(--err); }
  .badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
  .badge {
    font-size: 11.5px; padding: 2px 9px; border-radius: 999px;
    border: 1px solid var(--line); color: var(--muted);
  }
  .badge.on { color: var(--ok); border-color: var(--ok); }
  .plots { display: flex; flex-wrap: wrap; gap: 12px; }
  .plots figure { margin: 0; }
  .plots img { max-width: 100%; border-radius: 8px; background: white; }
  table { border-collapse: collapse; font-size: 13.5px; }
  td, th { padding: 3px 12px 3px 0; text-align: left; color: var(--text); }
  th { color: var(--muted); font-weight: 500; }
</style>
</head>
<body>
<header>
  <h1>femtools</h1>
  <span class="sub" id="version">connecting…</span>
</header>
<main>
  <div>
    <section>
      <h2>FSL script</h2>
      <textarea id="script" spellcheck="false"># two-node axial bar
NEW PROJECT bar
ADD NODE 1 0 0 0
ADD NODE 2 1 0 0
ADD MAT 1 TYPE=isotropic E=210e9 NU=0.3 RHO=7850
ADD PROP 1 TYPE=bar MAT=1 A=1e-4
ADD ELEM 1 TYPE=BAR2 NODES=1,2 PROP=1
SPC 1 ALL
SOLVE MODES N=3
MAC</textarea>
      <div>
        <button id="run">Run script</button>
        <button id="refresh" class="secondary">Refresh</button>
      </div>
      <pre id="log">ready.</pre>
    </section>
    <section style="margin-top:18px">
      <h2>Load model file</h2>
      <form class="loader" id="loadform">
        <input id="modelpath" type="text" spellcheck="false"
               placeholder="/path/to/model.ftproj | .json | .unv | .bdf | .inp | .k">
        <button type="submit">Load</button>
      </form>
      <p class="hint">Path on the machine running this server; stored results
        (.ftproj/.unv) are imported too.</p>
    </section>
    <section style="margin-top:18px">
      <h2>Capabilities</h2>
      <div class="badges" id="modules"></div>
    </section>
  </div>
  <div>
    <section>
      <h2>Model</h2>
      <div id="model">no model loaded</div>
      <h2 style="margin-top:16px">Results</h2>
      <div id="results">none</div>
    </section>
    <section style="margin-top:18px">
      <h2>Stress</h2>
      <div id="stress" style="overflow:auto"><em style="color:var(--muted)">run
        SOLVE STATIC to recover element stresses</em></div>
    </section>
    <section style="margin-top:18px">
      <h2>SPR nodal stress</h2>
      <div id="spr" style="overflow:auto"><em style="color:var(--muted)">run
        SOLVE STATIC to recover ZZ-SPR nodal stresses</em></div>
    </section>
    <section style="margin-top:18px">
      <h2>Plots</h2>
      <div class="plots" id="plots"><em style="color:var(--muted)">run a
        script to render plots</em></div>
    </section>
  </div>
</main>
<script>
const $ = (id) => document.getElementById(id);

async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function refreshStatus() {
  try {
    const s = await jget('/api/status');
    $('version').textContent = 'v' + s.version;
    $('modules').innerHTML = Object.entries(s.modules).map(
      ([k, v]) => `<span class="badge ${v ? 'on' : ''}">${k}</span>`).join('');
  } catch (e) { $('version').textContent = 'offline'; }
}

function renderModel(m) {
  if (!m || !m.loaded) { $('model').textContent = 'no model loaded'; return; }
  $('model').innerHTML = `<table>
    <tr><th>project</th><td>${m.name ?? '—'}</td></tr>
    <tr><th>nodes</th><td>${m.n_nodes}</td></tr>
    <tr><th>elements</th><td>${m.n_elements}</td></tr>
    <tr><th>materials / properties</th><td>${m.n_materials} / ${m.n_properties}</td></tr>
    <tr><th>SPCs</th><td>${m.n_spcs}</td></tr></table>`;
}

function renderResults(list) {
  if (!list || !list.length) { $('results').textContent = 'none'; return; }
  $('results').innerHTML = list.map(r => {
    const f = r.freq_hz ? ' — ' + r.freq_hz.map(x => x.toPrecision(5)).join(', ') + ' Hz' : '';
    return `<div><b>${r.name}</b> <span style="color:var(--muted)">(${r.type})</span>${f}</div>`;
  }).join('');
}

function fmtNum(x) {
  return (typeof x === 'number' && isFinite(x)) ? x.toPrecision(4) : '—';
}

function renderStress(data, message) {
  if (!data) {  // expected 400s: no model / no static result / missing kernel
    $('stress').innerHTML =
      `<em style="color:var(--muted)">${message || 'no stress data'}</em>`;
    return;
  }
  const comps = data.components || [];
  const header = '<tr><th>element</th><th>type</th>'
    + comps.map(c => `<th>s${c}</th>`).join('') + '<th>von Mises</th></tr>';
  const rows = (data.elements || []).map(e =>
    `<tr><td>${e.element}</td><td>${e.type}</td>`
    + comps.map((c, i) => `<td>${fmtNum((e.stress || [])[i])}</td>`).join('')
    + `<td><b>${fmtNum(e.von_mises)}</b></td></tr>`).join('');
  const shown = (data.elements || []).length;
  const note = `${data.n_elements} element${data.n_elements === 1 ? '' : 's'}`
    + (data.truncated ? ` (first ${shown} shown)` : '')
    + ` — max von Mises ${fmtNum(data.max_von_mises)}`
    + (data.result ? ` — result: ${data.result}` : '');
  $('stress').innerHTML = `<table>${header}${rows}</table>`
    + `<p class="hint">${note}</p>`;
}

async function refreshStress() {
  let payload;
  try {
    const r = await fetch('/api/stress?max_rows=15');
    payload = await r.json();  // both backends answer 400s with JSON too
  } catch (e) { renderStress(null, 'stress unavailable'); return; }
  if (payload && payload.ok) renderStress(payload);
  else renderStress(null, (payload && payload.error) || 'no stress data');
}

function renderSpr(data, message) {
  if (!data) {  // expected 400s: no static result / SPR kernel not installed
    $('spr').innerHTML =
      `<em style="color:var(--muted)">${message || 'no SPR data'}</em>`;
    return;
  }
  const comps = data.components || [];
  const header = '<tr><th>node</th><th>n_adj</th>'
    + comps.map(c => `<th>s${c}</th>`).join('') + '<th>von Mises</th></tr>';
  const rows = (data.nodes || []).map(n =>
    `<tr><td>${n.node}</td><td>${n.count ?? '—'}</td>`
    + comps.map((c, i) => `<td>${fmtNum((n.stress || [])[i])}</td>`).join('')
    + `<td><b>${fmtNum(n.von_mises)}</b></td></tr>`).join('');
  const shown = (data.nodes || []).length;
  const note = `${data.n_nodes} node${data.n_nodes === 1 ? '' : 's'}`
    + (data.truncated ? ` (first ${shown} shown)` : '')
    + ` — max von Mises ${fmtNum(data.max_von_mises)}`
    + (data.result ? ` — result: ${data.result}` : '');
  $('spr').innerHTML = `<table>${header}${rows}</table>`
    + `<p class="hint">${note}</p>`;
}

async function refreshSpr() {
  let payload;
  try {
    const r = await fetch('/api/spr?max_rows=15');
    payload = await r.json();
  } catch (e) { renderSpr(null, 'SPR unavailable'); return; }
  if (payload && payload.ok) renderSpr(payload);
  else renderSpr(null, (payload && payload.error) || 'no SPR data');
}

async function refreshData() {
  try {
    renderModel(await jget('/api/model'));
    renderResults((await jget('/api/results')).results);
  } catch (e) { /* server gone */ }
  refreshStress();
  refreshSpr();
}

function plotNames(results) {
  const names = ['mesh', 'mode', 'mac'];
  if ((results || []).some(r => /static/i.test(r.type || ''))) names.push('stress');
  return names;
}

function showPlots(names) {
  const stamp = Date.now();
  $('plots').innerHTML = names.map(n =>
    `<figure><img src="/api/plot/${n}?t=${stamp}" alt="${n}"
       onerror="this.parentElement.style.display='none'"></figure>`).join('');
}

$('run').onclick = async () => {
  $('log').textContent = 'running…';
  try {
    const r = await fetch('/api/script', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: $('script').value }),
    });
    const data = await r.json();
    if (data.ok) {
      $('log').innerHTML = `<span class="ok">ok</span> — executed `
        + `${data.executed.length} statements\\n` + data.executed.join('\\n');
      renderModel(data.model); renderResults(data.results || []);
      showPlots(plotNames(data.results));
    } else {
      $('log').innerHTML = `<span class="err">error</span>\\n${data.error}\\n\\nexecuted:\\n`
        + (data.executed || []).join('\\n');
      renderModel(data.model);
    }
    refreshStress();
    refreshSpr();
  } catch (e) { $('log').innerHTML = `<span class="err">request failed:</span> ${e}`; }
};
$('loadform').onsubmit = async (ev) => {
  ev.preventDefault();
  $('log').textContent = 'loading model…';
  try {
    const r = await fetch('/api/load', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: $('modelpath').value }),
    });
    const data = await r.json();
    if (data.ok) {
      $('log').innerHTML = `<span class="ok">ok</span> — loaded ${data.path} `
        + `(${data.format})`;
      renderModel(data.model); renderResults(data.results || []);
      showPlots((data.results || []).length ? plotNames(data.results) : ['mesh']);
    } else {
      $('log').innerHTML = `<span class="err">error</span>\\n${data.error}`;
    }
    refreshStress();
    refreshSpr();
  } catch (e) { $('log').innerHTML = `<span class="err">request failed:</span> ${e}`; }
};
$('refresh').onclick = () => { refreshStatus(); refreshData(); };
refreshStatus(); refreshData();
</script>
</body>
</html>
"""
