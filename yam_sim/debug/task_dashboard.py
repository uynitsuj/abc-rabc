"""Live task-evaluation dashboard state and HTML."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Any

from yam_sim.task_eval import TaskEvalResult


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float))


@dataclass(frozen=True)
class _HistoryPoint:
    wall_time: float
    step: int
    sim_time: float
    reward: float
    success: bool
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_time": self.wall_time,
            "step": self.step,
            "sim_time": self.sim_time,
            "reward": self.reward,
            "success": self.success,
            "metrics": self.metrics,
        }


class TaskEvalDashboardState:
    """Thread-safe state for a live task-eval dashboard."""

    def __init__(
        self,
        *,
        task_name: str,
        prompt: str,
        evaluator_name: str | None,
        debug_spec: dict[str, Any] | None = None,
        history_limit: int = 2048,
    ) -> None:
        self.task_name = task_name
        self.prompt = prompt
        self.evaluator_name = evaluator_name
        self.debug_spec = debug_spec
        self.history_limit = int(history_limit)
        self._lock = threading.Lock()
        self._history: deque[_HistoryPoint] = deque(maxlen=self.history_limit)
        self._current: dict[str, Any] | None = None

    @property
    def available(self) -> bool:
        return self.evaluator_name is not None

    def update(
        self,
        *,
        step: int,
        sim_time: float,
        result: TaskEvalResult | None,
    ) -> None:
        with self._lock:
            if result is None:
                self._current = {
                    "wall_time": time.time(),
                    "step": int(step),
                    "sim_time": float(sim_time),
                    "reward": None,
                    "success": None,
                    "metrics": {},
                }
                return

            info = result.to_info(squeeze=True)
            reward = float(info.pop("reward"))
            success = bool(info.pop("success"))
            point = _HistoryPoint(
                wall_time=time.time(),
                step=int(step),
                sim_time=float(sim_time),
                reward=reward,
                success=success,
                metrics=info,
            )
            self._history.append(point)
            self._current = point.to_dict()

    def snapshot(self, *, history_tail: int = 256) -> dict[str, Any]:
        with self._lock:
            history = list(self._history)
            if history_tail > 0:
                history = history[-int(history_tail):]
            current = self._current

        numeric_metric_keys: list[str] = []
        if current is not None:
            for key, value in current["metrics"].items():
                if _is_numeric_scalar(value):
                    numeric_metric_keys.append(key)

        return {
            "available": self.available,
            "task_name": self.task_name,
            "prompt": self.prompt,
            "evaluator_name": self.evaluator_name,
            "debug_spec": self.debug_spec,
            "history_limit": self.history_limit,
            "current": current,
            "history": [point.to_dict() for point in history],
            "numeric_metric_keys": numeric_metric_keys,
        }


TASK_DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>XDoF Task Debugger</title>
<style>
  :root {
    --bg: #0f1720;
    --panel: #16212d;
    --panel-2: #1c2a38;
    --text: #e6eef8;
    --muted: #96a8bd;
    --accent: #4fc3f7;
    --good: #4ade80;
    --bad: #fb7185;
    --border: #263445;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    background: radial-gradient(circle at top, #15202b, var(--bg) 55%);
    color: var(--text);
  }
  .shell {
    max-width: 1480px;
    margin: 0 auto;
    padding: 24px;
  }
  h1, h2, h3, p { margin: 0; }
  .topbar {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 16px;
    margin-bottom: 18px;
  }
  .subtitle { color: var(--muted); margin-top: 6px; }
  .links {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    align-items: center;
  }
  .links a, .mode-toggle {
    color: var(--accent);
    text-decoration: none;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.03);
    padding: 8px 10px;
    border-radius: 10px;
  }
  .mode-toggle {
    cursor: pointer;
    font: inherit;
    line-height: 1.2;
    color: var(--text);
  }
  .mode-toggle:hover {
    border-color: rgba(79,195,247,0.35);
    background: rgba(79,195,247,0.08);
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 18px;
  }
  .card, .panel {
    background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 14px 16px;
    backdrop-filter: blur(8px);
  }
  .card .label {
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
  }
  .card .value {
    font-size: 28px;
    font-weight: 700;
  }
  .success-true { color: var(--good); }
  .success-false { color: var(--bad); }
  .pill-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 16px;
  }
  .pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 999px;
    color: var(--text);
    font-size: 12px;
  }
  .pill-key {
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 11px;
  }
  .panels {
    display: grid;
    grid-template-columns: minmax(0, 1.9fr) minmax(300px, 0.72fr);
    gap: 16px;
    align-items: start;
  }
  .panel--plots { padding: 18px; }
  .panel--metrics {
    max-width: 380px;
    width: 100%;
    justify-self: end;
    position: sticky;
    top: 20px;
    max-height: calc(100vh - 40px);
    overflow: auto;
  }
  body.focus-plots .shell {
    max-width: 1720px;
  }
  body.focus-plots .panels {
    grid-template-columns: 1fr;
  }
  body.focus-plots .panel--metrics {
    display: none;
  }
  body.focus-plots .plots-grid {
    grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
  }
  .plots-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 16px;
  }
  .plot-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px;
  }
  .plot-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
  }
  .plot-title {
    font-size: 13px;
    color: var(--text);
    font-weight: 600;
  }
  .plot-value {
    font-size: 12px;
    font-weight: 700;
    color: var(--accent);
    background: rgba(79,195,247,0.08);
    border: 1px solid rgba(79,195,247,0.2);
    border-radius: 999px;
    padding: 4px 8px;
  }
  .plot-meta {
    margin-top: 8px;
    font-size: 11px;
    color: var(--muted);
    min-height: 14px;
  }
  canvas {
    width: 100%;
    height: 228px;
    display: block;
    background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.05));
    border-radius: 10px;
  }
  .metrics-table-wrap {
    max-height: 340px;
    overflow: auto;
    margin-top: 10px;
    border-top: 1px solid rgba(255,255,255,0.04);
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }
  table {
    width: 100%;
    border-collapse: collapse;
  }
  th, td {
    text-align: left;
    padding: 7px 6px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    vertical-align: top;
    font-size: 12px;
  }
  th { color: var(--muted); font-weight: 500; width: 38%; }
  .status {
    color: var(--muted);
    margin-top: 10px;
    font-size: 12px;
  }
  details.snapshot {
    margin-top: 14px;
    border-top: 1px solid rgba(255,255,255,0.06);
    padding-top: 12px;
  }
  details.snapshot summary {
    cursor: pointer;
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    list-style: none;
  }
  details.snapshot summary::-webkit-details-marker { display: none; }
  details.snapshot[open] summary { margin-bottom: 10px; }
  pre {
    white-space: pre-wrap;
    word-break: break-word;
    margin: 0;
    color: #d6e4f0;
    font-size: 11px;
    line-height: 1.5;
  }
  @media (max-width: 900px) {
    .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .panels { grid-template-columns: 1fr; }
    .plots-grid { grid-template-columns: 1fr; }
    .panel--metrics {
      max-width: none;
      position: static;
      max-height: none;
    }
  }
</style>
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <div>
        <h1>Task Eval Debugger</h1>
        <p class="subtitle" id="subtitle">Loading…</p>
      </div>
      <div class="links">
        <button class="mode-toggle" id="toggleMetricsBtn" type="button">Focus Plots</button>
        <a href="/" target="_blank" rel="noreferrer">Open VR Stream</a>
        <a href="/api/task-eval" target="_blank" rel="noreferrer">Raw JSON</a>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <div class="label">Step</div>
        <div class="value" id="stepValue">-</div>
      </div>
      <div class="card">
        <div class="label">Sim Time</div>
        <div class="value" id="simTimeValue">-</div>
      </div>
      <div class="card">
        <div class="label">Reward</div>
        <div class="value" id="rewardValue">-</div>
      </div>
      <div class="card">
        <div class="label">Success</div>
        <div class="value" id="successValue">-</div>
      </div>
    </div>

    <div class="pill-row" id="pillRow"></div>

    <div class="panels">
      <div class="panel panel--plots">
        <h2 style="font-size:16px; margin-bottom:12px;">Debug Plots</h2>
        <div class="plots-grid" id="plotsGrid">
          <div class="status">Waiting for evaluator spec…</div>
        </div>
      </div>

      <div class="panel panel--metrics">
        <h2 style="font-size:16px;">Current Metrics</h2>
        <div class="metrics-table-wrap">
          <table id="metricsTable"></table>
        </div>
        <div class="status" id="statusLine">Waiting for updates…</div>
        <details class="snapshot">
          <summary>Raw Snapshot</summary>
          <pre id="snapshotPre">{}</pre>
        </details>
      </div>
    </div>
  </div>

<script>
const LAYOUT_STORAGE_KEY = 'yam_task_debug_layout';
let lastPayload = null;

function fmt(value, digits=3) {
  if (value === null || value === undefined) return '-';
  if (typeof value === 'number') return Number(value).toFixed(digits);
  return String(value);
}

function summarize(value) {
  if (Array.isArray(value)) return JSON.stringify(value);
  if (value && typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function isCompactMetricValue(value) {
  if (value === null || value === undefined) return true;
  if (typeof value === 'number' || typeof value === 'boolean' || typeof value === 'string') return true;
  if (Array.isArray(value)) {
    return value.length <= 4 && value.every(item =>
      typeof item === 'number' || typeof item === 'boolean' || typeof item === 'string'
    );
  }
  return false;
}

function compactMetricEntries(metrics) {
  return Object.entries(metrics || {}).filter(([_, value]) => isCompactMetricValue(value));
}

function setPlotFocus(enabled) {
  document.body.classList.toggle('focus-plots', Boolean(enabled));
  const button = document.getElementById('toggleMetricsBtn');
  if (button) {
    button.textContent = enabled ? 'Show Metrics' : 'Focus Plots';
    button.setAttribute('aria-pressed', enabled ? 'true' : 'false');
  }
  try {
    localStorage.setItem(LAYOUT_STORAGE_KEY, enabled ? 'plots' : 'split');
  } catch (_) {}
}

function initLayoutToggle() {
  let enabled = false;
  try {
    enabled = localStorage.getItem(LAYOUT_STORAGE_KEY) === 'plots';
  } catch (_) {}
  setPlotFocus(enabled);

  const button = document.getElementById('toggleMetricsBtn');
  if (!button) return;
  button.addEventListener('click', () => {
    setPlotFocus(!document.body.classList.contains('focus-plots'));
    if (lastPayload) {
      renderDashboard(lastPayload);
    }
  });
}

function ensureCanvasSize(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  const displayWidth = Math.max(320, Math.round(rect.width));
  const displayHeight = Math.max(228, Math.round(rect.height));
  const renderWidth = Math.round(displayWidth * dpr);
  const renderHeight = Math.round(displayHeight * dpr);

  if (canvas.width !== renderWidth || canvas.height !== renderHeight) {
    canvas.width = renderWidth;
    canvas.height = renderHeight;
  }

  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: displayWidth, height: displayHeight };
}

function getCurrentMetricValue(current, key) {
  if (!current) return null;
  if (key === 'reward') return current.reward;
  if (key === 'success') return current.success ? 1 : 0;
  const metrics = current.metrics || {};
  return metrics[key];
}

function formatPlotValue(value, kind) {
  if (value === null || value === undefined) return 'no data';
  if (kind === 'bool') return value ? 'true' : 'false';
  if (typeof value === 'number') return fmt(value, 3);
  return summarize(value);
}

function formatThresholds(thresholds) {
  if (!Array.isArray(thresholds) || thresholds.length === 0) return '';
  return thresholds.map(t => {
    const dir = t.direction === 'gt' ? '≥' : (t.direction === 'lt' ? '≤' : '=');
    const label = t.label ? `${t.label}: ` : '';
    return `${label}${dir} ${fmt(Number(t.value), 2)}`;
  }).join(' · ');
}

function drawLinePlot(canvas, xs, ys, thresholds, kind, color) {
  const { ctx, width, height } = ensureCanvasSize(canvas);
  ctx.clearRect(0, 0, width, height);

  ctx.fillStyle = '#16212d';
  ctx.fillRect(0, 0, width, height);

  const padding = 28;
  const plotW = width - padding * 2;
  const plotH = height - padding * 2 - 6;

  const values = ys.filter(value => Number.isFinite(value));
  let minVal = values.length ? Math.min(...values) : 0;
  let maxVal = values.length ? Math.max(...values) : 1;
  if (Array.isArray(thresholds)) {
    for (const threshold of thresholds) {
      const thresholdValue = Number(threshold.value);
      if (Number.isFinite(thresholdValue)) {
        minVal = Math.min(minVal, thresholdValue);
        maxVal = Math.max(maxVal, thresholdValue);
      }
    }
  }
  if (kind === 'bool') {
    minVal = Math.min(minVal, 0);
    maxVal = Math.max(maxVal, 1);
  }
  const range = Math.max(maxVal - minVal, 1e-6);

  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padding + (plotH * i / 4);
    ctx.beginPath();
    ctx.moveTo(padding, y);
    ctx.lineTo(padding + plotW, y);
    ctx.stroke();
  }

  if (Array.isArray(thresholds)) {
    for (const threshold of thresholds) {
      const thresholdValue = Number(threshold.value);
      if (!Number.isFinite(thresholdValue)) continue;
      const y = padding + plotH - ((thresholdValue - minVal) / range) * plotH;
      ctx.save();
      ctx.setLineDash([6, 6]);
      ctx.strokeStyle = 'rgba(255,255,255,0.35)';
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(padding + plotW, y);
      ctx.stroke();
      ctx.restore();
    }
  }

  if (kind === 'bool') {
    for (const value of [0, 1]) {
      const y = padding + plotH - ((value - minVal) / range) * plotH;
      ctx.save();
      ctx.setLineDash([2, 4]);
      ctx.strokeStyle = 'rgba(255,255,255,0.14)';
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(padding + plotW, y);
      ctx.stroke();
      ctx.restore();
    }
  }

  if (ys.length > 0) {
    ctx.strokeStyle = color || '#4fc3f7';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ys.forEach((val, i) => {
      const x = padding + (ys.length === 1 ? plotW / 2 : plotW * i / (ys.length - 1));
      const y = padding + plotH - ((val - minVal) / range) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  ctx.fillStyle = '#96a8bd';
  ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
  ctx.textAlign = 'left';
  ctx.fillText(fmt(maxVal, 2), 6, padding + 10);
  ctx.fillText(fmt(minVal, 2), 6, padding + plotH);
}

function extractMetricSeries(history, key, xKey) {
  const xs = [];
  const ys = [];
  history.forEach(point => {
    const x = Number(point[xKey ?? 'step']);
    const raw = key === 'reward'
      ? point.reward
      : (key === 'success'
          ? (point.success ? 1 : 0)
          : (point.metrics ? point.metrics[key] : null));
    const y = typeof raw === 'number' ? Number(raw) : raw;
    if (Number.isFinite(x) && Number.isFinite(y)) {
      xs.push(x);
      ys.push(y);
    }
  });
  return { xs, ys };
}

function renderPills(current) {
  const pillRow = document.getElementById('pillRow');
  const metrics = current.metrics || {};
  const pills = [];

  if (Array.isArray(metrics.bottles_in_bin)) {
    pills.push(`<span class="pill"><span class="pill-key">in bin</span><span>${metrics.bottles_in_bin.length ? metrics.bottles_in_bin.join(', ') : 'none'}</span></span>`);
  }
  if (metrics.num_bottles_in_bin !== undefined) {
    pills.push(`<span class="pill"><span class="pill-key">count</span><span>${metrics.num_bottles_in_bin}</span></span>`);
  }
  if (metrics.max_bottles_in_bin_so_far !== undefined) {
    pills.push(`<span class="pill"><span class="pill-key">max so far</span><span>${metrics.max_bottles_in_bin_so_far}</span></span>`);
  }
  if (metrics.success_count !== undefined) {
    pills.push(`<span class="pill"><span class="pill-key">target</span><span>${metrics.success_count}</span></span>`);
  }
  pillRow.innerHTML = pills.join('');
}

function renderPlotCards(current, history, debugSpec, fallbackNumericKeys) {
  const container = document.getElementById('plotsGrid');
  const plots = Array.isArray(debugSpec?.plots) && debugSpec.plots.length
    ? debugSpec.plots
    : [
        { key: 'reward', title: 'Reward', color: '#4fc3f7', kind: 'line', thresholds: [] },
        { key: 'success', title: 'Success', color: '#34d399', kind: 'bool', thresholds: [] },
        ...fallbackNumericKeys
          .filter(key => key !== 'reward' && key !== 'success')
          .slice(0, 4)
          .map((key, idx) => ({
            key,
            title: key,
            color: ['#f59e0b', '#a78bfa', '#fb7185', '#7dd3fc'][idx % 4],
            kind: 'line',
            thresholds: [],
          })),
      ];
  const xKey = debugSpec?.x_key || 'step';

  const cards = [];
  plots.forEach((plot, idx) => {
    const plotId = `plotCanvas_${idx}`;
    const currentValue = getCurrentMetricValue(current, plot.key);
    const thresholdText = formatThresholds(plot.thresholds || []);
    cards.push(`
      <div class="plot-card">
        <div class="plot-head">
          <div class="plot-title">${plot.title || plot.key}</div>
          <div class="plot-value">${formatPlotValue(currentValue, plot.kind || 'line')}</div>
        </div>
        <canvas id="${plotId}"></canvas>
        <div class="plot-meta">${thresholdText}</div>
      </div>
    `);
  });
  container.innerHTML = cards.join('');

  plots.forEach((plot, idx) => {
    const canvas = document.getElementById(`plotCanvas_${idx}`);
    if (!canvas) return;
    const { xs, ys } = extractMetricSeries(history, plot.key, xKey);
    drawLinePlot(canvas, xs, ys, plot.thresholds || [], plot.kind || 'line', plot.color || null);
  });
}

function renderDashboard(data) {
  lastPayload = data;
  const subtitle = document.getElementById('subtitle');
  const statusLine = document.getElementById('statusLine');
  if (!data.available) {
    subtitle.textContent = `${data.task_name} · no evaluator configured`;
    statusLine.textContent = 'This task does not currently expose a task evaluator.';
    return;
  }

  const current = data.current || {};
  subtitle.textContent = `${data.task_name} · ${data.prompt} · ${data.evaluator_name}`;
  document.getElementById('stepValue').textContent = current.step ?? '-';
  document.getElementById('simTimeValue').textContent = current.sim_time !== undefined ? fmt(current.sim_time, 2) + 's' : '-';
  document.getElementById('rewardValue').textContent = current.reward !== undefined && current.reward !== null ? fmt(current.reward, 3) : '-';

  const successEl = document.getElementById('successValue');
  const success = current.success;
  successEl.textContent = success === null || success === undefined ? '-' : (success ? 'true' : 'false');
  successEl.className = 'value ' + (success ? 'success-true' : 'success-false');

  const metricsTable = document.getElementById('metricsTable');
  const metrics = current.metrics || {};
  metricsTable.innerHTML = compactMetricEntries(metrics)
    .map(([key, value]) => `<tr><th>${key}</th><td>${summarize(value)}</td></tr>`)
    .join('');
  renderPills(current);

  document.getElementById('snapshotPre').textContent = JSON.stringify(current, null, 2);
  statusLine.textContent = `history: ${data.history.length} points · updated ${current.wall_time ? new Date(current.wall_time * 1000).toLocaleTimeString() : '-'}`;

  const history = data.history || [];
  renderPlotCards(current, history, data.debug_spec, data.numeric_metric_keys || []);
}

async function refresh() {
  const response = await fetch('/api/task-eval');
  const data = await response.json();
  renderDashboard(data);
}

async function loop() {
  try {
    await refresh();
  } catch (err) {
    document.getElementById('statusLine').textContent = 'Failed to fetch task-eval state: ' + err;
  } finally {
    setTimeout(loop, 250);
  }
}

window.addEventListener('resize', () => {
  if (lastPayload) {
    renderDashboard(lastPayload);
  }
});

initLayoutToggle();
loop();
</script>
</body>
</html>
"""
