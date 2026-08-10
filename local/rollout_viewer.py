"""Local web viewer for failure-injection eval rollouts.

Serves an eval output directory (the harness's ``output_dir``) as a browsable
dashboard: per-episode video with seek markers at the slip-injection and rebin
moments, slip_* metrics as badges, and aggregate stat tiles. Reads
``results.jsonl`` live (the page polls), so it can be left open while a run is
still writing episodes; finished videos that are not yet in results.jsonl show
up as "recording" cards.

Stdlib only. Byte-range requests are supported so <video> seeking works.

Run:
    python local/rollout_viewer.py --root /nfs_us_2/karim/warp/eval_out/slip_rigup --port 8020
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path("/nfs_us_2/karim/warp/eval_out/slip_rigup")

# Video frames are appended once per control step (~29.4 Hz sim) and encoded at
# this fps, so video_time ~= slip_step / VIDEO_FPS.
VIDEO_FPS = 30.0


def _collect_results(root: Path) -> dict:
    episodes = []
    recorded_videos = set()
    results_path = root / "results.jsonl"
    if results_path.is_file():
        for line in results_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            video_path = rec.get("video_path")
            if not video_path:
                # video: false runs render offline (local/render_rollouts.py)
                # into the same seed_<n>.mp4 location -- attach if present.
                guess = root / rec["model_label"] / rec["task"] / f"seed_{rec['seed']}.mp4"
                if guess.is_file():
                    video_path = str(guess)
            video_url = None
            if video_path:
                try:
                    rel = Path(video_path).resolve().relative_to(root.resolve())
                    video_url = f"/files/{rel.as_posix()}"
                    recorded_videos.add(rel.as_posix())
                except ValueError:
                    pass
            rec["video_url"] = video_url
            episodes.append(rec)

    pending = []
    for mp4 in sorted(root.rglob("seed_*.mp4")):
        rel = mp4.resolve().relative_to(root.resolve()).as_posix()
        if rel not in recorded_videos:
            pending.append(
                {
                    "video_url": f"/files/{rel}",
                    "rel_path": rel,
                    "mtime": mp4.stat().st_mtime,
                    "size_mb": round(mp4.stat().st_size / 1e6, 1),
                }
            )
    return {
        "root": str(root),
        "generated_at": time.time(),
        "episodes": episodes,
        "pending_videos": pending,
        "video_fps": VIDEO_FPS,
    }


class ViewerHandler(BaseHTTPRequestHandler):
    root: Path = ROOT

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send_bytes(self, payload: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                self._send_bytes(INDEX_HTML.encode(), "text/html; charset=utf-8")
            elif path == "/api/results":
                payload = json.dumps(_collect_results(self.root)).encode()
                self._send_bytes(payload, "application/json")
            elif path.startswith("/files/"):
                self._serve_file(unquote(path[len("/files/"):]))
            else:
                self.send_error(404)
        except BrokenPipeError:
            pass

    def _serve_file(self, rel: str) -> None:
        target = (self.root / rel).resolve()
        if not str(target).startswith(str(self.root.resolve())) or not target.is_file():
            self.send_error(404)
            return
        size = target.stat().st_size
        ctype = "video/mp4" if target.suffix == ".mp4" else "application/octet-stream"
        range_header = self.headers.get("Range")
        start, end = 0, size - 1
        status = 200
        if range_header and range_header.startswith("bytes="):
            spec = range_header[len("bytes="):].split(",")[0].strip()
            s, _, e = spec.partition("-")
            if s:
                start = int(s)
                end = int(e) if e else size - 1
            elif e:  # suffix range: last N bytes
                start = max(0, size - int(e))
            end = min(end, size - 1)
            if start > end:
                self.send_error(416)
                return
            status = 206

        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(target, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except BrokenPipeError:
                    return
                remaining -= len(chunk)


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rollout Viewer — failure injection</title>
<style>
  :root {
    color-scheme: light;
    --surface: #fcfcfb; --card: #ffffff; --ink: #0b0b0b; --ink-2: #52514e;
    --line: #e4e3df; --accent: #2a78d6;
    --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
    --chip-bg: #f1f0ec;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --surface: #1a1a19; --card: #232322; --ink: #ffffff; --ink-2: #c3c2b7;
      --line: #3a3937; --accent: #3987e5; --chip-bg: #2e2d2b;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--surface); color: var(--ink);
    font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  header {
    display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap;
    padding: 1rem 1.5rem 0.4rem;
  }
  header h1 { font-size: 1.15rem; margin: 0; font-weight: 650; }
  header .root { color: var(--ink-2); font-family: ui-monospace, Menlo, monospace; font-size: 0.78rem; }
  header .poll { margin-left: auto; color: var(--ink-2); font-size: 0.78rem; }
  .tiles {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 0.6rem; padding: 0.6rem 1.5rem;
  }
  .tile {
    background: var(--card); border: 1px solid var(--line); border-radius: 10px;
    padding: 0.55rem 0.8rem;
  }
  .tile .v { font-size: 1.35rem; font-weight: 650; font-variant-numeric: tabular-nums; }
  .tile .k { color: var(--ink-2); font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; }
  .filters { display: flex; gap: 0.6rem; padding: 0 1.5rem 0.4rem; flex-wrap: wrap; }
  .filters select {
    background: var(--card); color: var(--ink); border: 1px solid var(--line);
    border-radius: 8px; padding: 0.3rem 0.5rem; font: inherit;
  }
  .grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(430px, 1fr));
    gap: 1rem; padding: 0.6rem 1.5rem 3rem;
  }
  .card {
    background: var(--card); border: 1px solid var(--line); border-radius: 12px;
    overflow: hidden; display: flex; flex-direction: column;
  }
  .card video { width: 100%; display: block; background: #000; }
  .card .body { padding: 0.6rem 0.9rem 0.8rem; display: flex; flex-direction: column; gap: 0.5rem; }
  .card .title-row { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
  .card .title { font-weight: 650; }
  .badge {
    display: inline-flex; align-items: center; gap: 0.3em;
    border-radius: 999px; padding: 0.08rem 0.6rem; font-size: 0.76rem; font-weight: 600;
    border: 1.5px solid; background: var(--chip-bg);
  }
  .b-good { color: var(--good); border-color: var(--good); }
  .b-warning { color: var(--warning); border-color: var(--warning); }
  .b-serious { color: var(--serious); border-color: var(--serious); }
  .b-critical { color: var(--critical); border-color: var(--critical); }
  .b-neutral { color: var(--ink-2); border-color: var(--line); }
  .chips { display: flex; flex-wrap: wrap; gap: 0.35rem; color: var(--ink-2); font-size: 0.78rem; }
  .chips .chip {
    background: var(--chip-bg); border-radius: 6px; padding: 0.1rem 0.45rem;
    font-variant-numeric: tabular-nums;
  }
  .timeline {
    position: relative; height: 18px; border-radius: 5px; background: var(--chip-bg);
    cursor: pointer;
  }
  .timeline .played { position: absolute; inset: 0 auto 0 0; width: 0; background: color-mix(in oklab, var(--accent) 30%, transparent); border-radius: 5px; }
  .timeline .mark {
    position: absolute; top: -3px; bottom: -3px; width: 4px; border-radius: 2px;
    transform: translateX(-2px);
  }
  .timeline .mark.slip { background: var(--serious); }
  .timeline .mark.rebin { background: var(--good); }
  .jump { display: flex; gap: 0.4rem; }
  .jump button {
    font: inherit; font-size: 0.78rem; font-weight: 600; cursor: pointer;
    border: 1px solid var(--line); background: var(--chip-bg); color: var(--ink);
    border-radius: 7px; padding: 0.22rem 0.6rem;
  }
  .jump button:hover { border-color: var(--accent); }
  .empty { padding: 2rem 1.5rem; color: var(--ink-2); }
  a { color: var(--accent); }
</style>
</head>
<body>
<header>
  <h1>Rollout viewer — grasp-slip failure injection</h1>
  <span class="root" id="root-path"></span>
  <span class="poll" id="poll-status">loading…</span>
</header>
<div class="tiles" id="tiles"></div>
<div class="filters">
  <select id="f-model"><option value="">all models</option></select>
  <select id="f-task"><option value="">all tasks</option></select>
  <select id="f-outcome"><option value="">all slip outcomes</option></select>
</div>
<div class="grid" id="grid"></div>
<div class="empty" id="empty" hidden></div>
<script>
"use strict";
let DATA = null, FPS = 30, lastPayload = "";
const $ = (id) => document.getElementById(id);
const fmt = (v, d=1) => (v === null || v === undefined) ? "—" : Number(v).toFixed(d);

function outcomeBadge(m) {
  if (!m || !m.slip_triggered) return ['no slip triggered', 'b-neutral', '○'];
  switch (m.slip_outcome) {
    case 'dropped': return ['verified drop', 'b-serious', '↓'];
    case 'accidental_bin': return ['accidental bin', 'b-warning', '⚠'];
    case 'not_released': return ['not released', 'b-warning', '⚠'];
    default: return ['slip pending', 'b-neutral', '…'];
  }
}

function median(xs) {
  if (!xs.length) return null;
  const s = [...xs].sort((a,b)=>a-b), m = s.length >> 1;
  return s.length % 2 ? s[m] : 0.5*(s[m-1]+s[m]);
}

function computeTiles(eps) {
  const n = eps.length;
  const succ = eps.filter(e => e.success === true).length;
  const trig = eps.filter(e => e.metrics && e.metrics.slip_triggered).length;
  const drops = eps.filter(e => e.metrics && e.metrics.slip_outcome === 'dropped');
  const rec = drops.filter(e => e.metrics.slip_recovery_time_s !== null && e.metrics.slip_recovery_time_s !== undefined);
  const roll = eps.filter(e => e.metrics && e.metrics.slip_rolled_off_table).length;
  const medRec = median(rec.map(e => e.metrics.slip_recovery_time_s));
  return [
    [n, 'episodes'],
    [n ? `${succ}/${n}` : '—', 'final success'],
    [n ? `${trig}/${n}` : '—', 'slips triggered'],
    [trig ? `${drops.length}/${trig}` : '—', 'verified drops'],
    [drops.length ? `${rec.length}/${drops.length}` : '—', 'recovered (rebinned)'],
    [medRec === null ? '—' : fmt(medRec) + ' s', 'median recovery'],
    [roll, 'rolled off table'],
  ];
}

function card(e) {
  const m = e.metrics || {};
  const el = document.createElement('div');
  el.className = 'card';
  const [oLabel, oClass, oIcon] = outcomeBadge(m);
  const succBadge = e.success === true
    ? '<span class="badge b-good">✓ success</span>'
    : (e.success === false ? '<span class="badge b-critical">✗ failure</span>'
                           : '<span class="badge b-neutral">unscored</span>');
  const rollBadge = m.slip_rolled_off_table ? '<span class="badge b-critical">⚠ rolled off table</span>' : '';
  const chips = [];
  if (m.slip_triggered) {
    chips.push(`slip @ ${fmt(m.slip_time_s)} s`);
    chips.push(`chunk offset ${m.slip_chunk_offset}`);
    if (m.slip_arm) chips.push(`${m.slip_arm} arm, ${m.slip_bottle}`);
    chips.push(m.slip_recovery_time_s != null ? `recovered in ${fmt(m.slip_recovery_time_s)} s` : 'not recovered');
  }
  if (m.num_bottles_in_bin !== undefined) chips.push(`${m.num_bottles_in_bin}/${m.num_active_bottles} in bin at end`);

  const slipT = m.slip_step != null ? m.slip_step / FPS : null;
  const rebinT = m.slip_rebin_step != null ? m.slip_rebin_step / FPS : null;

  el.innerHTML = `
    ${e.video_url ? `<video preload="metadata" muted controls src="${e.video_url}"></video>` : ''}
    <div class="body">
      <div class="title-row">
        <span class="title">${e.model_label} · seed ${e.seed}</span>
        ${succBadge}
        <span class="badge ${oClass}">${oIcon} ${oLabel}</span>
        ${rollBadge}
      </div>
      <div class="timeline"><div class="played"></div></div>
      <div class="jump"></div>
      <div class="chips">${chips.map(c => `<span class="chip">${c}</span>`).join('')}</div>
    </div>`;

  const video = el.querySelector('video');
  const tl = el.querySelector('.timeline');
  const played = el.querySelector('.played');
  const jump = el.querySelector('.jump');
  if (video && tl) {
    video.addEventListener('loadedmetadata', () => {
      const dur = video.duration;
      const addMark = (t, cls, label) => {
        if (t == null || !isFinite(dur) || dur <= 0) return;
        const mk = document.createElement('div');
        mk.className = `mark ${cls}`;
        mk.style.left = `${(100 * Math.min(t, dur) / dur).toFixed(2)}%`;
        mk.title = `${label} @ ${t.toFixed(1)} s`;
        tl.appendChild(mk);
        const b = document.createElement('button');
        b.textContent = `▶ ${label}`;
        b.onclick = () => { video.currentTime = Math.max(0, t - 1.5); video.play(); };
        jump.appendChild(b);
      };
      addMark(slipT, 'slip', 'slip');
      addMark(rebinT, 'rebin', 'rebin');
    });
    video.addEventListener('timeupdate', () => {
      if (video.duration) played.style.width = `${100 * video.currentTime / video.duration}%`;
    });
    tl.addEventListener('click', (ev) => {
      const r = tl.getBoundingClientRect();
      if (video.duration) { video.currentTime = video.duration * (ev.clientX - r.left) / r.width; video.play(); }
    });
  }
  return el;
}

function pendingCard(p) {
  const el = document.createElement('div');
  el.className = 'card';
  el.innerHTML = `
    <video preload="metadata" muted controls src="${p.video_url}"></video>
    <div class="body">
      <div class="title-row">
        <span class="title">${p.rel_path}</span>
        <span class="badge b-warning">⏺ recording / unscored</span>
        <span class="chips"><span class="chip">${p.size_mb} MB</span></span>
      </div>
    </div>`;
  return el;
}

function optionSet(sel, values) {
  const cur = sel.value;
  const keep = new Set([...sel.options].map(o => o.value));
  for (const v of values) if (!keep.has(v)) {
    const o = document.createElement('option'); o.value = v; o.textContent = v; sel.appendChild(o);
  }
  sel.value = cur;
}

function render() {
  if (!DATA) return;
  const eps = DATA.episodes.filter(e =>
    (!$('f-model').value || e.model_label === $('f-model').value) &&
    (!$('f-task').value || e.task === $('f-task').value) &&
    (!$('f-outcome').value || ((e.metrics||{}).slip_outcome || 'none') === $('f-outcome').value));

  optionSet($('f-model'), [...new Set(DATA.episodes.map(e => e.model_label))]);
  optionSet($('f-task'), [...new Set(DATA.episodes.map(e => e.task))]);
  optionSet($('f-outcome'), [...new Set(DATA.episodes.map(e => (e.metrics||{}).slip_outcome || 'none'))]);

  $('tiles').innerHTML = computeTiles(eps).map(([v,k]) =>
    `<div class="tile"><div class="v">${v}</div><div class="k">${k}</div></div>`).join('');

  const grid = $('grid');
  grid.innerHTML = '';
  eps.sort((a,b) => (a.model_label+a.task).localeCompare(b.model_label+b.task) || a.seed - b.seed);
  for (const e of eps) grid.appendChild(card(e));
  for (const p of DATA.pending_videos) grid.appendChild(pendingCard(p));
  $('empty').hidden = !!(eps.length || DATA.pending_videos.length);
  $('empty').textContent = 'No episodes yet — results.jsonl not written. This page refreshes automatically.';
}

async function poll() {
  try {
    const r = await fetch('/api/results');
    const text = await r.text();
    if (text !== lastPayload) {
      const anyPlaying = [...document.querySelectorAll('video')].some(v => !v.paused && !v.ended);
      DATA = JSON.parse(text);
      FPS = DATA.video_fps || 30;
      $('root-path').textContent = DATA.root;
      if (!anyPlaying) { lastPayload = text; render(); }
    }
    $('poll-status').textContent = `updated ${new Date().toLocaleTimeString()} · auto-refresh 8 s`;
  } catch (err) {
    $('poll-status').textContent = `refresh failed: ${err.message}`;
  }
}
poll();
setInterval(poll, 8000);
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a rollout inspection dashboard")
    parser.add_argument("--root", type=Path, default=ROOT, help="Eval output_dir to serve")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    ViewerHandler.root = args.root.resolve()
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    print(f"Serving {ViewerHandler.root} on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
