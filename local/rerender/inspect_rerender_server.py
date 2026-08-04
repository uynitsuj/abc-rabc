"""Synced multi-source camera inspector for re-rendered episodes.

Compares any number of render sources side-by-side, one column per camera
(top/left/right) and one stacked pane per source within each column. All videos
are synchronized by wall-clock time, with frame stepping and an 8x pixel
magnifier lens so per-frame render noise is easy to judge across sources.

Sources are given as NAME=PATH (repeatable), where PATH is a tree containing
.../<task>/episode_<uuid>/{top,left,right}_camera.mp4 :

    python scripts/inspect_rerender_server.py \
        --source original=/scratch/current/karimelrafi/original_15hz \
        --source mjwarp=/scratch/current/karimelrafi/rerendered_30hz \
        --source mujoco-GL=/scratch/current/karimelrafi/rerendered_mjgl_30hz \
        --port 8777

Then (from your laptop):
    ssh -L 8777:127.0.0.1:8777 <this-host>
    open http://127.0.0.1:8777
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

CAMERAS = ("top", "left", "right")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-source camera inspector.")
    p.add_argument("--source", action="append", default=[], metavar="NAME=PATH",
                   help="Render source as NAME=PATH; repeat for each source (display order preserved).")
    p.add_argument("--host", default="127.0.0.1", help="Bind host. Default 127.0.0.1")
    p.add_argument("--port", type=int, default=8777, help="Bind port. Default 8777")
    p.add_argument("--fps", type=float, default=30.0, help="Timeline fps for stepping/counter.")
    return p.parse_args()


def _parse_sources(specs: list[str]) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--source must be NAME=PATH, got {spec!r}")
        name, _, path = spec.partition("=")
        root = Path(path).resolve()
        if not root.is_dir():
            raise SystemExit(f"source {name!r} path is not a directory: {root}")
        sources.append((name, root))
    if not sources:
        raise SystemExit("At least one --source NAME=PATH is required.")
    return sources


def _is_within(path: Path, roots: list[Path]) -> bool:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        return False
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _episode_has_all_cams(ep_dir: Path) -> dict[str, str] | None:
    cams = {c: ep_dir / f"{c}_camera.mp4" for c in CAMERAS}
    if all(p.is_file() for p in cams.values()):
        return {c: str(p) for c, p in cams.items()}
    return None


def _discover_episodes(sources: list[tuple[str, Path]]) -> list[dict]:
    """Union of episodes across sources, keyed by relative episode path."""
    by_id: dict[str, dict] = {}
    order: list[str] = []
    for name, root in sources:
        for top in sorted(root.rglob("top_camera.mp4")):
            ep_dir = top.parent
            cams = _episode_has_all_cams(ep_dir)
            if cams is None:
                continue
            rel = str(ep_dir.relative_to(root))
            if rel not in by_id:
                by_id[rel] = {"id": rel, "episode": ep_dir.name, "sources": {}}
                order.append(rel)
            by_id[rel]["sources"][name] = cams
    return [by_id[k] for k in order]


_PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Multi-source render inspector</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:#0f1115; color:#e7ecf3; }
  header { padding:10px 14px; border-bottom:1px solid #2b3342; background:#171a21;
           display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  h1 { margin:0; font-size:16px; }
  select, button, label { font: inherit; }
  select { background:#202531; color:#e7ecf3; border:1px solid #2b3342; border-radius:8px; padding:6px 10px; }
  button { background:#2a3140; color:#e7ecf3; border:1px solid #3a4254;
           border-radius:8px; padding:6px 12px; cursor:pointer; }
  button:hover { background:#343c4d; }
  .grid { display:flex; gap:12px; padding:12px; justify-content:center; flex-wrap:wrap; align-items:flex-start; }
  .col { background:#171a21; border:1px solid #2b3342; border-radius:10px; padding:8px; }
  .col > h2 { margin:0 0 8px; font-size:14px; text-align:center; }
  .pane { margin-bottom:8px; }
  .pane .tag { font-size:11px; margin:0 0 3px; display:flex; justify-content:space-between; gap:8px; }
  .pane .tag .src { font-weight:600; }
  .pane .tag .sz { color:#5f6b7e; }
  video { display:block; background:#000; border-radius:6px; image-rendering: pixelated; }
  .miss { color:#6b7688; font-size:12px; padding:20px 8px; text-align:center; }
  .transport { padding:10px 14px; display:flex; gap:10px; align-items:center; flex-wrap:wrap;
               border-top:1px solid #2b3342; background:#171a21; position:sticky; bottom:0; }
  .frame { font-variant-numeric: tabular-nums; min-width:260px; color:#9da7b7; font-size:13px; }
  input[type=range] { flex:1; min-width:240px; }
  #lens { position:fixed; width:240px; height:240px; border:2px solid #4a90d9; border-radius:8px;
          pointer-events:none; display:none; image-rendering:pixelated; z-index:50;
          box-shadow:0 4px 20px rgba(0,0,0,.6); background:#000; }
  .hint { color:#5f6b7e; font-size:12px; }
</style></head>
<body>
<header>
  <h1>Render inspector</h1>
  <select id="ep"></select>
  <label class="hint">size <select id="zoom">
    <option value="1">100% (640)</option>
    <option value="1.5">150%</option>
    <option value="2">200%</option>
    <option value="0.75">75%</option>
  </select></label>
  <label class="hint"><input type="checkbox" id="smooth"> smooth (off = show pixels)</label>
  <span class="hint">space=play/pause · ←/→ step · shift=±10 · hover=magnify</span>
</header>

<div class="grid" id="grid"></div>

<div class="transport">
  <button id="playpause">▶︎ play</button>
  <button data-step="-10">⏪ -10</button>
  <button data-step="-1">◀ -1</button>
  <button data-step="1">+1 ▶</button>
  <button data-step="10">+10 ⏩</button>
  <span class="frame" id="frame"></span>
  <input type="range" id="scrub" min="0" max="0" value="0" step="1">
</div>

<canvas id="lens"></canvas>

<script>
const EPISODES = __EPISODES__;
const SOURCES = __SOURCES__;     // ordered source names
const FPS = __FPS__;
const CAMERAS = ["top","left","right"];
const COLORS = ["#7ec27e","#d9a14a","#5aa9e6","#c678dd","#e06c75","#56b6c2"];
const grid = document.getElementById('grid');
const epSel = document.getElementById('ep');
const zoomSel = document.getElementById('zoom');
const smoothChk = document.getElementById('smooth');
const frameLbl = document.getElementById('frame');
const scrub = document.getElementById('scrub');
const playBtn = document.getElementById('playpause');
const lens = document.getElementById('lens');
const lctx = lens.getContext('2d');

let videos = [];     // all <video> elements
let master = null;   // drives the timeline
let totalFrames = 0;
let playing = false;

function fileUrl(p){ return '/file?path=' + encodeURIComponent(p); }
function colorFor(name){ return COLORS[SOURCES.indexOf(name) % COLORS.length]; }

EPISODES.forEach((e,i)=>{
  const o=document.createElement('option'); o.value=i;
  o.textContent=e.id + '  ['+Object.keys(e.sources).join(', ')+']';
  epSel.appendChild(o);
});

function makeVideo(src){
  const v=document.createElement('video');
  v.src=fileUrl(src); v.preload='auto'; v.muted=true; v.playsInline=true;
  videos.push(v); attachLens(v); return v;
}

function buildGrid(ep){
  grid.innerHTML=''; videos=[]; master=null;
  CAMERAS.forEach(cam=>{
    const col=document.createElement('div'); col.className='col';
    col.innerHTML='<h2>'+cam+'</h2>';
    SOURCES.forEach(srcName=>{
      const pane=document.createElement('div'); pane.className='pane';
      const has = ep.sources[srcName];
      const tag=document.createElement('div'); tag.className='tag';
      tag.innerHTML='<span class="src" style="color:'+colorFor(srcName)+'">'+srcName+'</span>';
      pane.appendChild(tag);
      if(has){
        const v=makeVideo(has[cam]);
        pane.appendChild(v);
        if(cam==='top' && master===null) master=v;
      } else {
        const m=document.createElement('div'); m.className='miss'; m.textContent='(not available)';
        pane.appendChild(m);
      }
      col.appendChild(pane);
    });
    grid.appendChild(col);
  });
  applyZoom();
  if(master){
    master.addEventListener('loadedmetadata',()=>{
      totalFrames=Math.max(0, Math.round(master.duration*FPS));
      scrub.max=Math.max(0,totalFrames-1);
      updateLabel();
    },{once:true});
  }
}

function applyZoom(){
  const z=parseFloat(zoomSel.value);
  videos.forEach(v=>{ v.style.width=(640*z)+'px'; v.style.height=(480*z)+'px';
    v.style.imageRendering = smoothChk.checked ? 'auto' : 'pixelated'; });
}
zoomSel.addEventListener('change',applyZoom);
smoothChk.addEventListener('change',applyZoom);

function curFrame(){ return master? Math.round(master.currentTime*FPS):0; }
function updateLabel(){
  const t=master? master.currentTime:0;
  frameLbl.textContent='t='+t.toFixed(3)+'s  |  frame '+curFrame()+'/'+(totalFrames?totalFrames-1:0);
  scrub.value=curFrame();
}
function seekTime(t){ videos.forEach(v=>{ try{ v.currentTime=t; }catch(_){} }); }
function seekFrame(f){
  f=Math.max(0, Math.min((totalFrames||1)-1, f));
  seekTime((f+0.5)/FPS);
}
function step(n){ pause(); seekFrame(curFrame()+n); }

async function play(){
  playing=true; playBtn.textContent='⏸ pause';
  await Promise.all(videos.map(v=>v.play().catch(()=>{})));
  tick();
}
function pause(){ playing=false; playBtn.textContent='▶︎ play'; videos.forEach(v=>v.pause()); }
function togglePlay(){ playing?pause():play(); }

function tick(){
  if(!playing || !master) return;
  const t=master.currentTime;
  for(const v of videos){
    if(v!==master && Math.abs(v.currentTime-t)>0.05){ try{ v.currentTime=t; }catch(_){} }
  }
  updateLabel();
  if(master.ended) pause();
  requestAnimationFrame(tick);
}

playBtn.addEventListener('click',togglePlay);
document.querySelectorAll('[data-step]').forEach(b=>
  b.addEventListener('click',()=>step(parseInt(b.dataset.step))));
scrub.addEventListener('input',()=>{ pause(); seekFrame(parseInt(scrub.value)); });
document.addEventListener('seeked',updateLabel,true);
epSel.addEventListener('change',()=>{ pause(); buildGrid(EPISODES[parseInt(epSel.value)]); });

document.addEventListener('keydown',e=>{
  if(e.code==='Space'){ e.preventDefault(); togglePlay(); }
  else if(e.code==='ArrowRight'){ e.preventDefault(); step(e.shiftKey?10:1); }
  else if(e.code==='ArrowLeft'){ e.preventDefault(); step(e.shiftKey?-10:-1); }
});

// --- pixel magnifier lens ---
const LENS=240, MAG=8;
function attachLens(v){
  v.addEventListener('mousemove',ev=>{
    const r=v.getBoundingClientRect();
    const fx=(ev.clientX-r.left)/r.width, fy=(ev.clientY-r.top)/r.height;
    const vw=v.videoWidth, vh=v.videoHeight;
    if(!vw) return;
    const sw=LENS/MAG, sh=LENS/MAG;
    const sx=Math.max(0,Math.min(vw-sw, fx*vw - sw/2));
    const sy=Math.max(0,Math.min(vh-sh, fy*vh - sh/2));
    lens.width=LENS; lens.height=LENS; lctx.imageSmoothingEnabled=false;
    lctx.clearRect(0,0,LENS,LENS);
    try{ lctx.drawImage(v, sx,sy,sw,sh, 0,0,LENS,LENS); }catch(_){}
    lens.style.display='block';
    lens.style.left=(ev.clientX+18)+'px'; lens.style.top=(ev.clientY+18)+'px';
  });
  v.addEventListener('mouseleave',()=>{ lens.style.display='none'; });
}

if(EPISODES.length){ buildGrid(EPISODES[0]); } else { grid.textContent='No episodes found.'; }
</script>
</body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    roots: list[Path] = [Path(".")]
    episodes: list[dict] = []
    source_names: list[str] = []
    fps: float = 30.0

    def log_message(self, *_a) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._index()
        elif parsed.path == "/file":
            self._file(parsed)
        else:
            self.send_error(404, "Not Found")

    def _index(self) -> None:
        body = (
            _PAGE
            .replace("__EPISODES__", json.dumps(self.episodes))
            .replace("__SOURCES__", json.dumps(self.source_names))
            .replace("__FPS__", json.dumps(self.fps))
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, parsed) -> None:
        params = parse_qs(parsed.query)
        raw = params.get("path", [None])[0]
        if raw is None:
            self.send_error(400, "Missing path")
            return
        fp = Path(raw)
        if not _is_within(fp, self.roots):
            self.send_error(403, "Path outside roots")
            return
        size = fp.stat().st_size
        ctype = mimetypes.guess_type(str(fp))[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end, status = 0, size - 1, 200
        if rng and rng.startswith("bytes="):
            s, _, e = rng.split("=", 1)[1].partition("-")
            if s:
                start = int(s)
            if e:
                end = int(e)
            end = min(end, size - 1)
            if start > end or start >= size:
                self.send_error(416, "Range Not Satisfiable")
                return
            status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with fp.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)


def main() -> None:
    args = _parse_args()
    sources = _parse_sources(args.source)
    episodes = _discover_episodes(sources)
    _Handler.roots = [root for _, root in sources]
    _Handler.episodes = episodes
    _Handler.source_names = [name for name, _ in sources]
    _Handler.fps = args.fps
    print(f"Sources: {', '.join(name for name, _ in sources)}")
    print(f"Found {len(episodes)} episode(s):")
    for e in episodes:
        print(f"  {e['id']}  [{', '.join(e['sources'].keys())}]")
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"\nServing on http://{args.host}:{args.port}", flush=True)
    print(f"If remote:  ssh -L {args.port}:127.0.0.1:{args.port} <host>  then open the URL", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
