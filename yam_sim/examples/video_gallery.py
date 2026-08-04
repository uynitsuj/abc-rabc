"""Localhost gallery for browsing replay videos without copying them off-node."""

from __future__ import annotations

import argparse
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local MP4 gallery over HTTP.")
    parser.add_argument(
        "--root",
        action="append",
        dest="roots",
        default=[],
        help="Directory to scan recursively for MP4 files. Can be passed multiple times.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="Bind port. Default: 8765")
    parser.add_argument("--title", default="Replay Videos", help="Page title")
    parser.add_argument(
        "--include-glob",
        action="append",
        dest="include_globs",
        default=[],
        help=(
            "Only include videos whose filename or root-relative path matches this glob. "
            "Can be passed multiple times."
        ),
    )
    return parser.parse_args()


def _is_within_roots(path: Path, roots: list[Path]) -> bool:
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


def _matches_include_globs(video: Path, root: Path, include_globs: list[str]) -> bool:
    if not include_globs:
        return True
    rel = video.relative_to(root).as_posix()
    return any(
        Path(video.name).match(pattern) or video.match(pattern) or Path(rel).match(pattern)
        for pattern in include_globs
    )


def _discover_videos(roots: list[Path], include_globs: list[str] | None = None) -> list[Path]:
    include_globs = list(include_globs or [])
    videos: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        videos.extend(
            p
            for p in root.rglob("*.mp4")
            if p.is_file() and _matches_include_globs(p, root, include_globs)
        )
    videos.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return videos


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0
    return f"{int(num_bytes)}B"


def _render_index(title: str, roots: list[Path], videos: list[Path]) -> bytes:
    cards: list[str] = []
    for video in videos:
        rel_root = next((root for root in roots if _is_within_roots(video, [root])), None)
        rel = str(video.relative_to(rel_root)) if rel_root is not None else video.name
        url = f"/file?path={video.as_posix()}"
        meta = f"{_format_bytes(video.stat().st_size)} | {escape(str(video.parent))}"
        cards.append(
            "\n".join(
                [
                    '<article class="card" data-path="{}" data-url="{}" data-name="{}">'.format(
                        escape(str(video)),
                        escape(url, quote=True),
                        escape(video.name),
                    ),
                    f'  <h3>{escape(video.name)}</h3>',
                    f'  <div class="meta">{escape(rel)}</div>',
                    f'  <div class="meta">{meta}</div>',
                    '  <div class="links">',
                    '    <button class="play-button" type="button">Play Here</button>',
                    f'    <a href="{escape(url, quote=True)}" target="_blank">Open file</a>',
                    "  </div>",
                    "</article>",
                ]
            )
        )

    roots_html = "".join(f"<li>{escape(str(root))}</li>" for root in roots)
    cards_html = "\n".join(cards) if cards else '<p class="empty">No MP4 files found.</p>'

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0f1115;
      --panel: #171a21;
      --panel-2: #202531;
      --text: #e7ecf3;
      --muted: #9da7b7;
      --accent: #6fd3ff;
      --border: #2b3342;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, sans-serif;
      background: linear-gradient(180deg, #0b0d12, var(--bg));
      color: var(--text);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(15, 17, 21, 0.92);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--border);
      padding: 16px 20px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
    }}
    .sub {{
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 12px;
    }}
    .toolbar {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }}
    input {{
      min-width: 280px;
      flex: 1 1 320px;
      background: var(--panel-2);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
    }}
    button {{
      background: var(--accent);
      color: #071018;
      border: 0;
      border-radius: 10px;
      padding: 10px 14px;
      cursor: pointer;
      font-weight: 600;
    }}
    main {{
      padding: 20px;
    }}
    .viewer {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      margin-bottom: 18px;
    }}
    .viewer-title {{
      margin: 0 0 8px;
      font-size: 16px;
      line-height: 1.35;
      word-break: break-word;
    }}
    .viewer-meta {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
      word-break: break-word;
    }}
    .roots {{
      color: var(--muted);
      margin: 0 0 20px;
      padding-left: 18px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
    }}
    .card h3 {{
      margin: 0 0 8px;
      font-size: 16px;
      line-height: 1.35;
      word-break: break-word;
    }}
    .meta {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
      word-break: break-word;
    }}
    #player {{
      width: 100%;
      max-height: 420px;
      background: #000;
      border-radius: 10px;
      margin-top: 8px;
    }}
    .links {{
      margin-top: 8px;
      font-size: 13px;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }}
    a {{
      color: var(--accent);
    }}
    .empty {{
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(title)}</h1>
    <div class="sub">{len(videos)} videos indexed</div>
    <div class="toolbar">
      <input id="filter" type="search" placeholder="Filter by filename or path">
      <button onclick="location.reload()">Refresh</button>
    </div>
  </header>
  <main>
    <section class="viewer">
      <h2 class="viewer-title" id="player-title">Click a video to load it</h2>
      <div class="viewer-meta" id="player-meta">No video loaded.</div>
      <video id="player" controls preload="none"></video>
    </section>
    <ul class="roots">{roots_html}</ul>
    <section class="grid" id="grid">
      {cards_html}
    </section>
  </main>
  <script>
    const filter = document.getElementById('filter');
    const cards = Array.from(document.querySelectorAll('.card'));
    const player = document.getElementById('player');
    const playerTitle = document.getElementById('player-title');
    const playerMeta = document.getElementById('player-meta');
    filter.addEventListener('input', () => {{
      const needle = filter.value.trim().toLowerCase();
      for (const card of cards) {{
        const visible = !needle || card.dataset.path.toLowerCase().includes(needle) || card.querySelector('h3').textContent.toLowerCase().includes(needle);
        card.style.display = visible ? '' : 'none';
      }}
    }});
    for (const card of cards) {{
      const button = card.querySelector('.play-button');
      button.addEventListener('click', () => {{
        player.pause();
        player.removeAttribute('src');
        player.load();
        player.src = card.dataset.url;
        playerTitle.textContent = card.dataset.name;
        playerMeta.textContent = card.dataset.path;
        player.load();
        player.play().catch(() => {{}});
        card.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
        window.scrollTo({{ top: 0, behavior: 'smooth' }});
      }});
    }}
  </script>
</body>
</html>
"""
    return html.encode("utf-8")


class _GalleryHandler(BaseHTTPRequestHandler):
    roots: list[Path] = []
    title: str = "Replay Videos"
    include_globs: list[str] = []

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_index()
            return
        if parsed.path == "/file":
            self._serve_file(parsed)
            return
        self.send_error(404, "Not Found")

    def log_message(self, fmt: str, *args) -> None:
        return

    def _serve_index(self) -> None:
        body = _render_index(
            self.title,
            self.roots,
            _discover_videos(self.roots, self.include_globs),
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, parsed) -> None:
        params = parse_qs(parsed.query)
        raw_path = params.get("path", [None])[0]
        if raw_path is None:
            self.send_error(400, "Missing path")
            return
        file_path = Path(raw_path)
        if not _is_within_roots(file_path, self.roots):
            self.send_error(403, "Path is outside allowed roots")
            return
        size = file_path.stat().st_size
        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        range_header = self.headers.get("Range")

        start = 0
        end = size - 1
        status = 200
        if range_header and range_header.startswith("bytes="):
            spec = range_header.split("=", 1)[1]
            start_str, _, end_str = spec.partition("-")
            if start_str:
                start = int(start_str)
            if end_str:
                end = int(end_str)
            end = min(end, size - 1)
            if start > end or start >= size:
                self.send_error(416, "Requested Range Not Satisfiable")
                return
            status = 206

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        with file_path.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def main() -> None:
    args = _parse_args()
    roots = [Path(root).resolve() for root in (args.roots or ["."])]
    _GalleryHandler.roots = roots
    _GalleryHandler.title = args.title
    _GalleryHandler.include_globs = list(args.include_globs or [])
    server = ThreadingHTTPServer((args.host, args.port), _GalleryHandler)
    print(f"Serving {len(roots)} root(s) on http://{args.host}:{args.port}")
    for root in roots:
        print(f"  - {root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
