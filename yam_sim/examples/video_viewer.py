"""Dead-simple localhost MP4 viewer.

Serves either a single MP4 (--video) or every MP4 under a directory (--root)
in a plain HTML5 <video controls> element. No canvas tricks, no scrubber
synthesis — the browser's native video player handles playback and seeking.

Run:
    python -m yam_sim.examples.video_viewer --video /path/to/rollout.mp4
    python -m yam_sim.examples.video_viewer --root /path/to/videos --port 8766
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plain MP4 viewer.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Path to a single MP4 to view.")
    group.add_argument("--root", help="Directory to scan recursively for MP4s.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8766, help="Bind port. Default: 8766")
    parser.add_argument("--title", default="Video Viewer", help="Page title")
    return parser.parse_args()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root)
        return True
    except (FileNotFoundError, ValueError):
        return False


def _discover_videos(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.mp4") if p.is_file())


_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, sans-serif;
      background: #0f1115;
      color: #e7ecf3;
    }
    header {
      padding: 12px 16px;
      border-bottom: 1px solid #2b3342;
      background: #171a21;
      display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
    }
    h1 { margin: 0; font-size: 18px; }
    select {
      background: #202531; color: #e7ecf3;
      border: 1px solid #2b3342; border-radius: 8px;
      padding: 6px 10px; font: inherit; min-width: 320px;
    }
    main { padding: 16px; display: flex; justify-content: center; }
    video {
      max-width: 100%;
      max-height: calc(100vh - 100px);
      background: #000;
      border-radius: 8px;
    }
    .empty { color: #9da7b7; padding: 40px; text-align: center; }
    .path { color: #9da7b7; font-size: 12px; margin: 8px 16px; }
  </style>
</head>
<body>
  <header>
    <h1>__TITLE__</h1>
    __SELECTOR__
  </header>
  <div class="path" id="path"></div>
  <main>
    <video id="player" controls preload="auto" playsinline></video>
  </main>
  <script>
    const VIDEOS = __VIDEOS_JSON__;
    const player = document.getElementById('player');
    const pathEl = document.getElementById('path');
    const sel = document.getElementById('video-select');

    function fileUrl(path) { return '/file?path=' + encodeURIComponent(path); }

    function load(path) {
      pathEl.textContent = path;
      player.src = fileUrl(path);
      player.load();
    }

    if (sel) {
      sel.addEventListener('change', () => load(sel.value));
    }
    if (VIDEOS.length) {
      load(VIDEOS[0]);
    } else {
      pathEl.textContent = 'No videos found.';
    }
  </script>
</body>
</html>
"""


def _render_page(title: str, videos: list[str]) -> bytes:
    if len(videos) > 1:
        opts = "".join(
            f'<option value="{escape(v, quote=True)}">{escape(v)}</option>'
            for v in videos
        )
        selector = f'<select id="video-select">{opts}</select>'
    else:
        selector = ""
    body = (
        _PAGE
        .replace("__TITLE__", escape(title))
        .replace("__SELECTOR__", selector)
        .replace("__VIDEOS_JSON__", json.dumps(videos))
    )
    return body.encode("utf-8")


class _ViewerHandler(BaseHTTPRequestHandler):
    root: Path = Path(".")
    title: str = "Video Viewer"
    videos: list[str] = []

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_index()
        elif parsed.path == "/file":
            self._serve_file(parsed)
        else:
            self.send_error(404, "Not Found")

    def log_message(self, fmt: str, *args) -> None:
        return

    def _serve_index(self) -> None:
        body = _render_page(self.title, self.videos)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
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
        if not _is_within(file_path, self.root):
            self.send_error(403, "Path is outside allowed root")
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
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)


def main() -> None:
    args = _parse_args()
    if args.video:
        video = Path(args.video).resolve()
        if not video.is_file():
            raise SystemExit(f"--video is not a file: {video}")
        _ViewerHandler.root = video.parent
        _ViewerHandler.videos = [str(video)]
        print(f"Serving {video}")
    else:
        root = Path(args.root).resolve()
        if not root.is_dir():
            raise SystemExit(f"--root is not a directory: {root}")
        _ViewerHandler.root = root
        _ViewerHandler.videos = [str(p) for p in _discover_videos(root)]
        print(f"Serving {len(_ViewerHandler.videos)} video(s) from {root}")

    _ViewerHandler.title = args.title
    server = ThreadingHTTPServer((args.host, args.port), _ViewerHandler)
    print(f"  http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
