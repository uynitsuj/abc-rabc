#!/usr/bin/env bash
# Side-by-side episode viewer (port 8777). Server is pure stdlib; data (preview
# mp4s extracted from mcaps) lives outside the repo under ~/rerender_demos.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA=/home/karimelrafi/rerender_demos
exec /home/karimelrafi/miniforge3/bin/python "$HERE/inspect_rerender_server.py" \
  --source original-delivery=$DATA/original_preview \
  --source my-mjwarp310=$DATA/fixed_preview \
  --source mjgl-640=$DATA/out_mjgl640 \
  --host 0.0.0.0 --port 8777 --fps 30
