# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mujoco", "mujoco-warp", "warp-lang", "tyro"]
# ///
"""Re-render delivery sim episodes from mjgl into MuJoCo-Warp at the eval renderer.

The 30hz lerobot training data was rendered with mjgl; the ABC sim eval
(abc_minimal/eval_policy.py) renders with mjwarp. This produces mjwarp-rendered
staged episodes so policy training data matches the eval renderer.

Per delivery episode dir (must contain scene_assembled.xml + integration_state.npy):
  -> <out>/<ep_id>/{combined_camera-images-rgb.mp4, states_actions.bin,
                    episode_metadata.json, velocity_*.bin (carried forward)}

Resolved subtleties (see module docstring sections below):
  * CAMERA MAPPING: the delivery scene_assembled.xml defines cameras named EXACTLY
    `top`/`left`/`right` (plus overhead/left_side/right_side) with byte-identical
    local pos/quat/fovy and parent-body chain (top_camera_d405 -> top_camera_frame,
    left/right wrist cams) to the eval scene assets/put_bottles/put_bottle.xml.
    So the policy cameras map 1:1: top->top, left->left, right->right.
  * ASSETS: scene_assembled.xml references bottle meshes by ABSOLUTE path under
    `<xdof-sim>/xdof_sim/models/assets/` and other meshes by relative path. We point
    meshdir/texturedir at a local asset root (default the xdof-sim checkout, which
    holds the FULL bottle library bottle_0..18) and rewrite the absolute prefix to it.
  * QPOS: integration_state[:, 0] is sim-time; integration_state[:, 1:1+model.nq] is
    qpos (verified byte-identical to the sim_state.mcap /sim_state/qpos topic, and the
    per-frame ARM joints match the staged states_actions.bin proprioception at ZERO
    frame offset). We render qpos-only: forward() resolves xpos/xquat (incl. wrist-cam
    FK); qvel/ctrl do not affect a static rendered frame.
  * TIMING: integration_state has exactly one row per staged training frame (1:1, the
    nominal "30hz" data is the sim's 29.4hz states relabeled). So we render every
    integration_state row and the combined.mp4 frame count == states_actions.bin rows,
    a drop-in video swap that keeps the existing states_actions.bin.

Output video format is byte-compatible with lerobot_to_abc.py / export_mcap.py: the 3
cameras (top, left, right) each letterboxed to 224x168 (no-op for 4:3 sim cams),
stacked vertically, encoded with the strict x264 params + settb/setpts pts mapping
(pts=512*k, GOP30, timebase 1/15360) the trainer's synthesized frame mapping expects.

Run with the repo venv python directly (NOT `uv run`):
    .venv/bin/python rerender_mjwarp.py <delivery_root> <out_dir> --episodes ...
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import mujoco
import numpy as np
import tyro

# Reuse the EXACT encode params + filter the trainer expects (byte-true frame mapping).
from export_mcap import FPS, TIMESCALE, TICKS_PER_FRAME, X264, X264_STRICT_FFMPEG_ARGS

# Staged combined.mp4 per-cam geometry (SimEvalConfig camera_width/height; config.py:206-207).
OUT_W, OUT_H = 224, 168
# Native render resolution. Matches the existing mjgl LeRobot dataset's declared video
# shape [480, 640, 3] so the per-camera LeRobot videos are a drop-in renderer swap. The
# staged combined.mp4 is letterboxed down to 224x168 from these SAME rendered frames, so
# both formats are pixel-consistent (one render, two encodes).
RENDER_W, RENDER_H = 640, 480
# Policy camera names, top->bottom stack order in combined.mp4 (config camera_keys).
CAMERAS = ("top", "left", "right")
# LeRobot per-camera video feature keys (mirrors lerobot_to_abc.CAM_KEY / the mjgl dataset).
CAM_KEY = {c: f"{c}_camera-images-rgb" for c in CAMERAS}
# Plain CFR-30 x264 encode for the LeRobot per-cam videos: mirrors the mjgl dataset's
# info.json (libx264 / yuv420p / gop_size 10). NOT the strict pts=512*k combined mapping.
X264_LEROBOT = [
    "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-bf", "0",
    "-pix_fmt", "yuv420p", "-g", "10", "-keyint_min", "10",
    "-x264-params", "scenecut=0", "-movflags", "+faststart",
]
# Default full asset root: the local xdof-sim checkout (holds bottle_0..18 + garbage_can
# + i2rt_yam). The delivery's absolute mesh paths are rewritten onto this root.
DEFAULT_ASSET_ROOT = "/home/justinyu/xdof-sim/xdof_sim/models/assets"
# Absolute mesh-path prefix baked into the delivery XML (the collection machine's path).
DELIVERY_ASSET_PREFIX = (
    "/home/xdof/lab42/packages/market42/submodules/xdof-sim/xdof_sim/models/assets/"
)
# Exact letterbox from lerobot_to_abc.encode_percam / export_mcap.encode_aligned:
# fit-inside-224x168 (bicubic) + center pad (no-op for 4:3 sources) + even dims.
LETTERBOX_VF = (
    f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:flags=bicubic,"
    f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2"
)

# Sidecar files carried forward verbatim from the existing staged episode (frame
# alignment is identical, so per-frame velocity sidecars stay valid).
SIDECAR_GLOBS = ("velocity_*.bin",)


@dataclass
class Config:
    """Re-render delivery sim episodes into mjwarp staged episodes."""

    delivery_root: Annotated[Path, tyro.conf.Positional]
    """Dir containing episode_<uuid>/ subdirs (each with scene_assembled.xml + integration_state.npy)."""
    out_dir: Annotated[Path, tyro.conf.Positional]
    """Output staged dir; writes <out_dir>/<ep_id>/{combined.mp4, states_actions.bin, ...}."""
    staged_src: Path | None = None
    """Existing staged dir to copy states_actions.bin + velocity_*.bin from (drop-in swap).
    If None, no states/sidecars are copied (video-only output)."""
    episodes: str | None = None
    """Comma-separated episode_<uuid> names (or bare uuids) to render. None = all under delivery_root."""
    asset_root: str = DEFAULT_ASSET_ROOT
    """meshdir/texturedir root; the delivery's absolute mesh prefix is rewritten onto this."""
    task_name: str = "sim_put_the_plastic_bottles_in_the_bin"
    gpu_id: int = 0
    """CUDA device index for warp."""
    height: int = RENDER_H
    width: int = RENDER_W
    """Native render resolution (default 640x480 = mjgl LeRobot video shape). The staged
    combined.mp4 is letterboxed down to 224x168 from these frames."""
    percam_dir: Path | None = None
    """If set, ALSO write per-camera native-resolution CFR-30 mp4s
    <percam_dir>/<ep_id>/{top,left,right}_camera-images-rgb.mp4 for the LeRobot-v3 dataset
    (concatenated per file-group downstream). No letterbox — full render resolution."""
    no_combined: bool = False
    """Skip the staged combined.mp4 (only emit per-cam videos). Combined is emitted by default."""
    limit: int | None = None
    """Optional cap on number of episodes (after selection)."""
    overwrite: bool = False
    """Re-render even if combined.mp4 already exists."""


# ----------------------------------------------------------------------------- #
# Scene XML: rewrite asset paths so the delivery scene loads against a local root.
# ----------------------------------------------------------------------------- #
def build_scene_xml(xml_path: Path, asset_root: str) -> str:
    root = ET.fromstring(xml_path.read_text())
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError(f"{xml_path}: no <compiler> element")
    compiler.set("meshdir", asset_root)
    compiler.set("texturedir", asset_root)
    # mjwarp's put_data needs explicit collision buffer sizes (nconmax/njmax). The
    # delivery scene leaves them unset (-> -1); the eval scene (put_bottle.xml) sets
    # <size nconmax="512" njmax="4096"/>. Mirror those so the warp model matches eval.
    size = root.find("size")
    if size is None:
        size = ET.SubElement(root, "size")
    size.set("nconmax", "512")
    size.set("njmax", "4096")
    # Rewrite absolute file= refs (bottle meshes) onto the local asset root; relative
    # refs (i2rt_yam/*, task_water_bottles/garbage_can/*) already resolve via meshdir.
    for el in root.iter():
        f = el.get("file")
        if f and f.startswith(DELIVERY_ASSET_PREFIX):
            el.set("file", f"{asset_root.rstrip('/')}/{f[len(DELIVERY_ASSET_PREFIX):]}")
    return ET.tostring(root, encoding="unicode")


# ----------------------------------------------------------------------------- #
# MJWarp render wrapper (mirrors abc_minimal/eval_policy.py:MJWarpSim, qpos-only).
# ----------------------------------------------------------------------------- #
class MJWarpRenderer:
    def __init__(self, model: mujoco.MjModel, *, height: int, width: int, gpu_id: int):
        import mujoco_warp as mjw
        import warp as wp

        self.mjw, self.wp = mjw, wp
        self.model = model
        self.data = mujoco.MjData(model)
        self.height, self.width, self.nworld = height, width, 1
        wp.set_device(f"cuda:{gpu_id}")
        self.m_warp = mjw.put_model(model)
        self.d_warp = mjw.put_data(model, self.data, nworld=1, nconmax=model.nconmax, njmax=model.njmax)
        self.render_context = mjw.create_render_context(
            mjm=model,
            nworld=1,
            cam_res=(width, height),
            render_rgb=[True] * model.ncam,
            render_depth=[False] * model.ncam,
            use_textures=True,
            use_shadows=True,
        )
        # Camera name -> id (top/left/right); fail loudly if a policy cam is missing.
        self.cam_ids = {}
        for name in CAMERAS:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cid < 0:
                raise ValueError(f"Camera not found in scene: {name!r}")
            self.cam_ids[name] = cid

    def _copy(self, target, values, dtype):
        self.wp.copy(target, self.wp.from_numpy(np.asarray(values), dtype=dtype))

    def render_qpos(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
        """Set qpos, forward (FK for body-mounted/wrist cams), render -> {cam: HxWx3 RGB}."""
        self.mjw.reset_data(self.m_warp, self.d_warp)
        self._copy(self.d_warp.qpos, np.asarray(qpos, dtype=np.float32)[None], self.wp.float32)
        self.mjw.forward(self.m_warp, self.d_warp)
        self.mjw.refit_bvh(self.m_warp, self.d_warp, self.render_context)
        self.mjw.render(self.m_warp, self.d_warp, self.render_context)
        rgba = self.render_context.rgb_data.numpy().view(np.uint8).reshape(
            self.nworld, self.model.ncam, self.height, self.width, 4
        )
        rgb = rgba[0, :, :, :, :3][..., ::-1]  # BGR(A) -> RGB (matches eval render())
        return {name: rgb[cid].copy() for name, cid in self.cam_ids.items()}

    def close(self):
        try:
            self.wp.synchronize()
        except Exception:
            pass
        self.render_context = self.m_warp = self.d_warp = None


# ----------------------------------------------------------------------------- #
# Encoding (mirrors lerobot_to_abc.py: per-cam letterbox mp4 -> vstack strict pts).
# ----------------------------------------------------------------------------- #
def encode_percam(frames_hwc: np.ndarray, out_path: str) -> None:
    n, h, w, _ = frames_hwc.shape
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(FPS), "-i", "-", "-vsync", "0", "-vf", LETTERBOX_VF, *X264,
         "-threads", "1", out_path],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    try:
        view = np.ascontiguousarray(frames_hwc)
        for i in range(n):
            enc.stdin.write(view[i].tobytes())
    finally:
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError(f"per-cam encode failed: {out_path}")


def encode_lerobot_percam(frames_hwc: np.ndarray, out_path: str) -> None:
    """Native-resolution CFR-30 x264 encode for a LeRobot per-cam video (NO letterbox).

    Mirrors the mjgl dataset's info.json (libx264/yuv420p/gop10). One encode per camera per
    episode; the downstream reduce concatenates these in episode_index order into the
    per-file-group videos, so frame count MUST equal the episode length exactly.
    """
    n, h, w, _ = frames_hwc.shape
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(FPS), "-i", "-", "-vsync", "0", *X264_LEROBOT, "-threads", "1", out_path],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    try:
        view = np.ascontiguousarray(frames_hwc)
        for i in range(n):
            enc.stdin.write(view[i].tobytes())
    finally:
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError(f"lerobot per-cam encode failed: {out_path}")


def vstack_strict(percam_paths: list[str], combined: str) -> None:
    filt = (
        "".join(f"[{i}:v]" for i in range(len(percam_paths)))
        + f"vstack=inputs={len(percam_paths)}[v0];"
        + f"[v0]settb=expr=1/{TIMESCALE},setpts=N*{TICKS_PER_FRAME}[out]"
    )
    subprocess.run(
        ["ffmpeg", "-y", *sum((["-i", p] for p in percam_paths), []),
         "-filter_complex", filt, "-map", "[out]", *X264_STRICT_FFMPEG_ARGS, combined],
        capture_output=True, check=True,
    )


def probe_nframes(path: str) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    ).stdout.strip()
    return int(out)


# ----------------------------------------------------------------------------- #
# Per-episode render.
# ----------------------------------------------------------------------------- #
def render_episode(ep_dir: Path, cfg: Config) -> str:
    ep_id = ep_dir.name  # episode_<uuid>
    out_ep = cfg.out_dir / ep_id
    combined = out_ep / "combined_camera-images-rgb.mp4"

    integration_state = np.load(ep_dir / "integration_state.npy")  # (T, >=1+nq); col0=time
    xml = build_scene_xml(ep_dir / "scene_assembled.xml", cfg.asset_root)
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.timestep = 0.002  # match eval; irrelevant for qpos-only render but harmless
    nq = model.nq
    qpos_all = integration_state[:, 1:1 + nq]  # col0 = sim-time, then qpos (verified)
    n_frames = len(qpos_all)

    want_combined = not cfg.no_combined
    percam_ep = (cfg.percam_dir / ep_id) if cfg.percam_dir is not None else None
    percam_out = {c: percam_ep / f"{CAM_KEY[c]}.mp4" for c in CAMERAS} if percam_ep else {}

    # Skip if every requested output already exists with the right frame count.
    if not cfg.overwrite:
        combined_ok = (not want_combined) or (
            combined.exists() and probe_nframes(str(combined)) == n_frames)
        percam_ok = (percam_ep is None) or all(
            p.exists() and probe_nframes(str(p)) == n_frames for p in percam_out.values())
        if combined_ok and percam_ok:
            print(f"[skip] {ep_id}: outputs already have {n_frames} frames")
            return ep_id
    out_ep.mkdir(parents=True, exist_ok=True)
    if percam_ep is not None:
        percam_ep.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    renderer = MJWarpRenderer(model, height=cfg.height, width=cfg.width, gpu_id=cfg.gpu_id)
    try:
        frames = {c: np.empty((n_frames, cfg.height, cfg.width, 3), np.uint8) for c in CAMERAS}
        for k in range(n_frames):
            imgs = renderer.render_qpos(qpos_all[k])
            for c in CAMERAS:
                frames[c][k] = imgs[c]
    finally:
        renderer.close()
    render_s = time.time() - t0

    # Per-camera native-resolution LeRobot videos (no letterbox). Emitted from the SAME
    # rendered frames as the staged combined below, so the two formats are pixel-consistent.
    if percam_ep is not None:
        for c in CAMERAS:
            encode_lerobot_percam(frames[c], str(percam_out[c]))
            got = probe_nframes(str(percam_out[c]))
            if got != n_frames:  # hard-assert: a per-episode miscount corrupts the group concat
                raise RuntimeError(f"{ep_id}: percam {c} has {got} frames, expected {n_frames}")

    # Staged combined.mp4: letterbox the rendered frames 640x480 -> 224x168 per cam, vstack
    # with the strict pts mapping the ABC trainer's frame synthesis expects.
    if want_combined:
        with tempfile.TemporaryDirectory() as work:
            percam = []
            for c in CAMERAS:
                pc = str(Path(work) / f"{c}.mp4")
                encode_percam(frames[c], pc)
                percam.append(pc)
            vstack_strict(percam, str(combined))
        got = probe_nframes(str(combined))
        if got != n_frames:
            raise RuntimeError(f"{ep_id}: combined has {got} frames, expected {n_frames}")

    # Carry states_actions.bin + velocity_*.bin forward verbatim (drop-in swap).
    copied = []
    if cfg.staged_src is not None:
        src_ep = cfg.staged_src / ep_id
        sa = src_ep / "states_actions.bin"
        if sa.exists():
            sa_arr = np.fromfile(sa, dtype=np.float64)
            if sa_arr.size % 28 == 0 and sa_arr.size // 28 != n_frames:
                print(f"[WARN] {ep_id}: states_actions has {sa_arr.size // 28} frames vs {n_frames} rendered")
            (out_ep / "states_actions.bin").write_bytes(sa.read_bytes())
            copied.append("states_actions.bin")
        for pat in SIDECAR_GLOBS:
            for f in sorted(src_ep.glob(pat)):
                (out_ep / f.name).write_bytes(f.read_bytes())
                copied.append(f.name)

    # Staged episode_metadata.json (combined.mp4 is 224x168 per cam, matching lerobot_to_abc).
    if want_combined or cfg.staged_src is not None:
        meta = {
            "task_name": cfg.task_name,
            "cameras": list(CAMERAS),
            "camera_resolutions": {c: [OUT_W, OUT_H] for c in CAMERAS},
            "alignment": "mjwarp_rerender_integration_state_1to1",
            "num_steps": n_frames,
            "source_episode_id": ep_id.replace("episode_", ""),
            "renderer": "mjwarp",
            "asset_root": cfg.asset_root,
        }
        (out_ep / "episode_metadata.json").write_text(json.dumps(meta, indent=2))
    pc_note = f" +percam@{cfg.width}x{cfg.height}" if percam_ep is not None else ""
    print(f"[OK] {ep_id}: {n_frames} frames, render={render_s:.1f}s "
          f"({n_frames / render_s:.1f} fps){pc_note}, copied={copied}")
    return ep_id


def select_episodes(cfg: Config) -> list[Path]:
    if cfg.episodes:
        names = []
        for tok in cfg.episodes.split(","):
            tok = tok.strip()
            if not tok:
                continue
            names.append(tok if tok.startswith("episode_") else f"episode_{tok}")
        dirs = [cfg.delivery_root / n for n in names]
    else:
        dirs = sorted(p for p in cfg.delivery_root.glob("episode_*") if p.is_dir())
    dirs = [d for d in dirs if (d / "scene_assembled.xml").exists() and (d / "integration_state.npy").exists()]
    if cfg.limit is not None:
        dirs = dirs[: cfg.limit]
    return dirs


def main(cfg: Config) -> None:
    eps = select_episodes(cfg)
    print(f"{len(eps)} episodes -> {cfg.out_dir}  (asset_root={cfg.asset_root})")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for ep_dir in eps:
        try:
            render_episode(ep_dir, cfg)
            done += 1
        except Exception as e:  # keep going on a bad episode in batch mode
            print(f"[FAIL] {ep_dir.name}: {type(e).__name__}: {e}")
    print(f"rendered {done}/{len(eps)}")


if __name__ == "__main__":
    main(tyro.cli(Config))
