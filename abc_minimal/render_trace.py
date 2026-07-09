"""Offline qpos-trace -> video renderer for hardened put-bottles sim evals.

eval_policy.py's hardened scorer saves a per-world fp16 qpos trace
(qpos_trace_*.npz, shape (T, nq), one frame per 30 Hz control step) plus the
world's randomization record (bottle/bin scales and spawn poses) in
summary.json. This script rebuilds the EXACT MuJoCo model the eval used
(scene_xml with the recorded bottle_scales/bin_scale), then kinematically
replays the trace -- data.qpos = trace[t]; mj_forward; offscreen render --
with NO physics stepping and no policy. CPU-friendly: only an EGL/OSMesa GL
context is used (no mjwarp, no torch CUDA).

It also re-runs the hardened PutBottlesEvaluator over the trace and checks the
replayed placed_ever / in-bin enter events against the stored world record, so
every rendered video is provably the same rollout the scorer graded.

Usage:
  .venv/bin/python -m abc_minimal.render_trace \
    --trace local_eval_out/hard_warp_sh0/qpos_trace_008_s20260519.npz \
    --world-json local_eval_out/hard_warp_sh0/summary.json \
    --out trace_videos/warp_s20260519.mp4 --speed 2 --label "WARP seed 20260519"

--world-json accepts either a full summary.json (the world is auto-matched by
the trace filename, or via --world-index) or a single world-record JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TRACE_HZ = 30.0


def load_world_record(world_json: Path, trace_path: Path, world_index: int | None):
    """Return (world_record, scene_config_dict|None) from a summary.json or a bare record."""
    data = json.loads(Path(world_json).read_text())
    scene_cfg = (data.get("config") or {}).get("scene")
    if "worlds" not in data:
        return data, scene_cfg
    worlds = data["worlds"]
    if world_index is not None:
        matches = [w for w in worlds if w["world_index"] == world_index]
        if not matches:
            raise SystemExit(f"world_index {world_index} not in {world_json}")
        return matches[0], scene_cfg
    base = Path(trace_path).name
    matches = [
        w for w in worlds
        if w.get("qpos_trace_path") and Path(w["qpos_trace_path"]).name == base
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"could not uniquely match trace {base} in {world_json} "
            f"({len(matches)} matches); pass --world-index"
        )
    return matches[0], scene_cfg


def build_model(record: dict, scene_cfg: dict | None):
    """Rebuild the exact eval model: scene_xml() with the recorded scales."""
    from abc_minimal.config import PutBottlesSimConfig
    from abc_minimal.eval_policy import scene_xml

    scene = PutBottlesSimConfig(**scene_cfg) if scene_cfg else PutBottlesSimConfig()
    rand = record["randomization"]
    xml = scene_xml(
        scene,
        np.asarray(rand["bottle_scales"], dtype=np.float64),
        float(rand["bin_scale"]),
    )
    model = mujoco.MjModel.from_xml_string(xml)
    return model, scene


def _freejoint_adr(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise SystemExit(f"joint not found in rebuilt model: {name}")
    return int(model.jnt_qposadr[jid])


def verify_spawn_poses(model: mujoco.MjModel, trace: np.ndarray, record: dict) -> float:
    """Check trace frame 0 freejoint XY against the randomization record.

    Frame 0 is recorded AFTER the first 30 Hz control step (~34 ms of physics),
    so bottles have settled a few mm vertically; XY should still match spawn.
    Returns max XY deviation (m).
    """
    rand = record["randomization"]
    q0 = trace[0].astype(np.float64)
    worst = 0.0
    entries = list(rand["bottle_states"].items()) + [("bin_joint", rand["bin_state"])]
    for name, st in entries:
        adr = _freejoint_adr(model, name)
        dxy = float(np.linalg.norm(q0[adr : adr + 2] - np.asarray(st["pos"][:2])))
        worst = max(worst, dxy)
    return worst


def replay_scorer(model: mujoco.MjModel, scene, trace: np.ndarray, record: dict):
    """Re-run the hardened evaluator over the trace; return per-frame HUD data
    plus a validation dict against the stored world record.

    The eval loop called evaluate() once on the initial state BEFORE the first
    control step (that qpos is not in the trace), so trace[i] is evaluator step
    i+1. We offset the replayed step counter by 1 to reproduce the recorded
    event step numbers and grace/persistence gating exactly.
    """
    from abc_minimal.eval_policy import PutBottlesEvaluator

    ev = PutBottlesEvaluator(model, scene)
    ev.reset()
    ev._step = 1  # account for the pre-rollout evaluate() not present in the trace
    in_bin = np.zeros(len(trace), dtype=np.int32)
    placed = np.zeros(len(trace), dtype=np.int32)
    for i, q in enumerate(trace):
        r = ev.evaluate(q.astype(np.float32))
        in_bin[i] = r["num_bottles_in_bin"]
        placed[i] = r["placed_ever"]
    replay_events = [[int(s), n, e] for s, n, e in ev.inbin_events]
    rec_events = [[int(s), n, e] for s, n, e in record.get("inbin_events", [])]
    rec_placed = int(record["final_task_eval"]["placed_ever"])
    # fp16 trace quantization can shift a borderline in-bin crossing by one
    # frame (~0.5 mm on the eval margins), so also compare with +-1 step slack
    # after dropping single-frame enter/exit flickers on both sides.
    def _deflicker(events):
        out, i = [], 0
        while i < len(events):
            if (
                i + 1 < len(events)
                and events[i][2] == "enter"
                and events[i + 1][2] == "exit"
                and events[i][1] == events[i + 1][1]
                and events[i + 1][0] - events[i][0] <= 1
            ):
                i += 2
                continue
            out.append(events[i])
            i += 1
        return out

    rp, rc = _deflicker(replay_events), _deflicker(rec_events)
    events_match_1frame = len(rp) == len(rc) and all(
        a[1] == b[1] and a[2] == b[2] and abs(a[0] - b[0]) <= 1 for a, b in zip(rp, rc)
    )
    validation = {
        "replay_placed_ever": int(placed[-1]),
        "record_placed_ever": rec_placed,
        "placed_ever_match": int(placed[-1]) == rec_placed,
        "replay_enter_events": sum(1 for e in replay_events if e[2] == "enter"),
        "record_enter_events": sum(1 for e in rec_events if e[2] == "enter"),
        "events_match": replay_events == rec_events,
        "events_match_1frame": events_match_1frame,
        "num_active": int(r["num_active_bottles"]),
    }
    return in_bin, placed, validation


def record_placed_per_frame(record: dict, n_frames: int) -> np.ndarray:
    """Per-frame hardened placed count from the RECORD's placement_steps (the
    source of truth; step s in the record corresponds to trace frame s-1)."""
    steps = sorted(int(s) for s in record.get("placement_steps", []))
    placed = np.zeros(n_frames, dtype=np.int32)
    for s in steps:
        placed[max(0, s - 1):] += 1
    return placed


def draw_hud(frame: np.ndarray, label: str, t_s: float, in_bin: int, placed: int, active: int) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, max(16, frame.shape[0] // 22))
    lines = [
        (label, (10, 8)),
        (f"t={t_s:5.1f}s   in bin: {in_bin}   placed: {placed}/{active}", (10, frame.shape[0] - 8 - font.size)),
    ]
    for text, xy in lines:
        if not text:
            continue
        x, y = xy
        d.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0))
        d.text((x, y), text, font=font, fill=(255, 255, 90))
    return np.asarray(img)


def render_video(
    model: mujoco.MjModel,
    trace: np.ndarray,
    out_path: Path,
    cameras: list[str],
    speed: int,
    height: int,
    width: int,
    label: str,
    hud_in_bin: np.ndarray | None,
    hud_placed: np.ndarray | None,
    num_active: int,
    hold_frames: int = 0,
) -> None:
    import imageio.v2 as imageio

    for cam in cameras:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam) < 0:
            names = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                for i in range(model.ncam)
            ]
            raise SystemExit(f"camera {cam!r} not found; available: {names}")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=int(TRACE_HZ), macro_block_size=1, quality=8)
    shown_label = f"{label} [{speed}x]" if (label and speed > 1) else label
    try:
        indices = list(range(0, len(trace), speed))
        if indices[-1] != len(trace) - 1:
            indices.append(len(trace) - 1)  # always show the final state (last-step placements)
        for i in indices:
            data.qpos[:] = trace[i].astype(np.float64)
            mujoco.mj_forward(model, data)
            views = []
            for cam in cameras:
                renderer.update_scene(data, camera=cam)
                views.append(renderer.render())
            frame = np.concatenate(views, axis=1) if len(views) > 1 else views[0]
            if hud_in_bin is not None:
                frame = draw_hud(
                    frame, shown_label, i / TRACE_HZ,
                    int(hud_in_bin[i]), int(hud_placed[i]), num_active,
                )
            writer.append_data(frame)
        for _ in range(hold_frames):  # freeze last frame (pairs different-length rollouts)
            writer.append_data(frame)
    finally:
        writer.close()
        renderer.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", required=True, help="qpos_trace_*.npz from eval_policy")
    p.add_argument("--world-json", required=True, help="summary.json (auto-matches world by trace name) or a single world-record JSON")
    p.add_argument("--out", required=True, help="output .mp4")
    p.add_argument("--world-index", type=int, default=None)
    p.add_argument("--camera", default="overhead,left_side", help="comma-separated camera names (rendered side by side)")
    p.add_argument("--speed", type=int, default=1, help="playback speedup: keep every Nth frame, fps stays 30")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640, help="width PER CAMERA")
    p.add_argument("--label", default="", help="text burned into the top-left corner")
    p.add_argument("--no-hud", action="store_true", help="disable the label/time/bottle-count overlay")
    p.add_argument("--hold-seconds", type=float, default=0.0, help="freeze the last frame this many output seconds")
    args = p.parse_args()

    trace_path = Path(args.trace)
    trace = np.load(trace_path)["qpos"]
    record, scene_cfg = load_world_record(Path(args.world_json), trace_path, args.world_index)
    model, scene = build_model(record, scene_cfg)
    if model.nq != trace.shape[1]:
        raise SystemExit(f"model.nq={model.nq} != trace nq={trace.shape[1]}: wrong model/scene reconstruction")
    if len(trace) != int(record["steps"]):
        print(f"WARNING: trace has {len(trace)} frames but record says steps={record['steps']}", file=sys.stderr)

    spawn_dxy = verify_spawn_poses(model, trace, record)
    in_bin, placed, val = replay_scorer(model, scene, trace, record)
    print(
        f"reconstruction: seed={record['world_seed']} nq={model.nq} frames={len(trace)} "
        f"max spawn XY deviation={spawn_dxy * 1000:.1f}mm"
    )
    print(
        f"scorer replay: placed_ever {val['replay_placed_ever']} vs record {val['record_placed_ever']} "
        f"(match={val['placed_ever_match']}), enter events {val['replay_enter_events']} vs "
        f"{val['record_enter_events']}, exact event match={val['events_match']}, "
        f"1-frame-tolerant match={val['events_match_1frame']}"
    )
    if not val["placed_ever_match"] and val["events_match_1frame"]:
        print(
            "NOTE: placed_ever differs but all events match within 1 frame — fp16 trace "
            "quantization at an eval-margin boundary (record is the source of truth)",
            file=sys.stderr,
        )
    elif not val["placed_ever_match"]:
        print("WARNING: replayed placed_ever does not match the record — reconstruction is suspect", file=sys.stderr)

    # HUD: dense in-bin count from the replay; hardened "placed" count from the
    # RECORD's placement_steps so the overlay always agrees with the scored result.
    hud_placed = record_placed_per_frame(record, len(trace))
    render_video(
        model,
        trace,
        Path(args.out),
        [c.strip() for c in args.camera.split(",") if c.strip()],
        max(1, args.speed),
        args.height,
        args.width,
        args.label,
        None if args.no_hud else in_bin,
        None if args.no_hud else hud_placed,
        val["num_active"],
        hold_frames=int(round(args.hold_seconds * TRACE_HZ)),
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
