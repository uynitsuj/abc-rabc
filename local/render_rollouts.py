"""Offline-render eval rollouts to mp4 from their saved qpos trajectories.

Videos during rollout are expensive (per-step rendering while the policy
server idles); the harness instead saves per-step qpos (``save_states``) which
this script replays through the same batched mjwarp renderer afterwards. With
``early_stop_on_success`` the saved trajectories are already truncated at the
success moment, so renders end right after the placement.

Frames are batched across warp worlds: N frames of one episode are loaded as N
worlds and rendered in a single batched call.

Run:
    PYTHONPATH=. MUJOCO_GL=egl uv run --no-sync python local/render_rollouts.py \
        --root /nfs_us_2/karim/warp/eval_out/slip_rigup_1bottle_v3
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import yam_sim
from yam_sim.eval.rollout import per_world_frames
from yam_sim.eval.video import EpisodeVideoWriters
from yam_sim.task_specs import maybe_get_task_spec


def load_episodes(root: Path, models: list[str] | None, tasks: list[str] | None):
    episodes = []
    for line in (root / "results.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if models and rec["model_label"] not in models:
            continue
        if tasks and rec["task"] not in tasks:
            continue
        if not rec.get("states_path"):
            continue
        episodes.append(rec)
    return episodes


def forced_count_from_eval_mode(eval_mode: str | None) -> int | None:
    if not eval_mode:
        return None
    match = re.fullmatch(r"batched_fixed_count_(\d+)", eval_mode)
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Render saved eval qpos to mp4")
    parser.add_argument("--root", type=Path, required=True, help="Eval output_dir")
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--num-worlds", type=int, default=16, help="Frames per batched render")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    episodes = load_episodes(args.root, args.model, args.task)
    if not episodes:
        print("Nothing to render.")
        return
    print(f"{len(episodes)} episodes to render")

    by_task: dict[str, list[dict]] = {}
    for rec in episodes:
        by_task.setdefault(rec["task"], []).append(rec)

    for task_name, recs in by_task.items():
        spec = maybe_get_task_spec(task_name)
        env_task = spec.env_task if spec is not None else task_name
        prompt = spec.prompt if spec is not None else task_name
        count = forced_count_from_eval_mode(recs[0].get("eval_mode"))
        print(f"[{task_name}] env_task={env_task} force_object_count={count}")
        env = yam_sim.make_batched_env(
            task=env_task,
            prompt=prompt,
            num_worlds=args.num_worlds,
            force_object_count=count,
            camera_backend="mjwarp",
        )
        try:
            env.reset(seed=0)
            runtime = env._runtime
            for rec in recs:
                states_path = Path(rec["states_path"])
                video_path = states_path.parent / f"seed_{rec['seed']}.mp4"
                if video_path.exists() and not args.overwrite:
                    print(f"  seed {rec['seed']}: exists, skipping")
                    continue
                if not states_path.is_file():
                    print(f"  seed {rec['seed']}: missing {states_path}, skipping")
                    continue
                traj = np.load(states_path)
                if traj.shape[1] != env.model.nq:
                    print(
                        f"  seed {rec['seed']}: nq mismatch "
                        f"({traj.shape[1]} vs model {env.model.nq}), skipping"
                    )
                    continue
                writers = EpisodeVideoWriters([video_path], fps=args.fps)
                for start in range(0, len(traj), args.num_worlds):
                    frames = traj[start : start + args.num_worlds]
                    runtime.load_qpos_batch(frames)
                    runtime.forward()
                    obs = env.get_obs()
                    stitched = per_world_frames(obs, env.camera_names)[: len(frames)]
                    for frame in stitched:
                        writers.append(frame[None])
                writers.close()
                print(f"  seed {rec['seed']}: {len(traj)} frames -> {video_path.name}")
        finally:
            env.close()
    print("done")


if __name__ == "__main__":
    main()
