"""Run a robust policy eval matrix in yam-sim against openpi-served pi0 models.

Each (model checkpoint) x (task) cell is rolled out for N episodes using the
batched ``mjwarp`` env (``num_worlds`` episodes in parallel). One video is written
per episode, results are appended crash-safe to ``results.jsonl`` (resumable), and
a per-(model, task) success-rate summary is written at the end.

Prerequisite: start one openpi websocket server per checkpoint (see openpi
``scripts/serve_policy.py``), each on its own port.

Examples:
    # Inline: one model, two tasks, 8 episodes each, 4 in parallel.
    MUJOCO_GL=egl python -m yam_sim.examples.run_eval \
        --models pi0=0.0.0.0:8000 \
        --tasks bottles,sweep \
        --episodes 8 --num-worlds 4 \
        --output-dir /tmp/yam_eval

    # Compare two checkpoints.
    MUJOCO_GL=egl python -m yam_sim.examples.run_eval \
        --models base=0.0.0.0:8000,finetuned=0.0.0.0:8001 \
        --tasks bottles --episodes 20 --num-worlds 5 \
        --output-dir /tmp/yam_eval

    # Full job matrix from a config file.
    MUJOCO_GL=egl python -m yam_sim.examples.run_eval --config eval_config.yaml
"""

from __future__ import annotations

import argparse
import os

from yam_sim.eval.config import EvalConfig, ModelEndpoint, TaskSpecEntry
from yam_sim.eval.harness import run_eval

_DEFAULT_OUTPUT_DIR = "/tmp/yam_eval"


def parse_args():
    parser = argparse.ArgumentParser(description="Run a yam-sim policy eval matrix")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML/JSON eval config. Overrides inline flags when set.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated 'label=host:port' specs, e.g. base=0.0.0.0:8000,ft=0.0.0.0:8001",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task names/aliases, e.g. bottles,sweep",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--num-worlds", type=int, default=4)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Per-episode sim-time horizon; converted to chunks. Overrides max_chunks.",
    )
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scene", type=str, default="hybrid", choices=["eval", "training", "hybrid"])
    parser.add_argument(
        "--camera-backend",
        type=str,
        default="mjwarp",
        choices=["mjwarp", "madrona", "mujoco"],
        help="Observation renderer. 'mujoco' = MuJoCo-GL (matches training data, "
        "in-distribution, slower); 'mjwarp' = fast batched GPU render.",
    )
    parser.add_argument("--execute-chunk-dim", type=int, default=20)
    parser.add_argument("--prefix-length", type=int, default=3)
    parser.add_argument("--no-ttrtc", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--no-save-states",
        action="store_true",
        help="Disable saving the per-episode qpos trajectory (seed_<n>_qpos.npy).",
    )
    parser.add_argument("--jpeg-quality", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument(
        "--camera-gpu-id", type=int, default=None, help="GPU index for mjwarp rendering."
    )
    parser.add_argument(
        "--gpu", type=int, default=None, help="CUDA device for the harness process."
    )
    return parser.parse_args()


def _parse_models(spec: str) -> list[ModelEndpoint]:
    return [ModelEndpoint.parse(s) for s in spec.split(",") if s.strip()]


def _config_from_args(args) -> EvalConfig:
    if args.config is not None:
        import dataclasses

        from yam_sim.eval.config import load_eval_config

        config = load_eval_config(args.config)
        overrides: dict = {}
        # --models on the CLI fills in / overrides models from the file, so a
        # task-set config can stay checkpoint-independent.
        if args.models:
            overrides["models"] = _parse_models(args.models)
        if args.output_dir and args.output_dir != _DEFAULT_OUTPUT_DIR:
            overrides["output_dir"] = args.output_dir
        if args.max_seconds is not None:
            overrides["max_seconds"] = args.max_seconds
        if overrides:
            config = dataclasses.replace(config, **overrides)
        if not config.models:
            raise SystemExit(
                "No models defined: add 'models:' to the config or pass --models."
            )
        return config

    if not args.models or not args.tasks:
        raise SystemExit(
            "Provide --config, or both --models and --tasks for inline mode."
        )

    models = _parse_models(args.models)
    tasks = [TaskSpecEntry(task=t.strip()) for t in args.tasks.split(",") if t.strip()]

    return EvalConfig(
        models=models,
        tasks=tasks,
        output_dir=args.output_dir,
        episodes_default=args.episodes,
        num_worlds_default=args.num_worlds,
        max_seconds=args.max_seconds,
        scene=args.scene,
        camera_backend=args.camera_backend,
        execute_chunk_dim=args.execute_chunk_dim,
        prefix_length=args.prefix_length,
        use_ttrtc=not args.no_ttrtc,
        jpeg_quality=args.jpeg_quality,
        fps=args.fps,
        video=not args.no_video,
        save_states=not args.no_save_states,
        camera_gpu_id=args.camera_gpu_id,
        seed_base=args.seed_base,
    )


def main():
    args = parse_args()
    if args.gpu is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    config = _config_from_args(args)
    run_eval(config)


if __name__ == "__main__":
    main()
