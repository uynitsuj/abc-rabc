"""CLI for exporting delivered sim episodes into trainable dataset format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yam_sim.dataset_export.monitor import serve_monitor, MonitorConfig
from yam_sim.dataset_export.metadata import finalize_dataset_metadata
from yam_sim.dataset_export.pipeline import export_dataset, export_episode, upload_dataset
from yam_sim.dataset_export.s3_pipeline import export_s3_shard, finalize_s3_dataset
from yam_sim.dataset_export.types import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    ExportConfig,
)


def _add_render_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-name", required=True, help="Batch name recorded in collected metadata")
    parser.add_argument(
        "--render-backend",
        choices=("mujoco", "mjwarp", "madrona"),
        default="mjwarp",
        help="Renderer used for sim camera export",
    )
    parser.add_argument("--img-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--img-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--sim-batch-size", type=int, default=32)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--state-pad-size", type=int, default=32)
    parser.add_argument("--action-pad-size", type=int, default=32)


def _add_local_export_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-root", required=True, help="Root containing delivered episode dirs")
    parser.add_argument("--output-root", required=True, help="Local dataset root to write")


def _config_from_args(args) -> ExportConfig:
    return ExportConfig(
        batch_name=args.batch_name,
        fps=args.fps,
        image_width=args.img_width,
        image_height=args.img_height,
        render_backend=args.render_backend,
        sim_batch_size=args.sim_batch_size,
        gpu_id=args.gpu_id,
        state_pad_size=args.state_pad_size,
        action_pad_size=args.action_pad_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export delivered sim episodes into trainable BC data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export episode data and collected metadata locally")
    _add_local_export_paths(export_parser)
    _add_render_config_args(export_parser)

    episode_export_parser = subparsers.add_parser(
        "export-episode",
        help="Export exactly one delivered episode and write a result JSON",
    )
    episode_export_parser.add_argument("--episode-dir", required=True)
    episode_export_parser.add_argument("--output-root", required=True)
    episode_export_parser.add_argument("--result-json", required=True)
    episode_export_parser.add_argument("--source-delivery", default=None)
    _add_render_config_args(episode_export_parser)

    finalize_parser = subparsers.add_parser("finalize", help="Generate norm_stats and train/val metadata locally")
    finalize_parser.add_argument("--output-root", required=True)
    finalize_parser.add_argument("--n-states", type=int, default=14)
    finalize_parser.add_argument("--n-actions", type=int, default=14)
    finalize_parser.add_argument("--state-pad-size", type=int, default=32)
    finalize_parser.add_argument("--action-pad-size", type=int, default=32)
    finalize_parser.add_argument("--val-ratio", type=float, default=0.05)
    finalize_parser.add_argument("--seed", type=int, default=123)

    upload_parser = subparsers.add_parser("upload", help="Sync local data/metadata to S3")
    upload_parser.add_argument("--output-root", required=True)
    upload_parser.add_argument("--s3-prefix", required=True)
    upload_parser.add_argument("--aws-profile", default=None)
    upload_parser.add_argument("--aws-region", default=None)

    run_parser = subparsers.add_parser("run", help="Export locally, finalize metadata, and optionally upload")
    _add_local_export_paths(run_parser)
    _add_render_config_args(run_parser)
    run_parser.add_argument("--n-states", type=int, default=14)
    run_parser.add_argument("--n-actions", type=int, default=14)
    run_parser.add_argument("--val-ratio", type=float, default=0.05)
    run_parser.add_argument("--seed", type=int, default=123)
    run_parser.add_argument("--s3-prefix", default=None)
    run_parser.add_argument("--aws-profile", default=None)
    run_parser.add_argument("--aws-region", default=None)

    s3_export_parser = subparsers.add_parser(
        "s3-export",
        help="Download one shard from S3, export locally, and upload episodes plus shard metadata",
    )
    s3_export_parser.add_argument("--s3-input-prefix", required=True)
    s3_export_parser.add_argument("--s3-output-root", required=True)
    s3_export_parser.add_argument("--scratch-dir", required=True)
    s3_export_parser.add_argument("--shard-index", type=int, default=0)
    s3_export_parser.add_argument("--num-shards", type=int, default=1)
    s3_export_parser.add_argument("--source-aws-profile", default=None)
    s3_export_parser.add_argument("--output-aws-profile", default=None)
    s3_export_parser.add_argument("--source-region", default=None)
    s3_export_parser.add_argument("--output-region", default=None)
    s3_export_parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Load existing shard metadata from the output root and skip already-exported episodes",
    )
    _add_render_config_args(s3_export_parser)

    s3_finalize_parser = subparsers.add_parser(
        "s3-finalize",
        help="Merge shard metadata from S3, compute final metadata, and upload it",
    )
    s3_finalize_parser.add_argument("--s3-output-root", required=True)
    s3_finalize_parser.add_argument("--scratch-dir", required=True)
    s3_finalize_parser.add_argument("--aws-profile", default=None)
    s3_finalize_parser.add_argument("--aws-region", default=None)
    s3_finalize_parser.add_argument("--n-states", type=int, default=14)
    s3_finalize_parser.add_argument("--n-actions", type=int, default=14)
    s3_finalize_parser.add_argument("--state-pad-size", type=int, default=32)
    s3_finalize_parser.add_argument("--action-pad-size", type=int, default=32)
    s3_finalize_parser.add_argument("--val-ratio", type=float, default=0.05)
    s3_finalize_parser.add_argument("--seed", type=int, default=123)

    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Run a localhost dashboard for a dataset export",
    )
    monitor_parser.add_argument("--host", default="127.0.0.1")
    monitor_parser.add_argument("--port", type=int, default=8091)
    monitor_parser.add_argument("--s3-input-prefix", required=True)
    monitor_parser.add_argument("--s3-output-root", default=None)
    monitor_parser.add_argument("--staging-root", default=None)
    monitor_parser.add_argument("--far-root", default=None)
    monitor_parser.add_argument("--source-aws-profile", default=None)
    monitor_parser.add_argument("--staging-aws-profile", default=None)
    monitor_parser.add_argument("--far-aws-profile", default=None)
    monitor_parser.add_argument("--source-region", default=None)
    monitor_parser.add_argument("--staging-region", default=None)
    monitor_parser.add_argument("--far-region", default=None)
    monitor_parser.add_argument("--num-shards", type=int, default=1)
    monitor_parser.add_argument("--poll-interval", type=float, default=10.0)

    args = parser.parse_args()
    if args.command == "export":
        export_dataset(
            Path(args.input_root),
            Path(args.output_root),
            config=_config_from_args(args),
            max_episodes=args.max_episodes,
        )
        return

    if args.command == "export-episode":
        _trajectory, artifacts, metadata = export_episode(
            Path(args.episode_dir),
            Path(args.output_root),
            config=_config_from_args(args),
            source_delivery=args.source_delivery,
        )
        Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.result_json).write_text(
            json.dumps(
                {
                    "episode_dir": str(artifacts.episode_dir),
                    "states_actions_path": str(artifacts.states_actions_path),
                    "states_actions_bin_path": str(artifacts.states_actions_bin_path),
                    "initial_qpos_path": str(artifacts.initial_qpos_path),
                    "episode_metadata_path": str(artifacts.episode_metadata_path),
                    "video_paths": {key: str(value) for key, value in artifacts.video_paths.items()},
                    "metadata": metadata,
                },
                indent=2,
            )
        )
        return

    if args.command == "finalize":
        finalize_dataset_metadata(
            Path(args.output_root),
            n_states=args.n_states,
            n_actions=args.n_actions,
            state_pad_size=args.state_pad_size,
            action_pad_size=args.action_pad_size,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        return

    if args.command == "upload":
        upload_dataset(
            Path(args.output_root),
            args.s3_prefix,
            aws_profile=args.aws_profile,
            aws_region=args.aws_region,
        )
        return

    if args.command == "s3-export":
        summary = export_s3_shard(
            args.s3_input_prefix,
            args.s3_output_root,
            Path(args.scratch_dir),
            config=_config_from_args(args),
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            source_aws_profile=args.source_aws_profile,
            output_aws_profile=args.output_aws_profile,
            source_region=args.source_region,
            output_region=args.output_region,
            max_episodes=args.max_episodes,
            resume_existing=args.resume_existing,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "s3-finalize":
        outputs = finalize_s3_dataset(
            args.s3_output_root,
            Path(args.scratch_dir),
            aws_profile=args.aws_profile,
            aws_region=args.aws_region,
            n_states=args.n_states,
            n_actions=args.n_actions,
            state_pad_size=args.state_pad_size,
            action_pad_size=args.action_pad_size,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        print(json.dumps(outputs, indent=2))
        return

    if args.command == "monitor":
        monitor_root = args.s3_output_root or args.staging_root
        if not monitor_root:
            parser.error("monitor requires --s3-output-root or --staging-root")
        serve_monitor(
            host=args.host,
            port=args.port,
            config=MonitorConfig(
                s3_input_prefix=args.s3_input_prefix,
                staging_root=monitor_root,
                far_root=args.far_root,
                source_aws_profile=args.source_aws_profile,
                staging_aws_profile=args.staging_aws_profile,
                far_aws_profile=args.far_aws_profile,
                source_region=args.source_region,
                staging_region=args.staging_region,
                far_region=args.far_region,
                num_shards=args.num_shards,
                poll_interval_s=args.poll_interval,
            ),
        )
        return

    config = _config_from_args(args)
    export_dataset(
        Path(args.input_root),
        Path(args.output_root),
        config=config,
        max_episodes=args.max_episodes,
    )
    finalize_dataset_metadata(
        Path(args.output_root),
        n_states=args.n_states,
        n_actions=args.n_actions,
        state_pad_size=config.state_pad_size,
        action_pad_size=config.action_pad_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if args.s3_prefix:
        upload_dataset(
            Path(args.output_root),
            args.s3_prefix,
            aws_profile=args.aws_profile,
            aws_region=args.aws_region,
        )


if __name__ == "__main__":
    main()
