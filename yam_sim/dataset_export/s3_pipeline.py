"""S3-backed shard export and finalization orchestration."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
from typing import Any
import traceback

from yam_sim.dataset_export.metadata import finalize_dataset_metadata, write_json
from yam_sim.dataset_export.pipeline import export_episode_with_backend_lifecycle
from yam_sim.dataset_export.s3_source import discover_episode_sources, shard_episode_sources
from yam_sim.dataset_export.s3_utils import (
    copy_local_dir_to_s3,
    copy_local_file_to_s3,
    copy_s3_object_to_local,
    list_s3_objects,
    parse_s3_uri,
    sync_local_dir_to_s3,
)
from yam_sim.dataset_export.staging import cleanup_local_tree, stage_episode_locally
from yam_sim.dataset_export.types import ExportConfig


def _shard_tag(shard_index: int, num_shards: int) -> str:
    width = max(2, len(str(max(0, num_shards - 1))))
    return f"{shard_index:0{width}d}"


def _merge_unique_records(
    destination: dict[str, Any],
    payload: dict[str, Any],
    *,
    kind: str,
) -> None:
    for key, value in payload.items():
        existing = destination.get(key)
        if existing is None:
            destination[key] = value
            continue
        if existing != value:
            raise RuntimeError(f"Conflicting duplicate {kind} entry for {key}")


def _remote_metadata_uri(s3_output_root: str, filename: str) -> str:
    return parse_s3_uri(s3_output_root).child("metadata", filename).uri


def _remote_episode_data_uri(s3_output_root: str, episode_id: str) -> str:
    return parse_s3_uri(s3_output_root).child("data", episode_id).uri


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _ShardStatusWriter:
    """Best-effort S3 heartbeat for one export shard."""

    def __init__(
        self,
        *,
        s3_output_root: str,
        local_dir: Path,
        shard_index: int,
        num_shards: int,
        config: ExportConfig,
        source_prefix: str,
        output_aws_profile: str | None,
        output_region: str | None,
    ) -> None:
        self.s3_output_root = s3_output_root
        self.local_path = Path(local_dir) / f"status_shard_{_shard_tag(shard_index, num_shards)}.json"
        self.remote_uri = _remote_metadata_uri(s3_output_root, self.local_path.name)
        self.output_aws_profile = output_aws_profile
        self.output_region = output_region
        self.payload: dict[str, Any] = {
            "schema_version": 1,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "shard_tag": _shard_tag(shard_index, num_shards),
            "source_prefix": source_prefix,
            "output_root": s3_output_root,
            "batch_name": config.batch_name,
            "render_backend": config.render_backend,
            "image_width": config.image_width,
            "image_height": config.image_height,
            "fps": config.fps,
            "sim_batch_size": config.sim_batch_size,
            "gpu_id": config.gpu_id,
            "hostname": socket.gethostname(),
            "pod_name": os.environ.get("HOSTNAME"),
            "skypilot_node_rank": os.environ.get("SKYPILOT_NODE_RANK"),
            "skypilot_num_nodes": os.environ.get("SKYPILOT_NUM_NODES"),
            "started_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "phase": "initializing",
            "discovered_episodes": None,
            "assigned_episodes": None,
            "skipped_existing": 0,
            "attempted_episodes": 0,
            "completed_episodes": 0,
            "failed_episodes": 0,
            "current_episode": None,
            "current_episode_prefix": None,
            "last_completed_episode": None,
            "last_error": None,
        }

    def update(self, **updates: Any) -> None:
        self.payload.update(updates)
        self.payload["updated_at"] = _utc_now_iso()
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self.local_path.write_text(json.dumps(self.payload, indent=2, sort_keys=True) + "\n")
        try:
            copy_local_file_to_s3(
                self.local_path,
                self.remote_uri,
                aws_profile=self.output_aws_profile,
                aws_region=self.output_region,
            )
        except Exception as exc:  # pragma: no cover - status must not fail exports
            print(f"[s3-export] Warning: failed to upload shard status: {exc}")


def _load_remote_json_if_exists(
    s3_output_root: str,
    filename: str,
    local_dir: Path,
    *,
    aws_profile: str | None = None,
    aws_region: str | None = None,
) -> dict[str, Any] | list[Any] | None:
    uri = _remote_metadata_uri(s3_output_root, filename)
    try:
        local_path = copy_s3_object_to_local(
            uri,
            Path(local_dir) / filename,
            aws_profile=aws_profile,
            aws_region=aws_region,
        )
    except Exception:
        return None
    return json.loads(local_path.read_text())


def export_s3_shard(
    s3_input_prefix: str,
    s3_output_root: str,
    scratch_dir: Path,
    *,
    config: ExportConfig,
    shard_index: int = 0,
    num_shards: int = 1,
    source_aws_profile: str | None = None,
    output_aws_profile: str | None = None,
    source_region: str | None = None,
    output_region: str | None = None,
    max_episodes: int | None = None,
    resume_existing: bool = False,
) -> dict[str, Any]:
    """Run one shard of source-S3 -> local export -> destination-S3 upload."""
    discovered = discover_episode_sources(
        s3_input_prefix,
        aws_profile=source_aws_profile,
        aws_region=source_region,
    )
    assigned = shard_episode_sources(
        discovered,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    if max_episodes is not None:
        assigned = assigned[:max_episodes]

    shard_root = Path(scratch_dir) / f"shard_{_shard_tag(shard_index, num_shards)}"
    if shard_root.exists():
        shutil.rmtree(shard_root)
    source_root = shard_root / "source"
    dataset_root = shard_root / "dataset"
    metadata_root = shard_root / "metadata"
    source_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    shard_tag = _shard_tag(shard_index, num_shards)
    status = _ShardStatusWriter(
        s3_output_root=s3_output_root,
        local_dir=metadata_root,
        shard_index=shard_index,
        num_shards=num_shards,
        config=config,
        source_prefix=s3_input_prefix,
        output_aws_profile=output_aws_profile,
        output_region=output_region,
    )
    status.update(
        phase="resuming" if resume_existing else "starting",
        discovered_episodes=len(discovered),
        assigned_episodes=len(assigned),
    )
    existing_collected: dict[str, dict] = {}
    existing_camera_profiles: dict[str, dict] = {}
    if resume_existing:
        loaded_collected = _load_remote_json_if_exists(
            s3_output_root,
            f"collected_shard_{shard_tag}.json",
            metadata_root,
            aws_profile=output_aws_profile,
            aws_region=output_region,
        )
        if isinstance(loaded_collected, dict):
            existing_collected = loaded_collected
        loaded_camera_profiles = _load_remote_json_if_exists(
            s3_output_root,
            f"camera_profiles_shard_{shard_tag}.json",
            metadata_root,
            aws_profile=output_aws_profile,
            aws_region=output_region,
        )
        if isinstance(loaded_camera_profiles, dict):
            existing_camera_profiles = loaded_camera_profiles

    already_exported = set(existing_collected)
    assigned_to_run = [
        source for source in assigned if source.episode_name not in already_exported
    ]

    collected: dict[str, dict] = dict(existing_collected)
    camera_profiles: dict[str, dict] = dict(existing_camera_profiles)
    failures: list[dict[str, str]] = []
    status.update(
        phase="running" if assigned_to_run else "complete",
        skipped_existing=len(already_exported),
        attempted_episodes=len(assigned_to_run),
        completed_episodes=len(collected),
        failed_episodes=0,
    )
    if assigned_to_run:
        upload_data_root = dataset_root / "data"

        def submit_stage(
            executor: ThreadPoolExecutor,
            source,
        ) -> tuple[Any, Future[Path]] | None:
            if source is None:
                return None
            return (
                source,
                executor.submit(
                    stage_episode_locally,
                    source,
                    source_root,
                    aws_profile=source_aws_profile,
                    aws_region=source_region,
                ),
            )

        pending_upload: tuple[Any, Path, dict, Future[None]] | None = None

        def finalize_upload() -> None:
            nonlocal pending_upload
            if pending_upload is None:
                return
            source, exported_dir, metadata, future = pending_upload
            pending_upload = None
            try:
                future.result()
                profile = metadata.pop("_camera_profile", None)
                if profile is not None:
                    _merge_unique_records(
                        camera_profiles,
                        {metadata["camera_profile_id"]: profile},
                        kind="camera profile",
                    )
                _merge_unique_records(
                    collected,
                    {metadata["episode_id"]: metadata},
                    kind="episode metadata",
                )
                status.update(
                    phase="running",
                    completed_episodes=len(collected),
                    failed_episodes=len(failures),
                    last_completed_episode=metadata["episode_id"],
                    current_episode=None,
                    current_episode_prefix=None,
                    last_error=None,
                )
            except Exception as exc:
                failures.append(
                    {
                        "episode_name": source.episode_name,
                        "episode_prefix": source.episode_prefix,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                status.update(
                    phase="running",
                    completed_episodes=len(collected),
                    failed_episodes=len(failures),
                    current_episode=None,
                    current_episode_prefix=None,
                    last_error=str(exc),
                )
            finally:
                cleanup_local_tree(exported_dir, stop_at=upload_data_root)

        with ThreadPoolExecutor(max_workers=2) as executor:
            pending_stage = submit_stage(executor, assigned_to_run[0])

            for index, source in enumerate(assigned_to_run):
                staged_dir: Path | None = None
                exported_dir: Path | None = None
                try:
                    assert pending_stage is not None
                    stage_source, stage_future = pending_stage
                    if stage_source != source:
                        raise RuntimeError(
                            f"Stage/export pipeline desynchronized: expected {source.episode_name}, "
                            f"got {stage_source.episode_name}"
                        )

                    next_source = assigned_to_run[index + 1] if index + 1 < len(assigned_to_run) else None
                    pending_stage = submit_stage(executor, next_source)

                    status.update(
                        phase="staging",
                        current_episode=source.episode_name,
                        current_episode_prefix=source.episode_prefix,
                    )
                    staged_dir = stage_future.result()
                    status.update(
                        phase="exporting",
                        current_episode=source.episode_name,
                        current_episode_prefix=source.episode_prefix,
                    )
                    artifacts, metadata = export_episode_with_backend_lifecycle(
                        staged_dir,
                        dataset_root,
                        config=config,
                        source_delivery=source.source_delivery,
                    )
                    exported_dir = artifacts.episode_dir

                    finalize_upload()
                    pending_upload = (
                        source,
                        exported_dir,
                        metadata,
                        executor.submit(
                            copy_local_dir_to_s3,
                            exported_dir,
                            _remote_episode_data_uri(s3_output_root, metadata["episode_id"]),
                            aws_profile=output_aws_profile,
                            aws_region=output_region,
                        ),
                    )
                    status.update(
                        phase="uploading",
                        current_episode=metadata["episode_id"],
                        current_episode_prefix=source.episode_prefix,
                    )
                    exported_dir = None
                except Exception as exc:
                    failures.append(
                        {
                            "episode_name": source.episode_name,
                            "episode_prefix": source.episode_prefix,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    status.update(
                        phase="running",
                        completed_episodes=len(collected),
                        failed_episodes=len(failures),
                        current_episode=None,
                        current_episode_prefix=None,
                        last_error=str(exc),
                    )
                finally:
                    if exported_dir is not None:
                        cleanup_local_tree(exported_dir, stop_at=upload_data_root)
                    if staged_dir is not None:
                        cleanup_local_tree(staged_dir, stop_at=source_root)

            finalize_upload()

    status.update(
        phase="writing_metadata",
        completed_episodes=len(collected),
        failed_episodes=len(failures),
        current_episode=None,
        current_episode_prefix=None,
    )
    collected_path = write_json(metadata_root / f"collected_shard_{shard_tag}.json", collected)
    copy_local_file_to_s3(
        collected_path,
        _remote_metadata_uri(s3_output_root, collected_path.name),
        aws_profile=output_aws_profile,
        aws_region=output_region,
    )

    camera_profiles_path: Path | None = None
    if camera_profiles:
        camera_profiles_path = write_json(
            metadata_root / f"camera_profiles_shard_{shard_tag}.json",
            camera_profiles,
        )
        copy_local_file_to_s3(
            camera_profiles_path,
            _remote_metadata_uri(s3_output_root, camera_profiles_path.name),
            aws_profile=output_aws_profile,
            aws_region=output_region,
        )

    failures_path: Path | None = None
    if failures:
        failures_path = write_json(
            metadata_root / f"failures_shard_{shard_tag}.json",
            failures,
        )
        copy_local_file_to_s3(
            failures_path,
            _remote_metadata_uri(s3_output_root, failures_path.name),
            aws_profile=output_aws_profile,
            aws_region=output_region,
        )

    status.update(
        phase="complete" if not failures else "complete_with_failures",
        completed_episodes=len(collected),
        failed_episodes=len(failures),
        current_episode=None,
        current_episode_prefix=None,
        finished_at=_utc_now_iso(),
    )
    return {
        "discovered_episodes": len(discovered),
        "assigned_episodes": len(assigned),
        "skipped_existing": len(already_exported),
        "attempted_episodes": len(assigned_to_run),
        "exported_episodes": len(collected),
        "failed_episodes": len(failures),
        "collected_path": str(collected_path),
        "camera_profiles_path": str(camera_profiles_path) if camera_profiles_path else None,
        "failures_path": str(failures_path) if failures_path else None,
    }


def finalize_s3_dataset(
    s3_output_root: str,
    scratch_dir: Path,
    *,
    aws_profile: str | None = None,
    aws_region: str | None = None,
    n_states: int = 14,
    n_actions: int = 14,
    state_pad_size: int = 32,
    action_pad_size: int = 32,
    val_ratio: float = 0.05,
    seed: int = 123,
) -> dict[str, str]:
    """Merge shard metadata from S3, compute final metadata, and upload it."""
    metadata_prefix = parse_s3_uri(s3_output_root).child("metadata").uri
    metadata_objects = list_s3_objects(
        metadata_prefix,
        aws_profile=aws_profile,
        aws_region=aws_region,
    )
    collected_objects = [
        obj for obj in metadata_objects if Path(obj.key).name.startswith("collected_shard_")
    ]
    if not collected_objects:
        raise RuntimeError(f"No shard metadata found under {metadata_prefix}")

    camera_profile_objects = [
        obj for obj in metadata_objects if Path(obj.key).name.startswith("camera_profiles_shard_")
    ]

    finalize_root = Path(scratch_dir) / "finalize"
    if finalize_root.exists():
        shutil.rmtree(finalize_root)
    remote_meta_root = finalize_root / "remote_metadata"
    dataset_root = finalize_root / "dataset"
    metadata_root = dataset_root / "metadata"
    data_root = dataset_root / "data"
    remote_meta_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    collected: dict[str, dict] = {}
    for obj in sorted(collected_objects, key=lambda item: item.key):
        local_path = copy_s3_object_to_local(
            obj.uri,
            remote_meta_root / Path(obj.key).name,
            aws_profile=aws_profile,
            aws_region=aws_region,
        )
        _merge_unique_records(
            collected,
            json.loads(local_path.read_text()),
            kind="episode metadata",
        )

    camera_profiles: dict[str, dict] = {}
    for obj in sorted(camera_profile_objects, key=lambda item: item.key):
        local_path = copy_s3_object_to_local(
            obj.uri,
            remote_meta_root / Path(obj.key).name,
            aws_profile=aws_profile,
            aws_region=aws_region,
        )
        _merge_unique_records(
            camera_profiles,
            json.loads(local_path.read_text()),
            kind="camera profile",
        )

    write_json(metadata_root / "collected.json", collected)
    if camera_profiles:
        write_json(metadata_root / "camera_profiles.json", camera_profiles)

    for episode_id in sorted(collected):
        copy_s3_object_to_local(
            parse_s3_uri(s3_output_root).child("data", episode_id, "states_actions.npy").uri,
            data_root / episode_id / "states_actions.npy",
            aws_profile=aws_profile,
            aws_region=aws_region,
        )

    outputs = finalize_dataset_metadata(
        dataset_root,
        collected=collected,
        n_states=n_states,
        n_actions=n_actions,
        state_pad_size=state_pad_size,
        action_pad_size=action_pad_size,
        val_ratio=val_ratio,
        seed=seed,
    )
    sync_local_dir_to_s3(
        metadata_root,
        parse_s3_uri(s3_output_root).child("metadata").uri,
        aws_profile=aws_profile,
        aws_region=aws_region,
    )
    rendered_outputs = {name: str(path) for name, path in outputs.items()}
    if camera_profiles:
        rendered_outputs["camera_profiles"] = str(metadata_root / "camera_profiles.json")
    return rendered_outputs
