"""Localhost dashboard for monitoring S3-backed dataset exports."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import threading
import time
from typing import Any

from yam_sim.dataset_export.s3_source import discover_episode_sources, shard_episode_sources
from yam_sim.dataset_export.s3_utils import copy_s3_object_to_local, list_s3_objects, parse_s3_uri, run_aws


_EPISODE_DIR_RE = re.compile(r"--episode-dir\s+(\S+)")
_GPU_RE = re.compile(r"--gpu-id\s+(\d+)")
_DELIVERY_RE = re.compile(r"--source-delivery\s+(\S+)")
_SHARD_RE = re.compile(r"shard_(\d+)")
_RESUME_RE = re.compile(r"resume[_a-zA-Z0-9]*_shard(\d+)")
_STATUS_RE = re.compile(r"status_shard_(\d+)\.json$")
_COLLECTED_SHARD_RE = re.compile(r"collected_shard_(\d+)\.json$")
_FAILURES_SHARD_RE = re.compile(r"failures_shard_(\d+)\.json$")
_EPISODE_NAME_RE = re.compile(r"^episode_")


@dataclass(frozen=True)
class MonitorConfig:
    s3_input_prefix: str
    staging_root: str
    far_root: str | None
    source_aws_profile: str | None
    staging_aws_profile: str | None
    far_aws_profile: str | None
    source_region: str | None
    staging_region: str | None
    far_region: str | None
    num_shards: int
    poll_interval_s: float


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _aws_ls(
    s3_uri: str,
    *,
    aws_profile: str | None = None,
    aws_region: str | None = None,
) -> tuple[list[str] | None, str | None]:
    if not s3_uri.endswith("/"):
        s3_uri = s3_uri + "/"
    try:
        result = run_aws(
            ["s3", "ls", s3_uri],
            aws_profile=aws_profile,
            aws_region=aws_region,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        return None, stderr or stdout or str(exc)
    return result.stdout.splitlines(), None


def _count_direct_children(
    s3_uri: str,
    *,
    aws_profile: str | None = None,
    aws_region: str | None = None,
) -> tuple[int | None, str | None]:
    lines, error = _aws_ls(s3_uri, aws_profile=aws_profile, aws_region=aws_region)
    if lines is None:
        return None, error
    return len([line for line in lines if line.strip()]), None


def _list_metadata_entries(
    s3_uri: str,
    *,
    aws_profile: str | None = None,
    aws_region: str | None = None,
) -> dict[str, dict[str, Any]]:
    lines, error = _aws_ls(s3_uri, aws_profile=aws_profile, aws_region=aws_region)
    if lines is None:
        return {"_error": {"message": error or "unknown error"}}
    entries: dict[str, dict[str, Any]] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 4:
            continue
        date_str, time_str, size_str = parts[0], parts[1], parts[2]
        name = parts[3]
        entries[name] = {
            "timestamp": f"{date_str}T{time_str}",
            "size_bytes": int(size_str),
        }
    return entries


def _load_metadata_json_objects(
    s3_uri: str,
    pattern: re.Pattern[str],
    *,
    aws_profile: str | None = None,
    aws_region: str | None = None,
) -> tuple[dict[int, dict[str, Any]], str | None]:
    metadata_uri = parse_s3_uri(s3_uri)
    try:
        objects = list_s3_objects(
            metadata_uri.uri,
            aws_profile=aws_profile,
            aws_region=aws_region,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        return {}, stderr or stdout or str(exc)

    values: dict[int, dict[str, Any]] = {}
    cache_root = Path(os.environ.get("XDOF_EXPORT_MONITOR_CACHE", "/tmp/yam_export_monitor"))
    for obj in objects:
        name = PurePosixPath(obj.key).name
        match = pattern.match(name)
        if match is None:
            continue
        shard_index = int(match.group(1))
        try:
            local_path = copy_s3_object_to_local(
                obj.uri,
                cache_root / metadata_uri.bucket / obj.key,
                aws_profile=aws_profile,
                aws_region=aws_region,
            )
            payload = json.loads(local_path.read_text())
            if isinstance(payload, dict):
                values[shard_index] = payload
        except Exception as exc:
            values[shard_index] = {"_error": str(exc), "shard_index": shard_index}
    return values, None


def _load_metadata_json_lists(
    s3_uri: str,
    pattern: re.Pattern[str],
    *,
    aws_profile: str | None = None,
    aws_region: str | None = None,
) -> tuple[dict[int, list[dict[str, Any]]], str | None]:
    metadata_uri = parse_s3_uri(s3_uri)
    try:
        objects = list_s3_objects(
            metadata_uri.uri,
            aws_profile=aws_profile,
            aws_region=aws_region,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        return {}, stderr or stdout or str(exc)

    values: dict[int, list[dict[str, Any]]] = {}
    cache_root = Path(os.environ.get("XDOF_EXPORT_MONITOR_CACHE", "/tmp/yam_export_monitor"))
    for obj in objects:
        name = PurePosixPath(obj.key).name
        match = pattern.match(name)
        if match is None:
            continue
        shard_index = int(match.group(1))
        try:
            local_path = copy_s3_object_to_local(
                obj.uri,
                cache_root / metadata_uri.bucket / obj.key,
                aws_profile=aws_profile,
                aws_region=aws_region,
            )
            payload = json.loads(local_path.read_text())
            if isinstance(payload, list):
                values[shard_index] = [item for item in payload if isinstance(item, dict)]
            else:
                values[shard_index] = []
        except Exception as exc:
            values[shard_index] = [{"_error": str(exc), "shard_index": shard_index}]
    return values, None


def _failure_error_type(error: str | None) -> str:
    error = error or ""
    if "integration_state.npy shape" in error:
        return "integration_state_shape"
    if "No overlapping export window" in error:
        return "no_overlap_window"
    if "requires integration_state.npy" in error:
        return "missing_integration_state"
    return "other"


def _task_from_failure_record(record: dict[str, Any]) -> str:
    prefix = str(record.get("episode_prefix") or "")
    parts = PurePosixPath(prefix).parts
    for idx, part in enumerate(parts):
        if _EPISODE_NAME_RE.match(part) and idx > 0:
            return parts[idx - 1]
    return "unknown"


def _build_failure_breakdown(
    failures_by_shard: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = {}
    by_error_type = Counter()
    record_count = 0
    shard_count = 0
    for shard_index, records in failures_by_shard.items():
        shard_records = [record for record in records if "_error" not in record]
        if shard_records:
            shard_count += 1
        for record in shard_records:
            record_count += 1
            task = _task_from_failure_record(record)
            error_type = _failure_error_type(record.get("error"))
            by_error_type[error_type] += 1
            row = by_task.setdefault(
                task,
                {
                    "task": task,
                    "total": 0,
                    "integration_state_shape": 0,
                    "no_overlap_window": 0,
                    "missing_integration_state": 0,
                    "other": 0,
                    "shards": set(),
                },
            )
            row["total"] += 1
            row[error_type] += 1
            row["shards"].add(shard_index)

    rows = []
    for row in by_task.values():
        normalized = dict(row)
        normalized["shard_count"] = len(row["shards"])
        normalized.pop("shards", None)
        rows.append(normalized)
    rows.sort(key=lambda item: (-int(item["total"]), str(item["task"])))
    return {
        "record_count": record_count,
        "shard_count": shard_count,
        "by_task": rows,
        "by_error_type": dict(by_error_type),
    }


def _run_pgrep(pattern: str) -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af", pattern],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def _extract_arg(pattern: re.Pattern[str], cmd: str) -> str | None:
    match = pattern.search(cmd)
    return match.group(1) if match else None


def _extract_shard_id(cmd: str) -> str:
    match = _SHARD_RE.search(cmd)
    if match:
        return match.group(1)
    match = _RESUME_RE.search(cmd)
    if match:
        return match.group(1)
    return "?"


def _human_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "n/a"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {sec:02d}s"
    return f"{sec}s"


def _parse_elapsed(value: str) -> float | None:
    if not value:
        return None
    parts = value.split("-")
    days = 0
    time_part = value
    if len(parts) == 2:
        days = int(parts[0])
        time_part = parts[1]
    chunks = [int(chunk) for chunk in time_part.split(":")]
    if len(chunks) == 3:
        hours, minutes, seconds = chunks
    elif len(chunks) == 2:
        hours, minutes, seconds = 0, chunks[0], chunks[1]
    else:
        hours, minutes, seconds = 0, 0, chunks[0]
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _ps_rows(pattern: str) -> list[dict[str, Any]]:
    lines = _run_pgrep(pattern)
    rows: list[dict[str, Any]] = []
    for line in lines:
        pid_str, _, cmd = line.partition(" ")
        if not pid_str.isdigit():
            continue
        pid = int(pid_str)
        ps = subprocess.run(
            ["ps", "-o", "pid=,etime=,%cpu=,%mem=,args=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
        details = ps.stdout.strip()
        if not details:
            continue
        parts = details.split(None, 4)
        if len(parts) < 5:
            continue
        _, elapsed, cpu, mem, args = parts
        rows.append(
            {
                "pid": pid,
                "elapsed": elapsed,
                "elapsed_seconds": _parse_elapsed(elapsed),
                "cpu_percent": float(cpu),
                "mem_percent": float(mem),
                "cmd": args,
            }
        )
    return rows


def _build_active_episode_rows() -> list[dict[str, Any]]:
    rows = _ps_rows(r"yam_sim\.dataset_export\.cli export-episode")
    active: list[dict[str, Any]] = []
    for row in rows:
        cmd = row["cmd"]
        episode_dir = _extract_arg(_EPISODE_DIR_RE, cmd)
        episode_name = PurePosixPath(episode_dir).name if episode_dir else "unknown"
        active.append(
            {
                "pid": row["pid"],
                "shard": _extract_shard_id(cmd),
                "episode": episode_name,
                "delivery": _extract_arg(_DELIVERY_RE, cmd),
                "gpu_id": _extract_arg(_GPU_RE, cmd),
                "elapsed": row["elapsed"],
                "elapsed_seconds": row["elapsed_seconds"],
                "cpu_percent": row["cpu_percent"],
                "mem_percent": row["mem_percent"],
                "cmd": cmd,
            }
        )
    return active


class ExportRunMonitor:
    def __init__(self, config: MonitorConfig):
        self.config = config
        self._lock = threading.Lock()
        self._history: deque[dict[str, Any]] = deque(maxlen=720)
        self._snapshot: dict[str, Any] = {"status": "initializing"}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._source_totals = self._discover_totals()
        self._last_good_staged_count: int | None = None
        self._last_good_far_count: int | None = None
        self._last_good_metadata: dict[str, Any] = {}
        self._last_good_remote_statuses: dict[int, dict[str, Any]] = {}
        self._last_good_collected_counts: dict[int, dict[str, Any]] = {}
        self._last_good_failures: dict[int, list[dict[str, Any]]] = {}

    def _discover_totals(self) -> dict[str, Any]:
        sources = discover_episode_sources(
            self.config.s3_input_prefix,
            aws_profile=self.config.source_aws_profile,
            aws_region=self.config.source_region,
        )
        task_totals = Counter(source.source_delivery for source in sources)
        shard_totals = []
        for shard_index in range(self.config.num_shards):
            shard_totals.append(
                len(
                    shard_episode_sources(
                        sources,
                        shard_index=shard_index,
                        num_shards=self.config.num_shards,
                    )
                )
            )
        return {
            "total_episodes": len(sources),
            "task_totals": dict(sorted(task_totals.items())),
            "shard_totals": shard_totals,
        }

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._snapshot))

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            snapshot = self._collect_snapshot()
            with self._lock:
                self._snapshot = snapshot
            self._stop.wait(self.config.poll_interval_s)

    def _collect_snapshot(self) -> dict[str, Any]:
        staging_data_uri = parse_s3_uri(self.config.staging_root).child("data").uri
        staging_metadata_uri = parse_s3_uri(self.config.staging_root).child("metadata").uri
        staged_count, staging_count_error = _count_direct_children(
            staging_data_uri,
            aws_profile=self.config.staging_aws_profile,
            aws_region=self.config.staging_region,
        )
        staged_count_stale = False
        if staged_count is None and self._last_good_staged_count is not None:
            staged_count = self._last_good_staged_count
            staged_count_stale = True
        elif staged_count is not None:
            self._last_good_staged_count = staged_count
        far_count = None
        far_count_error = None
        if self.config.far_root:
            far_data_uri = parse_s3_uri(self.config.far_root).child("data").uri
            far_count, far_count_error = _count_direct_children(
                far_data_uri,
                aws_profile=self.config.far_aws_profile,
                aws_region=self.config.far_region,
            )
        far_count_stale = False
        if far_count is None and self._last_good_far_count is not None:
            far_count = self._last_good_far_count
            far_count_stale = True
        elif far_count is not None:
            self._last_good_far_count = far_count
        metadata_entries = _list_metadata_entries(
            staging_metadata_uri,
            aws_profile=self.config.staging_aws_profile,
            aws_region=self.config.staging_region,
        )
        metadata_stale = False
        if "_error" in metadata_entries and self._last_good_metadata:
            metadata_entries = self._last_good_metadata | metadata_entries
            metadata_stale = True
        elif "_error" not in metadata_entries:
            self._last_good_metadata = dict(metadata_entries)
        remote_statuses, remote_status_error = _load_metadata_json_objects(
            staging_metadata_uri,
            _STATUS_RE,
            aws_profile=self.config.staging_aws_profile,
            aws_region=self.config.staging_region,
        )
        remote_statuses_stale = False
        if remote_status_error and self._last_good_remote_statuses:
            remote_statuses = self._last_good_remote_statuses
            remote_statuses_stale = True
        elif not remote_status_error:
            self._last_good_remote_statuses = dict(remote_statuses)
        collected_counts_payloads, collected_counts_error = _load_metadata_json_objects(
            staging_metadata_uri,
            _COLLECTED_SHARD_RE,
            aws_profile=self.config.staging_aws_profile,
            aws_region=self.config.staging_region,
        )
        collected_counts_stale = False
        if collected_counts_error and self._last_good_collected_counts:
            collected_counts_payloads = self._last_good_collected_counts
            collected_counts_stale = True
        elif not collected_counts_error:
            self._last_good_collected_counts = dict(collected_counts_payloads)
        collected_counts = {
            shard_index: len(payload)
            for shard_index, payload in collected_counts_payloads.items()
            if "_error" not in payload
        }
        failure_payloads, failure_payloads_error = _load_metadata_json_lists(
            staging_metadata_uri,
            _FAILURES_SHARD_RE,
            aws_profile=self.config.staging_aws_profile,
            aws_region=self.config.staging_region,
        )
        failure_payloads_stale = False
        if failure_payloads_error and self._last_good_failures:
            failure_payloads = self._last_good_failures
            failure_payloads_stale = True
        elif not failure_payloads_error:
            self._last_good_failures = dict(failure_payloads)
        failure_breakdown = _build_failure_breakdown(failure_payloads)
        remote_shard_rows = self._build_remote_shard_rows(
            remote_statuses,
            collected_counts=collected_counts,
        )
        reported_completed_count = sum(row["completed_episodes"] for row in remote_shard_rows)
        reported_failed_count = sum(row["failed_episodes"] for row in remote_shard_rows)
        status_file_count = len(remote_statuses)
        active_exports = _build_active_episode_rows()
        export_parents = _ps_rows(r"yam_sim\.dataset_export\.cli s3-export")
        finalize_rows = _ps_rows(r"yam_sim\.dataset_export\.cli s3-finalize")
        sync_rows = _ps_rows(r"aws --profile .* s3 sync s3://.*sim_tasks_20260414_madrona_224")

        timestamp = time.time()
        self._history.append(
            {
                "timestamp": timestamp,
                "staged_count": staged_count,
                "far_count": far_count,
            }
        )
        speed_eps_per_min = None
        eta_seconds = None
        if len(self._history) >= 2 and staged_count is not None:
            oldest = next((item for item in self._history if item["staged_count"] is not None), None)
            newest = next((item for item in reversed(self._history) if item["staged_count"] is not None), None)
            if oldest and newest and newest["timestamp"] > oldest["timestamp"]:
                delta_eps = newest["staged_count"] - oldest["staged_count"]
                delta_t = newest["timestamp"] - oldest["timestamp"]
                if delta_eps > 0 and delta_t > 0:
                    speed_eps_per_min = delta_eps / delta_t * 60.0
                    remaining = self._source_totals["total_episodes"] - staged_count
                    if remaining > 0:
                        eta_seconds = remaining / (delta_eps / delta_t)

        effective_count = staged_count
        if effective_count is None:
            effective_count = reported_completed_count
        else:
            effective_count = max(effective_count, reported_completed_count)

        remote_active = [
            row for row in remote_shard_rows if row["phase"] not in ("complete", "complete_with_failures", "failed", "missing")
        ]
        remote_finished_count = len(
            [row for row in remote_shard_rows if row["phase"] in ("complete", "complete_with_failures")]
        )

        phase = "idle"
        if active_exports or remote_active:
            phase = "exporting"
        elif finalize_rows:
            phase = "finalizing"
        elif sync_rows:
            phase = "syncing_to_far"
        elif metadata_entries.get("collected.json") and far_count == self._source_totals["total_episodes"]:
            phase = "complete"
        elif metadata_entries.get("collected.json"):
            phase = "waiting_for_far_sync"
        elif remote_finished_count == self.config.num_shards and self.config.num_shards > 0:
            phase = "shards_complete"

        progress_fraction = 0.0
        if effective_count is not None and self._source_totals["total_episodes"]:
            progress_fraction = effective_count / self._source_totals["total_episodes"]
        far_fraction = 0.0
        if far_count is not None and self._source_totals["total_episodes"]:
            far_fraction = far_count / self._source_totals["total_episodes"]

        return {
            "generated_at": _utc_now_iso(),
            "phase": phase,
            "overall": {
                "total_episodes": self._source_totals["total_episodes"],
                "staged_count": staged_count,
                "effective_count": effective_count,
                "reported_completed_count": reported_completed_count,
                "reported_failed_count": reported_failed_count,
                "staging_count_error": staging_count_error,
                "staging_count_stale": staged_count_stale,
                "far_count": far_count,
                "far_count_error": far_count_error,
                "far_count_stale": far_count_stale,
                "remaining_count": None if effective_count is None else self._source_totals["total_episodes"] - effective_count,
                "progress_fraction": progress_fraction,
                "far_fraction": far_fraction,
                "speed_eps_per_min": speed_eps_per_min,
                "eta_seconds": eta_seconds,
            },
            "tasks": self._source_totals["task_totals"],
            "shards": {
                "configured_total": self.config.num_shards,
                "assigned_totals": self._source_totals["shard_totals"],
                "active_exports": active_exports,
                "parent_processes": export_parents,
                "remote_statuses": remote_shard_rows,
                "remote_status_error": remote_status_error,
                "remote_statuses_stale": remote_statuses_stale,
                "status_file_count": status_file_count,
                "collected_counts_error": collected_counts_error,
                "collected_counts_stale": collected_counts_stale,
                "failure_records_error": failure_payloads_error,
                "failure_records_stale": failure_payloads_stale,
            },
            "failures": failure_breakdown,
            "metadata": metadata_entries,
            "metadata_stale": metadata_stale,
            "watchers": {
                "finalize": finalize_rows,
                "sync": sync_rows,
            },
            "history": list(self._history),
            "config": {
                "s3_input_prefix": self.config.s3_input_prefix,
                "staging_root": self.config.staging_root,
                "far_root": self.config.far_root,
                "source_aws_profile": self.config.source_aws_profile,
                "staging_aws_profile": self.config.staging_aws_profile,
                "far_aws_profile": self.config.far_aws_profile,
                "num_shards": self.config.num_shards,
                "poll_interval_s": self.config.poll_interval_s,
            },
        }

    def _build_remote_shard_rows(
        self,
        statuses: dict[int, dict[str, Any]],
        *,
        collected_counts: dict[int, int],
    ) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        rows: list[dict[str, Any]] = []
        for shard_index, assigned_total in enumerate(self._source_totals["shard_totals"]):
            status = statuses.get(shard_index, {})
            completed = int(status.get("completed_episodes") or 0)
            completed = max(completed, collected_counts.get(shard_index, 0))
            failed = int(status.get("failed_episodes") or 0)
            updated_at = status.get("updated_at")
            updated_dt = _parse_iso_datetime(updated_at)
            if updated_dt is not None and updated_dt.tzinfo is None:
                updated_dt = updated_dt.replace(tzinfo=timezone.utc)
            stale_seconds = None
            if updated_dt is not None:
                stale_seconds = max(0.0, (now - updated_dt).total_seconds())
            phase = status.get("phase") or ("complete" if completed >= assigned_total and assigned_total else "missing")
            progress_fraction = completed / assigned_total if assigned_total else 1.0
            rows.append(
                {
                    "shard_index": shard_index,
                    "shard_tag": status.get("shard_tag") or f"{shard_index:02d}",
                    "phase": phase,
                    "assigned_episodes": assigned_total,
                    "attempted_episodes": int(status.get("attempted_episodes") or 0),
                    "completed_episodes": completed,
                    "failed_episodes": failed,
                    "skipped_existing": int(status.get("skipped_existing") or 0),
                    "current_episode": status.get("current_episode"),
                    "last_completed_episode": status.get("last_completed_episode"),
                    "last_error": status.get("last_error") or status.get("_error"),
                    "hostname": status.get("hostname") or status.get("pod_name"),
                    "gpu_id": status.get("gpu_id"),
                    "updated_at": updated_at,
                    "stale_seconds": stale_seconds,
                    "stale": stale_seconds is not None and stale_seconds > max(60.0, self.config.poll_interval_s * 6.0),
                    "progress_fraction": progress_fraction,
                }
            )
        return rows


_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Export Monitor</title>
  <style>
    :root {
      --bg: #f5f1e8;
      --panel: rgba(255,255,255,0.78);
      --panel-strong: rgba(255,255,255,0.92);
      --border: rgba(110, 88, 61, 0.18);
      --text: #1c1914;
      --muted: #6d6357;
      --accent: #de6a28;
      --accent-2: #0f8b8d;
      --accent-3: #f2b134;
      --good: #2e8b57;
      --warn: #c97a10;
      --bad: #b13c32;
      --shadow: 0 18px 46px rgba(87, 61, 29, 0.12);
      --radius: 24px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      --sans: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(222,106,40,0.22), transparent 24%),
        radial-gradient(circle at top right, rgba(15,139,141,0.18), transparent 28%),
        linear-gradient(180deg, #f7f3eb 0%, #efe7d8 100%);
    }
    .page {
      max-width: 1380px;
      margin: 0 auto;
      padding: 28px 24px 40px;
    }
    .hero {
      display: grid;
      grid-template-columns: 1.5fr 1fr;
      gap: 20px;
      margin-bottom: 20px;
    }
    .hero-card, .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(20px);
    }
    .hero-card {
      padding: 28px;
      position: relative;
      overflow: hidden;
    }
    .hero-card:before {
      content: "";
      position: absolute;
      inset: auto -60px -60px auto;
      width: 220px;
      height: 220px;
      background: radial-gradient(circle, rgba(242,177,52,0.28), transparent 68%);
      pointer-events: none;
    }
    h1 {
      margin: 0;
      font-size: clamp(2rem, 3vw, 3.25rem);
      line-height: 0.95;
      letter-spacing: -0.05em;
    }
    .sub {
      color: var(--muted);
      margin-top: 12px;
      font-size: 1rem;
      max-width: 44rem;
    }
    .phase {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(222,106,40,0.16);
      margin-top: 18px;
      font-weight: 700;
      letter-spacing: 0.01em;
    }
    .phase-dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--accent);
      box-shadow: 0 0 0 8px rgba(222,106,40,0.12);
    }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    .kpi {
      padding: 18px;
      border-radius: 18px;
      background: var(--panel-strong);
      border: 1px solid var(--border);
      min-height: 120px;
    }
    .kpi-label {
      color: var(--muted);
      font-size: 0.86rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
    }
    .kpi-value {
      margin-top: 10px;
      font-size: clamp(1.5rem, 2.4vw, 2.5rem);
      font-weight: 800;
      letter-spacing: -0.04em;
      line-height: 1;
    }
    .kpi-sub {
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.9rem;
    }
    .tabs {
      display: inline-flex;
      gap: 8px;
      padding: 8px;
      border-radius: 999px;
      background: rgba(255,255,255,0.65);
      border: 1px solid var(--border);
      margin-bottom: 18px;
    }
    .tab-btn {
      border: 0;
      background: transparent;
      color: var(--muted);
      padding: 10px 16px;
      border-radius: 999px;
      cursor: pointer;
      font: inherit;
      font-weight: 700;
    }
    .tab-btn.active {
      background: linear-gradient(135deg, rgba(222,106,40,0.92), rgba(209,88,44,0.92));
      color: white;
      box-shadow: 0 12px 28px rgba(222,106,40,0.26);
    }
    .tab { display: none; }
    .tab.active { display: block; }
    .section-grid {
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 18px;
    }
    .card {
      padding: 22px;
    }
    .card h2 {
      margin: 0 0 14px 0;
      font-size: 1.1rem;
      letter-spacing: -0.02em;
    }
    .progress-stack { display: grid; gap: 18px; }
    .bar-head {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      margin-bottom: 10px;
    }
    .bar-title {
      font-weight: 700;
      font-size: 0.98rem;
    }
    .bar-meta {
      color: var(--muted);
      font-size: 0.92rem;
    }
    .bar-shell {
      height: 18px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(64, 55, 40, 0.08);
      border: 1px solid rgba(64, 55, 40, 0.06);
    }
    .bar-fill {
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--accent), #ff8f43);
      transition: width 0.35s ease;
      min-width: 6px;
    }
    .bar-fill.teal {
      background: linear-gradient(90deg, var(--accent-2), #34b6b7);
    }
    .bar-caption {
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.9rem;
    }
    .active-grid {
      display: grid;
      gap: 12px;
    }
    .shard-card {
      border-radius: 18px;
      padding: 16px;
      background: rgba(255,255,255,0.85);
      border: 1px solid var(--border);
    }
    .shard-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 8px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(15,139,141,0.12);
      color: #0a5f60;
      font-size: 0.84rem;
      font-weight: 700;
    }
    .episode {
      font-family: var(--mono);
      font-size: 0.84rem;
      line-height: 1.45;
      color: #40372d;
      word-break: break-all;
      margin-top: 4px;
    }
    .muted { color: var(--muted); }
    .metric-row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .mini {
      border-radius: 14px;
      background: rgba(245,241,232,0.92);
      padding: 10px 12px;
      border: 1px solid rgba(110, 88, 61, 0.10);
    }
    .mini .label {
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
    }
    .mini .value {
      margin-top: 5px;
      font-weight: 700;
      font-family: var(--mono);
      font-size: 0.88rem;
    }
    .table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.93rem;
    }
    .table th, .table td {
      padding: 10px 8px;
      border-bottom: 1px solid rgba(110, 88, 61, 0.10);
      text-align: left;
      vertical-align: top;
    }
    .table th {
      color: var(--muted);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
    }
    .mono { font-family: var(--mono); }
    pre {
      margin: 0;
      padding: 14px;
      border-radius: 18px;
      background: #201b16;
      color: #f7f3eb;
      overflow: auto;
      font-family: var(--mono);
      font-size: 0.82rem;
      line-height: 1.45;
    }
    .note {
      margin-top: 12px;
      color: var(--muted);
      font-size: 0.88rem;
    }
    @media (max-width: 1100px) {
      .hero, .section-grid { grid-template-columns: 1fr; }
      .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 720px) {
      .page { padding: 20px 16px 30px; }
      .kpi-grid { grid-template-columns: 1fr; }
      .metric-row { grid-template-columns: 1fr; }
      .tabs { width: 100%; justify-content: space-between; }
      .tab-btn { flex: 1; }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="hero-card">
        <h1>Dataset Export Monitor</h1>
        <div class="sub">
          Live view of multi-node shard progress, active L40S workers, shard metadata, and the current throughput for the running sim dataset job.
        </div>
        <div class="phase"><span class="phase-dot"></span><span id="phase">Initializing…</span></div>
      </div>
      <div class="kpi-grid">
        <div class="kpi">
          <div class="kpi-label">Exported Episodes</div>
          <div class="kpi-value" id="stagedCount">-</div>
          <div class="kpi-sub" id="stagedSub">waiting for data</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Remaining</div>
          <div class="kpi-value" id="remainingCount">-</div>
          <div class="kpi-sub" id="remainingSub">until staging completes</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Speed</div>
          <div class="kpi-value" id="speed">-</div>
          <div class="kpi-sub">episodes / minute</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">ETA</div>
          <div class="kpi-value" id="eta">-</div>
          <div class="kpi-sub" id="etaSub">estimated from recent progress</div>
        </div>
      </div>
    </section>

    <div class="tabs">
      <button class="tab-btn active" data-tab="overview">Overview</button>
      <button class="tab-btn" data-tab="shards">Shards</button>
      <button class="tab-btn" data-tab="debug">Debug</button>
    </div>

    <section class="tab active" id="tab-overview">
      <div class="section-grid">
        <div class="card">
          <h2>Progress</h2>
          <div class="progress-stack">
            <div>
              <div class="bar-head">
                <div class="bar-title">Exported Episodes</div>
                <div class="bar-meta" id="stagingMeta">-</div>
              </div>
              <div class="bar-shell"><div class="bar-fill" id="stagingBar"></div></div>
              <div class="bar-caption" id="stagingCaption">Waiting for status…</div>
            </div>
            <div>
              <div class="bar-head">
                <div class="bar-title">Stage 2: Sync To FAR</div>
                <div class="bar-meta" id="farMeta">-</div>
              </div>
              <div class="bar-shell"><div class="bar-fill teal" id="farBar"></div></div>
              <div class="bar-caption" id="farCaption">FAR sync has not started yet.</div>
            </div>
          </div>
          <div class="note" id="updatedAt">Last update: -</div>
        </div>
        <div class="card">
          <h2>Active Work</h2>
          <div class="active-grid" id="activeExports"></div>
        </div>
      </div>
    </section>

    <section class="tab" id="tab-shards">
      <div class="card">
        <h2>Shard Progress</h2>
        <table class="table" id="shardTable">
          <thead>
            <tr>
              <th>Shard</th>
              <th>Phase</th>
              <th>Progress</th>
              <th>Failed</th>
              <th>Current Episode</th>
              <th>Node</th>
              <th>Updated</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
        <div class="note" id="shardStatusNote">Waiting for shard status files…</div>
      </div>
    </section>

    <section class="tab" id="tab-debug">
      <div class="section-grid">
        <div class="card">
          <h2>Task Totals</h2>
          <table class="table" id="taskTable">
            <thead><tr><th>Task</th><th>Total Episodes</th></tr></thead>
            <tbody></tbody>
          </table>
        </div>
        <div class="card">
          <h2>Failures By Task</h2>
          <table class="table" id="failureTaskTable">
            <thead>
              <tr>
                <th>Task</th>
                <th>Total</th>
                <th>State Shape</th>
                <th>No Window</th>
                <th>Missing State</th>
                <th>Other</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
          <div class="note" id="failureBreakdownNote">Waiting for failure records…</div>
        </div>
      </div>
      <div class="section-grid" style="margin-top:18px;">
        <div class="card">
          <h2>Metadata Checkpoints</h2>
          <table class="table" id="metadataTable">
            <thead><tr><th>Name</th><th>Timestamp</th><th>Size</th></tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
      <div class="section-grid" style="margin-top:18px;">
        <div class="card">
          <h2>Process State</h2>
          <pre id="processDump">Loading…</pre>
        </div>
        <div class="card">
          <h2>Run Config</h2>
          <pre id="configDump">Loading…</pre>
        </div>
      </div>
    </section>
  </div>

  <script>
    const tabs = document.querySelectorAll('.tab-btn');
    for (const btn of tabs) {
      btn.addEventListener('click', () => {
        for (const other of tabs) other.classList.remove('active');
        btn.classList.add('active');
        document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
        document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
      });
    }

    function fmtInt(value) {
      return value == null ? '-' : new Intl.NumberFormat().format(value);
    }

    function fmtPct(value) {
      if (value == null) return '-';
      return `${(value * 100).toFixed(1)}%`;
    }

    function fmtSize(bytes) {
      if (bytes == null) return '-';
      if (bytes < 1024) return `${bytes} B`;
      if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
      if (bytes < 1024 ** 3) return `${(bytes / (1024 ** 2)).toFixed(1)} MiB`;
      return `${(bytes / (1024 ** 3)).toFixed(2)} GiB`;
    }

    function fmtEta(seconds) {
      if (seconds == null || !Number.isFinite(seconds)) return '-';
      seconds = Math.max(0, Math.round(seconds));
      const hours = Math.floor(seconds / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      const secs = seconds % 60;
      if (hours) return `${hours}h ${String(minutes).padStart(2, '0')}m`;
      if (minutes) return `${minutes}m ${String(secs).padStart(2, '0')}s`;
      return `${secs}s`;
    }

    function fmtRate(rate) {
      if (rate == null || !Number.isFinite(rate)) return '-';
      return rate.toFixed(rate >= 10 ? 1 : 2);
    }

    function fmtAge(seconds) {
      if (seconds == null || !Number.isFinite(seconds)) return '-';
      seconds = Math.max(0, Math.round(seconds));
      if (seconds < 60) return `${seconds}s ago`;
      if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
      return `${Math.floor(seconds / 3600)}h ${String(Math.floor((seconds % 3600) / 60)).padStart(2, '0')}m ago`;
    }

    function renderActiveExports(active, watchers, remoteShards) {
      const container = document.getElementById('activeExports');
      container.innerHTML = '';
      const remoteActive = (remoteShards || []).filter(row => !['complete', 'complete_with_failures', 'failed', 'missing'].includes(row.phase));
      if (!remoteActive.length && !active.length && !watchers.finalize.length && !watchers.sync.length) {
        container.innerHTML = '<div class="muted">No active shard workers detected.</div>';
        return;
      }
      for (const row of remoteActive.slice(0, 12)) {
        const el = document.createElement('div');
        el.className = 'shard-card';
        el.innerHTML = `
          <div class="shard-head">
            <strong>Shard ${row.shard_index}</strong>
            <span class="chip">${row.phase.replaceAll('_', ' ')}</span>
          </div>
          <div class="episode">${row.current_episode ?? row.last_completed_episode ?? 'waiting for episode'}</div>
          <div class="muted" style="margin-top:6px;">${row.hostname ?? 'unknown node'} · ${fmtAge(row.stale_seconds)}</div>
          <div class="metric-row">
            <div class="mini"><div class="label">Done</div><div class="value">${fmtInt(row.completed_episodes)} / ${fmtInt(row.assigned_episodes)}</div></div>
            <div class="mini"><div class="label">Failed</div><div class="value">${fmtInt(row.failed_episodes)}</div></div>
          </div>
        `;
        container.appendChild(el);
      }
      for (const row of active) {
        const el = document.createElement('div');
        el.className = 'shard-card';
        el.innerHTML = `
          <div class="shard-head">
            <strong>Shard ${row.shard}</strong>
            <span class="chip">GPU ${row.gpu_id ?? 'n/a'}</span>
          </div>
          <div class="episode">${row.episode}</div>
          <div class="muted" style="margin-top:6px;">${row.delivery ?? 'unknown task'}</div>
          <div class="metric-row">
            <div class="mini"><div class="label">Elapsed</div><div class="value">${row.elapsed}</div></div>
            <div class="mini"><div class="label">PID</div><div class="value">${row.pid}</div></div>
          </div>
        `;
        container.appendChild(el);
      }
      for (const row of watchers.finalize) {
        const el = document.createElement('div');
        el.className = 'shard-card';
        el.innerHTML = `
          <div class="shard-head">
            <strong>Finalize</strong>
            <span class="chip">Metadata</span>
          </div>
          <div class="episode">pid ${row.pid}</div>
          <div class="muted" style="margin-top:6px;">${row.elapsed}</div>
        `;
        container.appendChild(el);
      }
      for (const row of watchers.sync) {
        const el = document.createElement('div');
        el.className = 'shard-card';
        el.innerHTML = `
          <div class="shard-head">
            <strong>FAR Sync</strong>
            <span class="chip">Upload</span>
          </div>
          <div class="episode mono">pid ${row.pid}</div>
          <div class="muted" style="margin-top:6px;">${row.elapsed}</div>
        `;
        container.appendChild(el);
      }
    }

    function renderShardTable(status) {
      const tbody = document.querySelector('#shardTable tbody');
      tbody.innerHTML = '';
      const rows = status.shards.remote_statuses || [];
      for (const row of rows) {
        const tr = document.createElement('tr');
        const stale = row.stale ? ' · stale' : '';
        const current = row.current_episode || row.last_completed_episode || '';
        const error = row.last_error ? `<div style="color:#b13c32; margin-top:4px;">${row.last_error}</div>` : '';
        tr.innerHTML = `
          <td class="mono">${row.shard_index}</td>
          <td>${row.phase.replaceAll('_', ' ')}${stale}</td>
          <td>${fmtInt(row.completed_episodes)} / ${fmtInt(row.assigned_episodes)} <span class="muted">(${fmtPct(row.progress_fraction)})</span></td>
          <td>${fmtInt(row.failed_episodes)}</td>
          <td class="mono">${current}${error}</td>
          <td class="mono">${row.hostname ?? '-'}</td>
          <td>${fmtAge(row.stale_seconds)}</td>
        `;
        tbody.appendChild(tr);
      }
      const note = document.getElementById('shardStatusNote');
      const statusCount = status.shards.status_file_count || 0;
      const stale = status.shards.remote_statuses_stale ? ' · using cached shard status' : '';
      const error = status.shards.remote_status_error ? ` · ${status.shards.remote_status_error}` : '';
      note.textContent = `${fmtInt(statusCount)} status files found for ${fmtInt(status.shards.configured_total)} configured shards${stale}${error}`;
    }

    function renderTables(status) {
      const taskTbody = document.querySelector('#taskTable tbody');
      taskTbody.innerHTML = '';
      for (const [task, total] of Object.entries(status.tasks)) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td class="mono">${task}</td><td>${fmtInt(total)}</td>`;
        taskTbody.appendChild(tr);
      }

      const failureTbody = document.querySelector('#failureTaskTable tbody');
      failureTbody.innerHTML = '';
      const failures = status.failures || {};
      const failureRows = failures.by_task || [];
      for (const row of failureRows) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td class="mono">${row.task}</td>
          <td>${fmtInt(row.total)}</td>
          <td>${fmtInt(row.integration_state_shape)}</td>
          <td>${fmtInt(row.no_overlap_window)}</td>
          <td>${fmtInt(row.missing_integration_state)}</td>
          <td>${fmtInt(row.other)}</td>
        `;
        failureTbody.appendChild(tr);
      }
      if (!failureRows.length) {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td colspan="6" class="muted">No failure records uploaded yet.</td>';
        failureTbody.appendChild(tr);
      }
      const failureNote = document.getElementById('failureBreakdownNote');
      const failureError = status.shards.failure_records_error ? ` · ${status.shards.failure_records_error}` : '';
      const failureStale = status.shards.failure_records_stale ? ' · using cached failure records' : '';
      failureNote.textContent = `${fmtInt(failures.record_count || 0)} failure records from ${fmtInt(failures.shard_count || 0)} shard files${failureStale}${failureError}`;

      const metaTbody = document.querySelector('#metadataTable tbody');
      metaTbody.innerHTML = '';
      const metadata = status.metadata || {};
      for (const [name, info] of Object.entries(metadata)) {
        if (name === '_error') continue;
        const tr = document.createElement('tr');
        tr.innerHTML = `<td class="mono">${name}</td><td>${info.timestamp}</td><td>${fmtSize(info.size_bytes)}</td>`;
        metaTbody.appendChild(tr);
      }
      if (metadata._error) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td colspan="3" style="color:#b13c32;">Using cached metadata view: ${metadata._error.message}</td>`;
        metaTbody.appendChild(tr);
      }

      document.getElementById('processDump').textContent = JSON.stringify({
        active_exports: status.shards.active_exports,
        shard_parents: status.shards.parent_processes,
        remote_statuses: status.shards.remote_statuses,
        failures: status.failures,
        watchers: status.watchers,
      }, null, 2);
      document.getElementById('configDump').textContent = JSON.stringify(status.config, null, 2);
      renderShardTable(status);
    }

    function render(status) {
      const overall = status.overall;
      document.getElementById('phase').textContent = status.phase.replaceAll('_', ' ');
      const displayedCount = overall.effective_count ?? overall.staged_count;
      document.getElementById('stagedCount').textContent = fmtInt(displayedCount);
      document.getElementById('stagedSub').textContent = `${fmtPct(overall.progress_fraction)} of ${fmtInt(overall.total_episodes)} total${overall.staging_count_stale ? ' · cached' : ''}`;
      document.getElementById('remainingCount').textContent = fmtInt(overall.remaining_count);
      document.getElementById('speed').textContent = fmtRate(overall.speed_eps_per_min);
      document.getElementById('eta').textContent = fmtEta(overall.eta_seconds);
      document.getElementById('etaSub').textContent = overall.staging_count_error ? `Staging auth issue: ${overall.staging_count_error}` : 'estimated from recent staged growth';

      const stagingPct = Math.max(0, Math.min(1, overall.progress_fraction || 0));
      document.getElementById('stagingBar').style.width = `${stagingPct * 100}%`;
      document.getElementById('stagingMeta').textContent = `${fmtInt(displayedCount)} / ${fmtInt(overall.total_episodes)}`;
      document.getElementById('stagingCaption').textContent =
        overall.staging_count_error
          ? `Using cached export count while S3 status is unavailable: ${overall.staging_count_error}`
          : `${fmtInt(overall.remaining_count)} episodes still need to export. ${fmtInt(overall.reported_failed_count)} reported failures.`;

      const farPct = Math.max(0, Math.min(1, overall.far_fraction || 0));
      document.getElementById('farBar').style.width = `${farPct * 100}%`;
      document.getElementById('farMeta').textContent =
        overall.far_count == null ? 'not started' : `${fmtInt(overall.far_count)} / ${fmtInt(overall.total_episodes)}${overall.far_count_stale ? ' · cached' : ''}`;
      document.getElementById('farCaption').textContent =
        overall.far_count_error ? `FAR auth/status unavailable: ${overall.far_count_error}` :
        overall.far_count == null ? 'FAR sync has not started yet.' :
        `${fmtInt(Math.max(0, overall.total_episodes - overall.far_count))} episodes still need to copy to FAR.`;

      document.getElementById('updatedAt').textContent = `Last update: ${status.generated_at}`;
      renderActiveExports(status.shards.active_exports, status.watchers, status.shards.remote_statuses);
      renderTables(status);
    }

    async function refresh() {
      try {
        const response = await fetch('/api/status');
        const status = await response.json();
        render(status);
      } catch (err) {
        console.error(err);
      }
    }

    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    server: "_MonitorServer"

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            body = _HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/status":
            payload = json.dumps(self.server.monitor.snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(404)
        self.end_headers()


class _MonitorServer(ThreadingHTTPServer):
    def __init__(self, addr: tuple[str, int], monitor: ExportRunMonitor):
        super().__init__(addr, _Handler)
        self.monitor = monitor


def serve_monitor(
    *,
    host: str,
    port: int,
    config: MonitorConfig,
) -> None:
    monitor = ExportRunMonitor(config)
    monitor.start()
    server = _MonitorServer((host, port), monitor)
    try:
        print(f"Export monitor listening on http://{host}:{port}")
        server.serve_forever()
    finally:
        monitor.stop()
        server.server_close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run a localhost dashboard for dataset export progress")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--s3-input-prefix", required=True)
    parser.add_argument("--s3-output-root", default=None)
    parser.add_argument("--staging-root", default=None)
    parser.add_argument("--far-root", default=None)
    parser.add_argument("--source-aws-profile", default=None)
    parser.add_argument("--staging-aws-profile", default=None)
    parser.add_argument("--far-aws-profile", default=None)
    parser.add_argument("--source-region", default=None)
    parser.add_argument("--staging-region", default=None)
    parser.add_argument("--far-region", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    args = parser.parse_args()
    monitor_root = args.s3_output_root or args.staging_root
    if not monitor_root:
        parser.error("requires --s3-output-root or --staging-root")

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


if __name__ == "__main__":
    main()
