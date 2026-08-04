"""Result records, aggregation, and serialization for the eval harness.

Each episode produces an :class:`EpisodeResult`, appended to ``results.jsonl`` as
it completes (crash-safe and resumable). At the end of a run the records are
aggregated into per-(model, task) success rates written to ``summary.json`` and
``summary.csv``.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EpisodeResult:
    """Outcome of a single evaluated episode."""

    model_label: str
    task: str
    seed: int | None
    success: bool | None
    reward: float | None
    num_chunks: int
    metrics: dict[str, Any] = field(default_factory=dict)
    video_path: str | None = None
    actions_path: str | None = None
    states_path: str | None = None
    # How this episode was evaluated. "batched_masked_count" means the batched env
    # used the MAX-count + parking mask, so object COUNT and pose were randomized
    # per world but mesh VARIANT and SCALE were fixed-canonical (a shared-model
    # approximation). None / "batched" = standard batched eval.
    # "batched_visual_groupN" = grouped visual-variety eval (N episodes share one
    # deterministic mesh-variant/scale/color config; poses vary).
    eval_mode: str | None = None
    # Grouped visual-variety bookkeeping (seed-deterministic, policy-independent):
    # the realized visual config (variants/scale/colors) for this episode's group,
    # the seed that produced it, and the group index -- so two eval runs can be
    # proven identical and downstream readers know the scene that was shown.
    scene_config: dict[str, Any] | None = None
    visual_seed: int | None = None
    visual_group_index: int | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def append_jsonl(path: str | Path, result: EpisodeResult) -> None:
    """Append a single episode result as one JSON line (created if needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(result.to_json()) + "\n")


def load_jsonl(path: str | Path) -> list[EpisodeResult]:
    """Load previously written episode results (for resume/aggregation)."""
    path = Path(path)
    if not path.exists():
        return []
    results: list[EpisodeResult] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            results.append(
                EpisodeResult(
                    model_label=data["model_label"],
                    task=data["task"],
                    seed=data.get("seed"),
                    success=data.get("success"),
                    reward=data.get("reward"),
                    num_chunks=data.get("num_chunks", 0),
                    metrics=data.get("metrics", {}),
                    video_path=data.get("video_path"),
                    actions_path=data.get("actions_path"),
                    states_path=data.get("states_path"),
                    eval_mode=data.get("eval_mode"),
                    scene_config=data.get("scene_config"),
                    visual_seed=data.get("visual_seed"),
                    visual_group_index=data.get("visual_group_index"),
                )
            )
    return results


def completed_keys(results: list[EpisodeResult]) -> set[tuple[str, str, int | None]]:
    """Return the ``(model_label, task, seed)`` triples already evaluated."""
    return {(r.model_label, r.task, r.seed) for r in results}


def _wald_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Normal-approximation 95% CI for a binomial success rate."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    half = z * math.sqrt(max(p * (1.0 - p), 0.0) / n)
    return (max(0.0, p - half), min(1.0, p + half))


def aggregate(results: list[EpisodeResult]) -> dict[str, Any]:
    """Aggregate episode results into per-(model, task), per-model, and overall stats."""
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for r in results:
        key = (r.model_label, r.task)
        cell = cells.setdefault(
            key,
            {"n": 0, "scored": 0, "successes": 0, "reward_sum": 0.0, "reward_n": 0},
        )
        cell["n"] += 1
        if r.success is not None:
            cell["scored"] += 1
            cell["successes"] += int(bool(r.success))
        if r.reward is not None:
            cell["reward_sum"] += float(r.reward)
            cell["reward_n"] += 1

    per_cell = []
    for (model_label, task), cell in sorted(cells.items()):
        scored = cell["scored"]
        successes = cell["successes"]
        success_rate = (successes / scored) if scored else None
        ci = _wald_ci(successes, scored) if scored else None
        mean_reward = (cell["reward_sum"] / cell["reward_n"]) if cell["reward_n"] else None
        per_cell.append(
            {
                "model_label": model_label,
                "task": task,
                "episodes": cell["n"],
                "scored": scored,
                "successes": successes,
                "success_rate": success_rate,
                "success_rate_ci95": ci,
                "mean_reward": mean_reward,
            }
        )

    per_model: dict[str, dict[str, Any]] = {}
    for row in per_cell:
        m = per_model.setdefault(
            row["model_label"], {"episodes": 0, "scored": 0, "successes": 0}
        )
        m["episodes"] += row["episodes"]
        m["scored"] += row["scored"]
        m["successes"] += row["successes"]
    for label, m in per_model.items():
        m["success_rate"] = (m["successes"] / m["scored"]) if m["scored"] else None

    total_scored = sum(row["scored"] for row in per_cell)
    total_successes = sum(row["successes"] for row in per_cell)
    overall = {
        "episodes": sum(row["episodes"] for row in per_cell),
        "scored": total_scored,
        "successes": total_successes,
        "success_rate": (total_successes / total_scored) if total_scored else None,
    }

    return {"per_cell": per_cell, "per_model": per_model, "overall": overall}


def write_summary(output_dir: str | Path, summary: dict[str, Any]) -> tuple[Path, Path]:
    """Write ``summary.json`` and ``summary.csv``; return their paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"

    json_path.write_text(json.dumps(summary, indent=2))

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model_label",
                "task",
                "episodes",
                "scored",
                "successes",
                "success_rate",
                "ci95_low",
                "ci95_high",
                "mean_reward",
            ]
        )
        for row in summary["per_cell"]:
            ci = row["success_rate_ci95"]
            writer.writerow(
                [
                    row["model_label"],
                    row["task"],
                    row["episodes"],
                    row["scored"],
                    row["successes"],
                    "" if row["success_rate"] is None else f"{row['success_rate']:.4f}",
                    "" if ci is None else f"{ci[0]:.4f}",
                    "" if ci is None else f"{ci[1]:.4f}",
                    "" if row["mean_reward"] is None else f"{row['mean_reward']:.4f}",
                ]
            )
    return json_path, csv_path


def format_table(summary: dict[str, Any]) -> str:
    """Render a compact per-(model, task) success-rate table for the console."""
    rows = summary["per_cell"]
    if not rows:
        return "(no results)"
    header = f"{'model':<16} {'task':<28} {'eps':>4} {'succ%':>7} {'reward':>8}"
    lines = [header, "-" * len(header)]
    for row in rows:
        sr = row["success_rate"]
        sr_str = "  n/a" if sr is None else f"{sr * 100:6.1f}"
        mr = row["mean_reward"]
        mr_str = "   n/a" if mr is None else f"{mr:8.3f}"
        lines.append(
            f"{row['model_label']:<16} {row['task']:<28} "
            f"{row['episodes']:>4} {sr_str:>7} {mr_str:>8}"
        )
    return "\n".join(lines)
