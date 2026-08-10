"""Configuration for the yam-sim policy eval harness.

An :class:`EvalConfig` describes a job matrix of (model endpoint) x (task) x N
episodes. It can be authored inline (via the CLI) or loaded from a YAML/JSON file.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from yam_sim.task_specs import SimTaskSpec, get_task_spec, maybe_get_task_spec

# Offset added to a visual GROUP index to form its construction (visual) seed in
# grouped visual-variety eval. Large enough that the visual RNG stream never
# overlaps any positional world seed (= seed_base + global episode index), so the
# two are independent and the whole scene is a pure function of the seeds.
VISUAL_SALT = 1_000_000


@dataclass(frozen=True)
class ModelEndpoint:
    """A running openpi websocket server hosting one pi0 checkpoint."""

    label: str
    host: str
    port: int = 8000
    api_key: str | None = None

    @classmethod
    def parse(cls, text: str) -> "ModelEndpoint":
        """Parse a ``label=host:port`` (or ``label=host``) spec string."""
        if "=" not in text:
            raise ValueError(
                f"Model spec must look like 'label=host:port', got {text!r}"
            )
        label, endpoint = text.split("=", 1)
        host, _, port = endpoint.partition(":")
        label = label.strip()
        host = host.strip()
        if not label or not host:
            raise ValueError(f"Model spec missing label or host: {text!r}")
        return cls(label=label, host=host, port=int(port) if port else 8000)


@dataclass(frozen=True)
class TaskSpecEntry:
    """One task to evaluate, with optional per-task overrides.

    ``task`` may be a registered task name/alias or a bare scene/env name (e.g.
    ``put_bottles``) that has no registered spec. In the latter case a spec is
    synthesized with no evaluator (the task is rolled out and recorded but not
    scored). ``prompt`` overrides the prompt sent to the policy — set it to match
    the prompt the checkpoint was trained with.
    """

    task: str
    episodes: int | None = None
    num_worlds: int | None = None
    max_chunks: int | None = None
    max_seconds: float | None = None
    prompt: str | None = None

    def resolve(self) -> SimTaskSpec:
        """Return the task spec, synthesizing one for unregistered env names."""
        spec = maybe_get_task_spec(self.task)
        if spec is None:
            spec = SimTaskSpec(
                name=self.task,
                env_task=self.task,
                prompt=self.prompt or self.task.replace("_", " "),
                evaluator_name=None,
            )
        elif self.prompt is not None:
            spec = dataclasses.replace(spec, prompt=self.prompt)
        return spec

    # Backwards-compatible alias.
    def resolved_spec(self) -> SimTaskSpec:
        return self.resolve()


@dataclass(frozen=True)
class EvalConfig:
    """Full eval job matrix plus global rollout/output settings.

    ``models`` may be left empty in a config file and supplied on the CLI instead,
    so a task-set config stays checkpoint-independent and reusable.
    """

    tasks: list[TaskSpecEntry] = field(default_factory=list)
    models: list[ModelEndpoint] = field(default_factory=list)
    output_dir: str = "/tmp/yam_eval"
    episodes_default: int = 10
    num_worlds_default: int = 4
    max_seconds: float | None = None  # global sim-time horizon; converted to chunks
    scene: str = "hybrid"
    # Observation renderer (physics always runs on mjwarp). "mujoco" = MuJoCo-GL,
    # matching the training-data renderer (in-distribution eval, slower); "mjwarp"
    # = fast batched GPU render (off-distribution for MuJoCo-GL-trained policies).
    camera_backend: str = "mjwarp"
    execute_chunk_dim: int = 20
    prefix_length: int = 3
    use_ttrtc: bool = True
    jpeg_quality: int = 0
    fps: int = 30
    video: bool = True
    # Save the full per-step MuJoCo qpos trajectory per episode (seed_<n>_qpos.npy,
    # shape (T, nq)) so completed rollouts can be deterministically re-rendered
    # offline (rerender_all.py) without re-simulating. Independent of `video`.
    save_states: bool = True
    camera_gpu_id: int | None = None
    seed_base: int = 0
    # Enable per-world object-COUNT randomization for variable-count tasks
    # (put_bottles/load_plates/hang_mug/turn_mug) by building the shared warp model
    # at MAX count and parking unused object slots. Variant+scale are fixed-per-
    # batch in this mode (a shared-model limit) -- recorded as ``eval_mode`` on
    # every result so downstream readers know these came from the approximation.
    mask_variable_count: bool = False
    # Pin the object count to this fixed value for every world (poses still
    # randomize per seed), e.g. put_bottles=4 in eval. The canonical eval count for
    # count-varying tasks whose training randomizes the count.
    force_object_count: int | None = None
    # Grouped visual-variety eval: every group of this many consecutive episodes
    # shares ONE deterministic visual config (mesh variant + scale + color), chosen
    # from visual_seed = VISUAL_SALT + global_group_index; poses still vary per
    # episode. The env is rebuilt once per group. Requires num_worlds == this value
    # and (for count-varying tasks) force_object_count set. None = off (one fixed
    # look for the whole run).
    visual_group_size: int | None = None
    # hang_mug only: use the scene's inline colored mugs (geoms directly on mug_1/2/3)
    # instead of reloading variant assets, so the geometry-based reward -- calibrated on
    # these mugs -- attributes tree contacts correctly. Pose-only, fixed count, one fixed
    # colored look; mutually exclusive with force_object_count / visual_group_size.
    mug_inline: bool = False
    # Stop stepping a batch once every recorded world's task success has held
    # for a short window (0.5 s), and truncate each world's saved qpos just
    # past its success moment. Big wall-clock win for tasks that finish early;
    # off by default to preserve the canonical fixed-horizon behavior.
    early_stop_on_success: bool = False
    # Send all active worlds' observations in ONE batched request per chunk
    # (one batched pi0 forward) instead of sequential per-world calls. Requires
    # the openpi server to support infer_batch (openpi branch
    # karim/batched-inference). Batched sampling draws RNG differently than
    # sequential, so results are not bit-identical across the two modes.
    batched_inference: bool = False
    # When failure injection is active: number of ADDITIONAL attempts to re-run
    # a seed whose injection did not produce a verified drop (slip_outcome
    # accidental_bin / not_released). Failed-injection attempts are not
    # recorded; the seed re-rolls (same scene, fresh policy sampling) in packed
    # retry batches until a verified outcome or attempts run out, at which
    # point the final attempt is recorded regardless. 0 = record first attempt.
    retry_failed_injection: int = 0
    # Weight + damp the free-jointed bin so recovery-phase arm bumps don't knock
    # it over (make_batched_env(bin_stabilize=...)), e.g. {mass_kg: 3.0, damping: 5.0}.
    # None = stock 0.15 kg undamped bin.
    bin_stabilize: dict[str, Any] | None = None
    # Extra randomization request keys merged into every env reset (construction
    # and per-world), passed through make_batched_env(extra_reset_options=...).
    # e.g. {bottle_spawn: opposite_bin} for put_bottles. None = defaults.
    reset_options: dict[str, Any] | None = None
    # Mid-rollout failure injection (eval/failure_injection.py). A mapping with a
    # ``type`` key (currently only "grasp_slip") plus that injector's config fields,
    # e.g. {type: grasp_slip, bin_edge_distance_m: 0.10, open_window_s: 0.5}.
    # None = no injection (nominal eval).
    failure_injection: dict[str, Any] | None = None
    # Pin ONE visual config (mesh variant + scale + color, seeded by this value) for the
    # whole run while sharding by POSITION via the standard seed_base/num_worlds loop.
    # Unlike visual_group_size (which ties visual_seed to seed_base//gs and forces
    # num_worlds==gs), this decouples the look from the position seeds, so a 100-episode
    # run can be split across GPUs (each shard a different seed_base, same look) and stay
    # scene-identical to a visual_group_size=100 single-batch run. Pair with
    # force_object_count for count-varying tasks. None = off.
    fixed_visual_seed: int | None = None

    def episodes_for(self, entry: TaskSpecEntry) -> int:
        return int(entry.episodes if entry.episodes is not None else self.episodes_default)

    def num_worlds_for(self, entry: TaskSpecEntry) -> int:
        return int(
            entry.num_worlds if entry.num_worlds is not None else self.num_worlds_default
        )

    def max_chunks_for(self, entry: TaskSpecEntry) -> int:
        if entry.max_chunks is not None:
            return int(entry.max_chunks)
        return int(entry.resolve().max_chunks)

    def horizon_chunks(self, entry: TaskSpecEntry, seconds_per_chunk: float) -> int:
        """Resolve the per-episode chunk horizon.

        Precedence (most specific wins): per-task ``max_seconds`` ->
        per-task ``max_chunks`` -> global ``max_seconds`` -> task spec ``max_chunks``.
        ``max_seconds`` is sim time and is converted to whole chunks via
        ``seconds_per_chunk`` (which the env's physics determines).
        """
        import math

        if entry.max_seconds is not None:
            return max(1, math.ceil(entry.max_seconds / seconds_per_chunk))
        if entry.max_chunks is not None:
            return int(entry.max_chunks)
        if self.max_seconds is not None:
            return max(1, math.ceil(self.max_seconds / seconds_per_chunk))
        return int(entry.resolve().max_chunks)


def _coerce_models(raw: list[Any]) -> list[ModelEndpoint]:
    models: list[ModelEndpoint] = []
    for item in raw:
        if isinstance(item, str):
            models.append(ModelEndpoint.parse(item))
        elif isinstance(item, dict):
            models.append(
                ModelEndpoint(
                    label=item["label"],
                    host=item["host"],
                    port=int(item.get("port", 8000)),
                    api_key=item.get("api_key"),
                )
            )
        else:
            raise TypeError(f"Unsupported model entry: {item!r}")
    return models


def _coerce_tasks(raw: list[Any]) -> list[TaskSpecEntry]:
    tasks: list[TaskSpecEntry] = []
    for item in raw:
        if isinstance(item, str):
            tasks.append(TaskSpecEntry(task=item))
        elif isinstance(item, dict):
            tasks.append(
                TaskSpecEntry(
                    task=item["task"],
                    episodes=item.get("episodes"),
                    num_worlds=item.get("num_worlds"),
                    max_chunks=item.get("max_chunks"),
                    max_seconds=item.get("max_seconds"),
                    prompt=item.get("prompt"),
                )
            )
        else:
            raise TypeError(f"Unsupported task entry: {item!r}")
    return tasks


def config_from_dict(data: dict[str, Any]) -> EvalConfig:
    """Build an :class:`EvalConfig` from a parsed YAML/JSON mapping.

    ``tasks`` is required; ``models`` is optional (may be supplied on the CLI).
    """
    if "tasks" not in data:
        raise ValueError("Eval config must define 'tasks'")
    known = {f for f in EvalConfig.__dataclass_fields__}  # type: ignore[attr-defined]
    kwargs: dict[str, Any] = {
        key: value
        for key, value in data.items()
        if key in known and key not in ("models", "tasks")
    }
    kwargs["tasks"] = _coerce_tasks(data["tasks"])
    if data.get("models"):
        kwargs["models"] = _coerce_models(data["models"])
    return EvalConfig(**kwargs)


def load_eval_config(path: str | Path) -> EvalConfig:
    """Load an eval config from a ``.yaml``/``.yml`` or ``.json`` file."""
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
    elif path.suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config extension: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError(f"Eval config must be a mapping, got {type(data)!r}")
    return config_from_dict(data)
