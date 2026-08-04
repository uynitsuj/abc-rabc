"""Task evaluator registry."""

from __future__ import annotations

from typing import TypeAlias

import mujoco

from yam_sim.task_eval.base import TaskEvaluator
from yam_sim.task_eval.bottles import BottlesInBinEvaluator, PutBottlesInBinEvaluator
from yam_sim.task_eval.mugs import HangMugOnRackEvaluator, MugFlipUprightEvaluator
from yam_sim.task_eval.plates import LoadPlatesInRackEvaluator
from yam_sim.task_eval.sweep import SweepAwayEvaluator
from yam_sim.task_specs import SimTaskSpec, maybe_get_task_spec

TaskEvaluatorFactory: TypeAlias = type[TaskEvaluator]


_TASK_EVALUATORS: dict[str, TaskEvaluatorFactory] = {
    "bottles_in_bin": BottlesInBinEvaluator,
    "put_bottles_in_bin": PutBottlesInBinEvaluator,
    "mug_flip_upright": MugFlipUprightEvaluator,
    "hang_mug_on_rack": HangMugOnRackEvaluator,
    "sweep_away": SweepAwayEvaluator,
    "load_plates_in_rack": LoadPlatesInRackEvaluator,
}


def make_task_evaluator(
    model: mujoco.MjModel,
    task: str | SimTaskSpec | None,
) -> TaskEvaluator | None:
    """Instantiate a task evaluator for a task name/spec if one is configured."""

    spec = task if isinstance(task, SimTaskSpec) else maybe_get_task_spec(task)
    if spec is None or spec.evaluator_name is None:
        return None

    try:
        evaluator_cls = _TASK_EVALUATORS[spec.evaluator_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown task evaluator {spec.evaluator_name!r} for task {spec.name!r}"
        ) from exc

    return evaluator_cls(model=model, spec=spec, **spec.evaluator_kwargs())
