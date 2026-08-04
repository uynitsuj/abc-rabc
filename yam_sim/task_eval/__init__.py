"""Task evaluation helpers for automatic reward/success scoring."""

from yam_sim.task_eval.base import TaskEvalResult, TaskEvaluator
from yam_sim.task_eval.debug_spec import EvalDebugSpec, PlotSpec, ThresholdSpec
from yam_sim.task_eval.bottles import BottlesInBinEvaluator
from yam_sim.task_eval.plates import LoadPlatesInRackEvaluator
from yam_sim.task_eval.sweep import SweepAwayEvaluator
from yam_sim.task_eval.registry import make_task_evaluator

__all__ = [
    "TaskEvalResult",
    "TaskEvaluator",
    "ThresholdSpec",
    "PlotSpec",
    "EvalDebugSpec",
    "BottlesInBinEvaluator",
    "LoadPlatesInRackEvaluator",
    "SweepAwayEvaluator",
    "make_task_evaluator",
]
