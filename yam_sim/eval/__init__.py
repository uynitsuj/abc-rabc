"""Policy evaluation harness for yam-sim.

Run a (model x task x N-episode) job matrix against trained pi0 checkpoints served
over openpi websocket servers, recording one video per episode and aggregating
per-(model, task) success rates.
"""

from __future__ import annotations

from yam_sim.eval.config import (
    EvalConfig,
    ModelEndpoint,
    TaskSpecEntry,
    config_from_dict,
    load_eval_config,
)
from yam_sim.eval.harness import run_eval
from yam_sim.eval.results import EpisodeResult, aggregate, write_summary

__all__ = [
    "EvalConfig",
    "ModelEndpoint",
    "TaskSpecEntry",
    "EpisodeResult",
    "config_from_dict",
    "load_eval_config",
    "run_eval",
    "aggregate",
    "write_summary",
]
