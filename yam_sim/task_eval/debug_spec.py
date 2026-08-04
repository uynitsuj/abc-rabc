"""Task-defined debug plot specifications for evaluator dashboards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ThresholdSpec:
    """A horizontal reference line on a plot."""

    value: float
    label: str | None = None
    direction: Literal["lt", "gt"] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"value": float(self.value)}
        if self.label is not None:
            out["label"] = self.label
        if self.direction is not None:
            out["direction"] = self.direction
        return out


@dataclass(frozen=True)
class PlotSpec:
    """A single scalar time-series plot."""

    key: str
    title: str | None = None
    color: str | None = None
    kind: Literal["line", "bool"] = "line"
    thresholds: list[ThresholdSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title if self.title is not None else self.key,
            "color": self.color,
            "kind": self.kind,
            "thresholds": [threshold.to_dict() for threshold in self.thresholds],
        }


@dataclass(frozen=True)
class EvalDebugSpec:
    """Schema for evaluator-specific dashboard rendering."""

    plots: list[PlotSpec]
    x_key: str = "step"

    def to_dict(self) -> dict[str, Any]:
        return {
            "x_key": self.x_key,
            "plots": [plot.to_dict() for plot in self.plots],
        }

