"""Explicit data-collection task groups used by yam-sim workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from yam_sim.task_registry import get_task_scene_xml


def _normalize_key(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


@dataclass(frozen=True)
class CollectionTaskGroup:
    """One data-collection task family used by the current sim setup."""

    name: str
    display_name: str
    env_task: str
    sim_task_names: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    randomization_mode: str = "registry"
    notes: str = ""

    @property
    def has_randomization(self) -> bool:
        return self.randomization_mode != "none"

    @property
    def scene_xml(self) -> Path | None:
        return get_task_scene_xml(self.env_task)


DATA_COLLECTION_TASKS: tuple[CollectionTaskGroup, ...] = (
    CollectionTaskGroup(
        name="chess",
        display_name="Chess",
        env_task="chess",
        sim_task_names=("set_up_chess_pieces_on_board",),
        aliases=("set up chess pieces on board",),
    ),
    CollectionTaskGroup(
        name="spelling_with_blocks",
        display_name="Spelling With Blocks",
        env_task="blocks",
        sim_task_names=(
            "spell_cat",
            "spell_dog",
            "spell_fish",
            "spell_bair",
            "spell_xdof",
            "spell_abc",
            "spell_yam",
            "spell_agi",
        ),
        aliases=("spelling blocks", "spelling with blocks", "blocks spelling"),
        notes="Collection family spans the per-word spell_* sim task specs.",
    ),
    CollectionTaskGroup(
        name="marker_in_drawer",
        display_name="Marker In Drawer",
        env_task="drawer",
        sim_task_names=(
            "put_markers_in_top_drawer",
            "put_markers_in_middle_drawer",
            "put_markers_in_bottom_drawer",
        ),
        aliases=("drawer", "markers in drawer", "marker in drawer"),
        notes="Collection family spans the top/middle/bottom drawer task specs.",
    ),
    CollectionTaskGroup(
        name="dish_rack",
        display_name="Dish Rack",
        env_task="dishrack",
        sim_task_names=("load_plates_into_dish_rack",),
        aliases=("dishrack", "load plates into dish rack"),
    ),
    CollectionTaskGroup(
        name="sweep",
        display_name="Sweep",
        env_task="sweep",
        sim_task_names=("sweep_away_paper_scraps_from_table",),
        aliases=("sweep away paper scraps from table",),
        notes="Uses the registry path with clustered trash pose randomization plus shared size perturbations.",
    ),
    CollectionTaskGroup(
        name="place_bottle_in_bin",
        display_name="Place Bottle In Bin",
        env_task="bottles",
        sim_task_names=("throw_plastic_bottles_in_bin",),
        aliases=("bottles", "bottle in bin", "place bottle in bin", "throw plastic bottles in bin"),
    ),
    CollectionTaskGroup(
        name="flip_mug",
        display_name="Flip Mug",
        env_task="mug_flip",
        sim_task_names=("turn_mug_right_side_up",),
        aliases=("mug flip", "flip mug", "turn mug right side up"),
    ),
    CollectionTaskGroup(
        name="mug_on_tree",
        display_name="Mug On Tree",
        env_task="mug_tree",
        sim_task_names=("hang_mug_on_mug_rack",),
        aliases=("mug tree", "mug on tree", "hang mug on mug rack"),
    ),
    CollectionTaskGroup(
        name="pouring_beads",
        display_name="Pouring Beads",
        env_task="pour",
        sim_task_names=("pouring",),
        aliases=("pour", "pouring", "pouring beads"),
    ),
    CollectionTaskGroup(
        name="random_object_handover",
        display_name="Random Object Handover",
        env_task="inhand_transfer",
        aliases=("handover", "random object handover", "inhand transfer", "random object transfer"),
        randomization_mode="special",
        notes="Uses the special inhand_transfer model-swapping randomizer path instead of TASK_RANDOMIZERS.",
    ),
)

_LOOKUP: dict[str, CollectionTaskGroup] = {}
for _task in DATA_COLLECTION_TASKS:
    for _alias in {_task.name, _task.display_name, _task.env_task, *_task.sim_task_names, *_task.aliases}:
        _LOOKUP[_normalize_key(_alias)] = _task


def list_data_collection_tasks() -> tuple[CollectionTaskGroup, ...]:
    """Return the explicit data-collection task families."""

    return DATA_COLLECTION_TASKS


def get_data_collection_task(name: str) -> CollectionTaskGroup:
    """Resolve a data-collection task family by name or alias."""

    key = _normalize_key(name)
    try:
        return _LOOKUP[key]
    except KeyError as exc:
        available = ", ".join(task.name for task in DATA_COLLECTION_TASKS)
        raise KeyError(
            f"Unknown data-collection task {name!r}. Available: {available}"
        ) from exc


def maybe_get_data_collection_task(name: str | None) -> CollectionTaskGroup | None:
    """Resolve a data-collection task family if present."""

    if not name:
        return None
    return _LOOKUP.get(_normalize_key(name))


__all__ = [
    "CollectionTaskGroup",
    "DATA_COLLECTION_TASKS",
    "get_data_collection_task",
    "list_data_collection_tasks",
    "maybe_get_data_collection_task",
]
