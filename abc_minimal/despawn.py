"""Option-2 despawn for the abc_minimal (MuJoCo Warp) evaluator.

Port of anchor/src/anchor/eval/despawn.py to this stack. Opt-in via
ABC_DESPAWN_FIRST_N > 0; the default path never imports or calls this.

WHY: the bin is small enough that the first bottles seated can pile and physically
block bottles 5-6 from ever fitting, censoring the top of the distribution. Removing
the FIRST N bottles that stable-seat clears the floor for the rest. The removed
bottles stay credited -- they are already latched placed_ever at the seat event --
so this fixes the sim, not the metric. The 0.26-cylinder scorer is untouched.

DIFFERENCE FROM THE ANCHOR VERSION, and why. The anchor stack runs CPU MuJoCo and
writes data.qpos directly, re-pinning EVERY sub-step to fight the stale contact and
warmstart forces a squeezing gripper leaves on a grasped bottle. This stack runs
MuJoCo Warp: state lives on GPU in d_warp, so direct data.qpos writes would silently
do nothing. Here we teleport once on latch, then check position every control step
and re-pin only on drift. 20 m from the bin there are no contacts to fight, so the
drift should be zero -- and the drift counter is therefore also the firing gate.
The anchor's notes record a stale despawn detector that reported zero while it was
firing, so the count is measured from the teleports themselves, not inferred.
"""
from __future__ import annotations

import os
import re

import numpy as np

DESPAWN_XY = 20.0     # metres; bin/table live within x in [0.3,0.9], y in [-0.65,0.65]
DESPAWN_Z = 0.5       # held aloft, velocity zeroed, out of the camera frustum
DRIFT_TOL = 0.01      # m; re-pin if a despawned bottle moves more than this


def enabled() -> int:
    try:
        return int(os.environ.get("ABC_DESPAWN_FIRST_N", "0"))
    except ValueError:
        return 0


class Despawner:
    """Tracks which bottles have been removed and keeps them removed."""

    def __init__(self, model, bottle_qpos_addrs, n_despawn: int):
        self.n = int(n_despawn)
        self.qadr = list(bottle_qpos_addrs)
        self.dofadr = self._dof_addrs(model)
        self.pinned: dict[int, np.ndarray] = {}   # bottle index -> pinned xyz
        self.teleports = 0                        # first-time removals (the firing count)
        self.repins = 0                           # drift corrections (should stay 0)
        self.max_drift = 0.0

    @staticmethod
    def _dof_addrs(model) -> list[int]:
        """Bottle freejoint dof addresses ordered by joint suffix, matching the
        evaluator's bottle_qpos_addrs ordering (see _bottle_addrs)."""
        ent = []
        for jj in range(model.njnt):
            m = re.fullmatch(r"bottle_(\d+)_joint", model.jnt(jj).name or "")
            if m:
                ent.append((int(m.group(1)), int(model.jnt_dofadr[jj])))
        ent.sort()
        return [e[1] for e in ent]

    def _spot(self, slot: int) -> np.ndarray:
        # distinct parking spots so removed bottles cannot collide with each other
        return np.array([DESPAWN_XY + 2.0 * slot, DESPAWN_XY, DESPAWN_Z], dtype=np.float64)

    def apply(self, qpos: np.ndarray, qvel: np.ndarray, placed_ever, placed_step) -> bool:
        """Mutate qpos/qvel in place. Returns True if anything changed."""
        changed = False

        # 1. newly-latched bottles, in the order they seated, up to N total
        if len(self.pinned) < self.n:
            order = sorted((int(s), int(i)) for i, s in placed_step.items())
            for _step, i in order:
                if len(self.pinned) >= self.n:
                    break
                if i in self.pinned:
                    continue
                spot = self._spot(len(self.pinned))
                a, d = self.qadr[i], self.dofadr[i]
                qpos[a:a + 3] = spot
                qpos[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)
                qvel[d:d + 6] = 0.0
                self.pinned[i] = spot
                self.teleports += 1
                changed = True

        # 2. keep them there; drift should be zero at 20 m (no contacts)
        for i, spot in self.pinned.items():
            a, d = self.qadr[i], self.dofadr[i]
            drift = float(np.linalg.norm(qpos[a:a + 3] - spot))
            self.max_drift = max(self.max_drift, drift)
            if drift > DRIFT_TOL:
                qpos[a:a + 3] = spot
                qpos[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)
                qvel[d:d + 6] = 0.0
                self.repins += 1
                changed = True
        return changed

    def stats(self) -> dict:
        return {
            "despawn_first_n": self.n,
            "despawn_teleports": self.teleports,
            "despawn_repins": self.repins,
            "despawn_max_drift_m": round(self.max_drift, 6),
            "despawn_pinned_idx": sorted(self.pinned),
        }
