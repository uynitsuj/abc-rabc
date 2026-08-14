"""Offline check of GraspSlipInjector's trigger/verify/track state machine.

Builds the single-world put_bottles env (CPU MuJoCo, no rendering, no policy
server), then drives the injector through a scripted qpos timeline by
teleporting bottle 0 and the bin directly: standing far from bin (no trigger)
-> held at the gripper 5 cm from the bin edge (trigger + forced-open window)
-> dropped to table level (verified) -> rolled past the table front edge
(roll-off) -> placed in the bin (rebin). Asserts the emitted slip_* record.

Run:  uv run python local/test_grasp_slip_injector.py
"""

from __future__ import annotations

import numpy as np
import mujoco

import yam_sim
from yam_sim.eval.failure_injection import GraspSlipConfig, GraspSlipInjector
from yam_sim.task_eval import make_task_evaluator
from yam_sim.task_specs import get_task_spec


class FakeBatchedEnv:
    """Minimal batched-env facade over a single-world MuJoCoYAMEnv."""

    def __init__(self, env):
        self.base_env = env
        self.model = env.model
        self.data = env.data
        self.num_worlds = 1
        self._task_spec = get_task_spec("put_plastic_bottles_in_bin")
        self._task_evaluator = make_task_evaluator(env.model, self._task_spec)
        self._gripper_indices = env._gripper_indices

    def qpos_batch(self) -> np.ndarray:
        return np.asarray(self.data.qpos, dtype=np.float64)[None, :].copy()


def set_freejoint(data, adr: int, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
    data.qpos[adr : adr + 3] = pos
    data.qpos[adr + 3 : adr + 7] = quat


def main() -> None:
    env = yam_sim.make_env(task="put_bottles", render_cameras=False)
    env.reset(
        seed=0,
        options={
            "bottle_count": 2,
            "randomize_scales": False,
            "randomize_variants": False,
        },
    )
    fake = FakeBatchedEnv(env)
    ev = fake._task_evaluator
    model, data = env.model, env.data

    bottle_adr = int(ev._bottle_qpos_addrs[0])
    bin_adr = int(ev._bin_qpos_adr)
    r_eff = 0.5 * (ev._bin_radius_x + ev._bin_radius_z)

    # Gripper center of arm 0 at the reset pose (finger-body midpoint).
    fk = mujoco.MjData(model)
    fk.qpos[:] = data.qpos
    mujoco.mj_kinematics(model, fk)
    robot = env.robot_names[0]
    fids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{robot}_link_{s}_finger")
        for s in ("left", "right")
    ]
    grip = 0.5 * (fk.xpos[fids[0]] + fk.xpos[fids[1]])
    print(f"reset gripper center: {np.round(grip, 3)}")

    cfg = GraspSlipConfig(
        bin_edge_distance_m=0.10,
        held_z_min=float(grip[2]) - 0.02,  # trigger height tied to the actual FK pose
        open_window_s=0.5,
        verify_window_s=1.5,
        drop_z_max=float(grip[2]) - 0.05,
    )
    injector = GraspSlipInjector(fake, cfg)
    dt = injector._control_dt
    open_steps = injector._open_steps
    grip_dim = injector._gripper_action_dims[0]

    bin_pos0 = data.qpos[bin_adr : bin_adr + 3].copy()
    bin_quat0 = data.qpos[bin_adr + 3 : bin_adr + 7].copy()

    def step(t: int, chunk_idx: int = 0, step_idx: int = 0) -> np.ndarray:
        actions = np.full((1, 14), 0.3, dtype=np.float32)
        injector.before_step(fake, actions, chunk_idx=chunk_idx, step_idx=step_idx)
        return actions

    # Geometry cross-check against the evaluator on the reset scene.
    centers, in_bin, edge = injector._bottle_geometry(fake.qpos_batch())
    ref = ev.evaluate_qpos_batch(fake.qpos_batch())
    assert np.array_equal(in_bin[0], ref.metrics["bottle_in_bin_mask"][0]), (
        in_bin,
        ref.metrics["bottle_in_bin_mask"],
    )
    print(f"geometry check ok (edge dists {np.round(edge[0], 3)})")

    # Phase A: bottles on the table far from the bin -> no trigger.
    for t in range(5):
        actions = step(t)
        assert actions[0, grip_dim] == np.float32(0.3)
    assert not injector._triggered[0]

    # Phase B: bottle 0 "held" at the gripper, bin moved so its footprint edge
    # sits ~5 cm from the bottle -> trigger, forced-open window.
    set_freejoint(data, bottle_adr, grip)
    offset = ev._bin_radius_x * (1.0 + 0.05 / r_eff)  # edge_dist ~= 0.05 along +x
    set_freejoint(
        data,
        bin_adr,
        (grip[0] + offset, grip[1], bin_pos0[2]),
        bin_quat0,
    )
    centers, in_bin, edge = injector._bottle_geometry(fake.qpos_batch())
    assert 0.0 < edge[0, 0] <= 0.10, f"test setup: edge_dist={edge[0, 0]}"

    trigger_t = 5
    actions = step(trigger_t, chunk_idx=3, step_idx=7)
    assert injector._triggered[0], "trigger did not fire"
    assert actions[0, grip_dim] == np.float32(1.0), "gripper not forced open"
    for t in range(trigger_t + 1, trigger_t + open_steps):
        actions = step(t)
        assert actions[0, grip_dim] == np.float32(1.0)
    actions = step(trigger_t + open_steps)
    assert actions[0, grip_dim] == np.float32(0.3), "override outlived open window"

    # Phase C: bottle falls to the table -> verified drop.
    set_freejoint(data, bottle_adr, (grip[0], grip[1], cfg.table_z + 0.035))
    step(trigger_t + open_steps + 1)
    assert injector._outcome[0] == "dropped", injector._outcome

    # Phase D: bottle rolls past the table front edge -> roll-off.
    set_freejoint(data, bottle_adr, (0.25, grip[1], cfg.table_z + 0.035))
    rolloff_t = trigger_t + open_steps + 2
    step(rolloff_t)
    assert injector._rolloff_step[0] == rolloff_t

    # Phase E: bottle placed inside the bin -> rebin recorded.
    bin_pos = data.qpos[bin_adr : bin_adr + 3]
    set_freejoint(data, bottle_adr, (bin_pos[0], bin_pos[1], bin_pos[2] + 0.05))
    rebin_t = rolloff_t + 1
    step(rebin_t)

    (record,) = injector.finalize()
    print(record)
    assert record["slip_triggered"] is True
    assert record["slip_step"] == trigger_t
    assert record["slip_chunk_offset"] == 7
    assert record["slip_arm"] == robot
    assert record["slip_bottle"] == ev.bottle_names[0]
    assert record["slip_outcome"] == "dropped"
    assert record["slip_rolled_off_table"] is True
    assert record["slip_rebin_step"] == rebin_t
    assert abs(record["slip_recovery_time_s"] - (rebin_t - trigger_t) * dt) < 1e-6

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
