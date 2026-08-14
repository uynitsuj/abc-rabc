"""Batched lockstep eval runner for abc_minimal's put-bottles task (MuJoCo Warp).

Design (all decisions measured/proven this week — see PROFILE_LOG + proofs):
  * ONE combined model: world-0's scene with every other world's scaled bottle/bin
    meshes added as extra assets; per-world `geom_dataid` rows (batch_sizes API)
    redirect each world's mesh geoms to its own copies. Proven exact to serial
    within batch-width float divergence (proof_2world_hetero.py).
  * EXACT per-seed initial states: run the *serial* `PutBottlesEnv.reset(seed)`
    on a throwaway CPU model per world and copy its qpos row into the batch.
    Identical rng consumption by construction — no reimplementation drift.
  * Lockstep stepping: all worlds share the step count (--no-early-stop protocol),
    so there is no divergence handling at all.
  * Vectorized evaluator: [B, nbottle] arrays for streaks/latches/events, plus the
    composite metric (tight-upright-ever), bin-tilt tracking, and despawn.
  * Policy is pluggable: `ZeroPolicy` (throughput/smoke), batched pi0 in-process
    (Thursday). infer(obs_batch) -> actions [B, H, nu].

Equivalence standard vs the serial runner is DISTRIBUTIONAL, never bitwise:
batch width changes contact reduction order; chaos amplifies it (measured).
"""
from __future__ import annotations

import dataclasses
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, "/scratch/warprm_eval/abc_rabc")
import mujoco  # noqa: E402

from abc_minimal.eval_policy import (  # noqa: E402
    ROOT,
    SCENE_XML,
    PutBottlesEnv,
    PutBottlesSimConfig,
    scene_xml,
)

TILT_DEG = 20.0


# --------------------------------------------------------------------------- #
# combined heterogeneous model
# --------------------------------------------------------------------------- #
def world_scales(scene: PutBottlesSimConfig, seed: int):
    """Reproduce exactly the two rng draws reset(seed) makes before pose sampling."""
    rng = np.random.default_rng(seed)
    bottle_scales = rng.uniform(*scene.bottle_scale_range, size=scene.bottle_count).astype(np.float32)
    bin_scale = float(rng.uniform(*scene.bin_scale_range))
    return bottle_scales, bin_scale


def _scaled(scene, name, base, bs, binsc):
    for idx in range(scene.bottle_count):
        if name.startswith(f"bottle_{idx}_"):
            return base * float(bs[idx])
    if name.startswith("water_bottle_"):
        return base * binsc
    return None


def combined_model(scene: PutBottlesSimConfig, seeds: list[int]):
    """One MjModel whose asset library holds every world's scaled meshes.
    Returns (model, dataid_rows[B, ngeom])."""
    scales = [world_scales(scene, s) for s in seeds]
    root = ET.fromstring(SCENE_XML.read_text())
    comp = root.find("compiler")
    comp.set("meshdir", str((ROOT / "assets" / "put_bottles" / "assets").resolve()))
    comp.set("texturedir", str((ROOT / "assets" / "put_bottles" / "assets").resolve()))
    asset = root.find("asset")
    fmt = lambda v: " ".join(f"{x:.9g}" for x in v)  # noqa: E731
    for mesh in list(root.findall("./asset/mesh")):
        name = mesh.get("name", "")
        base = np.asarray([float(v) for v in mesh.get("scale", "1 1 1").split()])
        s0 = _scaled(scene, name, base, *scales[0])
        if s0 is None:
            continue
        mesh.set("scale", fmt(s0))
        for w, sc in enumerate(scales[1:], start=1):
            dup = ET.SubElement(asset, "mesh", dict(mesh.attrib))
            dup.set("name", f"{name}__w{w}")
            dup.set("scale", fmt(_scaled(scene, name, base, *sc)))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))

    rows = np.tile(model.geom_dataid, (len(seeds), 1)).astype(np.int32)
    for w in range(1, len(seeds)):
        for g in range(model.ngeom):
            did = model.geom_dataid[g]
            if did < 0:
                continue
            mname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, did) or ""
            alt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, f"{mname}__w{w}")
            if alt >= 0:
                rows[w, g] = alt
    return model, rows


def serial_reset_rows(scene: PutBottlesSimConfig, seeds: list[int]):
    """EXACT initial states: serial reset(seed) per world on throwaway CPU envs.
    Returns qpos rows [B, nq] plus the envs' randomization records."""
    rows, records = [], []
    env = PutBottlesEnv(
        height=224, width=168, camera_keys=("top_camera-images-rgb",),
        prompt="", scene=scene, gpu_id=None,
    )
    env.obs = lambda: {}  # reset() ends with obs(); we discard it — skip camera renders
    for s in seeds:
        env.reset(seed=int(s))  # binds per-seed scaled model, samples poses, mj_forward
        rows.append(np.asarray(env.data.qpos, dtype=np.float64).copy())
        records.append(env.randomization)
    env.close()
    return np.stack(rows), records


# --------------------------------------------------------------------------- #
# vectorized evaluator (latch + paper geometry + composite + tilt + despawn)
# --------------------------------------------------------------------------- #
def _rotmat(q4):
    w, x, y, z = q4
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class BatchedEvaluator:
    """Port of PutBottlesEvaluator to [B] worlds + composite/tilt columns.
    Uses the SAME loose center-point live check as the serial evaluator (the
    runtime latch); paper-rule scoring stays offline via score_bottles on the
    saved traces, unchanged."""

    PERSIST = 15
    GRACE = 30

    def __init__(self, model: mujoco.MjModel, scene: PutBottlesSimConfig, nworld: int):
        self.scene = scene
        self.B = nworld
        names, addrs = [], []
        import re
        for j in range(model.njnt):
            m = re.fullmatch(r"bottle_(\d+)_joint", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "")
            if m:
                names.append((int(m.group(1)), int(model.jnt_qposadr[j])))
        names.sort()
        self.bottle_adr = np.array([a for _, a in names])
        jb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bin_joint")
        self.bin_adr = int(model.jnt_qposadr[jb])
        n = len(self.bottle_adr)
        self.streak = np.zeros((nworld, n), np.int32)
        self.placed_ever = np.zeros((nworld, n), bool)
        self.placed_step = np.full((nworld, n), -1, np.int32)
        self.comp_credit = np.zeros((nworld, n), bool)
        self.step = 0
        self._bin_up_ref = None      # [B, 3] body vector that is world-up at t0
        self.tip_step = np.full(nworld, -1, np.int32)
        self.max_tilt = np.zeros(nworld)

    def _bin_upright(self, q):  # q: [B, nq] -> [B] bool
        cos = np.empty(self.B)
        for w in range(self.B):
            R = _rotmat(q[w, self.bin_adr + 3:self.bin_adr + 7])
            if self._bin_up_ref is None:
                pass
            cos[w] = float((R @ self._ref[w])[2])
        return cos

    def prime(self, q0):
        self._ref = np.stack([
            _rotmat(q0[w, self.bin_adr + 3:self.bin_adr + 7]).T @ np.array([0.0, 0.0, 1.0])
            for w in range(self.B)])

    def update(self, q):  # q: [B, nq]
        self.step += 1
        binp = q[:, self.bin_adr:self.bin_adr + 3]                      # [B,3]
        pos = np.stack([q[:, a:a + 3] for a in self.bottle_adr], 1)     # [B,n,3]
        rel = pos - binp[:, None, :]
        # serial evaluator's live check: loose center cylinder
        inb = (np.hypot(rel[..., 0], rel[..., 1]) <= 0.155) & (rel[..., 2] >= -0.06) & (rel[..., 2] <= 0.26)
        self.streak = np.where(inb, self.streak + 1, 0)
        cos = self._bin_upright(q)
        tilt = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        self.max_tilt = np.maximum(self.max_tilt, tilt)
        newly_tipped = (tilt > TILT_DEG) & (self.tip_step < 0)
        self.tip_step[newly_tipped] = self.step
        if self.step >= self.GRACE:
            qual = self.streak >= self.PERSIST
            newly = qual & ~self.placed_ever
            self.placed_ever |= newly
            self.placed_step[newly] = self.step - self.PERSIST + 1
            upright = (tilt <= TILT_DEG)[:, None]
            self.comp_credit |= (qual & upright)

    def world_records(self, seeds):
        out = []
        for w in range(self.B):
            out.append(dict(
                world_seed=int(seeds[w]),
                placed_ever=int(self.placed_ever[w].sum()),
                placement_steps=sorted(int(s) for s in self.placed_step[w] if s > 0),
                composite=int(self.comp_credit[w].sum()),
                bin_tip_step=(int(self.tip_step[w]) if self.tip_step[w] > 0 else None),
                bin_max_tilt_deg=round(float(self.max_tilt[w]), 1),
            ))
        return out


class BatchedDespawner:
    """Vectorized Option-2 despawn: first N latched bottles per world get pinned
    20m away; drift-checked re-pin; per-world firing stats."""

    def __init__(self, model, bottle_qpos_adr, nworld, n_despawn):
        import re
        self.n = n_despawn
        self.B = nworld
        self.qadr = bottle_qpos_adr
        ent = []
        for jj in range(model.njnt):
            m = re.fullmatch(r"bottle_(\d+)_joint", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jj) or "")
            if m:
                ent.append((int(m.group(1)), int(model.jnt_dofadr[jj])))
        ent.sort()
        self.dofadr = np.array([e[1] for e in ent])
        self.pinned = np.zeros((nworld, len(self.qadr)), bool)
        self.slot = np.full((nworld, len(self.qadr)), -1, np.int32)
        self.teleports = np.zeros(nworld, np.int32)
        self.repins = np.zeros(nworld, np.int32)

    def _spot(self, slot):
        return np.array([20.0 + 2.0 * slot, 20.0, 0.5])

    def apply(self, q, v, ev: BatchedEvaluator) -> bool:
        if self.n == 0:
            return False
        changed = False
        for w in range(self.B):
            if self.pinned[w].sum() < self.n:
                order = [(ev.placed_step[w, i], i) for i in range(len(self.qadr))
                         if ev.placed_step[w, i] > 0 and not self.pinned[w, i]]
                for _, i in sorted(order):
                    if self.pinned[w].sum() >= self.n:
                        break
                    k = int(self.pinned[w].sum())
                    a, d = self.qadr[i], self.dofadr[i]
                    q[w, a:a + 3] = self._spot(k)
                    q[w, a + 3:a + 7] = (1, 0, 0, 0)
                    v[w, d:d + 6] = 0
                    self.pinned[w, i] = True
                    self.slot[w, i] = k
                    self.teleports[w] += 1
                    changed = True
            for i in np.flatnonzero(self.pinned[w]):
                a, d = self.qadr[i], self.dofadr[i]
                if np.linalg.norm(q[w, a:a + 3] - self._spot(self.slot[w, i])) > 0.01:
                    q[w, a:a + 3] = self._spot(self.slot[w, i])
                    q[w, a + 3:a + 7] = (1, 0, 0, 0)
                    v[w, d:d + 6] = 0
                    self.repins[w] += 1
                    changed = True
        return changed


# --------------------------------------------------------------------------- #
# the runner
# --------------------------------------------------------------------------- #
class ZeroPolicy:
    def __init__(self, nu, horizon=30):
        self.nu, self.h = nu, horizon

    def infer(self, obs_batch):
        B = obs_batch["state"].shape[0]
        return np.zeros((B, self.h, self.nu), np.float32)


@dataclasses.dataclass
class BatchedRunResult:
    records: list
    qpos_traces: np.ndarray  # [B, T, nq] fp16
    rand_records: list = None


def _jsonable(x):
    import json
    try:
        json.dumps(x)
        return x
    except TypeError:
        if isinstance(x, dict):
            return {k: _jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_jsonable(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        return str(x)


def write_trace_dir(out_root, arm, seeds, res: BatchedRunResult,
                    rand_records=None, shard=0, extra=None):
    """Emit the EXACT layout the serial eval stack produces, so score_bottles
    collect() and render_trace consume batched runs unchanged:
      <out_root>/fullhz_{arm}_sh{shard}/qpos_trace_{idx:03d}_s{seed}.npz  (key: qpos [T,nq] fp16)
      <out_root>/fullhz_{arm}_sh{shard}/summary.json
    collect() keys worlds by the seed parsed from the filename; summary.json is
    informational (collect() ignores it by design)."""
    import json
    out_dir = Path(out_root) / f"fullhz_{arm}_sh{shard}"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for w, seed in enumerate(seeds):
        p = out_dir / f"qpos_trace_{w:03d}_s{int(seed)}.npz"
        np.savez_compressed(p, qpos=res.qpos_traces[w])
        paths.append(str(p))
    summary = {
        "arm": arm,
        "batched": True,
        "num_worlds": len(seeds),
        "world_seeds": [int(s) for s in seeds],
        "records": _jsonable(res.records),
        "qpos_trace_paths": paths,
    }
    if rand_records is not None:
        summary["randomization"] = _jsonable(rand_records)
    if extra:
        summary.update(_jsonable(extra))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return out_dir


def run_batched(seeds, policy=None, steps=1800, execute_chunk=30, despawn_n=0,
                gpu=0, scene=None, trace=True):
    import mujoco_warp as mjw
    import warp as wp
    scene = scene or PutBottlesSimConfig()
    B = len(seeds)
    wp.set_device(f"cuda:{gpu}")

    model, dataid_rows = combined_model(scene, list(seeds))
    q0, rand_records = serial_reset_rows(scene, list(seeds))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    mw = mjw.put_model(model, batch_sizes={"geom_dataid": B})
    arr = mw.geom_dataid.numpy()
    arr[:] = dataid_rows
    wp.copy(mw.geom_dataid, wp.from_numpy(arr.astype(np.int32), dtype=wp.int32))
    dw = mjw.put_data(model, data, nworld=B, nconmax=model.nconmax, njmax=model.njmax)
    wp.copy(dw.qpos, wp.from_numpy(np.ascontiguousarray(q0, dtype=np.float32), dtype=wp.float32))
    qvel0 = np.zeros((B, model.nv), np.float32)
    wp.copy(dw.qvel, wp.from_numpy(qvel0, dtype=wp.float32))
    mjw.forward(mw, dw)

    # actuator/state mapping: serial env computes these by name; reuse one env's maps
    env = PutBottlesEnv(height=224, width=168, camera_keys=("top_camera-images-rgb",),
                        prompt="", scene=scene, gpu_id=None)
    env.obs = lambda: {}
    env.reset(seed=int(seeds[0]))
    ctrl_indices = list(env.ctrl_indices)
    qpos_indices = np.asarray(env.qpos_indices)
    gripper_idx = list(env.gripper_state_indices)
    env.close()
    # serial action_to_ctrl: gripper action columns scale by gripper_ctrl_max
    act_scale = np.ones(len(ctrl_indices), np.float32)
    for i in gripper_idx:
        act_scale[i] = scene.gripper_ctrl_max

    def get_state_rows(q):  # [B, nq] -> [B, K], serial get_state vectorized
        s = q[:, qpos_indices].astype(np.float32)
        for i in gripper_idx:
            s[:, i] = np.clip(s[:, i] / scene.gripper_ctrl_max, 0.0, 1.0)
        return s

    ev = BatchedEvaluator(model, scene, B)
    ev.prime(q0)
    desp = BatchedDespawner(model, ev.bottle_adr, B, despawn_n)
    policy = policy or ZeroPolicy(model.nu)
    decim = scene.control_decimation

    # image policies (BatchedPi0): world-aware batched render at replan boundaries.
    # Env camera keys are RAW MuJoCo names ("top","left","right"); the lerobot-style
    # names ("top_camera-images-rgb") are SERVER keys, mapped by policy.camera_key_map.
    needs_images = bool(getattr(policy, "needs_images", False))
    if needs_images:
        cam_names = list(getattr(policy, "camera_key_map", {"top": "top_camera-images-rgb"}))
        cam_ids = {}
        for name in cam_names:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            assert cid >= 0, f"camera not found in combined model: {name}"
            cam_ids[name] = cid
        H, W = 168, 224  # serial eval config: camera_height=168, camera_width=224
        ctx = mjw.create_render_context(mjm=model, nworld=B, cam_res=(W, H),
                                        render_rgb=[True] * model.ncam,
                                        render_depth=[False] * model.ncam,
                                        use_textures=True, use_shadows=True)

    def batched_obs():
        q = dw.qpos.numpy().astype(np.float64)
        if not needs_images:
            return {"state": q.astype(np.float32)}
        mjw.refit_bvh(mw, dw, ctx)
        mjw.render(mw, dw, ctx)
        rgba = ctx.rgb_data.numpy().view(np.uint8).reshape(B, model.ncam, H, W, 4)
        images = {name: np.ascontiguousarray(rgba[:, cid, :, :, :3].transpose(0, 3, 1, 2))
                  for name, cid in cam_ids.items()}  # [B,3,H,W] CHW like serial
        return policy.make_obs(get_state_rows(q), images)

    def replan():
        o = batched_obs()
        return policy.infer_batch(o) if needs_images else policy.infer(o)

    traces = []
    actions = replan()
    ai = 0
    for step in range(steps):
        if ai >= actions.shape[1]:
            actions = replan()
            ai = 0
        ctrl = np.zeros((B, model.nu), np.float32)
        ctrl[:, ctrl_indices] = actions[:, ai, :len(ctrl_indices)] * act_scale
        ai += 1
        wp.copy(dw.ctrl, wp.from_numpy(ctrl, dtype=wp.float32))
        for _ in range(decim):
            mjw.step(mw, dw)
        q = dw.qpos.numpy().astype(np.float64)
        ev.update(q)
        if despawn_n:
            v = dw.qvel.numpy().astype(np.float64)
            if desp.apply(q, v, ev):
                wp.copy(dw.qpos, wp.from_numpy(q.astype(np.float32), dtype=wp.float32))
                wp.copy(dw.qvel, wp.from_numpy(v.astype(np.float32), dtype=wp.float32))
        if trace:
            traces.append(q.astype(np.float16))

    recs = ev.world_records(seeds)
    if despawn_n:
        for w, r in enumerate(recs):
            r["despawn"] = dict(despawn_first_n=despawn_n,
                                despawn_teleports=int(desp.teleports[w]),
                                despawn_repins=int(desp.repins[w]))
    return BatchedRunResult(records=recs,
                            qpos_traces=np.stack(traces, axis=1) if trace else None,
                            rand_records=rand_records)


if __name__ == "__main__":
    import argparse
    import json
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[20260511 + k for k in range(4)])
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--despawn", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--arm", default=None, help="arm name -> serial-compatible trace dir")
    ap.add_argument("--trace-dir", default=None, help="root for fullhz_{arm}_sh{shard}/")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--pi0-config", default=None, help="openpi TrainConfig name -> in-process batched pi0")
    ap.add_argument("--pi0-ckpt", default=None, help="checkpoint params dir for --pi0-config")
    ap.add_argument("--obs-condition", type=float, default=None,
                    help="explicit condition per obs (nan = unconditional/CFG-null branch)")
    ap.add_argument("--cfg-weight", type=float, default=None,
                    help="CFG guidance weight w (needs velocity_condition ckpt); None = plain")
    a = ap.parse_args()
    policy = None
    if a.pi0_config:
        import os
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
        from abc_minimal.eval_policy import PI0_CAMERA_KEY_MAP, PI0_PROMPT
        from batched_policy import BatchedPi0
        policy = BatchedPi0(a.pi0_config, a.pi0_ckpt, PI0_PROMPT, PI0_CAMERA_KEY_MAP,
                            condition=a.obs_condition, cfg_weight=a.cfg_weight)
    t0 = time.perf_counter()
    res = run_batched(a.seeds, policy=policy, steps=a.steps, despawn_n=a.despawn, gpu=a.gpu)
    dt = time.perf_counter() - t0
    print(f"B={len(a.seeds)} steps={a.steps} wall={dt:.1f}s "
          f"({len(a.seeds)*a.steps/dt:.1f} world-steps/s incl. python)")
    for r in res.records:
        print(" ", json.dumps(r))
    if a.out:
        np.savez_compressed(a.out, qpos=res.qpos_traces, records=json.dumps(res.records))
    if a.arm and a.trace_dir:
        out_dir = write_trace_dir(a.trace_dir, a.arm, a.seeds, res,
                                  rand_records=res.rand_records, shard=a.shard,
                                  extra={"steps": a.steps, "despawn_n": a.despawn,
                                         "wall_seconds": round(dt, 1)})
        print(f"wrote {out_dir}/ ({len(a.seeds)} traces + summary.json)")
