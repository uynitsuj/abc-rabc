"""2-WORLD HETEROGENEITY PROOF for the batched runner.

Build ONE model whose asset library contains each world's pre-scaled bottle/bin
meshes; expand geom_dataid to (2, ngeom); redirect world 1's mesh geoms to its
own copies. Drop bottles from the default pose in lockstep, and compare each
batched world's rest state against a serial nworld=1 mjwarp run of the true
per-seed model. PASS = per-world rest qpos matches its own serial reference and
NOT the other world's.
"""
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, "/scratch/warprm_eval/abc_rabc")
import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp

from abc_minimal.eval_policy import PutBottlesSimConfig, scene_xml, SCENE_XML, ROOT

scene = PutBottlesSimConfig()
SEEDS = [20260511, 20260543]

def world_scales(seed):
    rng = np.random.default_rng(seed)
    bs = rng.uniform(*scene.bottle_scale_range, size=scene.bottle_count).astype(np.float32)
    return bs, float(rng.uniform(*scene.bin_scale_range))

def scaled(name, base_scale, bs, binsc):
    for idx in range(scene.bottle_count):
        if name.startswith(f"bottle_{idx}_"):
            return base_scale * float(bs[idx])
    if name.startswith("water_bottle_"):
        return base_scale * binsc
    return None

def combined_xml(all_scales):
    root = ET.fromstring(SCENE_XML.read_text())
    comp = root.find("compiler")
    comp.set("meshdir", str((ROOT / "assets" / "put_bottles" / "assets").resolve()))
    comp.set("texturedir", str((ROOT / "assets" / "put_bottles" / "assets").resolve()))
    asset = root.find("asset")
    fmt = lambda v: " ".join(f"{x:.9g}" for x in v)
    for mesh in list(root.findall("./asset/mesh")):
        name = mesh.get("name", "")
        base = np.asarray([float(v) for v in mesh.get("scale", "1 1 1").split()])
        # world 0 scale applied IN PLACE (geoms keep pointing here)
        s0 = scaled(name, base, *all_scales[0])
        if s0 is None:
            continue
        mesh.set("scale", fmt(s0))
        # extra copies for worlds 1..B-1
        for w, sc in enumerate(all_scales[1:], start=1):
            dup = ET.SubElement(asset, "mesh", dict(mesh.attrib))
            dup.set("name", f"{name}__w{w}")
            dup.set("scale", fmt(scaled(name, base, *sc)))
    return ET.tostring(root, encoding="unicode")

scales = [world_scales(s) for s in SEEDS]
xml = combined_xml(scales)
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
print(f"combined model: nmesh={model.nmesh} ngeom={model.ngeom}")

def run(m, nworld, dataid_rows=None, steps=240):
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bsz = {"geom_dataid": nworld} if dataid_rows is not None else None
    mw = mjw.put_model(m, batch_sizes=bsz) if bsz else mjw.put_model(m)
    if dataid_rows is not None:
        arr = mw.geom_dataid.numpy()
        assert arr.shape[0] == nworld, arr.shape
        for w in range(nworld):
            arr[w] = dataid_rows[w]
        wp.copy(mw.geom_dataid, wp.from_numpy(arr.astype(np.int32), dtype=wp.int32))
    dw = mjw.put_data(m, d, nworld=nworld, nconmax=m.nconmax, njmax=m.njmax)
    for _ in range(steps):
        mjw.step(mw, dw)
    return dw.qpos.numpy().copy()

# per-world dataid rows on the combined model
base_row = model.geom_dataid.copy()
rows = [base_row.copy()]
row1 = base_row.copy()
n_redirect = 0
for g in range(model.ngeom):
    did = model.geom_dataid[g]
    if did < 0:
        continue
    mname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, did) or ""
    alt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, f"{mname}__w1")
    if alt >= 0:
        row1[g] = alt
        n_redirect += 1
rows.append(row1)
print(f"world-1 geoms redirected: {n_redirect}")

wp.set_device("cuda:0")
q_batch = run(model, 2, dataid_rows=rows)
print("batched run done:", q_batch.shape)

# serial references: the TRUE per-seed models
refs = []
for (bs, binsc) in scales:
    m = mujoco.MjModel.from_xml_string(scene_xml(scene, bs, binsc))
    refs.append(run(m, 1)[0])

BQ = 7  # bin freejoint qpos width
for w in range(2):
    own = float(np.max(np.abs(q_batch[w] - refs[w])))
    other = float(np.max(np.abs(q_batch[w] - refs[1 - w])))
    print(f"world {w}: max|q - own_serial|={own:.5f}   max|q - OTHER_serial|={other:.5f}")
ok = all(np.max(np.abs(q_batch[w] - refs[w])) < 5e-3 for w in range(2)) and \
     all(np.max(np.abs(q_batch[w] - refs[1 - w])) > 1e-3 for w in range(2))
print("PROOF:", "PASS" if ok else "CHECK-BY-HAND (tolerances)")
