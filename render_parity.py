"""Render parity: serial env's own render (mjwarp B=1) vs the batched context,
same seed, same qpos, same cam. Decides the cam_res/reshape orientation
question empirically and validates the policy-input image path end to end.
"""
import sys

sys.path.insert(0, "/scratch/warprm_eval/abc_rabc")
import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp

from batched_runner import combined_model, serial_reset_rows
from abc_minimal.eval_policy import PutBottlesEnv, PutBottlesSimConfig

scene = PutBottlesSimConfig()
SEEDS = [20260511, 20260543]
H, W = 168, 224
wp.set_device("cuda:0")

# --- serial reference: the env's own render path ---
env = PutBottlesEnv(height=H, width=W, camera_keys=("top", "left", "right"),
                    prompt="", scene=scene, gpu_id=0)
env.reset(seed=SEEDS[0])
serial_imgs = env.render_cameras()          # {name: CHW uint8}
top_serial = serial_imgs["top"].astype(int)  # [3,H,W]
env.close()

# --- batched: same seed in world 0 ---
model, rows = combined_model(scene, SEEDS)
q0, _ = serial_reset_rows(scene, SEEDS)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
mw = mjw.put_model(model, batch_sizes={"geom_dataid": len(SEEDS)})
arr = mw.geom_dataid.numpy(); arr[:] = rows
wp.copy(mw.geom_dataid, wp.from_numpy(arr.astype(np.int32), dtype=wp.int32))
dw = mjw.put_data(model, data, nworld=len(SEEDS), nconmax=model.nconmax, njmax=model.njmax)
wp.copy(dw.qpos, wp.from_numpy(np.ascontiguousarray(q0, dtype=np.float32), dtype=wp.float32))
mjw.forward(mw, dw)

ctx = mjw.create_render_context(mjm=model, nworld=len(SEEDS), cam_res=(W, H),
                                render_rgb=[True] * model.ncam,
                                render_depth=[False] * model.ncam,
                                use_textures=True, use_shadows=True)
mjw.refit_bvh(mw, dw, ctx)
mjw.render(mw, dw, ctx)
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
flat = ctx.rgb_data.numpy().view(np.uint8)

def cand(shape_hw):
    h, w = shape_hw
    rgba = flat.reshape(len(SEEDS), model.ncam, h, w, 4)
    return rgba[0, cam_id, :, :, :3].transpose(2, 0, 1).astype(int)  # CHW

for label, hw in [("HW=(168,224)", (H, W)), ("HW=(224,168)", (W, H))]:
    try:
        img = cand(hw)
        if img.shape != top_serial.shape:
            img = img.transpose(0, 2, 1)
        d = np.abs(img - top_serial)
        print(f"{label}: mean|diff|={d.mean():.2f} frac>10={float((d.sum(0) > 30).mean()):.4f}")
    except Exception as e:
        print(f"{label}: reshape failed: {e}")

img = cand((H, W))
d = np.abs(img - top_serial)
ok = d.mean() < 8.0
print(f"PARITY {'PASS' if ok else 'FAIL'}: batched (H=168,W=224) vs serial mean|diff|={d.mean():.2f}")
sys.exit(0 if ok else 1)
