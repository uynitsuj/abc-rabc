# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mcap", "mcap-protobuf-support", "boto3"]
# ///
"""One-off probe: alignment shifts for the k=2 holdout episode (idx 3448)."""
import json
from pathlib import Path

import boto3
import numpy as np

from states_actions_from_mcap import read_position_streams

UUID = "episode_019e2c31-0b09-7e01-8940-112c35bc2243"

d = Path("/mnt/data/warp_rm/gt_validation/deliv") / UUID
man = json.load(open(Path(__file__).parent / "render_manifest_B.json"))
info = next(v for v in man.values() if v["uuid"] == UUID)
mc = Path("/tmp/probe_3448.mcap")
if not mc.exists():
    boto3.client("s3").download_file(
        "xdof-bair-abc",
        f"data/deliveries/{info['delivery']}/sim_put_the_plastic_bottles_in_the_bin/"
        f"{UUID}/output.mcap", str(mc))
integ = np.load(d / "integration_state.npy")
qpos = integ[:, 1:]
n_rows = integ.shape[0]
streams = read_position_streams(mc)
print("n_rows", n_rows, {t: len(s) for t, s in streams.items()})
lp = streams["/left-arm-proprio"][:, :6]
rp = streams["/right-arm-proprio"][:, :6]
m = n_rows // 2


def find6(t6, lo):
    best = None
    for off in range(lo, qpos.shape[1] - 5):
        dd = float(np.abs(qpos[m, off:off + 6] - t6).max())
        if best is None or dd < best[1]:
            best = (off, dd)
    return best


loff, ld = find6(lp[m - 1], 0)
roff, rd = find6(rp[m - 2], loff + 6)
print("loff", loff, round(ld, 8), "roff", roff, round(rd, 8))
for tag, s, off in (("L", lp, loff), ("R", rp, roff)):
    n = len(s)
    for shift in range(0, n_rows - n + 1):
        resid = float(np.abs(s - qpos[shift:shift + n, off:off + 6]).max())
        print(f"{tag} n={n} shift={shift} resid={resid:.3e}")
