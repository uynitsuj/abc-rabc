# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mcap", "mcap-protobuf-support", "boto3"]
# ///
"""Empirically determine mcap<->integration_state alignment for B episodes.

The A-validated rule (n_msgs == n_rows-1, prepend first) failed on at least one B ep with
n_msgs == n_rows. This probe downloads given episodes' output.mcap + integration_state.npy
and, using the 12 arm joints (6 per arm; grippers excluded — different qpos parameterization),
measures the residual between the proprio message stream and candidate integration_state row
alignments:

  A) msgs align to rows[1:]  (validated A rule: prepend first  -> sa rows == n_rows)
  B) msgs align to rows[:n_msgs] (aligned from row 0; if n_msgs==n_rows this is identity)
  C) msgs[:-1] align to rows[1:] (extra trailing message: drop last + prepend first)

integration_state row format: col 0 = time, cols 1:1+nq = qpos. Arm joints assumed at the
start of qpos: [left 6, left gripper?, right 6, ...] — we auto-locate the 6+6 arm columns by
matching the FIRST proprio message against the row-0.. qpos columns (search offset).
"""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from states_actions_from_mcap import read_position_streams

BUCKET = "xdof-bair-abc"
SUB = "sim_put_the_plastic_bottles_in_the_bin"

def probe(uuid: str, delivery: str):
    import boto3
    s3 = boto3.client("s3")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for fn in ("output.mcap", "integration_state.npy"):
            s3.download_file(BUCKET, f"data/deliveries/{delivery}/{SUB}/{uuid}/{fn}", str(td / fn))
        integ = np.load(td / "integration_state.npy")  # (n_rows, 1+nq+...)
        streams = read_position_streams(td / "output.mcap")
    lp = streams["/left-arm-proprio"][:, :6]   # (n_msgs, 6) arm joints only
    rp = streams["/right-arm-proprio"][:, :6]
    n_msgs, n_rows = lp.shape[0], integ.shape[0]
    qpos = integ[:, 1:]  # drop time col

    # locate left-arm 6 cols and right-arm 6 cols in qpos by matching a mid msg under rule A
    # try candidate offsets for left arm start and right arm start
    probe_msg = n_msgs // 2
    def find6(target6, row):
        best = None
        for off in range(0, min(60, qpos.shape[1] - 6)):
            d = np.abs(qpos[row, off:off+6] - target6).max()
            if best is None or d < best[1]:
                best = (off, d)
        return best
    # use rule-A hypothesized row (msg i -> row i+1), fall back to i if out of range
    row = probe_msg + 1 if probe_msg + 1 < n_rows else probe_msg
    loff, ld = find6(lp[probe_msg], row)
    roff_best = None
    for off in range(loff + 6, min(80, qpos.shape[1] - 6)):
        d = np.abs(qpos[row, off:off+6] - rp[probe_msg]).max()
        if roff_best is None or d < roff_best[1]:
            roff_best = (off, d)
    roff, rd = roff_best
    prop = np.concatenate([lp, rp], axis=1)  # (n_msgs, 12)
    def qrows(sl):
        return np.concatenate([qpos[sl, loff:loff+6], qpos[sl, roff:roff+6]], axis=1)

    out = {"uuid": uuid, "n_msgs": n_msgs, "n_rows": n_rows, "loff": loff, "roff": roff,
           "locate_resid": (round(float(ld), 8), round(float(rd), 8))}
    # rule A: msgs -> rows[1:1+n_msgs] (requires n_msgs <= n_rows-1)
    if n_msgs <= n_rows - 1:
        out["ruleA_prepend"] = round(float(np.abs(prop - qrows(slice(1, 1 + n_msgs))).max()), 8)
    # rule B: msgs -> rows[0:n_msgs]
    if n_msgs <= n_rows:
        out["ruleB_align0"] = round(float(np.abs(prop - qrows(slice(0, n_msgs))).max()), 8)
    # rule C: msgs[:-1] -> rows[1:], i.e. drop last msg (requires n_msgs-1 <= n_rows-1)
    if n_msgs - 1 <= n_rows - 1:
        out["ruleC_droplast_prepend"] = round(float(np.abs(prop[:-1] - qrows(slice(1, n_msgs))).max()), 8)
    print(json.dumps(out))

if __name__ == "__main__":
    man = json.loads(Path(sys.argv[1]).read_text())
    for uuid_arg in sys.argv[2:]:
        info = next(v for v in man.values() if v["uuid"] == uuid_arg)
        probe(uuid_arg, info["delivery"])
