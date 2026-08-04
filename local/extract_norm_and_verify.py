"""Extract the sim norm_stats from the lbm DiT-L checkpoint and verify the model loads.

- Writes {"norm_stats": {"state": {mean,std}, "actions": {mean,std}}} for the
  `sim_tasks_*` stat set (the mjwarp sim stats the model was trained with) so FT
  uses the SAME normalization as pretraining (mirrors the launcher's `ditl` mode).
- Builds DiTPolicy at the DiT-L config (1024/24/16) and runs load_pretrained to
  confirm 0 missing / 0 unexpected keys before we burn GPU hours.

  uv run python local/extract_norm_and_verify.py <ckpt> <out_norm_stats.json>
"""
import json
import sys

import torch

from abc_minimal.config import DiTConfig
from abc_minimal.dit import DiTPolicy, load_pretrained


def main():
    ckpt_path = sys.argv[1]
    out_path = sys.argv[2]

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    ns = ck["norm_stats"]
    sim_keys = [k for k in ns if k.startswith("sim_")]
    if len(sim_keys) != 1:
        raise SystemExit(f"expected exactly one sim_* norm_stats key, got {sim_keys}")
    key = sim_keys[0]
    sub = ns[key]
    blob = {
        "norm_stats": {
            "state": {"mean": list(sub["state"]["mean"]), "std": list(sub["state"]["std"])},
            "actions": {"mean": list(sub["actions"]["mean"]), "std": list(sub["actions"]["std"])},
        }
    }
    with open(out_path, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"[norm_stats] wrote {out_path} from checkpoint key '{key}'")

    # DiT-L config matches the launcher's DIT_L preset (1024/24/16).
    cfg = DiTConfig(hidden_size=1024, depth=24, num_heads=16)
    model = DiTPolicy(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"[model] DiTPolicy(1024/24/16) built: {n / 1e9:.3f}B params")
    load_pretrained(model, ckpt_path)  # raises on any missing/unexpected key
    print("[model] load_pretrained OK — 0 missing / 0 unexpected keys")


if __name__ == "__main__":
    main()
