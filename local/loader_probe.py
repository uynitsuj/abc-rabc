"""Probe the multiprocess DataLoader (workers + pin_memory + persistent) for RSS
growth — replicates train_loop's loader exactly, no model/GPU. Samples total RSS
(this process + all children) so a worker-side leak is visible.

  uv run python local/loader_probe.py <train_dir> <norm_stats> [num_workers] [batches] [pin]
"""
import os
import sys
from functools import partial

import numpy as np

from abc_minimal.config import DiTConfig
from abc_minimal.preprocess import load_norm_stats
from abc_minimal.train_loop import EpisodeDataset, MixtureDataset, collate
from torch.utils.data import DataLoader


def tree_rss_gb():
    pids = [os.getpid()]
    try:
        import subprocess
        out = subprocess.run(["bash", "-c", f"pgrep -P {os.getpid()}"], capture_output=True, text=True).stdout
        pids += [int(x) for x in out.split()]
    except Exception:
        pass
    total = 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1]); break
        except Exception:
            pass
    return total / 1024 / 1024


def main():
    train_dir = sys.argv[1]
    norm_path = sys.argv[2]
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    nbatches = int(sys.argv[4]) if len(sys.argv) > 4 else 600
    pin = (sys.argv[5] != "0") if len(sys.argv) > 5 else True

    cfg = DiTConfig(hidden_size=1024, depth=24, num_heads=16)
    ns = load_norm_stats(norm_path)
    ds = EpisodeDataset(train_dir, ns, train=True,
                        default_task_name="sim_put_the_plastic_bottles_in_the_bin",
                        mask_state_ratio=0.1, model_config=cfg)
    mix = MixtureDataset([ds], [1.0], len(ds))
    loader = DataLoader(mix, batch_size=48, shuffle=True, num_workers=nw,
                        collate_fn=partial(collate, camera_keys=cfg.camera_keys),
                        pin_memory=pin, drop_last=True,
                        persistent_workers=nw > 0,
                        prefetch_factor=2 if nw > 0 else None)
    print(f"nw={nw} pin={pin} nbatches={nbatches}", flush=True)
    it = iter(loader)
    for i in range(1, nbatches + 1):
        _ = next(it)
        if i % 50 == 0:
            print(f"  {i:4d} batches  tree_RSS={tree_rss_gb():.2f} GB", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
