"""Probe the EpisodeDataset for a per-sample RSS leak (single process).

Loops __getitem__ over random indices and prints RSS so we can see whether memory
grows linearly with samples (a leak) and isolate the source (torchcodec decoder).
"""
import os
import resource
import sys

import numpy as np

from abc_minimal.config import DiTConfig
from abc_minimal.preprocess import load_norm_stats
from abc_minimal.train_loop import EpisodeDataset


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def main():
    data_dir = sys.argv[1]
    norm_path = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    cfg = DiTConfig(hidden_size=1024, depth=24, num_heads=16)
    ns = load_norm_stats(norm_path)
    ds = EpisodeDataset(data_dir, ns, train=True,
                        default_task_name="sim_put_the_plastic_bottles_in_the_bin",
                        mask_state_ratio=0.1, model_config=cfg)
    rng = np.random.default_rng(0)
    print(f"len(ds)={len(ds)}  start RSS={rss_gb():.2f} GB", flush=True)
    for i in range(1, n + 1):
        idx = int(rng.integers(0, len(ds)))
        _ = ds[idx]
        if i % 50 == 0:
            print(f"  {i:4d} samples  RSS={rss_gb():.2f} GB", flush=True)
    print(f"end RSS={rss_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
