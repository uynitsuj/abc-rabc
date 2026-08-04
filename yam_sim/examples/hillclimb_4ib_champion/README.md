# Hill-Climb 4-IB Champion: C36 Reproduction

**Config:** 250c + exec15 + alpha=0.7 + all-green-bottles + compound visual fixes

**Results (25 seeds):** 68% 4-IB, 48% 5-IB, 24% 6-IB

## Quick Start

### 1. Setup

```bash
# Clone abc and install
git clone git@github.com:hgaurav2k/abc.git && cd abc
git checkout arthur/mar3/hillclimb-4ib
bash setup_dev.sh

# Login to W&B (for checkpoint download)
uv run python -c "import wandb; wandb.login(host='https://far.wandb.io')"
```

### 2. Run Evaluation

```bash
# Full 25-seed eval (requires GPU + MUJOCO_GL=egl)
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl bash path/to/run_eval.sh

# Quick 5-seed test
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl bash path/to/run_eval.sh 5 0
```

### 3. Replay Recorded Episode

The `actions.npy` file contains a 6-IB success rollout (seed=6, all 6 bottles deposited):

```python
import numpy as np, json

actions = np.load("actions.npy")           # (3750, 14)
qpos = np.load("initial_qpos.npy")        # (65,)
qvel = np.load("initial_qvel.npy")        # (58,)
config = json.load(open("config.json"))

print(f"Actions: {actions.shape}")
print(f"Bottles in bin: {config['num_bottles_in_bin']}")
print(f"Config: {config['num_chunks']}c exec{config['execute_dim']} alpha={config['bottle_alpha']}")
```

For visual replay, use `viser_replay.py` from the parent directory.

## Champion Config Details

| Parameter | Value | Why |
|-----------|-------|-----|
| num_chunks | 250 | Longer horizon = more pick-deposit cycles |
| execute_chunk_dim | 15 | Breaks mode collapse (re-plans every 15 steps) |
| bottle_opacity | 0.7 | Optimal: not too opaque, not too transparent |
| all_green_bottles | True | Fixes invisible bottle_3 (white on gray table) |
| match_training_resolution | True | 224x168 before JPEG matches training ETL |
| camera_noise | 3.0 | Visual fidelity compound effect |
| gaussian_blur | 3 | Visual fidelity compound effect |
| no_shadows | True | Visual fidelity compound effect |
| randomize_bottles | True | Random positions per seed |
| diffusion_steps | 5 | Standard inference quality |

## Checkpoint

`far-wandb/FAR-abc/far-abc-XGYkwlRVLgwVCKSU-last:v200`

Downloads automatically on first run (~17GB). Requires W&B login to `far.wandb.io`.

## Key Findings

1. **4 levers stack super-additively:** exec15 + horizon + alpha=0.7 + green = 2.8x baseline
2. **56% 4-IB ceiling** without visibility fix — broken to 68% by all-green-bottles
3. **bottle_3 was invisible** (3.5% contrast at alpha=0.5 on gray table)
4. **exec15 + prefix_length=1 conflict** — never combine both reductions
