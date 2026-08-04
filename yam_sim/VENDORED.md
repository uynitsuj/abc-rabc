# Vendored yam_sim

Vendored from the internal `yam_sim` repo (commit `221d38b`, 2026-08) so
abc-rabc is self-contained: sim environments, randomization, task evaluators,
batched mjwarp env, and eval harness for the six sim tasks — no second repo
needed to run eval.

Local changes vs upstream:

- `rendering/replay/renderer.py`: `_create_array_with_safe_sizes` accepts the
  trailing `batch_size` arg added in mujoco_warp 3.10 (works on 3.9 and 3.10).
- Pruned asset packs for tasks outside the release scope (~140 MB):
  `models/assets/{chess,chess2,task_chess,ball_sorting_toy,letters,building_blocks,blocks,jenga}`
  and `models/old_scenes/`. Their scene XMLs remain, so constructing those
  envs will fail on missing assets; the six release tasks are unaffected
  (verified by `local/rerender/smoke_test_envs.py`: build + randomized reset +
  step + batched mjwarp camera render for all six).

Extra deps beyond upstream abc-rabc: `gymnasium`, `opencv-python`
(declared in the root `pyproject.toml`).

Usage (from the repo root, so `import yam_sim` resolves here):

```python
import yam_sim
env = yam_sim.make_batched_env(scene="hybrid", task="put_bottles", num_worlds=8,
                               camera_backend="mjwarp", camera_gpu_id=0)
```

Eval entrypoints: `yam_sim/eval/harness.py` (`run_eval`), task suite config in
`yam_sim/eval/configs/sim_suite.yaml`, DiT policy glue in `local/dit_sim_eval.py`
(launched by `local/run_dit_eval.sh`).
