# ABC public simulator support for WARP-RM

This release contains the public simulator-side code needed to inspect the
WARP-RM paper's bottle-in-bin simulation results:

- `prepare.py` downloads the original public ABC bottles data, including the
  original 30 Hz image/action/state episodes.
- `abc_minimal/` and `eval_policy.py` provide the MuJoCo-Warp simulator and
  the Pi0 websocket policy client.
- `score_bottles.py` is the offline, full-horizon scorer used for the paper
  metrics.

It intentionally excludes private staging trees, raw source captures,
internal object storage, rerender workers, experiment logs, and intermediate
checkpoints.

## Reproduce the published table from public traces

The exact, canonical paper-A qpos traces are public at
[`uynitsuj/paper-sim-n128-traces`](https://huggingface.co/datasets/uynitsuj/paper-sim-n128-traces).
They consist of 128 matched worlds per arm (vanilla and WARP-RM), at the full
60-second horizon.

```bash
python -m venv .venv
.venv/bin/pip install numpy huggingface_hub
hf download uynitsuj/paper-sim-n128-traces --repo-type dataset \
  --local-dir paper-sim-n128-traces
.venv/bin/python score_bottles.py --trace-dir paper-sim-n128-traces --self-test
```

The self-test checks every reported paper-table cell and the robustness ladder.
For the paper metric, a bottle must have any of five points along its axis in a
rim-tight bin cylinder for at least 0.5 seconds and still be in the bin at the
60-second horizon. This is intentionally stricter than the evaluator's live
center-point count, which is only used for rollout control.

## Public data and re-rendered evaluation assets

`prepare.py --full` retrieves the original public ABC source episodes. The
WARP-RM paper dataset is published separately at
[`uynitsuj/sim-bottles-mjwarp-v1`](https://huggingface.co/datasets/uynitsuj/sim-bottles-mjwarp-v1).
It preserves those source episodes' actions, states, split assignment, scenes,
and camera setup, while replaying the available full qpos at 30 Hz through the
MuJoCo-Warp evaluation renderer. Consequently, it is trajectory-aligned with
the original source but has renderer-different pixels. It also contains the
portable full-state/scene supplement needed for audit and rerendering.

```bash
uv sync
uv run prepare.py --full
hf download uynitsuj/sim-bottles-mjwarp-v1 --repo-type dataset \
  --local-dir sim-bottles-mjwarp-v1
```

Two of the 2,438 source episodes do not have the required full state and scene
material and retain their original MJGL frames; this is documented in the
dataset card and manifest.

## Fresh rollout evaluation

The released policy parameter artifacts are at
[`uynitsuj/paper-sim-policy-checkpoints`](https://huggingface.co/uynitsuj/paper-sim-policy-checkpoints).
They are complete serving parameter trees, with optimizer and training state
omitted. Start the public
[OpenPI `release-candidate`](https://github.com/uynitsuj/openpi/tree/release-candidate)
server under the applicable upstream Pi0 terms, then point this evaluator at
it with `--policy-backend pi0`, save full qpos traces, and score them with
`score_bottles.py`.

The released canonical traces above are the definitive deterministic paper
table artifact. Fresh rollouts are a separate stochastic reproduction check;
they should be compared using the paired n=128 protocol rather than expected
to be bit-identical to the archived traces.

## License and third-party material

The code in this repository is licensed under Apache-2.0. The `put_bottles`
asset tree includes the i2rt YAM robot model under its bundled MIT license.
The optional ABC-DiT path uses DINOv3 code; DINOv3 weights are not included and
must be obtained under Meta's terms.

## Citation

```bibtex
@article{abc2026,
  title   = {Scalable Behavior Cloning with Open Data, Training, and Evaluation},
  year    = {2026},
  journal = {arXiv preprint},
  url     = {https://abc.bot/}
}
```
