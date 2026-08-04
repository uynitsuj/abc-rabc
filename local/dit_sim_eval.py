#!/usr/bin/env python3
"""Eval the finetuned ABC-DiT checkpoints in the yam_sim harness with the mjgl
(MuJoCo-GL) renderer -- the in-distribution renderer these models were trained on.

The yam_sim eval harness (yam_sim.eval.harness) drives a batched mjwarp env,
scores each episode with the task evaluators, and aggregates success rates. Its
built-in policy backend is the openpi/pi0 websocket client; its LBMPolicy backend
needs the (absent) `models.lbm` package. So this runner plugs a *local* DiT policy
-- the exact `abc_minimal.dit.DiTPolicy` the checkpoints were trained with -- into
the harness's per-(model,task) rollout entrypoint `_run_task_for_model`, which
already takes a `policy` object. Nothing in the shared harness is modified.

Run it from yam_sim's venv with abc-rabc on PYTHONPATH (see run_dit_eval.sh):
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=/home/karimelrafi/abc-rabc \
      /home/karimelrafi/yam_sim/.venv/bin/python local/dit_sim_eval.py ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# abc_minimal (the training repo) supplies the DiT architecture + CLIP + preprocess.
from abc_minimal.config import ClipConfig, DiTConfig
from abc_minimal.dit import CLIPTextEmbedder, DiTPolicy, load_pretrained, task_name_to_prompt
from abc_minimal.preprocess import normalize, parse_norm_stats, unnormalize

# yam_sim harness internals (reused verbatim; only the policy is swapped).
from yam_sim.eval.config import EvalConfig, ModelEndpoint, TaskSpecEntry
from yam_sim.eval.harness import _run_task_for_model
from yam_sim.eval.results import (
    aggregate,
    completed_keys,
    format_table,
    load_jsonl,
    write_summary,
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class BatchedDiTLocalPolicy:
    """Batched, in-process ABC-DiT policy matching the harness policy interface.

    `.infer(obs, action_prefix=, prefix_length=)` -> {"actions": (W, chunk, 14)}.
    Replicates abc_minimal/eval_policy.py's SimPolicy exactly (state normalize ->
    resize_pad + imagenet-normalize images -> CLIP task vec -> sample_actions[_rtc]
    -> unnormalize), but vectorized over W worlds instead of one obs at a time.
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        prompt: str,
        device: str = "cuda:0",
        diffusion_steps: int = 10,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        clip_cache_dir: str | None = None,
        norm_stats_path: str | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.diffusion_steps = diffusion_steps
        self.cfg = DiTConfig(hidden_size=hidden_size, depth=depth, num_heads=num_heads)
        self.model = DiTPolicy(self.cfg).to(self.device)
        # load_pretrained is strict (raises on any missing/unexpected key), so a
        # wrong architecture fails loudly rather than silently degrading.
        ckpt = load_pretrained(self.model, checkpoint)
        self.model.eval()
        if norm_stats_path:
            import json

            raw = json.loads(Path(norm_stats_path).read_text())
        else:
            raw = ckpt.get("norm_stats")
            if raw is None:
                raise ValueError(
                    f"{checkpoint} has no embedded norm_stats; pass --norm-stats-path"
                )
        self.norm_stats = parse_norm_stats(raw)

        clip_cfg = ClipConfig() if clip_cache_dir is None else ClipConfig(cache_dir=clip_cache_dir)
        self.embedder = CLIPTextEmbedder(clip_cfg, device=self.device)
        self.task_vec = self.embedder.encode([prompt]).to(self.device)  # (1, 512)
        self._mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    def set_task(self, task) -> None:  # harness/policy-interface compatibility
        return None

    def _prep_images(self, arr: np.ndarray) -> torch.Tensor:
        """(W, 3, H, Wd) uint8 -> (W, 3, 224, 224) resized+padded+imagenet-normalized.

        Identical math to abc_minimal.preprocess.resize_pad_normalize, batched."""
        x = torch.as_tensor(np.asarray(arr)).to(self.device).float()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if float(x.max()) > 1.0:
            x = x / 255.0
        _, _, h, w = x.shape
        target = 224
        ratio = max(w / target, h / target)
        nh = max(1, int(round(h / ratio)))
        nw = max(1, int(round(w / ratio)))
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
        pad_h0 = (target - nh) // 2
        pad_h1 = target - nh - pad_h0
        pad_w0 = (target - nw) // 2
        pad_w1 = target - nw - pad_w0
        x = F.pad(x, (pad_w0, pad_w1, pad_h0, pad_h1), value=0)
        return (x - self._mean) / (self._std + 1e-6)

    def _norm_prefix(self, action_prefix: np.ndarray, prefix_length: int, W: int) -> torch.Tensor:
        """(W, prefix_length, 14) raw joint prefix -> (W, chunk, 14) normalized+padded."""
        ap = np.asarray(action_prefix, dtype=np.float32)
        if ap.ndim == 2:  # (prefix_length, 14) -> broadcast to all worlds
            ap = np.repeat(ap[None], W, axis=0)
        pl = int(prefix_length)
        full = np.zeros((W, self.cfg.chunk_length, self.cfg.action_dim), dtype=np.float32)
        full[:, :pl] = normalize(ap[:, :pl], self.norm_stats["actions"])
        return torch.from_numpy(full).to(device=self.device, dtype=self.model.y_embedder.weight.dtype)

    @torch.no_grad()
    def infer(
        self,
        obs: dict,
        *,
        noise=None,
        action_prefix=None,
        prefix_length=None,
        latency=None,
    ) -> dict:
        state = np.asarray(obs["state"], dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]
            squeezed = True
        else:
            squeezed = False
        W = state.shape[0]

        state_n = normalize(state, self.norm_stats["state"])
        batch = {
            "state": torch.from_numpy(state_n).float().to(self.device),
            "actions": torch.zeros(
                W, self.cfg.chunk_length, self.cfg.action_dim, device=self.device
            ),
            "images": {cam: self._prep_images(obs["images"][cam]) for cam in self.cfg.camera_keys},
            "task_vec_clip": self.task_vec.expand(W, -1),
        }
        noise_t = None
        if noise is not None:
            noise_np = np.asarray(noise, dtype=np.float32)
            if noise_np.ndim == 2:
                noise_np = np.repeat(noise_np[None], W, axis=0)
            noise_t = torch.from_numpy(noise_np).to(
                device=self.device, dtype=self.model.y_embedder.weight.dtype
            )

        if action_prefix is None or prefix_length in (None, 0):
            actions = self.model.sample_actions(
                batch, num_steps=self.diffusion_steps, noise=noise_t
            )
        else:
            prefix_t = self._norm_prefix(action_prefix, prefix_length, W)
            actions = self.model.sample_actions_rtc(
                batch,
                prefix_t,
                prefix_length=int(prefix_length),
                num_steps=self.diffusion_steps,
                noise=noise_t,
            )

        actions_np = actions.float().detach().cpu().numpy()
        actions_np = unnormalize(actions_np, self.norm_stats["actions"]).astype(np.float32)
        if squeezed:
            return {"actions": actions_np[0]}
        return {"actions": actions_np}


def build_config(args, task_prompt: str) -> tuple[EvalConfig, TaskSpecEntry]:
    entry = TaskSpecEntry(
        task=args.task,
        prompt=task_prompt,
        episodes=args.episodes,
        num_worlds=args.num_worlds,
        max_seconds=args.max_seconds,
    )
    config = EvalConfig(
        tasks=[entry],
        models=[],  # policies are injected directly; harness model loop is bypassed
        output_dir=args.output_dir,
        episodes_default=args.episodes,
        num_worlds_default=args.num_worlds,
        max_seconds=args.max_seconds,
        scene=args.scene,
        camera_backend=args.camera_backend,  # "mujoco" == mjgl (in-distribution)
        execute_chunk_dim=args.execute_chunk_dim,
        prefix_length=args.prefix_length,
        use_ttrtc=args.use_ttrtc,
        jpeg_quality=args.jpeg_quality,
        video=args.video,
        save_states=args.save_states,
        camera_gpu_id=args.camera_gpu_id,
        seed_base=args.seed_base,
        force_object_count=args.force_object_count,
    )
    return config, entry


def main() -> None:
    ap = argparse.ArgumentParser(description="mjgl DiT sim eval (norabc vs rabc)")
    ap.add_argument("--bc-ckpt", required=True, help="norabc (vanilla BC) checkpoint .pt")
    ap.add_argument("--rabc-ckpt", required=True, help="RABC checkpoint .pt")
    ap.add_argument("--task", default="put_bottles")
    ap.add_argument(
        "--sim-task-name",
        default="sim_put_the_plastic_bottles_in_the_bin",
        help="training task_name; the CLIP prompt is task_name_to_prompt() of this",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--num-worlds", type=int, default=25)
    ap.add_argument("--max-seconds", type=float, default=128.0)
    ap.add_argument("--execute-chunk-dim", type=int, default=20)
    ap.add_argument("--prefix-length", type=int, default=3)
    ap.add_argument("--use-ttrtc", action="store_true", help="action-prefix RTC (default off)")
    ap.add_argument("--diffusion-steps", type=int, default=10)
    ap.add_argument("--scene", default="hybrid")
    ap.add_argument("--camera-backend", default="mujoco", choices=["mujoco", "mjwarp", "madrona"])
    ap.add_argument("--camera-gpu-id", type=int, default=0)
    ap.add_argument("--force-object-count", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=20260511)
    ap.add_argument("--jpeg-quality", type=int, default=0)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--save-states", action="store_true")
    ap.add_argument("--norm-stats-path", default=None)
    ap.add_argument(
        "--clip-cache-dir",
        default="/scratch/current/karimelrafi/abc_cache/_shared/clip",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--only",
        default=None,
        choices=[None, "bc", "rabc"],
        help="run only one arm (default both)",
    )
    args = ap.parse_args()

    prompt = task_name_to_prompt(args.sim_task_name)
    print(f"[dit-eval] CLIP prompt = {prompt!r}", flush=True)
    config, entry = build_config(args, prompt)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done = completed_keys(load_jsonl(results_path))

    arms = [("dit_bc", args.bc_ckpt), ("dit_rabc", args.rabc_ckpt)]
    if args.only == "bc":
        arms = [arms[0]]
    elif args.only == "rabc":
        arms = [arms[1]]

    for label, ckpt in arms:
        print(f"\n[dit-eval] === {label} === {ckpt}", flush=True)
        torch.manual_seed(0)  # same policy-noise stream for both arms
        np.random.seed(0)
        policy = BatchedDiTLocalPolicy(
            ckpt,
            prompt=prompt,
            device=args.device,
            diffusion_steps=args.diffusion_steps,
            clip_cache_dir=args.clip_cache_dir,
            norm_stats_path=args.norm_stats_path,
        )
        model = ModelEndpoint(label=label, host="local", port=0)
        _run_task_for_model(config, model, entry, policy, results_path, done)
        del policy
        torch.cuda.empty_cache()

    all_results = load_jsonl(results_path)
    summary = aggregate(all_results)
    json_path, csv_path = write_summary(out_dir, summary)
    print("\n" + format_table(summary))
    print(f"\n[dit-eval] summary -> {json_path}\n[dit-eval] results  -> {results_path}", flush=True)


if __name__ == "__main__":
    main()
