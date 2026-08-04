"""Base policy interface for running trained models in sim."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from termcolor import cprint

from yam_sim.transforms import Normalize, Unnormalize


def _download_wandb_checkpoint(
    artifact_ref: str, root: str = "checkpoints"
) -> Optional[str]:
    """Download a W&B artifact checkpoint. Requires wandb to be installed."""
    import wandb

    artifact_name = artifact_ref.split(":")[0].split("/")[-1]
    version = artifact_ref.split(":")[1]

    if version != "latest":
        download_path = os.path.join(root, artifact_name, f"version_{version}")
        if os.path.exists(download_path):
            return download_path

    api = wandb.Api()
    art = api.artifact(artifact_ref, type="checkpoint")
    version = art.version
    download_path = os.path.join(root, artifact_name, f"version_{version}")
    if os.path.exists(download_path):
        return download_path

    os.makedirs(download_path, exist_ok=True)
    print(f"Downloading artifact to {download_path}")
    art.download(root=download_path)
    return download_path


@dataclass
class PolicyConfig:
    """Shared config fields used by all policy types."""

    config: Optional[str] = None
    dir: Optional[str] = None
    ckpt_path: Optional[str] = None
    policy_type: Optional[str] = "lbm"
    model_size: Optional[str] = "dit_B"
    state_dim: Optional[int] = 14
    action_dim: Optional[int] = 14
    chunk_len: Optional[int] = 30
    mask_state: Optional[bool] = False
    unnormalize_actions: Optional[bool] = True
    norm_stats_path: Optional[str] = None
    normalize_preset: str = "clip"
    """Image normalization preset: 'clip' for current checkpoints, 'imagenet' for older ones."""


class BasePolicy:
    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg
        if self.cfg.ckpt_path is None and self.cfg.dir is not None:
            self.cfg.ckpt_path = self.cfg.dir
        self.device = "cuda:0"
        self._task_vec_h = None
        self._task_str: Optional[str] = None
        self.chunk_len = cfg.chunk_len
        self.action_dim = 14
        self.use_quantile_norm = False

        ckpt_path = self._resolve_checkpoint()
        print(f"Loading {str(self.cfg.policy_type).upper()} {self.cfg.model_size}...")
        self._build_model(ckpt_path)
        cprint(
            f"Loaded {str(self.cfg.policy_type).upper()} {self.cfg.model_size} "
            f"({self.model.compute_num_parameters()/1e6:.2f}M params)",
            "cyan",
        )
        ckpt = self._load_weights(ckpt_path)
        self._post_load_setup()
        self._load_norm_stats(ckpt)
        self.normalizer = Normalize(
            norm_stats=self.norm_stats,
            use_quantiles=self.use_quantile_norm,
            strict=False,
        )
        self.unnormalizer = Unnormalize(
            norm_stats=self.norm_stats,
            use_quantiles=self.use_quantile_norm,
            strict=False,
        )

    def _resolve_checkpoint(self) -> str:
        checkpoints_root = "checkpoints"
        ckpt_path: Optional[str] = None
        if self.cfg.ckpt_path is None:
            raise ValueError("No checkpoint provided")
        if self.cfg.ckpt_path.startswith("far-wandb/FAR-abc"):
            artifact_path = _download_wandb_checkpoint(
                artifact_ref=self.cfg.ckpt_path, root=checkpoints_root
            )
            ckpt_path = os.path.join(artifact_path, "last.pt")
            if not os.path.exists(ckpt_path):
                pts = [p for p in os.listdir(artifact_path) if p.endswith(".pt")]
                if not pts:
                    raise FileNotFoundError(
                        f"Could not find any .pt in {artifact_path}"
                    )
                ckpt_path = os.path.join(artifact_path, pts[0])
        else:
            ckpt_path = self.cfg.ckpt_path
        return ckpt_path

    def _load_weights(self, ckpt_path: str) -> dict:
        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )

        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt["state_dict"]

        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        state_dict = self._transform_state_dict(state_dict)

        missing_keys, unexpected_keys = self.model.load_state_dict(
            state_dict, strict=False
        )
        allowed_missing = {"clip_mean", "clip_std"}
        missing_keys = [k for k in missing_keys if k not in allowed_missing]
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "Error(s) in loading state_dict for "
                f"{self.model.__class__.__name__}: "
                f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
            )

        self.max_episode_length = ckpt.get("max_episode_length")
        lightweight_ckpt = {
            "norm_stats": ckpt.get("norm_stats"),
            "max_episode_length": ckpt.get("max_episode_length"),
            "step": ckpt.get("step"),
        }
        del ckpt
        self.model.eval()
        return lightweight_ckpt

    def _transform_state_dict(self, state_dict: dict) -> dict:
        """Optional checkpoint-key remapping hook for subclass compatibility."""
        return state_dict

    def _post_load_setup(self) -> None:
        pass

    def _load_norm_stats(self, ckpt: dict) -> None:
        if self.cfg.norm_stats_path is not None:
            with open(self.cfg.norm_stats_path, "r") as f:
                norm_stats_data = json.load(f)
            self.norm_stats = norm_stats_data.get("norm_stats", norm_stats_data)
        elif "norm_stats" in ckpt and ckpt["norm_stats"] is not None:
            self.norm_stats = ckpt["norm_stats"]
            if (
                isinstance(self.norm_stats, dict)
                and "state" not in self.norm_stats
                and "actions" not in self.norm_stats
            ):
                keys = list(self.norm_stats.keys())
                if len(keys) == 1:
                    self.norm_stats = self.norm_stats[keys[0]]
                elif len(keys) > 1:
                    import warnings

                    warnings.warn(
                        f"Checkpoint norm_stats has multiple task keys {keys}; using first key '{keys[0]}'. "
                        "Pass --norm-stats-path for explicit selection.",
                        stacklevel=2,
                    )
                    self.norm_stats = self.norm_stats[keys[0]]
                else:
                    raise ValueError("No norm stats found in checkpoint")
            if isinstance(self.norm_stats, dict):
                for k, v in self.norm_stats.items():
                    if isinstance(v, list) and len(v) == 1 and isinstance(v[0], dict):
                        self.norm_stats[k] = v[0]
        else:
            raise ValueError(
                "No norm stats found. Provide --norm-stats-path or ensure "
                "checkpoint contains norm_stats."
            )

    def set_task(self, task: Optional[str]) -> None:
        task_str = None if task is None else str(task).strip()
        if task_str is None or task_str == "":
            self._task_vec_h = None
            self._task_str = None
            return
        try:
            self._task_vec_h = self.model.encode_task_to_hidden(
                [task_str], device=self.device
            )
            self._task_str = task_str
        except Exception as e:
            print(f"Failed to compute task embedding for '{task}': {e}")
            self._task_vec_h = None
            self._task_str = None

    @torch.no_grad()
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        action_prefix: np.ndarray | None = None,
        prefix_length: int | None = None,
        latency: int | None = None,
    ) -> dict:
        """Run inference on the policy."""
        state_np = np.asarray(obs["state"], dtype=np.float32)
        if state_np.ndim == 1:
            batch_size = 1
            state_for_model = state_np[None, :]
        elif state_np.ndim == 2:
            batch_size = int(state_np.shape[0])
            state_for_model = state_np
        else:
            raise ValueError(
                f"Expected obs['state'] to have shape (D,) or (B, D), got {state_np.shape}"
            )

        prompts = self._normalize_prompts(obs.get("prompt"), batch_size=batch_size)

        desired_task = None
        if prompts and all(prompt == prompts[0] for prompt in prompts):
            task_trimmed = prompts[0].strip()
            desired_task = None if task_trimmed == "" else task_trimmed
        if self._task_str != desired_task:
            self.set_task(desired_task)

        inputs = {
            "state": state_for_model,
            "actions": np.zeros(
                (batch_size, self.chunk_len, self.action_dim), dtype=np.float32
            ),
        }
        inputs = self.normalizer(inputs)
        inputs = {
            "state": torch.from_numpy(inputs["state"]).float().to(self.device),
            "prompt": prompts,
        }

        images = obs.get("images", {})
        if isinstance(images, dict):
            images = dict(images)
            legacy_to_standard = {
                "cam_front": "top",
                "cam_top": "top",
                "rgb_front": "top",
                "rgb_top": "top",
                "front": "top",
                "cam_left": "left",
                "rgb_left": "left",
                "cam_right": "right",
                "rgb_right": "right",
            }
            for legacy, standard in legacy_to_standard.items():
                if standard not in images and legacy in images:
                    images[standard] = images[legacy]
        else:
            images = {}

        result = self._infer_actions(
            inputs,
            obs,
            images,
            noise=noise,
            action_prefix=action_prefix,
            prefix_length=prefix_length,
            latency=latency,
        )

        if isinstance(result, tuple):
            actions, infer_extras = result
        else:
            actions, infer_extras = result, {}

        actions = self.unnormalizer({"actions": actions})["actions"]
        self._check_action_bounds(actions)
        out = {"actions": actions, **infer_extras}
        if batch_size == 1 and out["actions"].ndim == 3:
            out["actions"] = out["actions"][0]
            if (
                "noise" in out
                and isinstance(out["noise"], np.ndarray)
                and out["noise"].ndim == 3
            ):
                out["noise"] = out["noise"][0]
        return out

    def _normalize_prompts(self, prompt_value, *, batch_size: int) -> list[str]:
        """Return a prompt list matching the current batch size."""
        if isinstance(prompt_value, str):
            return [prompt_value] * batch_size
        if prompt_value is None:
            return [""] * batch_size
        if isinstance(prompt_value, np.ndarray):
            prompt_value = prompt_value.tolist()
        if isinstance(prompt_value, (list, tuple)):
            prompts = [str(item) for item in prompt_value]
            if len(prompts) != batch_size:
                raise ValueError(
                    f"Expected {batch_size} prompts for batched inference, got {len(prompts)}"
                )
            return prompts
        raise TypeError(f"Unsupported prompt container type: {type(prompt_value)!r}")

    def _check_action_bounds(self, actions: np.ndarray) -> None:
        """Warn if unnormalized actions exceed expected physical bounds."""
        joint_pos_limit = 6.3
        max_val = np.abs(actions).max()
        if max_val > joint_pos_limit:
            import warnings

            warnings.warn(
                f"Action bounds exceeded: max |action| = {max_val:.3f} "
                f"(limit: {joint_pos_limit}). This may indicate a normalization bug.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _build_model(self, ckpt_path: str) -> None:
        raise NotImplementedError

    def _infer_actions(
        self,
        inputs,
        obs,
        images,
        *,
        noise=None,
        action_prefix=None,
        prefix_length=None,
        latency=None,
    ):
        raise NotImplementedError
