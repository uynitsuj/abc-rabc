"""LBM policy for sim inference.

Requires the model architecture package to be installed or on PYTHONPATH.
This module imports ``models.lbm.DiTPolicy``.
"""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from termcolor import cprint

from yam_sim.policy.base import BasePolicy, PolicyConfig


def _resize_with_pad_torch(images, height, width, mode="bilinear"):
    """Resize images with padding to preserve aspect ratio."""
    import torch.nn.functional as F

    if images.shape[-1] <= 4:
        channels_last = True
        images = images.permute(0, 3, 1, 2)
    else:
        channels_last = False

    batch_size, channels, cur_height, cur_width = images.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    antialias = mode == "bilinear"
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
        antialias=antialias,
    )

    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0, 1)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),
        mode="constant",
        value=0,
    )

    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)
    return padded_images


def _get_norm_stats(preset: str) -> tuple[list[float], list[float]]:
    preset = preset.lower()
    if preset == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif preset == "clip":
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    else:
        raise ValueError(f"Unknown normalization preset: {preset}")
    return mean, std


def _normalize_image(im: torch.Tensor, preset: str = "imagenet") -> torch.Tensor:
    """Normalize BCHW images using the configured preset."""
    mean, std = _get_norm_stats(preset)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=im.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32, device=im.device).view(1, 3, 1, 1)
    return (im - mean_t) / (std_t + 1e-6)


@dataclass
class LBMPolicyConfig(PolicyConfig):
    """Config for LBM-family policies."""

    camera_keys: Optional[List[str]] = None
    post_cond_layer_norm: Optional[bool] = None
    pre_vizemb_norm: Optional[bool] = None
    normalize_clip_for_cond: Optional[bool] = None
    pre_img_layer_norm: Optional[bool] = None
    use_attn_pool: Optional[bool] = None
    apool_num_queries: Optional[int] = None
    apool_num_heads: Optional[int] = None
    apool_mlp_ratio: Optional[int] = None
    use_vision_cross_attn: Optional[bool] = None
    vision_include_cls: Optional[bool] = None
    vision_cross_attn_stride: Optional[int] = None
    vision_pool_num_queries: Optional[int] = None
    vision_pool_num_heads: Optional[int] = None
    vision_pool_mlp_ratio: Optional[int] = None
    tau: float = 1.0
    num_diffusion_steps: int = 10
    use_torch_compile: bool = False
    deterministic: bool = False
    diffusion_steps: int = 10
    sampler: str = "euler"
    num_bins: int = 51
    v_min: float = -3.0
    v_max: float = 3.0
    prediction_type: str = "velocity"
    save_clip_features: bool = False
    action_future: Optional[int] = None


class LBMPolicy(BasePolicy):
    """Policy for LBM-family models."""

    cfg: LBMPolicyConfig

    def __init__(self, cfg: LBMPolicyConfig):
        self.tau = cfg.tau
        self.deterministic = cfg.deterministic
        self._clip_feat_dir: Optional[str] = None
        self._clip_feat_idx: int = 0
        self._clip_feat_buffer: List[dict] = []
        if self.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            cprint(
                "Deterministic mode: CUDA deterministic algorithms enabled",
                "yellow",
            )
        self.sampler = cfg.sampler
        super().__init__(cfg)

    def _transform_state_dict(self, state_dict: dict) -> dict:
        remapped: dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith("img_backbone.clip_model.") and hasattr(
                self.model, "clip_model"
            ):
                new_key = "clip_model." + key[len("img_backbone.clip_model.") :]
            if key.startswith("vision_tokens_proj.") and hasattr(
                self.model, "vision_memory_proj"
            ):
                new_key = "vision_memory_proj." + key[len("vision_tokens_proj.") :]
            remapped[new_key] = value
        return remapped

    @staticmethod
    def _load_saved_train_config(ckpt_path: str) -> dict:
        """Best-effort load of training config metadata for model reconstruction."""
        try:
            ckpt = torch.load(
                ckpt_path,
                map_location="cpu",
                mmap=True,
                weights_only=False,
            )
            train_config = ckpt.get("train_config")
            if isinstance(train_config, dict):
                return train_config
        except Exception:
            pass

        run_info_path = os.path.join(os.path.dirname(ckpt_path), "run_info.txt")
        if os.path.exists(run_info_path):
            try:
                with open(run_info_path, "r") as f:
                    content = f.read()
                marker = "train_config:\n"
                if marker in content:
                    json_blob = content.split(marker, 1)[1].split("\n\n", 1)[0]
                    train_config = json.loads(json_blob)
                    if isinstance(train_config, dict):
                        return train_config
            except Exception:
                pass

        return {}

    def _cfg_or_saved(self, name: str, saved_cfg: dict, default):
        value = getattr(self.cfg, name, None)
        if value is not None:
            return value
        return saved_cfg.get(name, default)

    def _build_model(self, ckpt_path: str) -> None:
        saved_cfg = self._load_saved_train_config(ckpt_path)
        resolved_model_size = saved_cfg.get("model_size", self.cfg.model_size)
        self.cfg.model_size = resolved_model_size
        model_cfgs = {
            "dit_S": {"hidden_size": 384, "depth": 12, "num_heads": 6},
            "dit_B": {"hidden_size": 768, "depth": 12, "num_heads": 12},
            "dit_L": {"hidden_size": 1024, "depth": 24, "num_heads": 16},
            "dit_xL": {"hidden_size": 1536, "depth": 32, "num_heads": 24},
        }
        mcfg = model_cfgs.get(resolved_model_size, model_cfgs["dit_B"])
        only_top_camera = self._cfg_or_saved("only_top_camera", saved_cfg, False)
        camera_keys = (
            self.cfg.camera_keys
            if self.cfg.camera_keys is not None
            else (["top"] if only_top_camera else ["top", "left", "right"])
        )

        post_cond_layer_norm = self._cfg_or_saved(
            "post_cond_layer_norm", saved_cfg, True
        )
        pre_vizemb_norm = self._cfg_or_saved("pre_vizemb_norm", saved_cfg, False)
        normalize_clip_for_cond = self._cfg_or_saved(
            "normalize_clip_for_cond", saved_cfg, False
        )
        pre_img_layer_norm = self._cfg_or_saved("pre_img_layer_norm", saved_cfg, False)
        use_attn_pool = self._cfg_or_saved("use_attn_pool", saved_cfg, False)
        apool_num_queries = self._cfg_or_saved("apool_num_queries", saved_cfg, 8)
        apool_num_heads = self._cfg_or_saved("apool_num_heads", saved_cfg, 8)
        apool_mlp_ratio = self._cfg_or_saved("apool_mlp_ratio", saved_cfg, 4)
        use_vision_cross_attn = self._cfg_or_saved(
            "use_vision_cross_attn", saved_cfg, False
        )
        vision_include_cls = self._cfg_or_saved("vision_include_cls", saved_cfg, True)
        vision_cross_attn_stride = self._cfg_or_saved(
            "vision_cross_attn_stride", saved_cfg, 1
        )
        vision_pool_num_queries = self._cfg_or_saved(
            "vision_pool_num_queries", saved_cfg, 21
        )
        vision_pool_num_heads = self._cfg_or_saved(
            "vision_pool_num_heads", saved_cfg, 8
        )
        vision_pool_mlp_ratio = self._cfg_or_saved(
            "vision_pool_mlp_ratio", saved_cfg, 4
        )
        action_future = self._cfg_or_saved("action_future", saved_cfg, 0)
        action_history = self._cfg_or_saved("action_history", saved_cfg, 0)

        if self.cfg.policy_type == "lbm":
            from models.lbm import DiTPolicy  # type: ignore[import]

            self.model = DiTPolicy(
                state_dim=14,
                action_dim=14,
                chunk_length=self.chunk_len,
                img_emb_dim=512,
                hidden_size=mcfg["hidden_size"],
                depth=mcfg["depth"],
                num_heads=mcfg["num_heads"],
                task_encoder="clip",
                camera_keys=camera_keys,
                num_diffusion_timesteps=self.cfg.diffusion_steps,
                device=self.device,
                post_cond_layer_norm=post_cond_layer_norm,
                pre_vizemb_norm=pre_vizemb_norm,
                normalize_clip_for_cond=normalize_clip_for_cond,
                pre_img_layer_norm=pre_img_layer_norm,
                use_attn_pool=use_attn_pool,
                apool_num_queries=apool_num_queries,
                apool_num_heads=apool_num_heads,
                apool_mlp_ratio=apool_mlp_ratio,
                use_vision_cross_attn=use_vision_cross_attn,
                vision_include_cls=vision_include_cls,
                vision_cross_attn_stride=vision_cross_attn_stride,
                vision_pool_num_queries=vision_pool_num_queries,
                vision_pool_num_heads=vision_pool_num_heads,
                vision_pool_mlp_ratio=vision_pool_mlp_ratio,
                action_history=action_history,
                action_future=action_future,
            ).to(self.device)
        elif self.cfg.policy_type == "lbm_ce":
            from models.lbm_distributional import DiTPolicyCE  # type: ignore[import]

            if use_attn_pool or use_vision_cross_attn:
                raise ValueError(
                    "Deployed lbm_ce does not support use_attn_pool or use_vision_cross_attn"
                )

            self.model = DiTPolicyCE(
                state_dim=14,
                action_dim=14,
                chunk_length=self.chunk_len,
                img_emb_dim=512,
                hidden_size=mcfg["hidden_size"],
                depth=mcfg["depth"],
                num_heads=mcfg["num_heads"],
                task_encoder="clip",
                camera_keys=camera_keys,
                num_diffusion_timesteps=self.cfg.diffusion_steps,
                device=self.device,
                post_cond_layer_norm=post_cond_layer_norm,
                pre_vizemb_norm=pre_vizemb_norm,
                normalize_clip_for_cond=normalize_clip_for_cond,
                pre_img_layer_norm=pre_img_layer_norm,
                num_bins=self.cfg.num_bins,
                v_min=self.cfg.v_min,
                v_max=self.cfg.v_max,
                prediction_type=self.cfg.prediction_type,
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported LBM policy type: {self.cfg.policy_type}")

    def _post_load_setup(self) -> None:
        if self.cfg.save_clip_features:
            import atexit
            from datetime import datetime

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._clip_feat_dir = f"clip_features/{ts}"
            os.makedirs(self._clip_feat_dir, exist_ok=True)
            cprint(
                f"Buffering CLIP features (will flush to {self._clip_feat_dir}/ on exit)",
                "yellow",
            )
            atexit.register(self.flush_clip_features)

        if bool(getattr(self.cfg, "use_torch_compile", False)) and hasattr(
            torch, "compile"
        ):
            try:
                torch.set_float32_matmul_precision("high")
                cprint(
                    "Compiling model.infer with torch.compile (max-autotune, fullgraph=True)",
                    "cyan",
                )
                self.model_infer = torch.compile(
                    self.model.infer, mode="max-autotune", fullgraph=True
                )
                cprint("Casting model (except CLIP) to bfloat16", "cyan")
                self.model.to(torch.bfloat16)
                try:
                    self.model.clip_model.to(torch.float32)
                    if hasattr(self.model, "enable_clip_compile"):
                        self.model.enable_clip_compile(mode="default", fullgraph=False)
                except Exception as exc:
                    print(f"Failed to keep CLIP in float32 or compile: {exc}")
            except Exception as exc:
                cprint(f"torch.compile failed; falling back to eager: {exc}", "red")

    def flush_clip_features(self) -> None:
        """Write all buffered frames to disk."""
        if not self._clip_feat_buffer or not self._clip_feat_dir:
            return
        os.makedirs(self._clip_feat_dir, exist_ok=True)
        cprint(
            f"Flushing {len(self._clip_feat_buffer)} frames to {self._clip_feat_dir}/...",
            "yellow",
        )
        for idx, frame_data in enumerate(self._clip_feat_buffer):
            noise = frame_data.pop("noise", None)
            np.savez_compressed(
                os.path.join(self._clip_feat_dir, f"frame_{idx:06d}.npz"),
                **frame_data,
            )
            if noise is not None:
                np.save(os.path.join(self._clip_feat_dir, f"noise_{idx:06d}.npy"), noise)
        cprint(f"Saved {len(self._clip_feat_buffer)} frames.", "green")
        self._clip_feat_buffer.clear()

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
        def _to_tensor(value):
            if not isinstance(value, torch.Tensor):
                value = torch.from_numpy(value.copy())
            value = value.float().to(self.device)
            if value.max() > 1.0:
                value = value / 255.0
            if value.ndim == 3:
                value = value.unsqueeze(0)
            return value

        preset = self.cfg.normalize_preset
        inputs["images"] = {
            key: _normalize_image(
                _resize_with_pad_torch(_to_tensor(value), 224, 224),
                preset=preset,
            )
            for key, value in images.items()
        }
        batch_size = int(inputs["state"].shape[0])

        action_prefix_tensor = None
        prefix_len = None
        if action_prefix is not None:
            action_prefix = np.asarray(action_prefix, dtype=np.float32)
            if action_prefix.ndim == 2:
                action_prefix = action_prefix[None, ...]
            if action_prefix.ndim != 3 or action_prefix.shape[0] != batch_size:
                raise ValueError(
                    f"Expected action_prefix shape (B, T, D) with B={batch_size}, got {action_prefix.shape}"
                )
            action_prefix_norm = self.normalizer({"actions": action_prefix})["actions"]
            action_prefix_padded = np.zeros(
                (batch_size, self.chunk_len, self.action_dim), dtype=np.float32
            )
            if prefix_length is None:
                prefix_len = np.full(
                    (batch_size,), action_prefix_norm.shape[1], dtype=np.int64
                )
            elif np.isscalar(prefix_length):
                prefix_len = np.full((batch_size,), int(prefix_length), dtype=np.int64)
            else:
                prefix_len = np.asarray(prefix_length, dtype=np.int64)
                if prefix_len.shape != (batch_size,):
                    raise ValueError(
                        f"Expected prefix_length shape ({batch_size},), got {prefix_len.shape}"
                    )

            for batch_idx in range(batch_size):
                cur_len = int(max(0, min(self.chunk_len, prefix_len[batch_idx])))
                action_prefix_padded[batch_idx, :cur_len] = action_prefix_norm[
                    batch_idx, :cur_len
                ]

            action_prefix_tensor = torch.from_numpy(action_prefix_padded).float().to(
                self.device
            )
            prefix_len = torch.from_numpy(prefix_len).to(
                device=self.device, dtype=torch.long
            )

        infer_params = inspect.signature(self.model.infer).parameters
        infer_kwargs = {"num_steps": self.model.num_diffusion_timesteps}
        if "tau" in infer_params:
            infer_kwargs["tau"] = self.tau
        if "sampler" in infer_params:
            infer_kwargs["sampler"] = self.sampler
        if "action_prefix" in infer_params and action_prefix_tensor is not None:
            infer_kwargs["action_prefix"] = action_prefix_tensor
            infer_kwargs["prefix_length"] = prefix_len
        if "latency" in infer_params and latency is not None and latency > 0:
            infer_kwargs["latency"] = latency

        if "noise" in infer_params:
            if noise is None:
                noise = torch.randn(
                    batch_size, self.chunk_len, self.action_dim, device=self.device
                )
            else:
                noise = np.asarray(noise, dtype=np.float32)
                if noise.ndim == 2:
                    noise = noise[None, ...]
                if noise.ndim != 3 or noise.shape[0] != batch_size:
                    raise ValueError(
                        f"Expected noise shape (B, T, D) with B={batch_size}, got {noise.shape}"
                    )
                noise = torch.from_numpy(noise).float().to(self.device)
            infer_kwargs["noise"] = noise
        else:
            noise = None

        if self._clip_feat_dir is not None:
            frame_data: dict[str, np.ndarray] = {}
            for cam_name in self.model.camera_keys:
                if cam_name in inputs["images"]:
                    frame_data[f"{cam_name}_img"] = (
                        inputs["images"][cam_name][0].detach().cpu().float().numpy()
                    )
            frame_data["state"] = inputs["state"][0].detach().cpu().float().numpy()
            self._clip_feat_buffer.append(frame_data)

            if self._clip_feat_idx == 0:
                metadata = {
                    "prompt": inputs.get("prompt", [""])[0],
                    "checkpoint": getattr(self.cfg, "dir", ""),
                    "camera_keys": list(self.model.camera_keys),
                    "chunk_length": self.model.chunk_length,
                    "action_dim": self.model.action_dim,
                    "model_size": getattr(self.cfg, "model_size", ""),
                }
                with open(
                    os.path.join(self._clip_feat_dir, "metadata.json"), "w"
                ) as f:
                    json.dump(metadata, f, indent=2)
            self._clip_feat_idx += 1

        actions = self.model.infer(inputs, **infer_kwargs)

        extras = {}
        if noise is not None:
            extras["noise"] = noise.cpu().numpy()
            if self._clip_feat_dir is not None:
                self._clip_feat_buffer[-1]["noise"] = (
                    noise[0].detach().cpu().float().numpy()
                )
        return actions.detach().cpu().numpy(), extras
