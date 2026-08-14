"""In-process batched pi0 policy for the batched runner (cow).

One process hosts BOTH stacks: warp/mjwarp (physics + world-aware render) and
jax/openpi (pi0). Memory cohabitation: set XLA_PYTHON_CLIENT_MEM_FRACTION<=0.35
BEFORE importing jax; warp's pool is modest (contact buffers) and physics lives
on cuda:0.

Batching strategy = the serial Policy.infer, unrolled around the batch axis:
  per-world input transforms (numpy, cheap)  ->  stack  ->  ONE jitted
  sample_actions call at [B]  ->  per-world output transforms.
The per-world transform loops preserve exact serial semantics (tokenization,
normalization, image handling); only the model call is batched — which is the
only part that costs anything.

Obs contract per world (mirrors Pi0Policy websocket client in eval_policy.py):
  {"state": f32[K], "prompt": str, <server_key>: CHW uint8}
"""
from __future__ import annotations

import numpy as np


class BatchedPi0:
    """Loads a trained openpi policy once; exposes infer_batch(obs_list)->[B,H,A]."""

    needs_images = True

    def __init__(self, config_name: str, ckpt_dir: str, prompt: str,
                 camera_key_map: dict[str, str], condition: float | None = None,
                 cfg_weight: float | None = None):
        import jax
        from openpi.policies import policy_config
        from openpi.training import config as _config

        self._jax = jax
        cfg = _config.get_config(config_name)
        sample_kwargs = {"cfg_weight": cfg_weight} if cfg_weight is not None else None
        self.policy = policy_config.create_trained_policy(cfg, ckpt_dir, sample_kwargs=sample_kwargs)
        self.prompt = prompt
        self.camera_key_map = dict(camera_key_map)
        self.condition = condition  # explicit o-bit/NaN override; None = config default

    def make_obs(self, state_rows: np.ndarray, images: dict[str, np.ndarray]):
        """state_rows [B,K] f32; images {env_key: [B,3,H,W] uint8} -> list of per-world
        server obs dicts (the exact websocket client payload)."""
        B = state_rows.shape[0]
        obs_list = []
        for w in range(B):
            o = {"state": np.asarray(state_rows[w], np.float32), "prompt": self.prompt}
            if self.condition is not None:
                o["condition"] = np.float32(self.condition)  # NaN = unconditional branch
            for env_key, server_key in self.camera_key_map.items():
                if env_key in images:
                    o[server_key] = np.ascontiguousarray(images[env_key][w])
            obs_list.append(o)
        return obs_list

    def infer_batch(self, obs_list) -> np.ndarray:
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        p = self.policy
        # per-world input transforms (may modify in place -> shallow copy first)
        ins = [p._input_transform(jax.tree.map(lambda x: x, o)) for o in obs_list]
        batch = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *ins)
        p._rng, rng = jax.random.split(p._rng)
        observation = _model.Observation.from_dict(batch)
        actions = p._sample_actions(rng, observation, **p._sample_kwargs)
        actions = np.asarray(actions)
        state_in = np.asarray(batch["state"])
        outs = []
        for w in range(len(obs_list)):
            ow = p._output_transform({"state": state_in[w], "actions": actions[w]})
            outs.append(np.asarray(ow["actions"], np.float32))
        return np.stack(outs)  # [B, H, A]
