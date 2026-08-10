"""Adapter that drives yam_sim rollouts from a remote openpi policy server.

The openpi server (started via ``scripts/serve_policy.py`` in the openpi repo) loads
the pi0 checkpoint, applies its own input/output transforms (incl. norm stats), and
exposes a websocket. This class repacks yam_sim observations to match the
``YamInputs`` schema in ``openpi/src/openpi/policies/yam_policy.py`` and forwards
them, returning actions in the shape yam_sim's ``run_policy`` loop expects.

Requires ``openpi-client`` to be importable:
    pip install -e /path/to/openpi/packages/openpi-client
"""

from __future__ import annotations

from typing import Optional

import numpy as np


_YAM_SIM_TO_OPENPI_CAM = {
    "top": "top_camera-images-rgb",
    "left": "left_camera-images-rgb",
    "right": "right_camera-images-rgb",
}


class OpenPIPolicy:
    """Drop-in replacement for ``LBMPolicy`` backed by an openpi websocket server."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        *,
        api_key: Optional[str] = None,
        action_dim: int = 14,
        chunk_len: Optional[int] = None,
        use_batch_infer: bool = False,
    ) -> None:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        self._client = WebsocketClientPolicy(host=host, port=port, api_key=api_key)
        self.action_dim = action_dim
        self.chunk_len = chunk_len  # filled in lazily on first inference if None
        # Batched inference: send all worlds in one request (requires a
        # batch-aware openpi server, see Policy.infer_batch). Requests are
        # padded up to the largest batch seen so far so the server's jitted
        # model compiles for ONE batch shape instead of one per active count.
        self.use_batch_infer = use_batch_infer
        self._batch_pad = 0

    def set_task(self, task: Optional[str]) -> None:
        """No-op; the prompt is sent inside ``infer`` per call."""

    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        action_prefix: np.ndarray | None = None,
        prefix_length: int | None = None,
        latency: int | None = None,
    ) -> dict:
        """Run inference on one or many worlds via the openpi server.

        The openpi websocket server processes a single observation per call, so a
        batched obs is split, sent sequentially, then re-stacked.
        """
        state_np = np.asarray(obs["state"], dtype=np.float32)
        if state_np.ndim == 1:
            batch_size = 1
            state_batch = state_np[None, :]
            squeezed = True
        elif state_np.ndim == 2:
            batch_size = int(state_np.shape[0])
            state_batch = state_np
            squeezed = False
        else:
            raise ValueError(
                f"Expected obs['state'] to have shape (D,) or (B, D), got {state_np.shape}"
            )

        prompts = self._prompts(obs.get("prompt"), batch_size)
        per_world_images = self._split_images(obs.get("images", {}), batch_size)

        samples: list[dict] = []
        for i in range(batch_size):
            sample = {
                "state": state_batch[i].astype(np.float32),
                "prompt": prompts[i],
            }
            for yam_key, openpi_key in _YAM_SIM_TO_OPENPI_CAM.items():
                if yam_key in per_world_images:
                    sample[openpi_key] = per_world_images[yam_key][i]
            samples.append(sample)

        action_chunks: list[np.ndarray] = []
        if self.use_batch_infer and batch_size > 1:
            self._batch_pad = max(self._batch_pad, batch_size)
            padded = samples + [samples[-1]] * (self._batch_pad - batch_size)
            responses = self._client.infer_batch(padded)[:batch_size]
            for response in responses:
                actions = np.asarray(response["actions"], dtype=np.float32)
                action_chunks.append(actions[:, : self.action_dim])
        else:
            for sample in samples:
                response = self._client.infer(sample)
                actions = np.asarray(response["actions"], dtype=np.float32)
                # openpi YamOutputs already trims to first 14 dims.
                action_chunks.append(actions[:, : self.action_dim])

        stacked = np.stack(action_chunks, axis=0)  # (B, T, 14)
        if self.chunk_len is None:
            self.chunk_len = stacked.shape[1]

        if squeezed:
            return {"actions": stacked[0]}
        return {"actions": stacked}

    @staticmethod
    def _prompts(value, batch_size: int) -> list[str]:
        if value is None:
            return [""] * batch_size
        if isinstance(value, str):
            return [value] * batch_size
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            if len(value) != batch_size:
                raise ValueError(
                    f"Expected {batch_size} prompts, got {len(value)}"
                )
            return [str(p) for p in value]
        raise TypeError(f"Unsupported prompt type: {type(value)!r}")

    @staticmethod
    def _split_images(images: dict, batch_size: int) -> dict[str, np.ndarray]:
        """Return ``{yam_cam_name: (B, 3, H, W) uint8}`` regardless of input shape."""
        out: dict[str, np.ndarray] = {}
        for name, arr in images.items():
            if name not in _YAM_SIM_TO_OPENPI_CAM:
                continue
            arr = np.asarray(arr)
            if arr.ndim == 3:
                arr = arr[None, ...]
            if arr.shape[0] != batch_size:
                raise ValueError(
                    f"Image '{name}' has batch {arr.shape[0]}, expected {batch_size}"
                )
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            out[name] = arr
        return out
