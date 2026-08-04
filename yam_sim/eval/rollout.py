"""Shared rollout mechanics for policy execution in yam-sim.

These helpers are factored out of ``yam_sim.examples.run_policy`` so the single
visualizer and the batched eval harness share one code path for observation
preparation, TTRTC action-prefix handling, and per-world frame assembly.
"""

from __future__ import annotations

import numpy as np


def jpeg_compress_image(img_chw: np.ndarray, quality: int = 75) -> np.ndarray:
    """Apply JPEG encode/decode to a CHW image to match training-data artifacts."""
    import cv2

    img_hwc = img_chw.transpose(1, 2, 0)
    img_bgr = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, compressed = cv2.imencode(".jpg", img_bgr, encode_param)
    decoded_bgr = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    return decoded_rgb.transpose(2, 0, 1)


def jpeg_compress_batch(image: np.ndarray, quality: int) -> np.ndarray:
    """JPEG-compress a single CHW image or a batch of BCHW images."""
    if quality <= 0:
        return image
    image = np.asarray(image)
    if image.ndim == 3:
        return jpeg_compress_image(image, quality=quality)
    if image.ndim == 4:
        return np.stack(
            [jpeg_compress_image(frame, quality=quality) for frame in image],
            axis=0,
        )
    raise ValueError(f"Unsupported image shape for JPEG compression: {image.shape}")


def prepare_policy_obs(obs: dict, *, prompt: str, jpeg_quality: int) -> dict:
    """Repack an env observation into the dict a policy's ``infer`` expects."""
    state = np.asarray(obs["state"], dtype=np.float32)
    batch_size = int(state.shape[0]) if state.ndim == 2 else 1
    images = {}
    for name, image in obs.get("images", {}).items():
        images[name] = jpeg_compress_batch(np.asarray(image), jpeg_quality)

    return {
        "state": state,
        "images": images,
        "prompt": prompt if batch_size == 1 else [prompt] * batch_size,
    }


def ttrtc_prefix(
    previous_chunk_actions: np.ndarray | None,
    init_q: np.ndarray,
    prefix_len: int,
    *,
    is_batched: bool,
) -> tuple[np.ndarray | None, int | None]:
    """Build the TTRTC action prefix and its length for the next inference call.

    On the first chunk (``previous_chunk_actions is None``) the prefix repeats the
    initial state; afterwards it is the tail of the previously executed actions.
    """
    if previous_chunk_actions is not None:
        if is_batched:
            prefix_length = min(prefix_len, previous_chunk_actions.shape[1])
            action_prefix = previous_chunk_actions[:, -prefix_length:, :]
        else:
            prefix_length = min(prefix_len, len(previous_chunk_actions))
            action_prefix = previous_chunk_actions[-prefix_length:]
    else:
        prefix_length = prefix_len
        if is_batched:
            action_prefix = np.repeat(init_q[:, None, :], prefix_length, axis=1)
        else:
            action_prefix = np.tile(init_q, (prefix_length, 1))
    return action_prefix, prefix_length


def slice_execute_actions(
    predicted_actions: np.ndarray,
    *,
    prefix_length: int | None,
    execute_dim: int,
    use_ttrtc: bool,
    is_batched: bool,
) -> np.ndarray:
    """Select the slice of predicted actions to execute this chunk."""
    if use_ttrtc and prefix_length is not None:
        start, stop = prefix_length, prefix_length + execute_dim
    else:
        start, stop = 0, execute_dim
    if is_batched:
        return predicted_actions[:, start:stop, :]
    return predicted_actions[start:stop]


def per_world_frames(obs: dict, camera_names: list[str]) -> np.ndarray:
    """Assemble per-world camera rows from an observation.

    Returns an array of shape ``(num_worlds, H, W * ncam, 3)`` (uint8), where each
    world's cameras are concatenated horizontally in ``camera_names`` order.
    """
    per_camera_batches = []
    for name in camera_names:
        image = np.asarray(obs["images"][name])
        if image.ndim == 3:  # (3, H, W) -> (1, 3, H, W)
            image = image[None, ...]
        per_camera_batches.append(image.transpose(0, 2, 3, 1))  # (B, H, W, 3)

    # Concatenate cameras along width for each world.
    return np.concatenate(per_camera_batches, axis=2)


def grid_from_obs(obs: dict, camera_names: list[str]) -> np.ndarray:
    """Stack every world's camera row vertically into one grid frame.

    Matches the legacy ``run_policy`` video layout (worlds stacked top-to-bottom).
    """
    rows = per_world_frames(obs, camera_names)  # (B, H, W*ncam, 3)
    return np.concatenate(list(rows), axis=0)
