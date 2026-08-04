"""Normalization transforms for state/action data."""

import numpy as np


def pad_to_dim(
    x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0
) -> np.ndarray:
    """Pad an array to the target dimension along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def _assert_quantile_stats(norm_stats) -> None:
    for k, v in norm_stats.items():
        if "q01" not in v or "q99" not in v:
            raise ValueError(
                f"Quantile stats must be provided if use_quantile_norm is True. "
                f"Key {k} is missing q01 or q99."
            )
        if np.array(v["q01"]).shape[-1] != np.array(v["q99"]).shape[-1]:
            raise ValueError(
                f"Quantile stats shape mismatch for key {k}: q01 and q99 have different shapes."
            )


class Normalize:
    def __init__(self, norm_stats=None, use_quantiles=False, strict=True):
        self.norm_stats = norm_stats
        self.use_quantiles = use_quantiles
        self.strict = strict
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data):
        if self.norm_stats is None:
            return data
        for key, value in data.items():
            if key in self.norm_stats:
                data[key] = self._normalize(value, self.norm_stats[key])
            elif self.strict:
                raise ValueError(f"Key {key} not found in norm stats")
        return data

    def _normalize(self, x, stats):
        if self.use_quantiles:
            return self._normalize_quantile(x, stats)
        mean = np.array(stats["mean"])[..., : x.shape[-1]]
        std = np.array(stats["std"])[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats):
        q01, q99 = np.array(stats["q01"]), np.array(stats["q99"])
        if q01.shape[-1] < x.shape[-1]:
            q01 = pad_to_dim(q01, x.shape[-1], axis=-1, value=0.0)
            q99 = pad_to_dim(q99, x.shape[-1], axis=-1, value=0.0)
        elif q01.shape[-1] > x.shape[-1]:
            q01 = q01[..., : x.shape[-1]]
            q99 = q99[..., : x.shape[-1]]
        scaled = (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        return np.clip(scaled, -1.0, 1.0)


class Unnormalize:
    def __init__(self, norm_stats=None, use_quantiles=False, strict=True):
        self.norm_stats = norm_stats
        self.use_quantiles = use_quantiles
        self.strict = strict
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data):
        if self.norm_stats is None:
            return data
        for key, value in data.items():
            if key in self.norm_stats:
                if self.use_quantiles:
                    data[key] = self._unnormalize_quantile(value, self.norm_stats[key])
                else:
                    data[key] = self._unnormalize(value, self.norm_stats[key])
            elif self.strict:
                raise ValueError(f"Key {key} not found in norm stats")
        return data

    def _unnormalize(self, x, stats):
        mean = np.array(stats["mean"])[..., : x.shape[-1]]
        std = np.array(stats["std"])[..., : x.shape[-1]]
        if mean.shape[-1] < x.shape[-1]:
            mean = pad_to_dim(mean, x.shape[-1], axis=-1, value=0.0)
            std = pad_to_dim(std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats):
        q01, q99 = np.array(stats["q01"]), np.array(stats["q99"])
        if q01.shape[-1] < x.shape[-1]:
            q01 = pad_to_dim(q01, x.shape[-1], axis=-1, value=0.0)
            q99 = pad_to_dim(q99, x.shape[-1], axis=-1, value=0.0)
        elif q01.shape[-1] > x.shape[-1]:
            q01 = q01[..., : x.shape[-1]]
            q99 = q99[..., : x.shape[-1]]
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
