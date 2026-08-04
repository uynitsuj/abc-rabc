"""Remote episode discovery and sharding for S3-backed exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import posixpath

from yam_sim.dataset_export.s3_utils import S3ObjectInfo, S3Uri, list_s3_objects, parse_s3_uri


@dataclass(frozen=True)
class S3EpisodeSource:
    """One deduplicated delivered episode stored under an S3 prefix."""

    source_root: S3Uri
    episode_prefix: str
    relative_episode_prefix: str
    files: tuple[S3ObjectInfo, ...]

    @property
    def episode_name(self) -> str:
        return PurePosixPath(self.relative_episode_prefix).name

    @property
    def source_delivery(self) -> str:
        parts = PurePosixPath(self.relative_episode_prefix).parts
        if len(parts) >= 2:
            return parts[0]
        root_parts = PurePosixPath(self.source_root.key).parts
        if root_parts:
            return root_parts[-1]
        return self.episode_name

    @property
    def prefix_uri(self) -> str:
        return self.source_root.child(self.relative_episode_prefix).uri

    def file_map(self) -> dict[str, S3ObjectInfo]:
        return {PurePosixPath(obj.key).name: obj for obj in self.files}


def _relative_key(key: str, prefix_key: str) -> str:
    prefix_key = prefix_key.strip("/")
    if not prefix_key:
        return key.strip("/")
    full_prefix = prefix_key + "/"
    if key == prefix_key:
        return ""
    if not key.startswith(full_prefix):
        raise ValueError(f"Key {key!r} is outside prefix {prefix_key!r}")
    return key[len(full_prefix) :].strip("/")


def _path_depth(value: str) -> int:
    if not value:
        return 0
    return len(PurePosixPath(value).parts)


def _select_shortest_episode_prefixes(
    output_objects: list[S3ObjectInfo],
    *,
    source_root: S3Uri,
) -> dict[str, tuple[str, str]]:
    chosen: dict[str, tuple[str, str]] = {}
    for obj in output_objects:
        episode_prefix = posixpath.dirname(obj.key)
        relative_prefix = _relative_key(episode_prefix, source_root.key)
        episode_name = PurePosixPath(relative_prefix).name
        current = chosen.get(episode_name)
        if current is None or _path_depth(relative_prefix) < _path_depth(current[1]):
            chosen[episode_name] = (episode_prefix, relative_prefix)
    return chosen


def discover_episode_sources(
    s3_input_prefix: str,
    *,
    aws_profile: str | None = None,
    aws_region: str | None = None,
) -> list[S3EpisodeSource]:
    """List delivered episodes under an S3 prefix and deduplicate nested copies."""
    source_root = parse_s3_uri(s3_input_prefix)
    objects = list_s3_objects(
        source_root.uri,
        aws_profile=aws_profile,
        aws_region=aws_region,
    )
    output_objects = [
        obj for obj in objects if PurePosixPath(obj.key).name == "output.mcap"
    ]
    selected = _select_shortest_episode_prefixes(output_objects, source_root=source_root)
    direct_children: dict[str, list[S3ObjectInfo]] = {}
    for obj in objects:
        direct_children.setdefault(posixpath.dirname(obj.key), []).append(obj)

    sources = [
        S3EpisodeSource(
            source_root=source_root,
            episode_prefix=episode_prefix,
            relative_episode_prefix=relative_prefix,
            files=tuple(sorted(direct_children.get(episode_prefix, []), key=lambda item: item.key)),
        )
        for episode_prefix, relative_prefix in selected.values()
    ]
    return sorted(sources, key=lambda source: source.relative_episode_prefix)


def shard_episode_sources(
    sources: list[S3EpisodeSource],
    *,
    shard_index: int,
    num_shards: int,
) -> list[S3EpisodeSource]:
    """Deterministically assign sorted sources to one shard."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards}), got {shard_index}"
        )
    ordered = sorted(sources, key=lambda source: source.relative_episode_prefix)
    return [source for idx, source in enumerate(ordered) if idx % num_shards == shard_index]
