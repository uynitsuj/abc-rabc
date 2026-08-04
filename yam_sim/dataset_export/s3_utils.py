"""AWS CLI helpers for dataset export pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import posixpath
import re
import subprocess
from typing import Sequence


_S3_LS_LINE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"(?P<size>\d+)\s+"
    r"(?P<key>.+)$"
)


def _normalize_s3_key(key: str) -> str:
    return key.strip("/")


@dataclass(frozen=True)
class S3Uri:
    """Parsed S3 URI with helpers for child paths."""

    bucket: str
    key: str = ""

    def __post_init__(self) -> None:
        if not self.bucket:
            raise ValueError("S3 bucket cannot be empty")
        object.__setattr__(self, "key", _normalize_s3_key(self.key))

    @property
    def uri(self) -> str:
        if not self.key:
            return f"s3://{self.bucket}"
        return f"s3://{self.bucket}/{self.key}"

    def child(self, *parts: str) -> "S3Uri":
        clean_parts = [self.key] if self.key else []
        clean_parts.extend(_normalize_s3_key(part) for part in parts if _normalize_s3_key(part))
        key = posixpath.join(*clean_parts) if clean_parts else ""
        return S3Uri(bucket=self.bucket, key=key)


@dataclass(frozen=True)
class S3ObjectInfo:
    """Single object returned by `aws s3 ls --recursive`."""

    bucket: str
    key: str
    size_bytes: int

    @property
    def uri(self) -> str:
        return S3Uri(self.bucket, self.key).uri


def parse_s3_uri(value: str) -> S3Uri:
    """Parse a `s3://bucket/key` URI."""
    if not value.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {value!r}")
    remainder = value[5:]
    bucket, sep, key = remainder.partition("/")
    if not bucket:
        raise ValueError(f"Invalid S3 URI {value!r}")
    return S3Uri(bucket=bucket, key=key if sep else "")


def run_aws(
    args: Sequence[str],
    *,
    aws_profile: str | None = None,
    aws_region: str | None = None,
    capture_output: bool = False,
    text: bool = False,
) -> subprocess.CompletedProcess:
    """Run an AWS CLI command with optional profile/region overrides."""
    cmd = ["aws"]
    if aws_profile:
        cmd.extend(["--profile", aws_profile])
    if aws_region:
        cmd.extend(["--region", aws_region])
    cmd.extend(args)
    return subprocess.run(cmd, check=True, capture_output=capture_output, text=text)


def list_s3_objects(
    s3_prefix: str,
    *,
    aws_profile: str | None = None,
    aws_region: str | None = None,
) -> list[S3ObjectInfo]:
    """List objects beneath an S3 prefix using the AWS CLI."""
    prefix = parse_s3_uri(s3_prefix)
    result = run_aws(
        ["s3", "ls", prefix.uri, "--recursive"],
        aws_profile=aws_profile,
        aws_region=aws_region,
        capture_output=True,
        text=True,
    )
    objects: list[S3ObjectInfo] = []
    for line in result.stdout.splitlines():
        match = _S3_LS_LINE.match(line.strip())
        if match is None:
            continue
        objects.append(
            S3ObjectInfo(
                bucket=prefix.bucket,
                key=match.group("key"),
                size_bytes=int(match.group("size")),
            )
        )
    return objects


def copy_s3_object_to_local(
    s3_uri: str,
    local_path: Path,
    *,
    aws_profile: str | None = None,
    aws_region: str | None = None,
) -> Path:
    """Download one S3 object to a local file path."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    run_aws(
        ["s3", "cp", s3_uri, str(local_path)],
        aws_profile=aws_profile,
        aws_region=aws_region,
    )
    return local_path


def copy_s3_dir_to_local(
    s3_prefix: str,
    local_dir: Path,
    *,
    aws_profile: str | None = None,
    aws_region: str | None = None,
) -> Path:
    """Download one S3 prefix tree into a local directory."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    run_aws(
        ["s3", "cp", s3_prefix.rstrip("/") + "/", str(local_dir), "--recursive"],
        aws_profile=aws_profile,
        aws_region=aws_region,
    )
    return local_dir


def copy_local_file_to_s3(
    local_path: Path,
    s3_uri: str,
    *,
    aws_profile: str | None = None,
    aws_region: str | None = None,
) -> None:
    """Upload one local file to S3."""
    run_aws(
        ["s3", "cp", str(local_path), s3_uri],
        aws_profile=aws_profile,
        aws_region=aws_region,
    )


def copy_local_dir_to_s3(
    local_dir: Path,
    s3_prefix: str,
    *,
    aws_profile: str | None = None,
    aws_region: str | None = None,
) -> None:
    """Upload a local directory tree to S3 without destination-side diffing."""
    local_dir = Path(local_dir)
    run_aws(
        ["s3", "cp", str(local_dir) + "/", s3_prefix.rstrip("/") + "/", "--recursive"],
        aws_profile=aws_profile,
        aws_region=aws_region,
    )


def sync_local_dir_to_s3(
    local_dir: Path,
    s3_prefix: str,
    *,
    aws_profile: str | None = None,
    aws_region: str | None = None,
) -> None:
    """Upload a local directory tree to S3."""
    local_dir = Path(local_dir)
    run_aws(
        ["s3", "sync", str(local_dir) + "/", s3_prefix.rstrip("/") + "/"],
        aws_profile=aws_profile,
        aws_region=aws_region,
    )
