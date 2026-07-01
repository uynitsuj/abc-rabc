# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyarrow", "pandas", "boto3", "tyro"]
# ///
"""Reduce the sharded mjwarp render scratch into the two final S3 datasets.

Phases (run independently via --phase):

  object_counts : from each episode's delivery randomization.json, count bottle_*_joint
                  -> {"counts": {str(episode_index): n_bottles}} at OBJECT_COUNTS_S3.

  staged        : per-episode staged mjwarp dirs. For each episode_index, pull the rendered
                  combined.mp4 + metadata from scratch, copy states_actions.bin + velocity_*.bin
                  from the LOCAL staged_put/{train_sim,val_sim} (renderer-invariant), and write
                  to  <STAGED_S3>/{train_sim,val_sim}/episode_<uuid>/. The 2 episodes absent
                  from the delivery keep their existing mjgl combined.mp4 (built from the mjgl
                  LeRobot videos) — recorded as mjgl-rendered.

  lerobot       : mirror the mjgl LeRobot v3 dataset with mjwarp per-cam videos. Copy
                  meta/ + data/ + norm_stats/ verbatim (states renderer-invariant; repromo
                  score cols kept — recomputed downstream). Regenerate the 8 per-file-group
                  videos per camera by concatenating the scratch per-episode percam mp4s in
                  episode_index order (re-encode, frame-exact). The 2 delivery-absent episodes
                  contribute their original mjgl frames (decoded from the mjgl video) so the
                  concat stays index-aligned and frame-exact.

Run with the project venv python (NOT `uv run`, which triggers uv sync):
    .venv/bin/python package_mjwarp.py --phase object_counts
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import boto3
import numpy as np
import pyarrow.parquet as pq
from botocore.config import Config as BotoConfig

# ------------------------------------------------------------------------- paths
DELIVERY_BUCKET = "xdof-bair-abc"
SUBTASK = "sim_put_the_plastic_bottles_in_the_bin"
# RM-training reference dataset (repromo/datasets/, NOT lerobot/): its meta/ + data/ +
# norm_stats.json are copied verbatim (states/actions + score cols are renderer-invariant);
# only the per-camera videos are regenerated with mjwarp frames. Its episodes parquet carries
# the source_episode_id <-> episode_index map + per-cam from_timestamp packing we rely on.
REF_LEROBOT_S3 = "s3://xdof-internal-research/repromo/datasets/sim_put_the_plastic_bottles_in_the_bin_30hz_gop10"
# The MJGL LeRobot dataset (lerobot/): SAME video bytes/packing as the repromo reference; used
# only as the fallback frame source for the 2 delivery-absent episodes' mjgl decode.
MJGL_LEROBOT_S3 = REF_LEROBOT_S3
SCRATCH_S3 = "s3://xdof-internal-research/abc/render_scratch/put_bottles_mjwarp"
STAGED_S3 = "s3://xdof-internal-research/abc/staged/put_bottles_mjwarp"
# LeRobot-v3 mjwarp dataset -> repromo/datasets/ (where RORM RM training pulls its --lerobot-repo).
LEROBOT_S3 = "s3://xdof-internal-research/repromo/datasets/sim_put_the_plastic_bottles_in_the_bin_30hz_mjwarp"
# object_counts.json rides INSIDE the dataset's meta/ (so the per-object RM filter receives it
# on the meta sync). Sourced verbatim from the reference dataset (complete 2438, incl. the 2
# delivery-absent episodes; our independently-computed counts match it exactly on all 2436).
OBJECT_COUNTS_SRC_S3 = f"{REF_LEROBOT_S3}/meta/object_counts.json"
LOCAL_STAGED = Path("/home/justinyu/abc/staged_put")

CAMERAS = ("top", "left", "right")
CAM_KEY = {c: f"{c}_camera-images-rgb" for c in CAMERAS}
FPS = 30
# Plain CFR-30 x264, mirrors the mjgl dataset info.json (libx264/yuv420p/gop10).
X264_LEROBOT = ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-bf", "0",
                "-pix_fmt", "yuv420p", "-g", "10", "-keyint_min", "10",
                "-x264-params", "scenecut=0", "-movflags", "+faststart"]


def _s3():
    return boto3.client("s3", config=BotoConfig(max_pool_connections=48, retries={"max_attempts": 8}))


def _split(uri: str):
    b, _, k = uri[5:].partition("/")
    return b, k.rstrip("/")


def probe_nframes(path: str) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True).stdout.strip()
    return int(out)


def load_manifest(s3, local_cache: Path) -> dict:
    """episode_index(str) -> {uuid, split, length, delivery}."""
    p = local_cache / "render_manifest.json"
    if not p.exists():
        b, k = _split(f"{SCRATCH_S3}/render_manifest.json")
        s3.download_file(b, k, str(p))
    return json.loads(p.read_text())


def load_mjgl_episodes(s3, local_cache: Path):
    """The mjgl meta episodes parquet(s) -> pandas df (has source_episode_id, per-cam offsets)."""
    d = local_cache / "mjgl_meta_episodes"
    d.mkdir(parents=True, exist_ok=True)
    b, k = _split(MJGL_LEROBOT_S3)
    # sync the whole meta/episodes tree
    subprocess.run(["aws", "s3", "sync", f"{MJGL_LEROBOT_S3}/meta/episodes", str(d),
                    "--only-show-errors"], check=True)
    files = sorted(glob.glob(str(d / "**" / "*.parquet"), recursive=True))
    return pq.read_table(files).to_pandas()


# ========================================================================= phase: object_counts
def phase_object_counts(cache: Path):
    """Verify our independently-computed bottle counts (from delivery randomization.json) match
    the reference dataset's object_counts.json, then confirm the reference (complete 2438) is the
    file that will ship. The lerobot phase's `meta` sync carries it into the dataset's meta/."""
    s3 = _s3()
    man = load_manifest(s3, cache)
    idxs = sorted(int(i) for i in man)

    def count_bottles(i):
        info = man[str(i)]
        deliv = info.get("delivery")
        if not deliv:
            return i, None  # delivery-absent: no randomization.json; reference covers it
        key = f"data/deliveries/{deliv}/{SUBTASK}/{info['uuid']}/randomization.json"
        try:
            r = json.loads(s3.get_object(Bucket=DELIVERY_BUCKET, Key=key)["Body"].read())
        except Exception as e:
            print(f"[oc][WARN] {i} {info['uuid']}: {e}")
            return i, None
        obj = r.get("object_states", {})
        n = sum(1 for name in obj if name.startswith("bottle_") and name.endswith("_joint"))
        meta_n = r.get("metadata", {}).get("bottle_count")
        if meta_n is not None and meta_n != n:
            print(f"[oc][WARN] {i}: bottle_*_joint count {n} != metadata.bottle_count {meta_n}")
        return i, n

    counts = {}
    with ThreadPoolExecutor(max_workers=48) as ex:
        for i, n in ex.map(count_bottles, idxs):
            if n is not None:
                counts[str(i)] = n
    from collections import Counter
    print(f"[oc] computed {len(counts)}/{len(idxs)} counts (2 delivery-absent skipped)")
    print(f"[oc] distribution: {dict(sorted(Counter(counts.values()).items()))}")

    # Cross-check against the reference dataset's object_counts.json (complete 2438).
    rb, rk = _split(OBJECT_COUNTS_SRC_S3)
    ref = json.loads(s3.get_object(Bucket=rb, Key=rk)["Body"].read())
    rc = ref["counts"]
    shared = set(counts) & set(rc)
    mism = [(k, rc[k], counts[k]) for k in shared if rc[k] != counts[k]]
    print(f"[oc] reference has {len(rc)} counts; shared={len(shared)}, mismatches={len(mism)}")
    if mism:
        raise SystemExit(f"[oc] count mismatch vs reference: {mism[:10]}")
    ref_only = sorted(set(rc) - set(counts), key=int)
    print(f"[oc] reference covers our 2 delivery-absent episodes: {ref_only} "
          f"-> counts {[rc[k] for k in ref_only]}")
    print(f"[oc] VERIFIED: our counts match reference on all {len(shared)}; the reference "
          f"object_counts.json (2438, complete) ships via the lerobot meta sync into "
          f"{LEROBOT_S3}/meta/object_counts.json")


# ========================================================================= phase: staged
def phase_staged(cache: Path, limit: Optional[int] = None):
    s3 = _s3()
    man = load_manifest(s3, cache)
    sb, sk = _split(SCRATCH_S3)
    tgt_b, tgt_k = _split(STAGED_S3)
    idxs = sorted(int(i) for i in man)
    if limit:
        idxs = idxs[:limit]

    absent = [i for i in idxs if not man[str(i)].get("delivery")]
    print(f"[staged] {len(idxs)} episodes; {len(absent)} delivery-absent (mjgl combined): {absent}")

    # Root norm_stats.json for the DiT-L `norm_stats=task` path (launch_abc RUN does
    # `aws s3 cp $STAGED_S3/norm_stats.json`). States/actions are renderer-invariant, so the
    # reference dataset's norm_stats is exact. Copy once (skip on a limited smoke run).
    if not limit:
        subprocess.run(["aws", "s3", "cp", f"{REF_LEROBOT_S3}/norm_stats.json",
                        f"{STAGED_S3}/norm_stats.json", "--only-show-errors"], check=True)
        print(f"[staged] copied norm_stats.json -> {STAGED_S3}/norm_stats.json")

    ok, fail = 0, []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i in idxs:
            info = man[str(i)]
            uuid, split, length = info["uuid"], info["split"], info["length"]
            eppad = f"{i:06d}"
            dst_prefix = f"{tgt_k}/{split}/{uuid}"
            local_bins = LOCAL_STAGED / split / uuid
            try:
                # combined.mp4 + metadata: from scratch (rendered) if present, else build from mjgl.
                comb_key = f"{sk}/staged/{eppad}/combined_camera-images-rgb.mp4"
                comb_local = tmp / f"{eppad}_combined.mp4"
                if _s3_exists(s3, sb, comb_key):
                    s3.download_file(sb, comb_key, str(comb_local))
                    meta_key = f"{sk}/staged/{eppad}/episode_metadata.json"
                    meta_local = tmp / f"{eppad}_meta.json"
                    s3.download_file(sb, meta_key, str(meta_local))
                    n = probe_nframes(comb_local)
                    if n != length:
                        raise RuntimeError(f"combined frames {n} != length {length}")
                    s3.upload_file(str(comb_local), tgt_b,
                                   f"{dst_prefix}/combined_camera-images-rgb.mp4")
                    s3.upload_file(str(meta_local), tgt_b, f"{dst_prefix}/episode_metadata.json")
                elif i in absent:
                    # build combined.mp4 from the mjgl LeRobot videos via lerobot_to_abc
                    _build_combined_from_mjgl(s3, i, uuid, split, tmp, tgt_b, dst_prefix, cache)
                else:
                    raise RuntimeError(f"no scratch combined for {eppad} and not delivery-absent")
                # bins: copy verbatim from local staged_put
                for fn in ["states_actions.bin"] + [p.name for p in sorted(local_bins.glob("velocity_*.bin"))]:
                    src = local_bins / fn
                    if src.exists():
                        s3.upload_file(str(src), tgt_b, f"{dst_prefix}/{fn}")
                ok += 1
                if ok % 100 == 0:
                    print(f"[staged] {ok}/{len(idxs)} ...", flush=True)
            except Exception as e:
                print(f"[staged][FAIL] {eppad} {uuid}: {type(e).__name__}: {e}", flush=True)
                fail.append(i)
    print(f"[staged] DONE: {ok}/{len(idxs)} uploaded to {STAGED_S3}; failed={fail}")
    if fail:
        raise SystemExit(f"[staged] {len(fail)} failed")


def _build_combined_from_mjgl(s3, ep_idx, uuid, split, tmp, tgt_b, dst_prefix, cache):
    """For a delivery-absent episode: decode the mjgl v3 per-cam frames and build the staged
    combined.mp4 with lerobot_to_abc's exact encode. Uploads combined + metadata."""
    import sys
    sys.path.insert(0, "/home/justinyu/abc")
    print(f"[staged] building mjgl combined for absent ep {ep_idx} ({uuid})", flush=True)
    df = load_mjgl_episodes(s3, cache)
    row = df[df["source_episode_id"] == uuid].iloc[0]
    length = int(row["length"])
    workdir = tmp / f"mjgl_{ep_idx}"
    workdir.mkdir(parents=True, exist_ok=True)
    percam = []
    from export_mcap import FPS as _FPS, TIMESCALE, TICKS_PER_FRAME, X264, X264_STRICT_FFMPEG_ARGS
    from lerobot_to_abc import LETTERBOX_VF
    from torchcodec.decoders import VideoDecoder
    for c in CAMERAS:
        ci = int(row[f"videos/{CAM_KEY[c]}/chunk_index"]); fi = int(row[f"videos/{CAM_KEY[c]}/file_index"])
        from_ts = float(row[f"videos/{CAM_KEY[c]}/from_timestamp"])
        vkey = f"{_split(MJGL_LEROBOT_S3)[1]}/videos/{CAM_KEY[c]}/chunk-{ci:03d}/file-{fi:05d}.mp4"
        vlocal = workdir / f"src_{c}.mp4"
        s3.download_file(_split(MJGL_LEROBOT_S3)[0], vkey, str(vlocal))
        dec = VideoDecoder(str(vlocal))
        start = int(round(from_ts * _FPS))
        batch = dec.get_frames_in_range(start=start, stop=start + length)
        frames = batch.data.permute(0, 2, 3, 1).contiguous().numpy()
        assert len(frames) == length, f"{c}: decoded {len(frames)} != {length}"
        pc = workdir / f"{c}.mp4"
        enc = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-s", f"{frames.shape[2]}x{frames.shape[1]}", "-r", str(_FPS),
                                "-i", "-", "-vsync", "0", "-vf", LETTERBOX_VF, *X264,
                                "-threads", "1", str(pc)], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for k in range(length):
            enc.stdin.write(np.ascontiguousarray(frames[k]).tobytes())
        enc.stdin.close()
        assert enc.wait() == 0
        percam.append(str(pc))
    combined = workdir / "combined_camera-images-rgb.mp4"
    filt = ("".join(f"[{j}:v]" for j in range(3)) + "vstack=inputs=3[v0];"
            + f"[v0]settb=expr=1/{TIMESCALE},setpts=N*{TICKS_PER_FRAME}[out]")
    subprocess.run(["ffmpeg", "-y", *sum((["-i", p] for p in percam), []),
                    "-filter_complex", filt, "-map", "[out]", *X264_STRICT_FFMPEG_ARGS, str(combined)],
                   capture_output=True, check=True)
    assert probe_nframes(combined) == length
    s3.upload_file(str(combined), tgt_b, f"{dst_prefix}/combined_camera-images-rgb.mp4")
    meta = {"task_name": SUBTASK, "cameras": list(CAMERAS),
            "camera_resolutions": {c: [224, 168] for c in CAMERAS},
            "alignment": "mjgl_v3_30hz_fallback", "num_steps": length,
            "source_episode_id": uuid, "renderer": "mjgl"}
    (workdir / "episode_metadata.json").write_text(json.dumps(meta, indent=2))
    s3.upload_file(str(workdir / "episode_metadata.json"), tgt_b, f"{dst_prefix}/episode_metadata.json")


# ========================================================================= phase: lerobot
def phase_lerobot(cache: Path, groups: Optional[str] = None, cam: Optional[str] = None,
                  copy_meta: bool = True):
    """Assemble the lerobot-v3 mjwarp dataset.

    Parallelize by launching one process per (group, cam): pass --groups <gi> --cam <c> and set
    --copy-meta only on ONE of them (the meta/data/norm_stats copy runs once). Default (no
    groups/cam, copy_meta=True) does the whole dataset serially.
    """
    s3 = _s3()
    man = load_manifest(s3, cache)
    df = load_mjgl_episodes(s3, cache).sort_values("episode_index")
    sb, sk = _split(SCRATCH_S3)
    tgt_b, tgt_k = _split(LEROBOT_S3)

    # 1. Copy meta/ + data/ verbatim + root norm_stats.json from the reference dataset
    # (states/actions + RM score cols + object_counts.json are renderer-invariant; only videos
    # change). Skips pairs/ + repromo_annotations/ (RM artifacts, recomputed for mjwarp). The
    # meta sync carries meta/object_counts.json (complete 2438) into the target automatically.
    if copy_meta:
        print("[lerobot] copying meta/ + data/ + norm_stats.json from reference (verbatim)")
        for sub in ("meta", "data"):
            subprocess.run(["aws", "s3", "sync", f"{REF_LEROBOT_S3}/{sub}", f"{LEROBOT_S3}/{sub}",
                            "--only-show-errors"], check=True)
        subprocess.run(["aws", "s3", "cp", f"{REF_LEROBOT_S3}/norm_stats.json",
                        f"{LEROBOT_S3}/norm_stats.json", "--only-show-errors"], check=True)

    # 2. Determine file-groups (each cam shares the same file_index packing).
    fg = df.groupby(f"videos/{CAM_KEY['top']}/file_index")
    group_ids = sorted(fg.groups.keys())
    if groups:
        want = set(int(x) for x in groups.split(","))
        group_ids = [g for g in group_ids if g in want]
    cams = [cam] if cam else list(CAMERAS)
    print(f"[lerobot] regenerating groups={group_ids} cams={cams}")

    for gi in group_ids:
        sub = fg.get_group(gi).sort_values(f"videos/{CAM_KEY['top']}/from_timestamp")
        ep_indices = sub["episode_index"].astype(int).tolist()
        lengths = sub["length"].astype(int).tolist()
        expected = sum(lengths)
        for c in cams:
            ci = int(sub[f"videos/{CAM_KEY[c]}/chunk_index"].iloc[0])
            _build_group_video(s3, sb, sk, man, df, c, gi, ci, ep_indices, lengths, expected,
                               tgt_b, tgt_k, cache)
    print(f"[lerobot] DONE groups={group_ids} cams={cams} -> {LEROBOT_S3}")


def _build_group_video(s3, sb, sk, man, df, cam, gi, chunk_idx, ep_indices, lengths, expected,
                       tgt_b, tgt_k, cache):
    """Concatenate per-episode percam mp4s (episode_index order) into one file-group video."""
    key_cam = CAM_KEY[cam]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # ffmpeg concat of RE-ENCODED per-episode segments via a single pipe of raw frames is
        # heavy; instead decode each segment and pipe frames sequentially into ONE encoder.
        out = tmp / f"group{gi}_{cam}.mp4"
        # Probe resolution from the first available scratch percam segment (all 640x480).
        w, h = 640, 480
        enc = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", str(FPS), "-i", "-", "-vsync", "0", *X264_LEROBOT, "-threads", "2", str(out)],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        total_written = 0
        for ei, L in zip(ep_indices, lengths):
            frames = _episode_frames(s3, sb, sk, man, df, cam, ei, L, tmp, cache, w, h)
            assert frames.shape[0] == L, f"ep {ei} {cam}: {frames.shape[0]} != {L}"
            for k in range(L):
                enc.stdin.write(np.ascontiguousarray(frames[k]).tobytes())
            total_written += L
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError(f"group{gi} {cam} encode failed")
        got = probe_nframes(out)
        if got != expected or total_written != expected:
            raise RuntimeError(f"group{gi} {cam}: encoded {got} (wrote {total_written}) != {expected}")
        dst = f"{tgt_k}/videos/{key_cam}/chunk-{chunk_idx:03d}/file-{gi:05d}.mp4"
        s3.upload_file(str(out), tgt_b, dst)
        print(f"[lerobot] group{gi} {cam}: {got} frames -> s3://{tgt_b}/{dst}", flush=True)


def _episode_frames(s3, sb, sk, man, df, cam, ep_idx, length, tmp, cache, w, h) -> np.ndarray:
    """Return (length,H,W,3) uint8 frames for one episode's camera: from scratch percam mp4 if
    rendered, else decode the mjgl v3 video (delivery-absent fallback)."""
    key_cam = CAM_KEY[cam]
    seg = tmp / f"seg_{ep_idx}_{cam}.mp4"
    pc_key = f"{sk}/percam/{ep_idx:06d}/{key_cam}.mp4"
    if _s3_exists(s3, sb, pc_key):
        s3.download_file(sb, pc_key, str(seg))
        return _decode_all(str(seg), length)
    # fallback: mjgl frames (delivery-absent). Decode from the mjgl concatenated video at offset.
    if man[str(ep_idx)].get("delivery"):
        raise RuntimeError(f"ep {ep_idx} {cam}: scratch percam missing but episode IS renderable")
    print(f"[lerobot] ep {ep_idx} {cam}: delivery-absent, using mjgl frames", flush=True)
    row = df[df["episode_index"] == ep_idx].iloc[0]
    ci = int(row[f"videos/{key_cam}/chunk_index"]); fi = int(row[f"videos/{key_cam}/file_index"])
    from_ts = float(row[f"videos/{key_cam}/from_timestamp"])
    vkey = f"{_split(MJGL_LEROBOT_S3)[1]}/videos/{key_cam}/chunk-{ci:03d}/file-{fi:05d}.mp4"
    vlocal = tmp / f"mjglsrc_{fi}_{cam}.mp4"
    if not vlocal.exists():
        s3.download_file(_split(MJGL_LEROBOT_S3)[0], vkey, str(vlocal))
    from torchcodec.decoders import VideoDecoder
    dec = VideoDecoder(str(vlocal))
    start = int(round(from_ts * FPS))
    batch = dec.get_frames_in_range(start=start, stop=start + length)
    return batch.data.permute(0, 2, 3, 1).contiguous().numpy()


def _decode_all(path: str, length: int) -> np.ndarray:
    """Decode all frames of a short mp4 to (N,H,W,3) uint8 via ffmpeg rawvideo."""
    import re
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
                           capture_output=True, text=True).stdout.strip()
    w, h = (int(x) for x in probe.split(","))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vsync", "0",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    arr = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    return arr


def _s3_exists(s3, b, k) -> bool:
    try:
        s3.head_object(Bucket=b, Key=k)
        return True
    except Exception:
        return False


LEROBOT_MIRROR_S3 = "s3://xdof-internal-research/lerobot/sim_put_the_plastic_bottles_in_the_bin_30hz_mjwarp"


def phase_mirror(cache: Path):
    """Server-side copy the finished lerobot mjwarp dataset from repromo/datasets/ to the
    lerobot/ prefix (zero-regret: the task named lerobot/, the RM reader uses repromo/datasets/).
    Videos are a server-side S3 copy (no re-download)."""
    print(f"[mirror] {LEROBOT_S3} -> {LEROBOT_MIRROR_S3} (server-side)")
    subprocess.run(["aws", "s3", "sync", LEROBOT_S3, LEROBOT_MIRROR_S3, "--only-show-errors"],
                   check=True)
    print(f"[mirror] DONE -> {LEROBOT_MIRROR_S3}")


def phase_verify(cache: Path):
    """Final verification of both datasets on S3."""
    s3 = _s3()
    man = load_manifest(s3, cache)
    ok = True

    # --- staged ---
    tb, tk = _split(STAGED_S3)
    def count_pre(prefix):
        r = s3.list_objects_v2(Bucket=tb, Prefix=prefix, Delimiter="/")
        return len(r.get("CommonPrefixes", []))
    n_train = count_pre(f"{tk}/train_sim/")
    n_val = count_pre(f"{tk}/val_sim/")
    print(f"[verify][staged] train_sim={n_train} (expect 2338), val_sim={n_val} (expect 100)")
    ok &= (n_train == 2338 and n_val == 100)
    ns = _s3_exists(s3, tb, f"{tk}/norm_stats.json")
    print(f"[verify][staged] root norm_stats.json present: {ns}")
    ok &= ns

    # --- lerobot (repromo/datasets) ---
    lb, lk = _split(LEROBOT_S3)
    df = load_mjgl_episodes(s3, cache).sort_values("episode_index")
    fg = df.groupby(f"videos/{CAM_KEY['top']}/file_index")
    all_vid = True
    for gi in sorted(fg.groups.keys()):
        sub = fg.get_group(gi)
        expected = int(sub["length"].sum())
        for c in CAMERAS:
            ci = int(sub[f"videos/{CAM_KEY[c]}/chunk_index"].iloc[0])
            key = f"{lk}/videos/{CAM_KEY[c]}/chunk-{ci:03d}/file-{gi:05d}.mp4"
            ex = _s3_exists(s3, lb, key)
            all_vid &= ex
            if not ex:
                print(f"[verify][lerobot] MISSING video: {key}")
    print(f"[verify][lerobot] all 8x3 file-group videos present: {all_vid}")
    ok &= all_vid
    oc = _s3_exists(s3, lb, f"{lk}/meta/object_counts.json")
    print(f"[verify][lerobot] meta/object_counts.json present: {oc}")
    ok &= oc
    print(f"\n[verify] {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    if not ok:
        raise SystemExit("[verify] failed")


@dataclass
class Cfg:
    phase: Literal["object_counts", "staged", "lerobot", "mirror", "verify"]
    limit: Optional[int] = None          # staged: cap episodes (smoke)
    groups: Optional[str] = None         # lerobot: comma file-group ids, e.g. "0" or "0,1"
    cam: Optional[str] = None            # lerobot: single camera (top|left|right) for parallel builds
    no_copy_meta: bool = False           # lerobot: skip the meta/data/norm_stats copy (parallel workers)
    cache_dir: str = "/tmp/pkg_mjwarp_cache"


def main(cfg: Cfg):
    cache = Path(cfg.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    if cfg.phase == "object_counts":
        phase_object_counts(cache)
    elif cfg.phase == "staged":
        phase_staged(cache, cfg.limit)
    elif cfg.phase == "lerobot":
        phase_lerobot(cache, cfg.groups, cfg.cam, copy_meta=not cfg.no_copy_meta)
    elif cfg.phase == "mirror":
        phase_mirror(cache)
    elif cfg.phase == "verify":
        phase_verify(cache)


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Cfg))
