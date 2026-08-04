#!/usr/bin/env python3
"""Re-render one sim episode's cameras at 30 Hz with the faithful mjwarp 3.10 eval
renderer and write a DROP-IN delivery-format output.mcap:

  * camera topics (/top-camera, /left-wrist-camera, /right-wrist-camera) are
    replaced with re-rendered 30 Hz frames, one foxglove.CompressedVideo (H264
    access unit) message per frame;
  * every non-camera topic (proprio / command / leader / instruction / ...) is
    copied verbatim from the source output.mcap.

Run under abc-rabc/.venv (mujoco_warp 3.10). Requires ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import threading
from pathlib import Path

import numpy as np
import mujoco
import warp as wp
import mujoco_warp as mjw

# scene camera name -> delivery mcap topic
CAM_TOPIC = {
    "top": "/top-camera/image-raw",
    "left": "/left-wrist-camera/image-raw",
    "right": "/right-wrist-camera/image-raw",
}
AUD = b"\x00\x00\x00\x01\x09"  # 4-byte start code + H264 access-unit-delimiter NAL


# ── rendering (batched, faithful 3.10 eval config) ──────────────────────────────
# Original collection root ".../xdof_sim/models/" -> local yam_sim checkout. yam_sim
# is the asset set the published dataset + eval renderer use (correct dark grippers);
# the xdof-sim checkout's i2rt_yam gripper renders bright blue (wrong). yam_sim has the
# full task_dishrack plate/dish_rack set too; we bypass yam_sim's own normalizer only
# because its legacy_asset_prefixes step mis-maps plate_19/20 + dish_rack_10.
LOCAL_MODELS = "/home/karimelrafi/yam_sim/yam_sim/models/"
_ROOT_RE = re.compile(r"/(?:[^\"'<>\s]+/)*xdof_sim/models/")
_HERE = Path(__file__).resolve().parent
_ALIASES_JSON = str(_HERE / "dishrack_aliases.json")


def _load_dishrack_aliases() -> dict:
    import json
    try:
        return json.load(open(_ALIASES_JSON))
    except Exception:
        return {}


_DISHRACK_ALIASES = _load_dishrack_aliases()


STUB_OBJ = str(_HERE / "_stub" / "cube.obj")
STUB_PNG = str(_HERE / "_stub" / "white.png")


def load_render_model(raw_scene_xml: str) -> "mujoco.MjModel":
    """Compile a render-faithful model straight from the delivered scene_assembled.xml:

      1. rewrite the asset root to the local yam_sim checkout (matches the published
         dataset + eval renderer; xdof-sim's i2rt_yam gripper renders wrong/blue);
      2. apply dishrack/plate variant-directory aliases (DishRack050 -> dish_rack_12)
         so task_dishrack assets resolve locally — WITHOUT yam_sim's buggy
         legacy_asset_prefixes remap (which sends plate_19/20 + dish_rack_10 to an
         empty assets_robocasa layout);
      3. repoint any missing mesh/texture FILE at a tiny stand-in instead of removing
         scene elements. Removing/re-adding <mesh>/<texture> re-indexes mjwarp's
         texcoord arrays and corrupts texture sampling on OTHER (rendered) geoms —
         the blue-gripper bug. Missing meshes are collision-only (group 3, which the
         renderer's enabled_geom_groups=[0,1,2] never draws); missing textures are
         phantom refs on flat materials. The stubs are invisible / neutral, so
         rendered pixels are unchanged.

    Nothing about joints/qpos changes, so mjSTATE_INTEGRATION still applies.
    """
    import os as _os
    import xml.etree.ElementTree as ET

    xml = _ROOT_RE.sub(LOCAL_MODELS, Path(raw_scene_xml).read_text())
    for kind, aliases in _DISHRACK_ALIASES.items():
        for alias, canonical in aliases.items():
            xml = xml.replace(f"task_dishrack/{kind}/{alias}/",
                              f"task_dishrack/{kind}/{canonical}/")
    root = ET.fromstring(xml)
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    texturedir = compiler.get("texturedir", "") if compiler is not None else ""

    def _resolve(f, d):
        return f if _os.path.isabs(f) else _os.path.join(d, f)

    for mesh in root.iter("mesh"):
        f = mesh.get("file")
        if f and not _os.path.exists(_resolve(f, meshdir)):
            mesh.set("file", STUB_OBJ)
    for tex in root.iter("texture"):
        f = tex.get("file")
        if f and not _os.path.exists(_resolve(f, texturedir)):
            tex.set("file", STUB_PNG)

    # Safety net: a jointed body whose only mass came from a stubbed collision mesh
    # still gets mass from the stub cube, but guarantee no zero-mass compile error.
    for body in root.iter("body"):
        if any(c.tag in ("joint", "freejoint") for c in body) and body.find("inertial") is None:
            body.insert(0, ET.Element("inertial", {
                "pos": "0 0 0", "mass": "0.1", "diaginertia": "1e-3 1e-3 1e-3"}))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def render_cameras(scene_xml: str, states: np.ndarray, *, w: int, h: int,
                   gpu_id: int, batch: int) -> tuple[dict[str, list[np.ndarray]], dict[str, int]]:
    m = load_render_model(scene_xml)
    d = mujoco.MjData(m)
    ssz = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION)
    if states.shape[1] != ssz:
        raise ValueError(f"state width {states.shape[1]} != mjSTATE_INTEGRATION {ssz}")
    cam_idx = {}
    for name in CAM_TOPIC:
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if i < 0:
            raise ValueError(f"camera {name!r} missing from scene")
        cam_idx[name] = i

    # materialize qpos per frame (setState handles the full integration layout)
    T = states.shape[0]
    qpos = np.empty((T, m.nq), np.float32)
    for i in range(T):
        mujoco.mj_setState(m, d, states[i], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_kinematics(m, d)
        qpos[i] = d.qpos

    wp.set_device(f"cuda:{gpu_id}")
    nworld = min(batch, T)
    m_warp = mjw.put_model(m)
    d_warp = mjw.put_data(m, d, nworld=nworld,
                          nconmax=max(int(m.nconmax), 4096),
                          njmax=max(int(m.njmax), 4096))
    # identical config to abc_minimal/eval_policy.py:MJWarpSim
    rc = mjw.create_render_context(
        mjm=m, nworld=nworld, cam_res=(w, h),
        render_rgb=[True] * m.ncam, render_depth=[False] * m.ncam,
        use_textures=True, use_shadows=True,
    )
    qdev = wp.array(qpos, dtype=wp.float32, device=d_warp.qpos.device)

    frames = {name: [] for name in CAM_TOPIC}
    for start in range(0, T, nworld):
        stop = min(start + nworld, T)
        n = stop - start
        wp.copy(d_warp.qpos[:n], qdev[start:stop])
        mjw.kinematics(m_warp, d_warp)
        mjw.com_pos(m_warp, d_warp)
        mjw.camlight(m_warp, d_warp)
        mjw.refit_bvh(m_warp, d_warp, rc)
        mjw.render(m_warp, d_warp, rc)
        rgba = rc.rgb_data.numpy().view(np.uint8).reshape(nworld, m.ncam, h, w, 4)
        for name, ci in cam_idx.items():
            for k in range(n):
                frames[name].append(rgba[k, ci, :, :, :3][..., ::-1].copy())  # BGR->RGB
    return frames, cam_idx


# ── H264 encode -> per-frame access units ───────────────────────────────────────
def encode_access_units(frames: list[np.ndarray], *, w: int, h: int, fps: float,
                        gop: int = 10) -> list[bytes]:
    """Encode frames to an Annex-B H264 stream (AUD per frame) and split into
    one access unit per frame. Returns list[bytes] of length len(frames)."""
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", str(gop),
         "-x264-params", "aud=1:scenecut=0:keyint=%d:min-keyint=%d" % (gop, gop),
         "-f", "h264", "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    def _feed():
        for fr in frames:
            proc.stdin.write(np.ascontiguousarray(fr).tobytes())
        proc.stdin.close()

    t = threading.Thread(target=_feed, daemon=True)
    t.start()
    stream = proc.stdout.read()
    proc.wait()
    t.join()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.read().decode()[:500]}")

    # split on AUD start codes; each AU starts with an AUD NAL
    parts = stream.split(AUD)
    aus = [AUD + p for p in parts[1:]]  # parts[0] is empty / pre-first-AUD
    if len(aus) != len(frames):
        raise RuntimeError(f"AU split mismatch: {len(aus)} AUs != {len(frames)} frames")
    return aus


# ── dynamic foxglove.CompressedVideo proto from the source mcap schema ──────────
def build_compressed_video_class(schema_data: bytes):
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    fds = descriptor_pb2.FileDescriptorSet.FromString(schema_data)
    pool = descriptor_pool.DescriptorPool()
    added, files = set(), {f.name: f for f in fds.file}

    def add(name):
        if name in added or name not in files:
            return
        for dep in files[name].dependency:
            add(dep)
        pool.Add(files[name])
        added.add(name)

    for n in list(files):
        add(n)
    desc = pool.FindMessageTypeByName("foxglove.CompressedVideo")
    return message_factory.GetMessageClass(desc)


# ── write drop-in mcap ──────────────────────────────────────────────────────────
def write_mcap(src_mcap: Path, out_mcap: Path, cam_aus: dict[str, list[bytes]],
               frame_log_times: np.ndarray) -> dict:
    from mcap.reader import make_reader
    from mcap.writer import Writer

    topic_cam = {v: k for k, v in CAM_TOPIC.items()}
    with open(src_mcap, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        schemas = {s.id: s for s in summary.schemas.values()}
        channels = {c.id: c for c in summary.channels.values()}

        # CompressedVideo schema (reuse from source) + dynamic message class
        cv_schema = next(s for s in schemas.values() if s.name == "foxglove.CompressedVideo")
        CV = build_compressed_video_class(cv_schema.data)

        out_mcap.parent.mkdir(parents=True, exist_ok=True)
        with open(out_mcap, "wb") as of:
            w = Writer(of)
            w.start(profile="", library="render_to_mcap")

            new_schema = {}
            for sid, s in schemas.items():
                new_schema[sid] = w.register_schema(name=s.name, encoding=s.encoding, data=s.data)
            new_chan = {}
            for cid, c in channels.items():
                new_chan[cid] = w.register_channel(
                    topic=c.topic, message_encoding=c.message_encoding,
                    schema_id=new_schema[c.schema_id], metadata=dict(c.metadata))

            # pass through every non-camera message verbatim
            f.seek(0)
            reader2 = make_reader(f)
            n_pass = 0
            for schema, channel, message in reader2.iter_messages():
                if channel.topic in topic_cam:
                    continue
                w.add_message(
                    channel_id=new_chan[channel.id],
                    log_time=message.log_time,
                    data=message.data,
                    publish_time=message.publish_time,
                    sequence=message.sequence,
                )
                n_pass += 1

            # inject re-rendered camera frames
            n_cam = 0
            for name, topic in CAM_TOPIC.items():
                cid = next(c.id for c in channels.values() if c.topic == topic)
                aus = cam_aus[name]
                for seq, (au, lt) in enumerate(zip(aus, frame_log_times)):
                    lt = int(lt)
                    msg = CV()
                    msg.timestamp.seconds = lt // 1_000_000_000
                    msg.timestamp.nanos = lt % 1_000_000_000
                    msg.frame_id = name
                    msg.data = au
                    msg.format = "h264"
                    w.add_message(
                        channel_id=new_chan[cid], log_time=lt,
                        data=msg.SerializeToString(), publish_time=lt, sequence=seq)
                    n_cam += 1
            w.finish()
    return {"passthrough_msgs": n_pass, "camera_msgs": n_cam}


def camera_frame_log_times(src_mcap: Path, n_frames: int) -> np.ndarray:
    """Evenly space n_frames log_times across the source top-camera log-time span
    so re-rendered frames stay aligned with the passed-through proprio timeline."""
    from mcap.reader import make_reader
    ts = []
    with open(src_mcap, "rb") as f:
        for _s, ch, m in make_reader(f).iter_messages(topics=["/top-camera/image-raw"]):
            ts.append(m.log_time)
    if len(ts) < 2:
        raise RuntimeError("source mcap has <2 top-camera frames; cannot derive timeline")
    return np.linspace(min(ts), max(ts), n_frames).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", required=True, help="source delivery dir")
    ap.add_argument("--scene", required=True, help="normalized scene xml")
    ap.add_argument("--out-mcap", required=True)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    ep = Path(args.episode_dir)
    states = np.load(ep / "integration_state.npy")
    src_mcap = ep / "output.mcap"

    frames, _ = render_cameras(args.scene, states, w=args.width, h=args.height,
                               gpu_id=args.gpu_id, batch=args.batch)
    T = len(frames["top"])
    cam_aus = {name: encode_access_units(frames[name], w=args.width, h=args.height, fps=args.fps)
               for name in CAM_TOPIC}
    log_times = camera_frame_log_times(src_mcap, T)
    stats = write_mcap(src_mcap, Path(args.out_mcap), cam_aus, log_times)
    print(f"Done: {T} frames/cam -> {args.out_mcap}  "
          f"(camera_msgs={stats['camera_msgs']}, passthrough={stats['passthrough_msgs']})")


if __name__ == "__main__":
    main()
