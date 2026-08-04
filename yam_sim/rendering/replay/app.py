"""CLI entrypoint for modular episode replay."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Must be set before mujoco / yam_sim are imported.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

from yam_sim.rendering.replay.camera_providers import create_camera_provider
from yam_sim.rendering.replay.episode import load_episode_context
from yam_sim.rendering.replay.runtime import create_replay_session
from yam_sim.rendering.replay.viewer import ReplayViewer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a raw, delivered, or dataset episode through yam-sim + Viser"
    )
    parser.add_argument("episode_dir", help="Path to a raw, delivered, or dataset episode dir")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument(
        "--camera-source",
        choices=("recorded", "sim", "none"),
        default="recorded",
        help="Recorded MP4s, live simulated cameras, or disable sidebar cameras",
    )
    parser.add_argument(
        "--sim-render-backend",
        choices=("mujoco", "mjwarp", "madrona"),
        default="madrona",
        help="Backend for --camera-source sim",
    )
    parser.add_argument("--render-width", type=int, default=640, help="Sim camera render width")
    parser.add_argument("--render-height", type=int, default=480, help="Sim camera render height")
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU to use for MJWarp/Madrona rendering")
    parser.add_argument(
        "--replay-mode",
        choices=("auto", "physics", "qpos"),
        default="auto",
        help="Physics re-step or direct state replay when replay states are available",
    )
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)
    context = load_episode_context(
        episode_dir,
        load_recorded_cameras=(args.camera_source == "recorded"),
    )

    print(f"\nCreating replay session (scene={context.scene}, task={context.task}, mode={args.replay_mode}) ...")
    if context.instruction:
        print(f"Instruction: {context.instruction!r}")
    session, control_hz = create_replay_session(context, mode=args.replay_mode)
    print(f"Control Hz: {control_hz:.1f}")
    print(f"\nAction timeline: {len(session.actions)} steps, {session.duration_s:.1f}s at {session.timeline_hz:.1f} Hz")
    if session.sim_states is not None:
        label = "Sim qpos" if session.has_exact_qpos else "Replay states"
        print(f"{label}: {len(session.sim_states)} frames (dim={session.sim_states.shape[1]})")
    else:
        print("No replay states found - only physics re-step mode available")
    if context.rand_state is not None:
        print(f"Randomization restored: {len(context.rand_state.object_states)} objects")
    else:
        print("No randomization.json found - using default scene layout")

    camera_provider = create_camera_provider(
        source=args.camera_source,
        sim_backend=args.sim_render_backend,
        episode_camera_frames=context.streams.camera_frames,
        episode_camera_ts=context.streams.camera_ts,
        env=session.env,
        width=args.render_width,
        height=args.render_height,
        gpu_id=args.gpu_id,
    )

    viewer = ReplayViewer(
        session,
        camera_provider,
        port=args.port,
        speed=args.speed,
    )
    viewer.run()


if __name__ == "__main__":
    main()
