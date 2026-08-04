"""Robot and camera configuration dataclasses for the YAM bimanual system."""

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CameraConfig:
    """Configuration for a single camera."""

    serial: str = ""
    height: int = 480
    width: int = 640
    fps: int = 30
    socket: Optional[str] = None


@dataclass
class LeaderConfig:
    """Configuration for a leader (teleoperation) device."""

    device_name: str = "/dev/ttyUSB0"
    control_rate: int = 400
    hardware: str = "gello"
    servo_ids: Optional[List[int]] = None


@dataclass
class FollowerConfig:
    """Configuration for a follower (robot arm) device."""

    channel: str = ""
    control_rate: int = 30
    use_sim: bool = False


@dataclass
class RobotConfig:
    """Configuration for a single robot arm."""

    leader: LeaderConfig = field(default_factory=LeaderConfig)
    follower: FollowerConfig = field(default_factory=FollowerConfig)
    root_pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    root_ori: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    init_q: List[float] = field(default_factory=lambda: [0.0] * 7)


@dataclass
class PolicyConfig:
    """Configuration for policy inference timing."""

    dt: float = 0.03333  # ~30 Hz


@dataclass
class RobotSystemConfig:
    """Full system configuration: cameras, robots, policy."""

    cameras: Dict[str, CameraConfig] = field(default_factory=dict)
    robots: Dict[str, RobotConfig] = field(default_factory=dict)
    policy: PolicyConfig = field(default_factory=PolicyConfig)


def _base_config() -> RobotSystemConfig:
    """Base configuration with all common settings."""
    return RobotSystemConfig(
        cameras={
            "top": CameraConfig(
                serial="335122272485",
                height=480,
                width=640,
                fps=30,
                socket="top_rgb",
            ),
            "left": CameraConfig(
                serial="230322276861",
                height=480,
                width=640,
                fps=30,
                socket="left_rgb",
            ),
            "right": CameraConfig(
                serial="218622274707",
                height=480,
                width=640,
                fps=30,
                socket="right_rgb",
            ),
        },
        robots={
            "left": RobotConfig(
                leader=LeaderConfig(device_name="/dev/ttyUSB0", control_rate=400),
                follower=FollowerConfig(channel="can_l_foll", control_rate=30),
                root_pos=[0.0, 0.3, 0.0],
                root_ori=[1.0, 0.0, 0.0, 0.0],
                init_q=[
                    -0.20656902,
                    0.47283894,
                    0.99431604,
                    -0.7043946,
                    -0.30842298,
                    -0.32864118,
                    0.9987507,
                ],
            ),
            "right": RobotConfig(
                leader=LeaderConfig(device_name="/dev/ttyUSB1", control_rate=400),
                follower=FollowerConfig(channel="can_r_foll", control_rate=30),
                root_pos=[0.0, -0.3, 0.0],
                root_ori=[1.0, 0.0, 0.0, 0.0],
                init_q=[
                    0.20160982,
                    0.39005876,
                    1.1182956,
                    -0.8726253,
                    0.13332571,
                    0.42629892,
                    0.9895317,
                ],
            ),
        },
        policy=PolicyConfig(dt=0.03333),
    )


def get_i2rt_config() -> RobotSystemConfig:
    """Get the real-robot configuration."""
    return _base_config()


def get_i2rt_sim_config() -> RobotSystemConfig:
    """Get the simulation configuration (same as real but with use_sim=True)."""
    config = _base_config()
    for robot in config.robots.values():
        robot.follower.use_sim = True
    init_q_json = os.environ.get("DEPLOY_INIT_Q")
    if init_q_json:
        init_q = json.loads(init_q_json)
        robot_names = list(config.robots.keys())
        dof = len(init_q) // len(robot_names)
        for i, name in enumerate(robot_names):
            config.robots[name].init_q = list(init_q[i * dof : (i + 1) * dof])
    return config


# Symmetric flat init_q used by the cbox station in robots_realtime.
_FLAT_INIT_Q = [0.0, 0.5, 1.0, -1.0, 0.0, 0.0, 1.0]


def get_viser_ik_config() -> RobotSystemConfig:
    """Get the config for interactive Viser IK teleoperation.

    Uses a symmetric flat init_q (both arms identical) so the robot
    starts in an upright, centred pose suitable for interactive control.
    The default sim config uses the bbox station's asymmetric init_q
    which is calibrated for real-robot replay.
    """
    config = copy.deepcopy(get_i2rt_sim_config())
    for robot in config.robots.values():
        robot.init_q = list(_FLAT_INIT_Q)
    return config
