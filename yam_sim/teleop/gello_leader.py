"""GELLO leader node for teleoperation.

Reads joint positions from physical GELLO hardware (Dynamixel servos) and
publishes them as target actions over ZMQ for a sim or real follower.

Ported from abc/deploy/rllg2 for standalone use with yam-sim.

Requirements:
    pip install yam-sim[teleop] dynamixel-sdk

Usage:
    # Left leader on /dev/ttyUSB0 (default GELLO hardware)
    uv run python -m yam_sim.teleop.gello_leader \
        --name left --device /dev/ttyUSB0 --control-rate 400

    # Right leader on /dev/ttyUSB1 (default GELLO hardware)
    uv run python -m yam_sim.teleop.gello_leader \
        --name right --device /dev/ttyUSB1 --control-rate 400

    # CLAPD hardware (macOS USB serial)
    uv run python -m yam_sim.teleop.gello_leader \
        --name left --device /dev/cu.usbserial-FTAAMOEB --hardware clapd
    uv run python -m yam_sim.teleop.gello_leader \
        --name right --device /dev/cu.usbserial-FTAAMNKG --hardware clapd
"""

from __future__ import annotations

import argparse
import subprocess
import time

import numpy as np

from yam_sim.teleop.dynamixel import Dynamixel
from yam_sim.teleop.leader_robot import Robot
from yam_sim.teleop.node import Node

# Hardware presets: joint_signs and default servo IDs per hardware type.
# "gello" is the original GELLO arm; "clapd" is the CLAPD arm variant.
HARDWARE_PRESETS: dict[str, dict] = {
    "gello": {
        "joint_signs": [1, -1, -1, -1, 1, 1, 1],
        "default_servo_ids": {
            "_default": [40, 41, 42, 43, 44, 45, 46],
        },
    },
    "clapd": {
        "joint_signs": [1, -1, -1, 1, 1, 1, 1],
        "default_servo_ids": {
            "left": [20, 21, 22, 23, 24, 25, 26],
            "right": [30, 31, 32, 33, 34, 35, 36],
        },
    },
}


class GelloLeaderNode(Node):

    calibration_joint_pos = np.array([0.0] * 7)
    calibration_joint_pos[-1] = 0.357  # gripper

    num_arm_joints = 7

    LEADER_MAX_GRIPPER = 0.33736414
    LEADER_MIN_GRIPPER = -0.6040225
    FOLLOWER_MAX_GRIPPER = 1.0
    FOLLOWER_MIN_GRIPPER = -0.1

    def __init__(
        self,
        name: str,
        control_rate: float,
        device_name: str,
        initial_pos: list | None = None,
        servo_ids: list[int] | None = None,
        hardware: str = "gello",
    ):
        super().__init__(name, control_rate)

        self.device_name = device_name
        self.initial_pos = None
        if initial_pos is not None:
            self.initial_pos = np.array(initial_pos)

        # Apply hardware preset
        if hardware not in HARDWARE_PRESETS:
            raise ValueError(
                f"Unknown hardware '{hardware}'. "
                f"Available: {list(HARDWARE_PRESETS.keys())}"
            )
        preset = HARDWARE_PRESETS[hardware]
        self.joint_signs = np.array(preset["joint_signs"])

        # Create publisher for leader actions
        self.leader_topic_name = f"{self._name}_actions"
        self.create_publisher(self.leader_topic_name)

        # Setup leader robot
        leader_dynamixel = Dynamixel.Config(
            baudrate=4_000_000, device_name=device_name
        ).instantiate()
        if servo_ids is None:
            id_map = preset["default_servo_ids"]
            servo_ids = id_map.get(name, id_map.get("_default", id_map[next(iter(id_map))]))
        self.leader = Robot(leader_dynamixel, servo_ids=servo_ids)

        self.joint_min = (
            np.array([-120.0, 0.0, 0.0, -75.0, -85.0, -115.0]) * np.pi / 180.0
        )
        self.joint_max = (
            np.array([120.0, 180.0, 180.0, 75.0, 85.0, 115.0]) * np.pi / 180.0
        )
        self.joint_range = self.joint_max - self.joint_min
        self.joint_min += self.joint_range * 0.01
        self.joint_max -= self.joint_range * 0.01
        self.joint_max[2] -= self.joint_range[2] * 0.3

        self.prev_time = None
        self.prev_gripper_position = None
        self.prev_time_q = None
        self.qd = np.zeros(6)
        self.kp = np.array([0.8, 6, 6, 1.3, 0.5, 0.5]) / 2
        self.kd = 0.04 * 1.3 / 2

        self._get_dynamixel_offsets()
        self.verify_latency_timer()

    def verify_latency_timer(self) -> None:
        suffix = self.device_name.split("/")[-1]
        command = f"cat /sys/bus/usb-serial/devices/{suffix}/latency_timer"
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, check=True
            )
            ttyUSB_latency_timer = int(result.stdout)
            if ttyUSB_latency_timer != 1:
                raise Exception(
                    f"Latency timer of {suffix} is {ttyUSB_latency_timer}, should be 1. Run:\n"
                    f"  echo 1 | sudo tee /sys/bus/usb-serial/devices/{suffix}/latency_timer"
                )
        except subprocess.CalledProcessError:
            print(f"Warning: Could not verify latency timer for {suffix}")

    def _get_dynamixel_offsets(self, verbose=True):
        """Calibrate Dynamixel servos to match follower arm joint positions.

        Before launching, place the leader arm roughly in the calibration position
        (within +/-90 degrees per joint).
        """
        # Warm up
        for _ in range(10):
            self._read_raw_position()

        def _get_error(calibration_joint_pos, offset, index, joint_state):
            joint_sign_i = self.joint_signs[index]
            joint_i = joint_sign_i * (joint_state[index] - offset)
            start_i = calibration_joint_pos[index]
            return np.abs(joint_i - start_i)

        self.joint_offsets = []
        curr_pos = self._read_raw_position()
        print(f"Calibrating from raw position: {curr_pos}")
        for i in range(self.num_arm_joints):
            best_offset = 0
            best_error = 1e9
            for offset in np.linspace(-20 * np.pi, 20 * np.pi, 20 * 4 * 2 + 1):
                error = _get_error(
                    self.calibration_joint_pos, offset, i, curr_pos
                )
                if error < best_error:
                    best_error = error
                    best_offset = offset
            self.joint_offsets.append(best_offset)

        self.joint_offsets = np.asarray(self.joint_offsets)
        if verbose:
            print(
                "Calibration offsets: ",
                [f"{x:.3f}" for x in self.joint_offsets],
            )

    def _read_raw_position(self) -> np.ndarray:
        positions, elapsed_time = self.leader.read_position()
        return (np.array(positions) / 2048 - 1) * np.pi

    def read_position(self) -> np.ndarray:
        raw_positions = self._read_raw_position()
        pos = self.joint_signs * (raw_positions - self.joint_offsets)
        pos[-1] = self.get_gripper_pos(pos[-1])
        return pos

    def get_gripper_pos(self, leader_gripper_pos: float) -> float:
        gripper_position = (leader_gripper_pos - self.LEADER_MIN_GRIPPER) / (
            self.LEADER_MAX_GRIPPER - self.LEADER_MIN_GRIPPER
        ) * (self.FOLLOWER_MAX_GRIPPER - self.FOLLOWER_MIN_GRIPPER) + self.FOLLOWER_MIN_GRIPPER
        return np.clip(
            gripper_position, self.FOLLOWER_MIN_GRIPPER, self.FOLLOWER_MAX_GRIPPER
        )

    def get_gripper_torque(self, curr_pos: float) -> float:
        gripper_position = curr_pos[-1]
        gripper_current = 0.1 * (1.06 - gripper_position)
        if self.prev_time is not None:
            dt = time.perf_counter() - self.prev_time
            gripper_vel = (gripper_position - self.prev_gripper_position) / dt
            gripper_current -= 0.005 * gripper_vel
        self.prev_gripper_position = gripper_position
        self.prev_time = time.perf_counter()
        return gripper_current

    def get_joint_limit_barrier_torque(self, position: np.ndarray) -> np.ndarray:
        exceed_max_mask = position > self.joint_max
        tau_l = -0.3 * (position - self.joint_max) * exceed_max_mask
        exceed_min_mask = position < self.joint_min
        tau_l += -0.3 * (position - self.joint_min) * exceed_min_mask
        return tau_l * self.joint_signs[:-1]

    def get_pos_control_torque(self, q, q_des) -> np.ndarray:
        arm_torque = self.kp * (q_des - q)
        beta = 0.7
        if self.prev_time_q is not None:
            dt = time.perf_counter() - self.prev_time_q
            q_d = (q - self.prev_q) / dt
            self.qd = beta * self.qd + (1 - beta) * q_d
            arm_torque -= self.kd * self.qd
        self.prev_q = q
        self.prev_time_q = time.perf_counter()
        return arm_torque * self.joint_signs[:-1]

    def set_arm_torque(self, position: np.ndarray) -> None:
        arm_torque = np.zeros(7)
        arm_torque[-1] = self.get_gripper_torque(position)
        arm_torque[:-1] += self.get_joint_limit_barrier_torque(position[:-1])
        self.set_arm_current(arm_torque)

    def set_arm_current(self, arm_torque: np.ndarray) -> None:
        arm_current = (arm_torque * 1158.73).astype(int)
        self.leader.set_current(arm_current)

    def set_initial_pos(self):
        start_time = time.time()
        t = time.time() - start_time
        T = 2.0
        while t <= T:
            q = self.read_position()[:-1]
            q_des = (t / T) * self.initial_pos + (1 - t / T) * q
            arm_torque = np.zeros(7)
            arm_torque[:-1] = self.get_pos_control_torque(q, q_des)
            self.set_arm_current(arm_torque)
            t = time.time() - start_time
            time.sleep(1 / 60)

    def get_follower_target_pos(self) -> np.ndarray:
        target_pos = self.read_position()
        gripper = (target_pos[-1] - self.FOLLOWER_MIN_GRIPPER) / (
            self.FOLLOWER_MAX_GRIPPER - self.FOLLOWER_MIN_GRIPPER
        )
        if gripper < 0.2:
            target_pos[-1] = self.FOLLOWER_MIN_GRIPPER
        return target_pos

    def initial_bootup(self, *args, **kwargs) -> None:
        if self.initial_pos is not None:
            print(f"[{self._name}] Setting initial position to {self.initial_pos}")
            self.set_initial_pos()

        initial_pos = self.get_follower_target_pos()
        self.publish(
            self.leader_topic_name, initial_pos, extras={"type": "interp"}
        )
        print(f"[{self._name}] Leader ready, publishing to {self.leader_topic_name}")

    def tick(self) -> None:
        target_pos = self.get_follower_target_pos()
        self.publish(
            self.leader_topic_name, target_pos, extras={"type": "servo"}
        )
        self.set_arm_torque(target_pos)

    def on_shutdown(self) -> None:
        try:
            self.leader._disable_torque()
        except (ConnectionError, OSError):
            pass
        print(f"[{self._name}] Leader shutdown.")


def main():
    parser = argparse.ArgumentParser(description="GELLO leader node for teleoperation")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Leader name, used as ZMQ topic prefix (e.g. 'left' or 'right')",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Serial device path (e.g. /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--control-rate",
        type=float,
        default=400,
        help="Control loop rate in Hz (default: 400)",
    )
    parser.add_argument(
        "--initial-pos",
        type=float,
        nargs="+",
        default=None,
        help="Optional 6D initial arm position to move to on startup",
    )
    parser.add_argument(
        "--servo-ids",
        type=int,
        nargs="+",
        default=None,
        help="Dynamixel servo IDs (default: 40-46 for gello, 20-26/30-36 for clapd)",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        choices=list(HARDWARE_PRESETS.keys()),
        default="gello",
        help="Hardware type preset (default: gello)",
    )
    args = parser.parse_args()

    leader = GelloLeaderNode(
        name=args.name,
        control_rate=args.control_rate,
        device_name=args.device,
        initial_pos=args.initial_pos,
        servo_ids=args.servo_ids,
        hardware=args.hardware,
    )
    leader.run()


if __name__ == "__main__":
    main()
