"""Low-level robot interface for GELLO leader arms (Dynamixel servo batch control).

Ported from abc/deploy/rllg2 for standalone use with yam-sim.
Requires: pip install dynamixel-sdk
"""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import Union

import numpy as np
from dynamixel_sdk import (
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
    GroupSyncRead,
    GroupSyncWrite,
)

from yam_sim.teleop.dynamixel import Dynamixel, OperatingMode, ReadAttribute


class MotorControlType(Enum):
    PWM = auto()
    POSITION_CONTROL = auto()
    DISABLED = auto()
    UNKNOWN = auto()


class Robot:
    def __init__(self, dynamixel: Dynamixel, baudrate=1_000_000, servo_ids=None):
        if servo_ids is None:
            servo_ids = [1, 2, 3, 4, 5]
        self.servo_ids = servo_ids
        self.dynamixel = dynamixel

        self.position_reader = GroupSyncRead(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            ReadAttribute.POSITION.value,
            4,
        )
        for id in self.servo_ids:
            self.position_reader.addParam(id)

        self.velocity_reader = GroupSyncRead(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            ReadAttribute.VELOCITY.value,
            4,
        )
        for id in self.servo_ids:
            self.velocity_reader.addParam(id)

        self.pos_writer = GroupSyncWrite(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            self.dynamixel.ADDR_GOAL_POSITION,
            4,
        )
        for id in self.servo_ids:
            self.pos_writer.addParam(id, [2048])

        self.pwm_writer = GroupSyncWrite(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            self.dynamixel.ADDR_GOAL_PWM,
            2,
        )
        for id in self.servo_ids:
            self.pwm_writer.addParam(id, [2048])

        self._disable_torque()
        self.motor_control_state = MotorControlType.DISABLED

        self.current_writer = GroupSyncWrite(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            102,  # Goal current address
            2,
        )

    def set_current(self, currents):
        if not hasattr(self, "_arm_current_mode_set"):
            self.set_arm_current_control()
            self._arm_current_mode_set = True
        for servo_id, current in zip(self.servo_ids, currents):
            current = np.clip(current, -900, 900)
            param_goal_current = [DXL_LOBYTE(current), DXL_HIBYTE(current)]
            if not self.current_writer.addParam(servo_id, param_goal_current):
                raise RuntimeError(
                    f"Failed to set current for Dynamixel with ID {servo_id}"
                )
            self.current_writer.txPacket()
            self.current_writer.clearParam()

    def read_position(self, tries=2):
        s = time.perf_counter()
        result = self.position_reader.txRxPacket()
        elapsed = time.perf_counter() - s
        if result != 0:
            if tries > 0:
                return self.read_position(tries=tries - 1)
            else:
                print("failed to read position!")
        positions = []
        for id in self.servo_ids:
            position = self.position_reader.getData(
                id, ReadAttribute.POSITION.value, 4
            )
            if position > 2**31:
                position -= 2**32
            positions.append(position)
        return positions, elapsed

    def read_velocity(self):
        self.velocity_reader.txRxPacket()
        velocities = []
        for id in self.servo_ids:
            velocity = self.velocity_reader.getData(
                id, ReadAttribute.VELOCITY.value, 4
            )
            if velocity > 2**31:
                velocity -= 2**32
            velocities.append(velocity)
        return velocities

    def set_goal_pos(self, action):
        if self.motor_control_state is not MotorControlType.POSITION_CONTROL:
            self._set_position_control()
        for i, motor_id in enumerate(self.servo_ids):
            data_write = [
                DXL_LOBYTE(DXL_LOWORD(action[i])),
                DXL_HIBYTE(DXL_LOWORD(action[i])),
                DXL_LOBYTE(DXL_HIWORD(action[i])),
                DXL_HIBYTE(DXL_HIWORD(action[i])),
            ]
            self.pos_writer.changeParam(motor_id, data_write)
        self.pos_writer.txPacket()

    def set_pwm(self, action):
        if self.motor_control_state is not MotorControlType.PWM:
            self._set_pwm_control()
        for i, motor_id in enumerate(self.servo_ids):
            data_write = [
                DXL_LOBYTE(DXL_LOWORD(action[i])),
                DXL_HIBYTE(DXL_LOWORD(action[i])),
            ]
            self.pwm_writer.changeParam(motor_id, data_write)
        self.pwm_writer.txPacket()

    def _disable_torque(self):
        for motor_id in self.servo_ids:
            self.dynamixel._disable_torque(motor_id)

    def _enable_torque(self):
        for motor_id in self.servo_ids:
            self.dynamixel._enable_torque(motor_id)

    def _set_pwm_control(self):
        self._disable_torque()
        for motor_id in self.servo_ids:
            self.dynamixel.set_operating_mode(motor_id, OperatingMode.PWM)
        self._enable_torque()
        self.motor_control_state = MotorControlType.PWM

    def _set_position_control(self):
        self._disable_torque()
        for motor_id in self.servo_ids:
            self.dynamixel.set_operating_mode(motor_id, OperatingMode.POSITION)
        self._enable_torque()
        self.motor_control_state = MotorControlType.POSITION_CONTROL

    def set_arm_current_control(self):
        for servo_id in self.servo_ids:
            self.dynamixel._disable_torque(servo_id)
            dxl_comm_result, dxl_error = self.dynamixel.packetHandler.write1ByteTxRx(
                self.dynamixel.portHandler,
                servo_id,
                self.dynamixel.OPERATING_MODE_ADDR,
                0,  # Current control mode
            )
            if dxl_comm_result != 0:
                print(
                    f"Failed to set current control mode for motor {servo_id}: {dxl_comm_result}"
                )
            self.dynamixel._enable_torque(servo_id)
            print(f"Servo (ID {servo_id}) set to current control mode")

    def limit_pwm(self, limit: Union[int, list, np.ndarray]):
        if isinstance(limit, int):
            limits = [limit] * len(self.servo_ids)
        else:
            limits = limit
        self._disable_torque()
        for motor_id, lim in zip(self.servo_ids, limits):
            self.dynamixel.set_pwm_limit(motor_id, lim)
        self._enable_torque()
