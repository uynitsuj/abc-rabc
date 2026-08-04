"""Dynamixel servo communication layer.

Ported from abc/deploy/rllg2 for standalone use with yam-sim.
Requires: pip install dynamixel-sdk
"""

from __future__ import annotations

import enum
import math
import os
from dataclasses import dataclass

from dynamixel_sdk import (
    COMM_SUCCESS,
    PacketHandler,
    PortHandler,
)


class ReadAttribute(enum.Enum):
    TEMPERATURE = 146
    VOLTAGE = 145
    VELOCITY = 128
    POSITION = 132
    CURRENT = 126
    PWM = 124
    HARDWARE_ERROR_STATUS = 70
    HOMING_OFFSET = 20
    BAUDRATE = 8


class OperatingMode(enum.Enum):
    VELOCITY = 1
    POSITION = 3
    CURRENT_CONTROLLED_POSITION = 5
    PWM = 16
    UNKNOWN = -1


class Dynamixel:
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_VELOCITY_LIMIT = 44
    ADDR_GOAL_PWM = 100
    OPERATING_MODE_ADDR = 11
    POSITION_I = 82
    POSITION_P = 84
    ADDR_ID = 7

    @dataclass
    class Config:
        baudrate: int = 57600
        protocol_version: float = 2.0
        device_name: str = ""
        dynamixel_id: int = 1

        def instantiate(self) -> "Dynamixel":
            return Dynamixel(self)

    def __init__(self, config: Config):
        self.config = config
        self.connect()

    def connect(self):
        if self.config.device_name == "":
            for port_name in os.listdir("/dev"):
                if any(tag in port_name for tag in ("ttyUSB", "ttyACM", "cu.usbserial")):
                    self.config.device_name = "/dev/" + port_name
                    print(f"using device {self.config.device_name}")
        self.portHandler = PortHandler(self.config.device_name)
        self.packetHandler = PacketHandler(self.config.protocol_version)
        if not self.portHandler.openPort():
            raise Exception(f"Failed to open port {self.config.device_name}")
        if not self.portHandler.setBaudRate(self.config.baudrate):
            raise Exception(f"Failed to set baudrate to {self.config.baudrate}")

        self.operating_modes = [None for _ in range(64)]
        self.torque_enabled = [None for _ in range(64)]
        return True

    def disconnect(self):
        self.portHandler.closePort()

    def set_goal_position(self, motor_id, goal_position):
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
            self.portHandler, motor_id, self.ADDR_GOAL_POSITION, goal_position
        )

    def set_pwm_value(self, motor_id: int, pwm_value, tries=3):
        if self.operating_modes[motor_id] is not OperatingMode.PWM:
            self._disable_torque(motor_id)
            self.set_operating_mode(motor_id, OperatingMode.PWM)
        if not self.torque_enabled[motor_id]:
            self._enable_torque(motor_id)
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
            self.portHandler, motor_id, self.ADDR_GOAL_PWM, pwm_value
        )
        if dxl_comm_result != COMM_SUCCESS:
            if tries <= 1:
                raise ConnectionError(
                    f"dxl_comm_result: {self.packetHandler.getTxRxResult(dxl_comm_result)}"
                )
            else:
                print(f"dynamixel pwm setting failure, retrying ({tries - 1} left)")
                self.set_pwm_value(motor_id, pwm_value, tries=tries - 1)
        elif dxl_error != 0:
            raise ConnectionError(
                f"dynamixel error: {self.packetHandler.getTxRxResult(dxl_error)}"
            )

    def read_position(self, motor_id: int):
        pos = self._read_value(motor_id, ReadAttribute.POSITION, 4)
        if pos > 2**31:
            pos -= 2**32
        return pos

    def read_velocity(self, motor_id: int):
        pos = self._read_value(motor_id, ReadAttribute.VELOCITY, 4)
        if pos > 2**31:
            pos -= 2**32
        return pos

    def read_current(self, motor_id: int):
        current = self._read_value(motor_id, ReadAttribute.CURRENT, 2)
        if current > 2**15:
            current -= 2**16
        return current

    def read_temperature(self, motor_id: int):
        return self._read_value(motor_id, ReadAttribute.TEMPERATURE, 1)

    def read_hardware_error_status(self, motor_id: int):
        return self._read_value(motor_id, ReadAttribute.HARDWARE_ERROR_STATUS, 1)

    def _enable_torque(self, motor_id):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler, motor_id, self.ADDR_TORQUE_ENABLE, 1
        )
        self._process_response(dxl_comm_result, dxl_error, motor_id)
        self.torque_enabled[motor_id] = True

    def _disable_torque(self, motor_id):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler, motor_id, self.ADDR_TORQUE_ENABLE, 0
        )
        self._process_response(dxl_comm_result, dxl_error, motor_id)
        self.torque_enabled[motor_id] = False

    def _process_response(self, dxl_comm_result: int, dxl_error: int, motor_id: int):
        if dxl_comm_result != COMM_SUCCESS:
            raise ConnectionError(
                f"dxl_comm_result for motor {motor_id}: "
                f"{self.packetHandler.getTxRxResult(dxl_comm_result)}"
            )
        elif dxl_error == 128:
            pass  # voltage error, ignore
        elif dxl_error != 0:
            raise ConnectionError(
                f"dynamixel error for motor {motor_id}: "
                f"{self.packetHandler.getTxRxResult(dxl_error)}"
            )

    def set_operating_mode(self, motor_id: int, operating_mode: OperatingMode):
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
            self.portHandler, motor_id, self.OPERATING_MODE_ADDR, operating_mode.value
        )
        self._process_response(dxl_comm_result, dxl_error, motor_id)
        self.operating_modes[motor_id] = operating_mode

    def set_pwm_limit(self, motor_id: int, limit: int):
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
            self.portHandler, motor_id, 36, limit
        )
        self._process_response(dxl_comm_result, dxl_error, motor_id)

    def _read_value(self, motor_id, attribute: ReadAttribute, num_bytes: int, tries=10):
        try:
            if num_bytes == 1:
                value, dxl_comm_result, dxl_error = self.packetHandler.read1ByteTxRx(
                    self.portHandler, motor_id, attribute.value
                )
            elif num_bytes == 2:
                value, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(
                    self.portHandler, motor_id, attribute.value
                )
            elif num_bytes == 4:
                value, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(
                    self.portHandler, motor_id, attribute.value
                )
        except Exception:
            if tries == 0:
                raise
            return self._read_value(motor_id, attribute, num_bytes, tries=tries - 1)

        if dxl_comm_result != COMM_SUCCESS:
            if tries <= 1:
                raise ConnectionError(
                    f"dxl_comm_result {dxl_comm_result} for servo {motor_id}"
                )
            return self._read_value(motor_id, attribute, num_bytes, tries=tries - 1)
        if dxl_error != 0:
            raise ConnectionError(f"dxl_error {dxl_error} for servo {motor_id}")
        return value
