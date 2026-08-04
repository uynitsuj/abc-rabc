"""Launch the MuJoCo sim follower for GELLO teleoperation.

This starts the sim follower node that subscribes to GELLO leader ZMQ topics
and drives the MuJoCo simulation.

Usage:
    # Terminal 1: Start the sim follower
    MUJOCO_GL=egl python -m yam_sim.examples.teleop_sim --scene hybrid

    # Terminal 2+3: Start GELLO leaders (requires hardware + dynamixel_sdk)
    # The leaders publish joint positions to ZMQ topics "left_actions" / "right_actions".
    # See yam_sim/teleop/gello_leader.py for the expected interface.
"""

from __future__ import annotations


def main():
    from yam_sim.teleop.sim_follower import main as sim_main

    sim_main()


if __name__ == "__main__":
    main()
