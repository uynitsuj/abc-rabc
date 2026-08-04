"""Base Node class with ZMQ pub/sub for teleop communication."""

from __future__ import annotations

import signal
import time
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import zmq

from yam_sim.teleop import communication as comms


class Node(ABC):
    def __init__(self, name: str, control_rate: float = -1, verbose: bool = True):
        self._name = name
        self._control_rate = control_rate
        self._period = 1.0 / control_rate
        self._verbose = verbose
        self._zmq_context = zmq.Context()
        self._publishers = dict()
        self._subscribers = dict()

    def create_publisher(self, topic: str, linger=None, send_timeout=None) -> None:
        self._publishers[topic] = comms.create_publisher(
            self._zmq_context, topic, linger, send_timeout
        )

    def publish(self, topic: str, message: np.ndarray, extras: dict | None = None) -> None:
        if topic not in self._publishers:
            raise ValueError(f"Publisher for topic '{topic}' not created.")
        comms.publish(self._publishers[topic], message, extras or {})

    def create_subscriber(self, topic: str, conflate=None) -> None:
        self._subscribers[topic] = comms.create_subscriber(
            self._zmq_context, topic, conflate
        )

    def subscribe(self, topic: str, block: bool = True) -> Tuple[np.ndarray, dict]:
        if topic not in self._subscribers:
            raise ValueError(f"Subscriber for topic '{topic}' not created.")
        if self._subscribers[topic].poll(timeout=0) or block:
            return comms.subscribe(self._subscribers[topic])
        else:
            return None, {}

    def run(self, *args, **kwargs) -> None:
        """Run the node's main loop."""
        try:
            next_t = time.perf_counter()
            self.initial_bootup(*args, **kwargs)
            while True:
                self.tick()
                if self._control_rate < 0:
                    continue
                next_t += self._period
                while True:
                    remaining = next_t - time.perf_counter()
                    if remaining <= 0.0:
                        next_t = time.perf_counter()
                        break
                    if remaining > 3e-4:
                        time.sleep(remaining - 1e-4)
        except KeyboardInterrupt:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            self.on_shutdown()
            for _, pub in self._publishers.items():
                pub.close()
            for _, sub in self._subscribers.items():
                sub.close()
            self._zmq_context.term()

    @abstractmethod
    def initial_bootup(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def tick(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def on_shutdown(self) -> None:
        pass
