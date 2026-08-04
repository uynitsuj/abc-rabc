"""ZMQ IPC serialization for numpy arrays with metadata."""

from __future__ import annotations

import json
import struct
import time
from typing import Tuple

import numpy as np
import zmq


def create_publisher(
    context: zmq.Context,
    topic: str,
    linger: int = None,
    send_timeout: int = None,
    bind: bool = True,
) -> zmq.Socket:
    publisher = context.socket(zmq.PUB)
    if linger is not None:
        publisher.setsockopt(zmq.LINGER, linger)
    if send_timeout is not None:
        publisher.setsockopt(zmq.SNDTIMEO, send_timeout)
    if "://" not in topic:
        publisher.bind(f"ipc:///tmp/{topic}")
    else:
        publisher.bind(topic)
    return publisher


def publish(
    publisher: zmq.Socket,
    message: np.ndarray,
    extras: dict | None = None,
) -> None:
    header = {
        "timestamp": time.perf_counter(),
        "shape": list(message.shape),
        "dtype": message.dtype.str,
        "extras": extras or {},
    }
    header_b = json.dumps(header).encode("utf-8")
    payload = np.ascontiguousarray(message).tobytes()
    buffer = struct.pack("!I", len(header_b)) + header_b + payload
    publisher.send(buffer, copy=False)


def create_subscriber(
    context: zmq.Context, topic: str, conflate: int = None
) -> zmq.Socket:
    subscriber = context.socket(zmq.SUB)
    if conflate is not None:
        subscriber.setsockopt(zmq.CONFLATE, conflate)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
    if "://" not in topic:
        subscriber.connect(f"ipc:///tmp/{topic}")
    else:
        subscriber.connect(topic)
    return subscriber


def subscribe(subscriber: zmq.Socket) -> Tuple[np.ndarray, dict]:
    buffer_b = subscriber.recv()
    (hlen,) = struct.unpack("!I", buffer_b[:4])
    header_b = buffer_b[4 : 4 + hlen]
    header = json.loads(header_b.decode("utf-8"))
    payload = buffer_b[4 + hlen :]
    message = (
        np.frombuffer(payload, dtype=np.dtype(header["dtype"]))
        .reshape(tuple(header["shape"]))
        .copy()
    )
    extras = header.get("extras", {}) or {}
    extras["timestamp"] = header.get("timestamp")
    return message, extras
