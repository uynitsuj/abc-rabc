# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mcap", "mcap-protobuf-support"]
# ///
"""Reconstruct the ABC states_actions array for a delivery episode from its output.mcap.

Shared by build_states_actions_B.py (staged bins) and package_B.py (lerobot data
parquets), so the two can never diverge.

Empirically reverse-engineered against A's staged episodes (which came from the mjgl
LeRobot dataset): the 28-col states_actions is
    state(14)   = [ /left-arm-proprio.position(7) | /right-arm-proprio.position(7) ]
    actions(14) = [ /left-command-state.position(7) | /right-command-state.position(7) ]
where each 7-vec is [6 arm joints, 1 gripper] (matches export_mcap's
[L-arm6, L-grip, R-arm6, R-grip] ordering).

Row-count alignment (verified on 5 A episodes, max abs diff < 1e-6 = float32 noise):
    n_msgs == n_integration_state_rows - 1   (the sim emits proprio/command one step
    after the initial integration_state row), and the staged array duplicates the FIRST
    source row at the front:
        sa = vstack([src[0:1], src])          -> len == n_integration_state_rows
This makes states_actions rows == integration_state rows == render frame count, so the
mjwarp-rendered video is a drop-in match.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Topic -> (dim, block). Order within state/actions matches export_mcap STATE/ACTION_TOPICS.
STATE_TOPICS = ("/left-arm-proprio", "/right-arm-proprio")   # 7 + 7 = 14
ACTION_TOPICS = ("/left-command-state", "/right-command-state")  # 7 + 7 = 14
ALL_TOPICS = STATE_TOPICS + ACTION_TOPICS
POS_DIM = 7  # 6 arm joints + 1 gripper per arm


def read_position_streams(mcap_path: str | Path) -> dict[str, np.ndarray]:
    """topic -> (n_msgs, POS_DIM) float64, sorted by mcap log_time."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    buf: dict[str, list] = {t: [] for t in ALL_TOPICS}
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, message, decoded in reader.iter_decoded_messages():
            if channel.topic in buf:
                buf[channel.topic].append((message.log_time, list(decoded.position)))
    out = {}
    for t in ALL_TOPICS:
        rows = sorted(buf[t], key=lambda x: x[0])
        arr = np.array([p for _, p in rows], dtype=np.float64)
        out[t] = arr
    return out


def _candidate_shifts(n: int, n_rows: int) -> list[int]:
    """Valid head shifts for a topic with n msgs over n_rows frames.

    A stream occupies rows [s, s+n): head-fill s copies of the first msg, tail-fill
    (n_rows - n - s) copies of the last. Empirical family (qpos-verified across the
    corpus): s in {0, 1} and tail-fill in {0, 1}. Ordered prepend-first (s=1) first
    — the common validated rule — so degenerate ties resolve to it.
    """
    total = n_rows - n
    if not 0 <= total <= 2:
        raise ValueError(f"n_msgs={n} vs n_rows={n_rows}: fill {total} outside [0,2]")
    cands = [s for s in (1, 0) if s <= total and 0 <= total - s <= 1]
    if not cands:
        raise ValueError(f"n_msgs={n} vs n_rows={n_rows}: no valid shift in {{0,1}}")
    return cands


def _align_topic(stream: np.ndarray, n_rows: int, shift: int) -> np.ndarray:
    tail = n_rows - len(stream) - shift
    if shift == 0 and tail == 0:
        return stream
    return np.vstack([stream[0:1]] * shift + [stream] + [stream[-1:]] * tail)


def _verify_arm_shifts(streams: dict, qpos: np.ndarray, n_rows: int,
                       mcap_path, tol: float = 1e-5) -> dict[str, int]:
    """Empirically pick + verify each arm's head shift against qpos arm columns.

    The arm columns' position within qpos varies per episode (object joints shift
    the layout), so locate the left 6-block by minimizing the residual at a
    mid-episode row, then the right 6-block strictly to its right (left precedes
    right in qpos — the ordered search disambiguates similar joint values). For
    each arm, the first candidate shift whose FULL-stream max residual is < tol
    wins; none passing raises (never silently misalign).
    """
    m_row = n_rows // 2

    def _find6(target6, lo):
        best = None
        for off in range(lo, qpos.shape[1] - 6 + 1):
            d = float(np.abs(qpos[m_row, off:off + 6] - target6).max())
            if best is None or d < best[1]:
                best = (off, d)
        return best

    lp = streams[STATE_TOPICS[0]][:, :6]
    rp = streams[STATE_TOPICS[1]][:, :6]
    s0_l = min(1, n_rows - len(lp))
    s0_r = min(1, n_rows - len(rp))
    loff, _ = _find6(lp[m_row - s0_l], 0)
    roff, _ = _find6(rp[m_row - s0_r], loff + 6)

    shifts: dict[str, int] = {}
    for topic, stream, off in ((STATE_TOPICS[0], lp, loff), (STATE_TOPICS[1], rp, roff)):
        n = len(stream)
        errs = {}
        for s in _candidate_shifts(n, n_rows):
            resid = float(np.abs(stream - qpos[s:s + n, off:off + 6]).max())
            errs[s] = resid
            if resid < tol:
                shifts[topic] = s
                break
        else:
            raise ValueError(f"{mcap_path}: qpos alignment verification FAILED for "
                             f"{topic} (n={n}, n_rows={n_rows}): residuals {errs} "
                             f"all > {tol} (loff={loff}, roff={roff})")
    return shifts


def reconstruct_states_actions(mcap_path: str | Path, n_integration_rows: int,
                               qpos: np.ndarray | None = None) -> np.ndarray:
    """Return (n_integration_rows, 28) float64 = [state14 | actions14].

    Alignment rule (empirically established, see probe_alignment_B.py / probe_3448.py):
    each topic's messages occupy rows [s_t, s_t + n_t) of the integration_state, with
    head-fill s_t copies of the first message and tail-fill (n_rows - n_t - s_t)
    copies of the last, s_t and tail-fill each in {0, 1}. Observed variants:
      * all topics s=1, no tail (the common case; validated byte-exact vs A's staged),
      * all topics s=0 (stream starts AT the initial row; qpos resid 1.2e-07),
      * mixed per-topic (left s=0 / right s=1, etc.; 11 B eps),
      * s=1 + tail-fill 1 (stream truncated at the END; idx 3448: shift-2 hypothesis
        fails at 1.1e-01, shift-1 passes at 1.2e-07).
    When ``qpos`` (integration_state[:, 1:]) is given, BOTH arms' proprio shifts are
    chosen + verified over their full streams (arm columns located per episode);
    command topics have no qpos ground truth and reuse their arm's verified shift
    (clamped to their own valid candidate set). Without qpos, the prepend-first
    default (s = min(1, fill)) applies.
    """
    streams = read_position_streams(mcap_path)
    for t in ALL_TOPICS:
        if streams[t].size and streams[t].shape[1] != POS_DIM:
            raise ValueError(f"{mcap_path}: {t} position dim {streams[t].shape[1]} != {POS_DIM}")
        _candidate_shifts(len(streams[t]), n_integration_rows)  # raises if out of family
    if qpos is not None:
        arm_shifts = _verify_arm_shifts(streams, qpos, n_integration_rows, mcap_path)
    else:
        arm_shifts = {t: min(1, n_integration_rows - len(streams[t])) for t in STATE_TOPICS}
    arm_of = {STATE_TOPICS[0]: STATE_TOPICS[0], STATE_TOPICS[1]: STATE_TOPICS[1],
              ACTION_TOPICS[0]: STATE_TOPICS[0], ACTION_TOPICS[1]: STATE_TOPICS[1]}
    aligned = []
    for t in ALL_TOPICS:
        cands = _candidate_shifts(len(streams[t]), n_integration_rows)
        pref = arm_shifts[arm_of[t]]
        s_t = pref if pref in cands else cands[0]
        aligned.append(_align_topic(streams[t], n_integration_rows, s_t))
    sa = np.concatenate(aligned, axis=1)  # (n_rows, 28)
    assert sa.shape == (n_integration_rows, 28), (sa.shape, n_integration_rows)
    return sa
