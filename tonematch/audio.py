"""Audio I/O and utility functions."""

from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

EPS = 1e-12


def load_audio(path: str, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    """Load audio as mono float32. Optionally resample."""
    x, sr = sf.read(path, always_2d=True, dtype="float32")
    x = x.mean(axis=1)
    if sample_rate is not None and sr != sample_rate:
        x = resample(x, sr, sample_rate)
        sr = sample_rate
    return x.astype(np.float32), sr


def save_audio(path: str, x: np.ndarray, sample_rate: int) -> None:
    sf.write(path, np.asarray(x, dtype=np.float32), sample_rate)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + EPS))


def rms_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(rms(x) + EPS)


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def lin_to_db(lin: float) -> float:
    return float(20.0 * np.log10(abs(lin) + EPS))


def match_rms(x: np.ndarray, target_rms: float) -> tuple[np.ndarray, float]:
    """Scale x to a target RMS. Returns (scaled, gain_linear)."""
    g = target_rms / (rms(x) + EPS)
    return x * g, g


def short_term_rms(x: np.ndarray, sr: int, win_s: float = 0.05, hop_s: float = 0.01) -> np.ndarray:
    """Short-term RMS envelope (linear)."""
    win = max(1, int(win_s * sr))
    hop = max(1, int(hop_s * sr))
    n = 1 + max(0, (len(x) - win)) // hop
    if n <= 0:
        return np.array([rms(x)])
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx]
    return np.sqrt(np.mean(np.square(frames), axis=1) + EPS)


def active_segment(x: np.ndarray, sr: int, duration_s: float) -> np.ndarray:
    """Return the most active (loudest sustained) segment of the signal."""
    if len(x) <= int(duration_s * sr):
        return x
    hop_s = 0.25
    env = short_term_rms(x, sr, win_s=1.0, hop_s=hop_s)
    seg_frames = int(duration_s / hop_s)
    if seg_frames >= len(env):
        return x
    # cumulative energy over sliding window
    c = np.cumsum(env**2)
    scores = c[seg_frames:] - c[:-seg_frames]
    best = int(np.argmax(scores))
    start = int(best * hop_s * sr)
    return x[start : start + int(duration_s * sr)]


def remove_silence(x: np.ndarray, sr: int, threshold_db: float = -45.0) -> np.ndarray:
    """Drop frames whose short-term level is far below the signal's peak level."""
    env = short_term_rms(x, sr, win_s=0.05, hop_s=0.05)
    env_db = 20.0 * np.log10(env + EPS)
    ref = np.percentile(env_db, 95)
    keep = env_db > (ref + threshold_db)
    if not np.any(keep):
        return x
    hop = int(0.05 * sr)
    mask = np.zeros(len(x), dtype=bool)
    for i, k in enumerate(keep):
        if k:
            mask[i * hop : (i + 1) * hop] = True
    mask[len(env) * hop :] = keep[-1]
    return x[mask[: len(x)]]
