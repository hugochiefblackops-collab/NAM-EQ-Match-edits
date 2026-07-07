"""Candidate ranking: find the NAM capture + input gain whose nonlinear
character best matches the target recording.

Two-stage search:
  1. Coarse: short DI segment, coarse gain grid, all captures.
  2. Refine: longer segment, finer gain grid, top-N captures only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio import active_segment, db_to_lin
from .features import (
    ToneFingerprint,
    eq_penalty,
    extract_fingerprint,
    nonlinear_distance,
)


@dataclass
class CandidateResult:
    capture: object
    gain_db: float
    score: float
    nl_distance: float
    eq_penalty: float
    breakdown: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return getattr(self.capture, "name", str(self.capture))


def evaluate_gain(
    capture,
    di: np.ndarray,
    sr: int,
    gain_db: float,
    target_fp: ToneFingerprint,
    eq_weight: float = 0.25,
) -> CandidateResult:
    y = capture.process(di * db_to_lin(gain_db))
    fp = extract_fingerprint(y, sr)
    nl = nonlinear_distance(fp, target_fp)
    pen = eq_penalty(fp, target_fp)
    return CandidateResult(
        capture=capture,
        gain_db=gain_db,
        score=nl["total"] + eq_weight * pen,
        nl_distance=nl["total"],
        eq_penalty=pen,
        breakdown=nl,
    )


def best_gain_for_capture(
    capture,
    di: np.ndarray,
    sr: int,
    target_fp: ToneFingerprint,
    gains_db: np.ndarray,
    progress=None,
) -> CandidateResult:
    best: CandidateResult | None = None
    for g in gains_db:
        r = evaluate_gain(capture, di, sr, float(g), target_fp)
        if best is None or r.score < best.score:
            best = r
        if progress:
            progress()
    return best


def rank_captures(
    captures: list,
    di: np.ndarray,
    target: np.ndarray,
    sr: int,
    gain_range_db: tuple[float, float] = (-12.0, 12.0),
    coarse_seg_s: float = 8.0,
    refine_seg_s: float = 16.0,
    refine_top: int = 5,
    progress_cb=None,
) -> list[CandidateResult]:
    """Return CandidateResults sorted best-first."""
    if not captures:
        return []

    lo, hi = gain_range_db
    coarse_gains = np.linspace(lo, hi, 7)
    di_coarse = active_segment(di, sr, coarse_seg_s)
    di_refine = active_segment(di, sr, refine_seg_s)
    target_fp = extract_fingerprint(target, sr)

    n_total = len(captures) * len(coarse_gains)
    done = [0]

    def tick():
        done[0] += 1
        if progress_cb:
            progress_cb(done[0] / (n_total + refine_top * 5), "searching")

    # Stage 1: coarse
    coarse = [
        best_gain_for_capture(c, di_coarse, sr, target_fp, coarse_gains, progress=tick)
        for c in captures
    ]
    coarse.sort(key=lambda r: r.score)

    # Stage 2: refine top-N with finer gains on longer audio
    results = []
    for r in coarse[: max(1, refine_top)]:
        step = (hi - lo) / 6.0
        fine = np.clip(r.gain_db + np.linspace(-step, step, 5), lo, hi)
        rr = best_gain_for_capture(r.capture, di_refine, sr, target_fp, fine, progress=tick)
        results.append(rr)
    results.extend(coarse[max(1, refine_top) :])
    results.sort(key=lambda r: r.score)
    return results
