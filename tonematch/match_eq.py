"""Match-EQ: design an impulse response that corrects the linear (EQ/cab)
difference between the tuned candidate rig and the target recording.

This is the "Tone Match" half: after the NAM capture + input gain reproduce
the nonlinear character, the residual LTAS difference is linear and can be
captured in a FIR filter, exported as a .wav IR loadable in the NAM plugin's
IR slot (or any IR loader).
"""

from __future__ import annotations

import numpy as np

from .audio import EPS
from .features import ltas, log_freq_grid, smooth_spectrum_db


def design_match_ir(
    candidate: np.ndarray,
    target: np.ndarray,
    sr: int,
    n_taps: int = 2048,
    max_gain_db: float = 18.0,
    octave_fraction: float = 6.0,
    f_lo_taper: float = 50.0,
    f_hi_taper: float = 11000.0,
    minimum_phase: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Design a FIR that makes `candidate` match `target` spectrally.

    Returns (ir, grid_freqs, correction_db).
    """
    grid = log_freq_grid(sr)
    fc, pc = ltas(candidate, sr)
    ft, pt = ltas(target, sr)
    c_db = smooth_spectrum_db(fc, pc, grid, octave_fraction)
    t_db = smooth_spectrum_db(ft, pt, grid, octave_fraction)
    corr_db = t_db - c_db
    corr_db -= np.median(corr_db[(grid > 200) & (grid < 4000)])  # 0 dB midband

    # Taper the correction toward 0 dB at the extremes (don't boost rumble/hiss)
    lo_fade = np.clip(np.log2(grid / (f_lo_taper / 2)) / np.log2(2.0), 0.0, 1.0)
    hi_fade = np.clip(np.log2((f_hi_taper * 2) / grid) / np.log2(2.0), 0.0, 1.0)
    corr_db = corr_db * lo_fade * hi_fade
    corr_db = np.clip(corr_db, -max_gain_db, max_gain_db)

    ir = fir_from_log_spectrum(grid, corr_db, sr, n_taps, minimum_phase)
    return ir, grid, corr_db


def fir_from_log_spectrum(
    grid: np.ndarray,
    gain_db: np.ndarray,
    sr: int,
    n_taps: int = 2048,
    minimum_phase: bool = True,
) -> np.ndarray:
    """Build a FIR from a gain curve defined on a log-frequency grid."""
    n_fft = 4 * n_taps
    f_lin = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    # interpolate in log-f domain; clamp ends
    lf = np.log10(np.maximum(f_lin, grid[0] / 2))
    g_db = np.interp(lf, np.log10(grid), gain_db, left=gain_db[0], right=gain_db[-1])
    mag = 10.0 ** (g_db / 20.0)

    if minimum_phase:
        h = _minimum_phase_fir(mag, n_taps)
    else:
        # linear phase: symmetric IR
        h_full = np.fft.irfft(mag)
        h_full = np.roll(h_full, n_fft // 2)
        mid = n_fft // 2
        h = h_full[mid - n_taps // 2 : mid + n_taps // 2]
        h = h * np.hanning(len(h))
    return h.astype(np.float32)


def _minimum_phase_fir(mag: np.ndarray, n_taps: int) -> np.ndarray:
    """Minimum-phase FIR from a magnitude response via the real cepstrum."""
    n_fft = 2 * (len(mag) - 1)
    log_mag = np.log(np.maximum(mag, 1e-8))
    ceps = np.fft.irfft(log_mag)
    w = np.zeros(n_fft)
    w[0] = 1.0
    w[1 : n_fft // 2] = 2.0
    w[n_fft // 2] = 1.0
    min_phase_spec = np.exp(np.fft.rfft(ceps * w))
    h = np.fft.irfft(min_phase_spec)[:n_taps]
    # gentle fade-out on the tail to avoid truncation ripple
    fade = int(0.25 * n_taps)
    h[-fade:] *= 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade)))
    return h


def apply_fir(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve

    return fftconvolve(x, ir, mode="full")[: len(x)].astype(np.float32)
