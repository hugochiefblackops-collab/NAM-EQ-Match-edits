"""Feature extraction and distances for tone matching.

Two kinds of features:

1. LTAS (long-term average spectrum) — linear, fully correctable by a match
   IR/EQ afterwards. Used to *generate* the correction, and only lightly
   penalized during model ranking (extreme corrections indicate a bad base
   match).

2. Nonlinear / dynamic fingerprint — things an EQ can NOT fix: crest factor,
   dynamic-range compression, saturation density (spectral flatness of the
   fizz region), spectral flux, and MFCC texture statistics. These drive the
   ranking of candidate NAM captures and the input-gain search.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.fft import dct
from scipy.signal import welch, stft

from .audio import EPS, remove_silence, short_term_rms

# ----------------------------------------------------------------------------
# Spectral helpers
# ----------------------------------------------------------------------------


def ltas(x: np.ndarray, sr: int, n_fft: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Long-term average power spectrum via Welch. Returns (freqs, power)."""
    nper = min(n_fft, len(x))
    f, p = welch(x, fs=sr, nperseg=nper, noverlap=nper // 2, window="hann")
    return f, p


def log_freq_grid(sr: int, f_lo: float = 30.0, pts_per_octave: int = 24) -> np.ndarray:
    f_hi = 0.95 * sr / 2
    n_oct = np.log2(f_hi / f_lo)
    n = int(np.ceil(n_oct * pts_per_octave)) + 1
    return f_lo * 2.0 ** (np.arange(n) / pts_per_octave)


def smooth_spectrum_db(
    f: np.ndarray, p: np.ndarray, grid: np.ndarray, octave_fraction: float = 6.0
) -> np.ndarray:
    """Interpolate a power spectrum onto a log grid and smooth by 1/N octave."""
    p_db = 10.0 * np.log10(p + EPS)
    g_db = np.interp(grid, f, p_db)
    # Gaussian smoothing in log-frequency domain
    pts_per_octave = 1.0 / np.log2(grid[1] / grid[0])
    sigma = pts_per_octave / octave_fraction / 2.355  # FWHM = 1/N octave
    radius = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / max(sigma, 1e-6)) ** 2)
    k /= k.sum()
    padded = np.pad(g_db, radius, mode="edge")
    return np.convolve(padded, k, mode="valid")


# ----------------------------------------------------------------------------
# Mel / MFCC (no librosa dependency)
# ----------------------------------------------------------------------------


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, f_lo: float, f_hi: float) -> np.ndarray:
    m = np.linspace(_hz_to_mel(f_lo), _hz_to_mel(f_hi), n_mels + 2)
    hz = _mel_to_hz(m)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        a, b, c = bins[i], bins[i + 1], bins[i + 2]
        if b > a:
            fb[i, a:b] = (np.arange(a, b) - a) / (b - a)
        if c > b:
            fb[i, b:c] = (c - np.arange(b, c)) / (c - b)
    return fb


# ----------------------------------------------------------------------------
# Fingerprint
# ----------------------------------------------------------------------------


@dataclass
class ToneFingerprint:
    crest_db: float
    dyn_range_db: float
    flatness_fizz: float
    flatness_mid: float
    flux: float
    mfcc_std: np.ndarray = field(repr=False)
    mfcc_dstd: np.ndarray = field(repr=False)
    ltas_grid: np.ndarray = field(repr=False)
    ltas_db: np.ndarray = field(repr=False)


def extract_fingerprint(x: np.ndarray, sr: int) -> ToneFingerprint:
    x = remove_silence(x, sr)
    x = x / (np.max(np.abs(x)) + EPS)  # level-invariant

    # --- Dynamics -----------------------------------------------------------
    env = short_term_rms(x, sr, win_s=0.05, hop_s=0.01)
    env_db = 20.0 * np.log10(env + EPS)
    dyn_range_db = float(np.percentile(env_db, 95) - np.percentile(env_db, 10))

    # Crest factor over 400 ms windows
    win = int(0.4 * sr)
    n = max(1, len(x) // win)
    crests = []
    for i in range(n):
        seg = x[i * win : (i + 1) * win]
        r = np.sqrt(np.mean(seg**2) + EPS)
        pk = np.max(np.abs(seg)) + EPS
        if r > 10 ** (-50 / 20):  # skip near-silence
            crests.append(20.0 * np.log10(pk / r))
    crest_db = float(np.mean(crests)) if crests else 12.0

    # --- STFT-based ---------------------------------------------------------
    n_fft = 2048
    hop = 512
    f, _, Z = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
    S = np.abs(Z) ** 2  # (freq, time)

    # keep active frames only
    frame_e = S.sum(axis=0)
    active = frame_e > (np.percentile(frame_e, 95) * 10 ** (-40 / 10))
    if np.sum(active) >= 8:
        S = S[:, active]

    def band_flatness(f_a: float, f_b: float) -> float:
        sel = (f >= f_a) & (f <= f_b)
        band = S[sel] + EPS
        gm = np.exp(np.mean(np.log(band), axis=0))
        am = np.mean(band, axis=0)
        return float(np.mean(gm / am))

    flatness_fizz = band_flatness(2000, min(8000, 0.45 * sr))
    flatness_mid = band_flatness(400, 2000)

    # Spectral flux (normalized): how fast the spectrum changes → attack/dirt
    Sn = S / (S.sum(axis=0, keepdims=True) + EPS)
    flux = float(np.mean(np.sqrt(np.sum(np.diff(Sn, axis=1) ** 2, axis=0))))

    # --- MFCC texture -------------------------------------------------------
    fb = mel_filterbank(sr, n_fft, n_mels=40, f_lo=60.0, f_hi=min(10000.0, 0.45 * sr))
    mel = np.log(fb @ S + EPS)  # (mels, time)
    mfcc = dct(mel, type=2, axis=0, norm="ortho")[1:20]  # drop c0 → EQ-tilt tolerant
    mfcc_std = np.std(mfcc, axis=1)
    dm = np.diff(mfcc, axis=1)
    mfcc_dstd = np.std(dm, axis=1) if dm.shape[1] > 1 else np.zeros(mfcc.shape[0])

    # --- LTAS ---------------------------------------------------------------
    grid = log_freq_grid(sr)
    fw, pw = ltas(x, sr)
    ltas_db = smooth_spectrum_db(fw, pw, grid)
    ltas_db -= np.median(ltas_db)  # level-invariant

    return ToneFingerprint(
        crest_db=crest_db,
        dyn_range_db=dyn_range_db,
        flatness_fizz=flatness_fizz,
        flatness_mid=flatness_mid,
        flux=flux,
        mfcc_std=mfcc_std,
        mfcc_dstd=mfcc_dstd,
        ltas_grid=grid,
        ltas_db=ltas_db,
    )


# ----------------------------------------------------------------------------
# Distances
# ----------------------------------------------------------------------------


def _rel(a: float, b: float) -> float:
    return abs(a - b) / (abs(b) + 1e-6)


def _vec_dist(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))


def nonlinear_distance(cand: ToneFingerprint, target: ToneFingerprint) -> dict:
    """EQ-invariant-ish distance between two fingerprints. Lower = better.

    Returns a dict with a 'total' key and per-component breakdown.
    """
    d = {
        "crest": 1.0 * _rel(cand.crest_db, target.crest_db),
        "dyn_range": 1.0 * _rel(cand.dyn_range_db, target.dyn_range_db),
        "flatness_fizz": 1.0 * _rel(cand.flatness_fizz, target.flatness_fizz),
        "flatness_mid": 0.5 * _rel(cand.flatness_mid, target.flatness_mid),
        "flux": 0.5 * _rel(cand.flux, target.flux),
        "mfcc_texture": 0.8 * _vec_dist(cand.mfcc_std, target.mfcc_std),
        "mfcc_delta": 0.8 * _vec_dist(cand.mfcc_dstd, target.mfcc_dstd),
    }
    d["total"] = float(sum(d.values()))
    return d


def eq_correction_db(cand: ToneFingerprint, target: ToneFingerprint) -> np.ndarray:
    """Required EQ correction (dB, on cand.ltas_grid) to map candidate → target."""
    return target.ltas_db - cand.ltas_db


def eq_penalty(cand: ToneFingerprint, target: ToneFingerprint) -> float:
    """Penalty for how much linear correction the candidate would still need.

    Mild — the match IR fixes this — but a rig that needs ±18 dB of EQ was
    probably the wrong pick.
    """
    corr = eq_correction_db(cand, target)
    grid = cand.ltas_grid
    sel = (grid >= 80) & (grid <= 8000)
    return float(np.mean(np.abs(corr[sel])) / 18.0)
