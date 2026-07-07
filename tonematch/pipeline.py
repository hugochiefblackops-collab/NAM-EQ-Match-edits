"""End-to-end tone match pipeline.

Given:
  * target: an isolated guitar recording (the tone you want)
  * di:     your clean DI (or any clean guitar performance)
  * a library of .nam captures

Produces:
  * best capture + input gain (drive) setting
  * a match IR (.wav) correcting the residual EQ/cab difference
  * a rendered preview of your DI through the full matched chain
  * a JSON report + comparison spectrum plot
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

from .audio import db_to_lin, lin_to_db, load_audio, match_rms, rms, save_audio
from .features import extract_fingerprint
from .match_eq import apply_fir, design_match_ir
from .search import CandidateResult, rank_captures

DEFAULT_SR = 48000


@dataclass
class MatchOutput:
    ranked: list
    best: CandidateResult
    ir_path: str
    render_path: str
    target_ref_path: str
    report_path: str
    plot_path: str | None
    report: dict
    renders: list = field(default_factory=list)  # one dict per rendered rig


def run_match(
    target_path: str,
    di_path: str,
    captures: list,
    out_dir: str,
    sr: int = DEFAULT_SR,
    gain_range_db: tuple[float, float] = (-12.0, 12.0),
    refine_top: int = 5,
    render_top: int = 1,
    ir_taps: int = 2048,
    progress_cb=None,
) -> MatchOutput:
    os.makedirs(out_dir, exist_ok=True)

    def report_progress(frac, msg):
        if progress_cb:
            progress_cb(frac, msg)

    report_progress(0.0, "loading audio")
    target, _ = load_audio(target_path, sr)
    di, _ = load_audio(di_path, sr)
    # normalize DI to a sane nominal level (~ -18 dBFS RMS) so gain search is meaningful
    di, _ = match_rms(di, 10 ** (-18 / 20))
    target_peak_norm = target / (np.max(np.abs(target)) + 1e-9)

    # --- 1) rank captures & find input gain (the NAM / dynamics half) -------
    ranked = rank_captures(
        captures,
        di,
        target,
        sr,
        gain_range_db=gain_range_db,
        refine_top=refine_top,
        progress_cb=lambda f, m: report_progress(0.05 + 0.75 * f, m),
    )
    if not ranked:
        raise ValueError("No captures could be evaluated - check your .nam files/folder.")
    best = ranked[0]

    # --- 2+3) render the top-K rigs, each with its own match IR --------------
    n_render = max(1, min(int(render_top), len(ranked)))
    renders = []
    best_rendered, best_grid, best_corr = None, None, None
    for i, r in enumerate(ranked[:n_render]):
        report_progress(0.85 + 0.1 * i / n_render, f"rendering rig {i + 1}/{n_render}: {r.name}")
        safe = re.sub(r"[^A-Za-z0-9]+", "-", r.name).strip("-")[:40] or f"rig{i + 1}"
        rig_dir = os.path.join(out_dir, f"rank{i + 1:02d}_{safe}") if n_render > 1 else out_dir
        os.makedirs(rig_dir, exist_ok=True)

        amped = r.capture.process(di * db_to_lin(r.gain_db))
        ir, grid, corr_db = design_match_ir(amped, target, sr, n_taps=ir_taps)
        rendered = apply_fir(amped, ir)
        rendered, out_gain = match_rms(rendered, rms(target_peak_norm) * 0.5)
        peak = np.max(np.abs(rendered)) + 1e-9
        if peak > 0.99:
            rendered *= 0.99 / peak
            out_gain *= 0.99 / peak

        rig_ir_path = os.path.join(rig_dir, "match_ir.wav")
        rig_render_path = os.path.join(rig_dir, "matched_render.wav")
        save_audio(rig_ir_path, ir, sr)
        save_audio(rig_render_path, rendered, sr)
        renders.append(
            {
                "rank": i + 1,
                "name": r.name,
                "file": getattr(r.capture, "path", None),
                "input_gain_db": round(r.gain_db, 2),
                "output_gain_db": round(lin_to_db(out_gain), 2),
                "score": round(r.score, 4),
                "ir": os.path.abspath(rig_ir_path),
                "render": os.path.abspath(rig_render_path),
            }
        )
        if i == 0:
            best_rendered, best_grid, best_corr = rendered, grid, corr_db

    # --- 4) write shared artifacts -------------------------------------------
    report_progress(0.96, "writing outputs")
    target_ref_path = os.path.join(out_dir, "target_reference.wav")
    save_audio(target_ref_path, target_peak_norm * 0.5, sr)
    ir_path = renders[0]["ir"]
    render_path = renders[0]["render"]

    plot_path = None
    try:
        plot_path = _plot_match(
            os.path.join(out_dir, "match_plot.png"), best_rendered, target, sr, best_grid, best_corr
        )
    except Exception as e:  # matplotlib optional
        print(f"[warn] plot skipped: {e}")

    report = {
        "best_model": {
            "name": best.name,
            "file": getattr(best.capture, "path", None),
            "input_gain_db": round(best.gain_db, 2),
            "output_gain_db": renders[0]["output_gain_db"],
            "score": round(best.score, 4),
            "nonlinear_distance": round(best.nl_distance, 4),
            "eq_penalty": round(best.eq_penalty, 4),
            "breakdown": {k: round(v, 4) for k, v in best.breakdown.items()},
        },
        "ranking": [
            {
                "name": r.name,
                "file": getattr(r.capture, "path", None),
                "input_gain_db": round(r.gain_db, 2),
                "score": round(r.score, 4),
            }
            for r in ranked
        ],
        "renders": renders,
        "match_ir": os.path.abspath(ir_path),
        "how_to_use": (
            "In the NAM plugin: load the model file, set Input gain to "
            f"{best.gain_db:+.1f} dB, load match_ir.wav in the IR slot "
            "(disable any other cab), then adjust Output to taste."
        ),
        "sample_rate": sr,
    }
    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    report_progress(1.0, "done")
    return MatchOutput(
        ranked=ranked,
        best=best,
        ir_path=os.path.abspath(ir_path),
        render_path=os.path.abspath(render_path),
        target_ref_path=os.path.abspath(target_ref_path),
        report_path=os.path.abspath(report_path),
        plot_path=os.path.abspath(plot_path) if plot_path else None,
        report=report,
        renders=renders,
    )


def _plot_match(path, rendered, target, sr, grid, corr_db):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fp_r = extract_fingerprint(rendered, sr)
    fp_t = extract_fingerprint(target, sr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.semilogx(fp_t.ltas_grid, fp_t.ltas_db, label="target", lw=2)
    ax1.semilogx(fp_r.ltas_grid, fp_r.ltas_db, label="matched render", lw=1.5, alpha=0.85)
    ax1.set_ylabel("LTAS (dB, median-normalized)")
    ax1.set_title("Tone match result")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    ax2.semilogx(grid, corr_db, color="tab:green", lw=1.5)
    ax2.set_ylabel("Match IR correction (dB)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_xlim(30, sr / 2)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
