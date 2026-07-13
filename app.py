"""NAM EQ Matcher GUI (Gradio).

Run:  python app.py   then open http://127.0.0.1:7860
"""

from __future__ import annotations

import os
import tempfile

import gradio as gr

from tonematch.nam_backend import load_captures
from tonematch.pipeline import run_match

T3K_CACHE = os.path.abspath("t3k_cache")

DESCRIPTION = """
# 🎸 NAM EQ Matcher — clone a guitar tone from a recording

1. **Target**: a recording with the tone you want — an isolated guitar track, or a **full mix**
   (tick *Extract guitar stem* and Demucs will demix it first).
2. **DI**: your own clean (unamped) guitar take.
3. **NAM library**: a folder of `.nam` captures — or search **TONE3000** below and download
   only the shortlist you want to try.

NAM EQ Matcher reamps your DI through every capture, searches the input gain that matches the
target's **saturation & compression character**, then designs a **match IR** that corrects
the remaining EQ/cab difference. You get the winning model + gain setting + IR to load
straight into the NAM plugin.
"""


# ---------------------------------------------------------------------------
# TONE3000 integration
# ---------------------------------------------------------------------------

_t3k_client = None


def _get_t3k():
    global _t3k_client
    if _t3k_client is None:
        from tonematch.tone3000 import T3KClient

        _t3k_client = T3KClient()
    return _t3k_client


def t3k_connect(key):
    from tonematch.tone3000 import T3KError

    client = _get_t3k()
    try:
        client.connect(key.strip() or None)
        return "✅ Connected to TONE3000."
    except T3KError as e:
        return f"❌ {e}"


def _rig_label(r):
    ts = r.get("tone_stack") or {}
    eq = f", EQ B{ts['bass']:g}/M{ts['middle']:g}/T{ts['treble']:g}" if ts else ""
    return f"#{r['rank']} {r['name']} (gain {r['input_gain_db']:+.1f} dB{eq})"


def t3k_search(query, amps_text, gear, sort):
    from tonematch.tone3000 import T3KError

    client = _get_t3k()
    amps = [a for a in (amps_text or "").split(",") if a.strip()]
    try:
        tones = client.search_tones_refined(
            query or "", amps=amps, gear=gear, sort=sort, page_size=30
        )
    except T3KError as e:
        raise gr.Error(str(e))
    rows = [
        [
            i + 1,
            t["id"],
            t["title"],
            ", ".join(m["name"] for m in t.get("makes", [])) or "-",
            t.get("_matched_on", "-"),
            t["gear"],
            t["models_count"],
            t["downloads_count"],
        ]
        for i, t in enumerate(tones)
    ]
    hint = f" (refined by: {', '.join(a.strip() for a in amps)})" if amps else ""
    status = f"{len(tones)} NAM tones found{hint}. Nothing downloaded yet — pick rows below."
    return rows, tones, status


def t3k_download(tones, rows_text):
    from tonematch.tone3000 import T3KError

    if not tones:
        raise gr.Error("Search first, then pick rows to download.")
    try:
        picks = [int(s) for s in rows_text.replace(",", " ").split()]
    except ValueError:
        raise gr.Error("Row numbers, please — e.g.  1 3 7")
    bad = [p for p in picks if not 1 <= p <= len(tones)]
    if bad:
        raise gr.Error(f"Row(s) out of range: {bad}")

    client = _get_t3k()
    downloaded = []
    try:
        for p in picks:
            downloaded.extend(client.download_tone(tones[p - 1]["id"], T3K_CACHE))
    except T3KError as e:
        raise gr.Error(str(e))
    status = f"⬇️ Downloaded {len(downloaded)} .nam file(s) to `{T3K_CACHE}` — folder set below."
    return status, T3K_CACHE


def do_match(target, di, models_dir, model_files, demix, stem, gain_lo, gain_hi, refine_top, render_top, limit, device, preview_s, progress=gr.Progress()):
    if target is None or di is None:
        raise gr.Error("Please provide both a target recording and a DI track.")

    stem_path = None
    if demix:
        try:
            from tonematch.stems import extract_stem

            stem_dir = tempfile.mkdtemp(prefix="tonematch_stems_")
            target = extract_stem(
                target,
                stem_dir,
                stem=stem,
                progress_cb=lambda f, m: progress(0.1 * f, desc=f"Demucs: {m}"),
            )
            stem_path = target
        except ImportError as e:
            raise gr.Error(str(e))

    sources = []
    if models_dir and os.path.isdir(models_dir.strip()):
        sources.append(models_dir.strip())
    if model_files:
        sources.extend(f.name if hasattr(f, "name") else f for f in model_files)
    if not sources:
        raise gr.Error("Point me at a folder of .nam files or upload some.")

    from tonematch.nam_backend import resolve_device

    dev = resolve_device(device)
    progress(0.0, desc=f"Loading NAM captures ({dev})...")
    captures, load_errors = [], []
    for s in sources:
        captures.extend(
            load_captures(s, limit=int(limit) if limit else None, errors_out=load_errors, device=dev)
        )
    # dedupe by path
    seen, uniq = set(), []
    for c in captures:
        if c.path not in seen:
            seen.add(c.path)
            uniq.append(c)
    captures = uniq[: int(limit)] if limit else uniq
    if not captures:
        detail = "\n".join(
            f"• {os.path.basename(p)} — {msg}" for p, msg in load_errors[:5]
        ) or "No .nam files were found at the given locations."
        raise gr.Error(
            f"Could not load any NAM captures:\n{detail}\n\n"
            "Run `python -m tonematch.doctor your_model.nam` for a full diagnosis."
        )

    out_dir = tempfile.mkdtemp(prefix="tonematch_")

    def cb(frac, msg):
        progress(min(frac, 1.0), desc=f"{msg} ({len(captures)} captures)")

    result = run_match(
        target,
        di,
        captures,
        out_dir,
        gain_range_db=(float(gain_lo), float(gain_hi)),
        refine_top=max(int(refine_top), int(render_top)),
        render_top=int(render_top),
        preview_s=float(preview_s),
        progress_cb=cb,
    )

    rows = [
        [i + 1, r.name, f"{r.gain_db:+.1f}", f"{r.score:.4f}", f"{r.nl_distance:.4f}", f"{r.eq_penalty:.4f}"]
        for i, r in enumerate(result.ranked[:15])
    ]
    best = result.report["best_model"]
    ts = result.renders[0]["tone_stack"]
    summary = (
        f"### 🏆 Best match: **{best['name']}**\n"
        f"- Input gain: **{best['input_gain_db']:+.1f} dB**\n"
        f"- Plugin EQ suggestion: **Bass {ts['bass']:g} · Middle {ts['middle']:g} · Treble {ts['treble']:g}**\n"
        f"- Model file: `{best['file']}`\n\n"
        f"**{result.report['how_to_use']}**"
    )
    if len(result.renders) > 1:
        summary += "\n\n### Rendered rigs (each with IR + EQ + hybrid)\n" + "\n".join(
            f"- {_rig_label(r)}" for r in result.renders
        )
    files = [result.report_path]
    for r in result.renders:
        if r.get("nam_copy"):
            files.append(r["nam_copy"])
        files.extend([r["settings_txt"], r["ir"], r["render"], r["tone_stack"]["render"],
                      r["hybrid"]["gentle_ir"], r["hybrid"]["render"]])
    if result.plot_path:
        files.append(result.plot_path)
    if stem_path:
        files.append(stem_path)
    return (
        summary,
        rows,
        result.target_ref_path,
        result.render_path,
        result.renders[0]["tone_stack"]["render"],
        result.renders[0]["hybrid"]["render"],
        result.plot_path,
        files,
    )


with gr.Blocks(title="NAM EQ Matcher") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            target_in = gr.Audio(label="Target recording (isolated guitar)", type="filepath")
            di_in = gr.Audio(label="Your DI track (clean)", type="filepath")
            models_dir_in = gr.Textbox(
                label="NAM models folder",
                placeholder=r"e.g. C:\Users\you\Documents\NAM models",
            )
            model_files_in = gr.Files(label="...or upload .nam files", file_types=[".nam"])
            with gr.Accordion("🔍 Search TONE3000 (download only what you need)", open=False):
                t3k_key_in = gr.Textbox(
                    label="Publishable API key",
                    placeholder="t3k_pub_...  (tone3000.com → Settings → API Keys; blank if already connected)",
                    type="password",
                )
                t3k_connect_btn = gr.Button("Connect TONE3000 account")
                t3k_status = gr.Markdown()
                with gr.Row():
                    t3k_query_in = gr.Textbox(label="Search", placeholder="e.g. british crunch, high gain...", scale=3)
                    t3k_gear_in = gr.Dropdown(
                        ["any", "amp", "full-rig", "pedal", "outboard"], value="any", label="Gear", scale=1
                    )
                    t3k_sort_in = gr.Dropdown(
                        ["downloads-all-time", "trending", "best-match", "newest"],
                        value="downloads-all-time",
                        label="Sort",
                        scale=1,
                    )
                t3k_amps_in = gr.Textbox(
                    label="Amps you're looking for (optional, comma-separated)",
                    placeholder="e.g. Marshall JCM800, Friedman BE-100, 5150 — runs one search per amp and ranks make-metadata hits first",
                )
                t3k_search_btn = gr.Button("Search catalog (no downloads)")
                t3k_results = gr.Dataframe(
                    headers=["#", "ID", "Title", "Make", "Matched on", "Gear", "Models", "Downloads"],
                    label="Results (metadata only)",
                    interactive=False,
                )
                t3k_tones_state = gr.State([])
                with gr.Row():
                    t3k_rows_in = gr.Textbox(label="Rows to download", placeholder="e.g. 1 3 7", scale=3)
                    t3k_dl_btn = gr.Button("Download selected", scale=1)
            with gr.Row():
                demix_in = gr.Checkbox(
                    label="Target is a full mix — extract guitar stem first (Demucs)", value=False
                )
                stem_in = gr.Dropdown(
                    ["guitar", "other", "guitar+other"],
                    value="guitar",
                    label="Stem",
                    scale=0,
                )
            with gr.Accordion("Advanced", open=False):
                gain_lo = gr.Slider(-24, 0, value=-12, step=1, label="Input gain search min (dB)")
                gain_hi = gr.Slider(0, 24, value=12, step=1, label="Input gain search max (dB)")
                refine_top = gr.Slider(1, 10, value=5, step=1, label="Captures to refine (stage 2)")
                render_top = gr.Slider(
                    1, 10, value=1, step=1,
                    label="Top rigs to render — each gets its own match IR",
                )
                limit = gr.Number(value=0, precision=0, label="Max captures to load (0 = all)")
                device_in = gr.Dropdown(
                    ["auto", "cpu", "cuda"], value="auto",
                    label="Processing device (auto = GPU if available)",
                )
                preview_s_in = gr.Slider(
                    0, 120, value=30, step=5,
                    label="Render length (s) - loudest section of your DI; 0 = full DI (slow)",
                )
            go = gr.Button("Match my tone", variant="primary")
        with gr.Column():
            summary_out = gr.Markdown()
            table_out = gr.Dataframe(
                headers=["#", "Capture", "Gain (dB)", "Score", "NL dist", "EQ penalty"],
                label="Ranking",
                interactive=False,
            )
            target_audio_out = gr.Audio(label="A: Target (reference)")
            render_audio_out = gr.Audio(label="B: NAM + match IR (#1, closest match)")
            eq_audio_out = gr.Audio(label="C: NAM + plugin EQ only (#1, most natural)")
            hybrid_audio_out = gr.Audio(label="D: NAM + plugin EQ + gentle IR (#1, recommended)")
            plot_out = gr.Image(label="Spectrum match")
            files_out = gr.Files(label="Downloads (per rig: .nam copy, settings.txt, IRs, renders; plus report & plot)")

    t3k_connect_btn.click(t3k_connect, inputs=[t3k_key_in], outputs=[t3k_status])
    t3k_search_btn.click(
        t3k_search,
        inputs=[t3k_query_in, t3k_amps_in, t3k_gear_in, t3k_sort_in],
        outputs=[t3k_results, t3k_tones_state, t3k_status],
    )
    t3k_dl_btn.click(
        t3k_download,
        inputs=[t3k_tones_state, t3k_rows_in],
        outputs=[t3k_status, models_dir_in],
    )

    go.click(
        do_match,
        inputs=[target_in, di_in, models_dir_in, model_files_in, demix_in, stem_in, gain_lo, gain_hi, refine_top, render_top, limit, device_in, preview_s_in],
        outputs=[summary_out, table_out, target_audio_out, render_audio_out, eq_audio_out, hybrid_audio_out, plot_out, files_out],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
