"""CLI for NAM EQ Matcher.

Example:
    python match.py --target song_guitar_stem.wav --di my_di.wav \
        --models ./nam_models --out ./results
"""

from __future__ import annotations

import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description="Match a guitar tone from a recording using NAM captures.")
    ap.add_argument("--target", required=True, help="Isolated guitar recording to match (wav/flac/mp3)")
    ap.add_argument("--di", required=True, help="Your clean DI track")
    ap.add_argument("--models", required=True, help="Folder of .nam files (or a single .nam)")
    ap.add_argument("--out", default="results", help="Output directory")
    ap.add_argument("--gain-range", type=float, nargs=2, default=(-12, 12), metavar=("LO", "HI"))
    ap.add_argument("--refine-top", type=int, default=5, help="Captures to refine in stage 2")
    ap.add_argument("--render-top", type=int, default=1, help="Render the top N rigs, each with its own match IR")
    ap.add_argument("--limit", type=int, default=None, help="Max captures to load")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="Processing device for NAM inference (auto = GPU if available)")
    ap.add_argument("--preview-s", type=float, default=30.0,
                    help="Render only the loudest N seconds of the DI per rig (0 = full DI, slow)")
    ap.add_argument("--demix", action="store_true", help="Target is a full mix: extract guitar stem with Demucs first")
    ap.add_argument("--stem", default="guitar", choices=["guitar", "other", "guitar+other"], help="Stem to extract with --demix")
    args = ap.parse_args()

    target_path = args.target
    if args.demix:
        from tonematch.stems import extract_stem

        print(f"Extracting '{args.stem}' stem with Demucs (first run downloads the model)...")
        target_path = extract_stem(
            args.target, args.out, stem=args.stem,
            progress_cb=lambda f, m: print(f"  {m}"),
        )
        print(f"Stem written to {target_path}")

    from tonematch.nam_backend import load_captures, resolve_device
    from tonematch.pipeline import run_match

    dev = resolve_device(args.device)
    captures = load_captures(args.models, limit=args.limit, device=dev)
    if not captures:
        print("No .nam captures loaded.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(captures)} capture(s) on {dev}. Searching...")

    last = [""]

    def progress(frac, msg):
        bar = f"[{int(frac * 100):3d}%] {msg}"
        if bar != last[0]:
            print("\r" + bar + " " * 20, end="", flush=True)
            last[0] = bar

    out = run_match(
        target_path,
        args.di,
        captures,
        args.out,
        sr=args.sr,
        gain_range_db=tuple(args.gain_range),
        refine_top=max(args.refine_top, args.render_top),
        render_top=args.render_top,
        preview_s=args.preview_s,
        progress_cb=progress,
    )
    print()
    ts = out.renders[0]["tone_stack"]
    print(f"\nBest match: {out.best.name}")
    print(f"  input gain : {out.best.gain_db:+.1f} dB")
    print(f"  plugin EQ  : Bass {ts['bass']:g}, Middle {ts['middle']:g}, Treble {ts['treble']:g}")
    print(f"  score      : {out.best.score:.4f}")
    if len(out.renders) > 1:
        print(f"\nRendered {len(out.renders)} rigs (each with match IR, plugin EQ, and hybrid):")
        for r in out.renders:
            t = r["tone_stack"]
            print(
                f"  #{r['rank']} {r['name']}  gain {r['input_gain_db']:+.1f} dB  "
                f"EQ B{t['bass']:g}/M{t['middle']:g}/T{t['treble']:g}  ->  {r['settings_txt']}"
            )
    print(f"\nOutputs in {args.out}:")
    print(f"  match IR   : {out.ir_path}")
    print(f"  render     : {out.render_path}")
    print(f"  report     : {out.report_path}")
    if out.plot_path:
        print(f"  plot       : {out.plot_path}")
    print(f"\n{out.report['how_to_use']}")


if __name__ == "__main__":
    main()
