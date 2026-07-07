# NAM EQ Matcher

Clone a guitar tone from a finished recording.

Combines **NAM** (nonlinear amp character: gain structure, saturation, compression, sag)
with a **EQ Match**-style matched impulse response (linear EQ/cab correction).

## How it works

You provide three things:

1. **Target** — a recording with the tone you want: an isolated guitar track, or a full mix
   (built-in Demucs demixing extracts the guitar stem for you)
2. **DI** — your own clean guitar take (try to match the original recording both in content and lenght, or ideally similar playing style/register,)
3. **NAM library** — a folder of `.nam` captures, or use the built-in **TONE3000 search**
   to browse the catalog (metadata only) and download just a shortlist

The pipeline:

1. **Dynamic/saturation match (NAM half).** Your DI is reamped through every capture in
   the library across a grid of input gains. Each result is compared to the target using an
   *EQ-invariant fingerprint*: crest factor, dynamic range compression, spectral flatness of
   the fizz region, spectral flux, and MFCC texture statistics. These capture what an EQ
   cannot fix — clipping behavior, compression feel, harmonic density. A two-stage search
   (coarse over all captures, fine over the top 5) finds the best capture + drive setting.
2. **Frequency match (EQ Match half).** The winning rig's output is compared to the
   target's 1/6-octave smoothed long-term spectrum. The difference (clipped to ±18 dB,
   tapered at the extremes) is turned into a minimum-phase FIR and exported as
   `match_ir.wav` — load it in the NAM plugin's IR slot as your "cabinet".
3. **Outputs.** Best model + input gain, the match IR, a rendered preview of *your* DI
   through the full matched chain, a comparison spectrum plot, and a JSON report.
   Set *Top rigs to render* (GUI) or `--render-top N` (CLI) to get the top-N captures
   instead of just the winner — each gets its own folder (`rank01_...`, `rank02_...`)
   with its own `match_ir.wav` and `matched_render.wav`, so you can A/B them by ear.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Use

**GUI:**

```bash
python app.py
# open http://127.0.0.1:7860
```

**CLI:**

```bash
python match.py --target song_stem.wav --di my_di.wav --models ./nam_models --out ./results

# render the top 5 rigs, each with its own match IR:
python match.py --target song_stem.wav --di my_di.wav --models ./nam_models --render-top 5

# full mix? extract the guitar stem first with Demucs:
python match.py --target full_song.mp3 --demix --di my_di.wav --models ./nam_models --out ./results
```

**TONE3000 search (no bulk downloads):**

Create a free publishable API key at tone3000.com → Settings → API Keys (`t3k_pub_...`).
Searching returns metadata only; you download just the tones you pick (a `.nam` is usually
well under 1 MB). Matching itself always needs the files — the matcher must reamp your DI
through each capture. First connect opens your browser for TONE3000 login (OAuth); tokens
are cached in `~/.tonematch/`. Rate limit: 100 requests/min.

In the GUI: open the *Search TONE3000* accordion → connect → search → enter row numbers →
Download selected (goes to `./t3k_cache`, and the models-folder field is filled in for you).

If you know what amps you're after, list them in the *Amps you're looking for* field
(comma-separated, up to 6). NAM EQ Matcher runs one catalog query per amp, merges the results,
and ranks tones whose **make metadata** matches first (then title, tags, description) — the
*Matched on* column shows why each result surfaced.

CLI:

```bash
python -m tonematch.tone3000 connect --key t3k_pub_xxx
python -m tonematch.tone3000 search "crunch" --amps "Marshall,JCM800,Friedman" --gear amp
python -m tonematch.tone3000 download 12345 67890 --out ./t3k_cache
python match.py --target stem.wav --di my_di.wav --models ./t3k_cache
```

**Test (no torch needed):**

```bash
python -m tests.test_synthetic
```

## Using the result

In the NAM plugin: load the winning `.nam` file, set **Input** to the reported gain (dB),
load `match_ir.wav` in the **IR slot** (disable any other cab), adjust Output to taste.

## Tips

- The better your target isolation, the better the match. For full mixes use `--demix`
  (or the GUI checkbox): Demucs `htdemucs_6s` extracts a dedicated **guitar** stem. If the
  guitar sounds thin/incomplete in that stem, try `guitar+other`. First run downloads the
  model; CPU separation takes roughly the length of the song. Expect residual bleed to
  slightly skew the EQ match — trust your ears over the plot.
- Play a DI similar in register and intensity to the target part — the matcher compares
  statistics, not aligned samples, but similar material makes them comparable.
- More diverse `.nam` libraries → better odds one capture nails the clipping character.
- Search is CPU-heavy: ~10 reamps per capture. Use "Max captures" or a curated subfolder
  for quick passes.

## Project layout

The Python package keeps the internal name `tonematch` (renaming it would break
imports and your venv); the product name is **NAM EQ Matcher**.

```
tonematch/
  audio.py        I/O, resampling, envelopes, segment selection
  features.py     tone fingerprint (dynamics, saturation, texture) + LTAS
  match_eq.py     matched-IR design (1/6-oct smoothing, min-phase FIR)
  nam_backend.py  .nam loading (neural-amp-modeler) + MockAmp for tests
  search.py       two-stage capture ranking + input-gain search
  stems.py        Demucs guitar-stem extraction (optional dependency)
  tone3000.py     TONE3000 API client (OAuth PKCE, search, selective download)
  pipeline.py     end-to-end orchestration + reports
app.py            Gradio GUI
match.py          CLI
tests/            synthetic end-to-end test
```
