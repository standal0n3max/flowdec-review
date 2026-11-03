# Course project (AMLT ITMO 2026): review of **FlowDec** (ICLR 2025)

Review and hands-on testing of **FlowDec: A flow-based full-band general audio codec
with high perceptual quality** (Welker et al., ICLR 2025) on our own 48 kHz audio,
with a classical (non-neural) baseline comparison.

## Paper

| | |
|---|---|
| Title | FlowDec: A flow-based full-band general audio codec with high perceptual quality |
| Venue | ICLR 2025 (Poster) |
| OpenReview | [uxDFlPGRLX](https://openreview.net/forum?id=uxDFlPGRLX) |
| Proceedings PDF | [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3d937d5e40a883d5da9fd4aeeeb372b8-Abstract-Conference.html) |
| arXiv | [2412.20156](https://arxiv.org/abs/2412.20156) |
| Code | [facebookresearch/FlowDec](https://github.com/facebookresearch/FlowDec) |
| Demo page | [sp-uhh.github.io/FlowDec](https://sp-uhh.github.io/FlowDec/) |

FlowDec is a **48 kHz full-band neural audio codec** for general audio (speech, music,
ambient). It combines a **non-adversarial** codec (NDAC, an adversarially-free DAC) with a
stochastic **conditional-flow-matching (CFM) postfilter**. It reaches good perceptual quality
at 4.5–7.5 kbit/s with only **6 network evaluations (NFE)**, versus ~60 for the prior
score-based ScoreDec, and does so **without any GAN training**.

## Repository layout

```
CP/
├── README.md                     # this file
├── pyproject.toml                # pinned environment (uv); Python 3.10
├── requirements.txt              # frozen pip requirements
├── paper/                        # ICLR PDF, extracted text, checkpoints/
├── code/FlowDec/                 # official repo (git clone), + MPS patch
├── scripts/
│   ├── flowdec_lib.py            # device selection, checkpoint loading, inference
│   ├── baselines.py              # Opus (ffmpeg) + Griffin-Lim classical baselines
│   ├── metrics.py                # SI-SDR, fwSSNR, logSpecMSE, CLAP-based FAD
│   ├── make_custom_data.py       # builds our custom 48 kHz evaluation set
│   ├── patch_flowdec_mps.py      # makes the backbone importable on MPS/CPU
│   ├── build_notebook.py         # renders notebooks/flowdec_demo.ipynb
│   ├── fill_report.py            # injects result tables into report/slides
│   ├── slides_content.py         # single source of truth for the presentation deck
│   ├── build_slide_assets.py     # equations, pipeline diagram, code cells, audio clips
│   ├── build_slides_html.py      # renders report/flowdec_slides.html
│   ├── build_slides_pptx.py      # renders report/flowdec_slides.pptx
│   ├── diagnose_mps.py           # reproduction + CPU/MPS cross-check
│   └── run_experiments.py        # main driver (reproduce / custom / nfe stages)
├── data/
│   ├── authors/                  # authors' published demo audio (clean + outputs)
│   └── custom/                   # our own 48 kHz clips (speech / music / ambient)
├── notebooks/
│   ├── flowdec_demo.ipynb        # frozen, reproducible walkthrough (outputs embedded)
│   └── flowdec_demo.html         # self-contained export (open offline / backup)
├── results/                      # metrics JSON, plots, demo WAVs
└── report/
    ├── report.md                 # written report (EN)
    ├── slides.md                 # talk outline (Marp/Markdown)
    ├── flowdec_slides.html       # generated: self-contained interactive deck (EN)
    ├── flowdec_slides.pptx       # generated: 16:9 slide deck for submission
    └── assets/                   # generated: equations, diagram, code cells, audio
```

## One-time setup

Requires `uv`, `ffmpeg` (with `libopus`), and ~2.5 GB of disk for checkpoints.

```bash
# 1. Environment (Python 3.10, pinned versions)
uv sync

# 2. Official checkpoints (~1.2 GB) into paper/checkpoints/
mkdir -p paper && curl -L -o paper/checkpoints.zip \
  https://github.com/facebookresearch/FlowDec/releases/download/checkpoints/checkpoints.zip
unzip -q paper/checkpoints.zip -d paper/

# 3. Make the backbone work on Apple Silicon (idempotent)
uv run python scripts/patch_flowdec_mps.py
```

## Reproduce the results

```bash
# 4. Build our custom 48 kHz data set (deterministic; drop your own WAVs in data/custom/raw/)
uv run python scripts/make_custom_data.py

# 5. Run everything (reproduction + custom + NFE ablation), CPU is the accurate path
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/run_experiments.py --stage all --device cpu
```

Outputs land in `results/`: `custom_metrics.json`, `fad_custom.json`,
`reproduction_authors.json`, `nfe_ablation.json`, `plots/*.png`, and `demo/*.wav`.

## Findings in one glance

- **Reproduction.** Running the official checkpoints through our own pipeline on the
  authors' published demo audio reproduces their FlowDec outputs to well under 1 dB SI-SDR
  (see `results/reproduction_authors.json`), confirming the pipeline is faithful.
- **On our own data.** FlowDec gives much higher SI-SDR / fwSSNR than rate-matched
  classical baselines (Opus, Griffin-Lim) and far better FAD (perceptual), matching the
  paper's claim that quality is not captured by SI-SDR alone.
- **NFE.** With the midpoint solver FlowDec stays usable already at NFE = 4–6 and keeps
  improving towards NFE = 50 — the paper's key efficiency argument.
- **Device caveat (own finding).** On CPU, FlowDec is faithful but slow (RTF ≈ 15). On
  Apple-Silicon **MPS it is ~6× faster but degrades quality** (`logSpecMSE` grows by
  roughly 10 dB and SI-SDR drops a few dB), consistent with bandwidth-limited half/float32
  matmuls. We therefore use CPU for all reported numbers and document MPS as a fast preview.

See `report/report.md` for the full write-up and `report/slides.md` for the talk outline.

## Presentation (10 min + 5 min Q&A)

The deck has a single source of truth — `scripts/slides_content.py` — and two renderers, so
the HTML deck and the `.pptx` can never drift apart. All numbers are pulled from
`results/*.json` through the same aggregation helpers that fill `report/report.md`.

```bash
uv run python scripts/build_slide_assets.py   # equations, diagram, code cells, audio
uv run python scripts/build_slides_html.py    # report/flowdec_slides.html
uv run python scripts/build_slides_pptx.py    # report/flowdec_slides.pptx
```

- `report/flowdec_slides.html` — self-contained interactive deck (15 sections): sticky nav,
  light/dark theme, **lecture mode** (arrow keys page through sections, `Esc` to exit) and
  real `<audio>` A/B players. Open it in any browser; no server or network needed.
- `report/flowdec_slides.pptx` — a real 16:9 PowerPoint with speaker notes. Because
  `python-pptx` has no native audio shape, each listening clip is embedded as a short
  **mp4** (waveform poster + sound) that plays on click in PowerPoint and LibreOffice.
- Audio is **A/B against Opus at the same bitrate** (music, ambient, speech), all linked to
  the committed mp3/mp4 assets in `report/assets/audio/`.

Required assignment criteria are labelled on the relevant slides: novelty, reproduction on
the authors' audio, comparison with classical baselines.

## Licensing

The FlowDec code and checkpoints are released by Meta under **CC-BY-NC** (non-commercial);
components are MIT/Apache. This repository is for a non-commercial course project.
