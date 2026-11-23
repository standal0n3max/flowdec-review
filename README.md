# Course project (AMLT ITMO 2026): review of **FlowDec** (ICLR 2025)

Review and hands-on testing of **FlowDec: A flow-based full-band general audio codec
with high perceptual quality** (Welker et al., ICLR 2025) on our own 48 kHz audio,
with a classical (non-neural) baseline comparison.

The talk is [report/AMLT_CP_flowdec.pdf](report/AMLT_CP_flowdec.pdf). Listening clips
for that talk are in `report/assets/audio/`.

## Paper

| | |
|---|---|
| Title | FlowDec: A flow-based full-band general audio codec with high perceptual quality |
| Venue | ICLR 2025 (Poster) |
| OpenReview | [uxDFlPGRLX](https://openreview.net/forum?id=uxDFlPGRLX) |
| Proceedings | [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3d937d5e40a883d5da9fd4aeeeb372b8-Abstract-Conference.html) |
| arXiv | [2503.01485](https://arxiv.org/abs/2503.01485) |
| Code | [facebookresearch/FlowDec](https://github.com/facebookresearch/FlowDec) |
| Demo page | [sp-uhh.github.io/FlowDec](https://sp-uhh.github.io/FlowDec/) |

FlowDec is a **48 kHz full-band neural audio codec** for general audio (speech, music,
ambient). It combines a **non-adversarial** codec (NDAC, an adversarially-free DAC) with a
stochastic **conditional-flow-matching (CFM) postfilter**. It reaches good perceptual quality
at 4.5–7.5 kbit/s with only **6 network evaluations (NFE)**, versus ~60 for the prior
score-based ScoreDec, and does so **without any GAN training**.

## Repository layout

```
├── README.md
├── pyproject.toml                 # pinned environment (uv); Python 3.10
├── requirements.txt
├── scripts/
│   ├── flowdec_lib.py             # device selection, checkpoint loading, inference
│   ├── baselines.py               # Opus (ffmpeg) + Griffin-Lim
│   ├── metrics.py                 # SI-SDR, fwSSNR, logSpecMSE, CLAP-based FAD
│   ├── make_custom_data.py        # builds the 48 kHz evaluation set
│   ├── patch_flowdec_mps.py       # makes the backbone importable on MPS/CPU
│   ├── diagnose_mps.py            # reproduction + CPU/MPS cross-check
│   ├── run_experiments.py         # reproduce / custom / nfe
│   └── build_notebook.py          # renders notebooks/flowdec_demo.ipynb
├── notebooks/flowdec_demo.ipynb
├── data/custom/                   # speech / music / ambient, 48 kHz
├── results/                       # metrics JSON and plots
└── report/
    ├── AMLT_CP_flowdec.pdf        # submitted talk
    └── assets/audio/              # A/B clips used in the talk
```

## One-time setup

Requires `uv`, `ffmpeg` (with `libopus`), and ~2.5 GB of disk for checkpoints.

```bash
uv sync

mkdir -p paper && curl -L -o paper/checkpoints.zip \
  https://github.com/facebookresearch/FlowDec/releases/download/checkpoints/checkpoints.zip
unzip -q paper/checkpoints.zip -d paper/

uv run python scripts/patch_flowdec_mps.py
```

## Reproduce the results

```bash
uv run python scripts/make_custom_data.py

PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/run_experiments.py --stage all --device cpu
```

Outputs land in `results/`: `custom_metrics.json`, `fad_custom.json`,
`reproduction_authors.json`, `nfe_ablation.json`, `plots/*.png`, and `demo/*.wav`.
CPU is the accurate path. On Apple Silicon, MPS is about 6× faster and degrades the
postfilter, so the reported numbers are CPU numbers.

## Findings

- The official checkpoints, run through this pipeline on the authors' demo audio, match
  their FlowDec outputs closely (`results/reproduction_authors.json`). The postfilter is
  stochastic, so sample-for-sample equality is not expected.
- On our own 48 kHz clips FlowDec beats rate-matched Opus and Griffin-Lim on both signal
  metrics and FAD. Against codec-only NDAC the picture reverses: NDAC is the closer
  waveform, FlowDec sounds more natural (lower FAD).
- Quality levels off by NFE = 6. More steps change little.

## Licensing

The FlowDec code and checkpoints are released by Meta under **CC-BY-NC** (non-commercial);
components are MIT/Apache. This repository is a non-commercial course project and does not
redistribute model weights.
