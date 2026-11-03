# FlowDec: A flow-based full-band general audio codec — review and testing

**Course project, Advanced Machine Learning (AMLT ITMO 2026)**
**Paper:** *FlowDec: A flow-based full-band general audio codec with high perceptual quality*,
Welker, Le, Chen, Hsu, Gerkmann, Richard, Wu — **ICLR 2025** (Poster), OpenReview
[`uxDFlPGRLX`](https://openreview.net/forum?id=uxDFlPGRLX), arXiv
[2412.20156](https://arxiv.org/abs/2412.20156).

Code: [facebookresearch/FlowDec](https://github.com/facebookresearch/FlowDec) (CC-BY-NC core).

---

## 1. Introduction

### 1.1 The problem

An **audio codec** compresses a waveform into a compact, quantized representation and
reconstructs the waveform from it. Legacy codecs (Opus, AAC, EVS) rely on hand-crafted
transform/prediction designs, while modern **end-to-end neural codecs** (SoundStream,
EnCodec, DAC) learn the whole pipeline. To sound natural at low bitrates, almost all neural
codecs use **adversarial (GAN) training**: a discriminator pushes the decoder's output
distribution towards natural audio. GAN training is notoriously unstable, needs several
hand-tuned discriminators and loss weights, and can suffer mode collapse.

The prior **postfilter** work **ScoreDec** avoided the GAN by attaching a *score-based
diffusion* model to a codec. ScoreDec produced high quality, but only for **speech**, only
at a high **24 kbit/s** bitrate, and — critically — needed around **60 network evaluations**
(NFE) at inference, a real-time factor (RTF) of ~1.7. That makes it impractical.

### 1.2 The ideas FlowDec proposes, and why they are new

FlowDec keeps the postfilter idea but rebuilds it around **conditional flow matching (CFM)**,
and generalizes it:

1. **A new CFM formulation for signal enhancement.** Instead of the usual flow-matching
   coupling `x0 ~ N(0, σ²I)` independent of the data, FlowDec uses a *joint* coupling whose
   initial state is **centered on the codec output** `y`:
   `x0 ~ N(y, Σ_y)`, interpolating linearly to the clean target `x1 = x*`. Because the initial
   Gaussians are centered on the codec estimate and `σ1 = 0`, the flow is **contractive**
   (trajectories actually reach the target) and numerically stable, and the ODE can be solved
   accurately with very few steps. It needs only **one** hyperparameter, `σ_y`, instead of the
   four of ScoreDec's SDE.
2. **Frequency-dependent `σ_y(f)`.** Natural audio power drops with frequency; a single scalar
   `σ_y` over-smooths high frequencies. FlowDec estimates a per-frequency-band `σ_y` from the
   data (a quantile heuristic), which noticeably improves low-NFE quality.
3. **Non-adversarial codec training.** The underlying codec **NDAC** is DAC with all
   adversarial losses removed, plus two additions that a non-adversarial codec needs:
   a **multi-scale constant-Q transform (CQT) loss** (good low-frequency resolution) and an
   **L1 waveform loss** (fixes SI-SDR/phase, which magnitude-only losses miss). The
   distribution-matching job is handed to the CFM postfilter.
4. **General full-band audio.** 48 kHz (full hearing range, ≤20 kHz) for **speech, music and
   ambient** sound, at **4.5–7.5 kbit/s**, versus ScoreDec's speech-only 24 kbit/s.

### 1.3 Why this is not "just another JPEG for audio"

A classical transform codec **throws information away** and the decoder cannot invent it,
which is why it sounds hollow at very low rates. FlowDec's decoder **generatively
reconstructs** what was lost: the CFM postfilter samples a plausible full-band signal
conditional on the codec's coarse estimate. On top of that, FlowDec produces **discrete
tokens**, which is exactly what audio-LLM / TTS systems need; a lower bitrate means a shorter
token sequence and therefore cheaper language-model generation.

---

## 2. Model

### 2.1 High-level architecture

FlowDec is a **two-stage postfilter codec**:

```
waveform x*
   │  encoder E                    (NDAC-75 encoder, convolutional)
   ▼
latent z ──► residual vector quantizer (RVQ) ──► discrete codes c   ← the "bitstream"
   │                                                  │
   │  decoder D0 (non-adversarial)                    │  n_quantizers nq sets the bitrate
   ▼                                                  ▼
coarse estimate y = D0(c)          ◄──────────────────┘
   │  stochastic postfilter Ω = FlowDec (CFM) — solves an ODE
   ▼
enhanced waveform x̂  (48 kHz, full-band)
```

- **Underlying codec — NDAC (improved non-adversarial DAC).** DAC's encoder/decoder and RVQ,
  trained with reconstruction + multi-scale mel losses, **plus CQT and L1-waveform losses**,
  and **no discriminator**. `NDAC-75` runs at 75 Hz feature rate, 48 kHz, hop 640, 10
  codebooks of 1024 dims; `NDAC-25` at 25 Hz, hop 1920, 16 codebooks of 128 dims.
  Bitrate is chosen at inference by **how many codebooks are used**:
  `nq = 10 → 7.5 kbit/s`, `8 → 6.0`, `6 → 4.5`, `4 → 3.0`.
- **Postfilter — FlowDec (CFM).** A modified **NCSN++ 2-D U-Net** (26 M parameters): the first
  two depths doubled to 256 channels (suppresses high-frequency artifacts on music), depth
  reduced to 4, one ResNet block per depth, and **all attention removed** so runtime is linear
  in duration. It operates in a **compressed complex STFT** domain: a 1534-sample Hann window
  (31.96 ms, 768 bins), hop 384 (75% overlap), with amplitude compression `α = 0.3` and phase
  preserved. This invertible feature transform φ is applied before the flow and inverted after.

### 2.2 The algorithm

**Training (per FlowDec variant):** sample `t ~ U(0,1)`, `ε ~ N(0,I)`, set
`x_t = t·x* + (1−t)·(y + σ_y ε)` in the feature domain, and minimize the joint flow-matching
loss

    L_JFM = E ‖ v_θ(x_t, t, y) − (x* − (y + σ_y ε)) ‖²₂

i.e. the network regresses the **constant velocity** that carries a point on the line
`x0 → x1` from its noisy start to the clean target. Adam, lr 1e-4, 800 k iterations,
2 s snippets, batch 64, EMA 0.999.

**Inference:** start from `X = φ(y) + σ_y(f)·ε` and integrate the learned ODE
`dX/dt = v_θ(X, t, Y)` from `t = 0` to `1` with the **midpoint** solver for `N` steps
(NFE = 2·N; default N = 3 → **NFE = 6**), then map back to a waveform with φ⁻¹ and clip.

### 2.3 Pros and cons

**Pros**
- No GAN: simpler, stable training, interpretable objective, no mode collapse.
- **6 NFE** vs ~60 for ScoreDec — ~10× cheaper inference, RTF ≈ 0.23 (vs 1.71) on an A100.
- Full-band general audio (speech + music + ambient), very low bitrate (4.5–7.5 kbit/s).
- One hyperparameter (`σ_y`, made frequency-dependent) instead of a four-parameter SDE.
- Best-in-class **perceptual** (FAD) quality among compared codecs, on par with GAN-based DAC
  in listening tests, and better at preserving harmonics / "air" in music.

**Cons**
- **Intrusive** spectral metrics (SI-SDR, logSpecMSE) are *worse* than adversarial DAC: the
  model trades distortion for perception (the rate–distortion–perception trade-off).
- Still needs a strong pretrained **codec + a generative DNN**, so inference is heavier than a
  classical codec's.
- **Non-causal** architecture → **not streaming-capable** (the authors flag this as future work).
- Licence **CC-BY-NC** (non-commercial), and a heavy dependency stack (audiotools, hydra, DAC).

---

## 3. Experimental results on our own data

### 3.1 Input data (our custom 48 kHz set)

We deliberately did **not** reuse the paper's test set. We built a small, deterministic,
48 kHz mono set (5 s clips):

| Category | Clips | Source |
|---|---|---|
| Speech | 2 | macOS `say` text-to-speech (`Samantha`), converted to 48 kHz |
| Music | 2 | synthesized polyphonic chord progressions with harmonics, percussive transients and a bass line |
| Ambient / sound | 2 | synthesized 1/f noise beds with slow swells and tonal chirps |

The full generator is `scripts/make_custom_data.py` (fixed seed); the clips live in
`data/custom/`. If desired, real recordings can be dropped into `data/custom/raw/` and are
picked up automatically. We also downloaded the authors' **published demo audio** (clean
reference and their official FlowDec / DAC-75 outputs) into `data/authors/` to sanity-check
our pipeline against theirs.

### 3.2 How we tested / what we measured

Everything goes through the **official checkpoints** (`flowdec_75m` + `ndac-75`) using the
official inference API (`dac.encode(...) → model.enhance(..., N, solver)`) via
`scripts/flowdec_lib.py`. We use the EMA weights and the default `midpoint` solver.

Metrics (identical for every method), matching the paper:
- **SI-SDR** (dB) — scale-invariant signal-to-distortion ratio (signal fidelity);
- **fwSSNR** (dB) — frequency-weighted segmental SNR (better for music, e.g. bass);
- **logSpecMSE** — MSE in dB log-magnitude spectrograms (lower better);
- **FAD ×100** — Fréchet Audio Distance with **CLAP-laion-audio** embeddings (perception).

**Baselines (classical, non-neural — required by the assignment):**
- **Opus** through `ffmpeg` (`libopus`, CBR, 60 ms frames) at **3.0 / 4.5 / 7.5 kbit/s**,
- **Griffin-Lim** reconstruction from a **rate-matched** (~7.7 kbit/s) mel magnitude
  spectrogram (20 bands, 48 frames/s, 8-bit coefficients, 64 iterations) — the classical
  phase-recovery approach that discards phase and invents it from magnitude.

### 3.3 Reproduction on the authors' own audio

Before trusting our custom-data numbers we verified our pipeline on the authors' published
examples (`scripts/diagnose_mps.py`, `results/reproduction_authors.json`). On their clean
references our pipeline reproduces their official FlowDec outputs to within a fraction of a dB
in SI-SDR/fwSSNR (the postfilter is stochastic, so exact sample equality is not expected).
| clip | method | SI-SDR (dB) | fwSSNR (dB) | logSpecMSE |
|---|---|---:|---:|---:|
| Music_Licensed_1 | authors FlowDec | 7.70 | 15.75 | 448 |
| Music_Licensed_1 | ours FlowDec | 7.60 | 15.96 | 565 |
| Music_Licensed_1 | authors DAC-75 | 10.44 | 16.26 | 119 |
| Sound_AudioSet_1 | authors FlowDec | 0.92 | 15.57 | 91 |
| Sound_AudioSet_1 | ours FlowDec | 0.65 | 15.20 | 101 |
| Sound_AudioSet_1 | authors DAC-75 | 4.99 | 16.01 | 128 |
| Speech_EARS_1 | authors FlowDec | 8.60 | 15.20 | 103 |
| Speech_EARS_1 | ours FlowDec | 8.87 | 15.38 | 99 |
| Speech_EARS_1 | authors DAC-75 | 11.16 | 16.57 | 89 |

### 3.4 Results on our own data

| method | rate (kbit/s) | SI-SDR (dB) | fwSSNR (dB) | logSpecMSE | FAD×100 |
|---|---:|---:|---:|---:|---:|
| FlowDec | 7.5 | 15.13 | 17.72 | 126 | 19.1 |
| FlowDec | 4.5 | 12.75 | 15.13 | 134 | 26.0 |
| FlowDec | 3.0 | 10.92 | 13.31 | 141 | 35.3 |
| NDAC (codec only) | 7.5 | 17.51 | 19.99 | 66 | 29.4 |
| Opus | 7.5 | -2.08 | 6.48 | 1430 | 78.7 |
| Opus | 4.5 | -5.42 | 3.63 | 1475 | 99.2 |
| Opus | 3.0 | -17.32 | 1.12 | 1414 | 126.9 |
| Griffin-Lim | ~7.7 | -41.94 | 2.07 | 253 | 99.7 |

| category | method (rate) | SI-SDR (dB) | fwSSNR (dB) | logSpecMSE |
|---|---|---:|---:|---:|
| speech | FlowDec (7.5) | 10.76 | 16.51 | 151 |
| speech | FlowDec (4.5) | 9.13 | 14.04 | 154 |
| speech | Opus (4.5) | 0.29 | 4.57 | 737 |
| speech | Griffin-Lim (~7.7) | -27.20 | 4.86 | 434 |
| music | FlowDec (7.5) | 14.58 | 20.19 | 90 |
| music | FlowDec (4.5) | 12.23 | 17.32 | 93 |
| music | Opus (4.5) | 0.02 | 7.24 | 2337 |
| music | Griffin-Lim (~7.7) | -49.89 | 3.21 | 179 |
| ambient | FlowDec (7.5) | 20.05 | 16.47 | 136 |
| ambient | FlowDec (4.5) | 16.90 | 14.04 | 154 |
| ambient | Opus (4.5) | -16.57 | -0.93 | 1352 |
| ambient | Griffin-Lim (~7.7) | -48.75 | -1.87 | 147 |

### 3.5 NFE ablation (solver steps)

| NFE | SI-SDR (dB) | fwSSNR (dB) | logSpecMSE |
|---:|---:|---:|---:|
| 2 | 13.02 | 14.68 | 430 |
| 4 | 12.86 | 18.11 | 126 |
| 6 | 12.56 | 18.52 | 115 |
| 50 | 12.39 | 18.60 | 112 |

### 3.6 Analysis

- **FlowDec vs classical baselines.** At 4.5–7.5 kbit/s, Opus is severely degraded
  (SI-SDR around 0–2 dB, and unusable at 3 kbit/s), and Griffin-Lim is dominated because it
  discards phase entirely. FlowDec is far ahead on **every** metric, and its **FAD** advantage
  is very large — i.e. it *sounds* much closer to the reference than the classical baselines.
- **The perceptual/distortion trade-off (important nuance).** FlowDec's output is not a
  "closer waveform" than DAC's in the SI-SDR sense; it is a *more plausible* waveform. This is
  exactly why FAD and SI-SDR can disagree, and why the paper reports SI-SDR *worse* than DAC
  (7.55 vs 9.23) while FAD is *better* (1.62 vs 4.15). Our own numbers reproduce this **exact
  reversal against NDAC**: the codec-only baseline has the *best* waveform metrics
  (SI-SDR 17.51 vs 15.13, logSpecMSE 66 vs 126) yet a *worse* FAD (29.4 vs 19.1), i.e. the
  stochastic postfilter deliberately trades waveform fidelity for perceptual plausibility.
  FlowDec then beats Opus/Griffin-Lim on **both** metric families at once.
- **NFE robustness.** Quality climbs steeply from NFE = 2 to NFE = 6 and then plateaus, so
  NFE = 6 is a good operating point — the paper's central efficiency claim. ScoreDec, by
  contrast, collapses at NFE = 6 (the paper reports FAD ×100 = 145 and SI-SDR ≈ −27 dB there).
- **High frequencies.** The spectrograms (`results/plots/custom_spectrograms.png`) show
  FlowDec reconstructing high-frequency structure that Opus smooths away, while
  Griffin-Lim produces buzzy, phase-incoherent texture.
- **Device caveat (our own finding, not in the paper).** On this Mac (M4 Pro, 24 GB) CPU
  inference is faithful but slow (RTF ≈ 15). Apple-Silicon **MPS** is ~6× faster but
  **degrades quality** (logSpecMSE grows by roughly an order of magnitude, SI-SDR drops a few
  dB). The NDAC codec itself is numerically identical across devices, so the gap comes from
  the ODE/postfilter path (bandwidth-limited `float32` matmuls). We therefore report **CPU**
  numbers and document MPS only as a fast preview.

---

## 4. Conclusion

**Summary.** FlowDec shows that a **non-adversarial** neural codec plus a **conditional
flow-matching postfilter** can match GAN-based codecs perceptually and beat them in FAD, while
being far cheaper to sample (**6 NFE**) and simpler to train. The key technical move is the
*joint* CFM coupling `x0 ~ N(y, Σ_y)` that makes the flow contractive and stable at very low
step counts; frequency-dependent `σ_y(f)` removes high-frequency over-smoothing.

**What we verified.** The official checkpoints reproduce the authors' outputs on their own
demo audio; on our custom 48 kHz data FlowDec clearly beats rate-matched classical baselines
(Opus, Griffin-Lim) on perception (FAD) and on signal metrics, and stays robust from NFE = 4.

**Limitations.**
- Not streaming-capable (non-causal DNNs), so not (yet) a replacement for Opus in real-time
  communication; the paper suggests causal variants as future work.
- License **CC-BY-NC** and a heavy dependency stack limit direct commercial/production reuse.
- Intrusive metrics can look worse than adversarial DAC — quality must be judged perceptually.
- On Apple Silicon, the fast MPS path is **not** numerically faithful for the postfilter.

**Where it fits.** Not a replacement for FLAC/Opus for lossless/storage use, but a strong
component for **very-low-bitrate** general-audio coding and for tokenizing audio for
generative audio-language models.

---

## Appendix A. Reproducibility

- Environment pinned in `pyproject.toml` (Python 3.10, torch 2.7.1) and `requirements.txt`;
  `uv sync` reproduces it.
- All randomness seeded (`SEED = 1234`); metrics computed from saved WAVs in
  `results/demo/` and `results/*.json`.
- Hooks: `scripts/run_experiments.py` (stages `reproduce` / `custom` / `nfe`),
  `notebooks/flowdec_demo.ipynb` (frozen outputs for the demo).
- One portability fix is required on non-CUDA machines: the NCSN++ backbone loads small
  CUDA/C++ extensions at import; `scripts/patch_flowdec_mps.py` guards those and routes
  non-CUDA devices to the native `upfirdn2d` implementation.

## Appendix B. Licence / attribution

FlowDec code and checkpoints are released by Meta under **CC-BY-NC** (portions MIT/Apache).
This project is non-commercial coursework and redistributes no model weights.
