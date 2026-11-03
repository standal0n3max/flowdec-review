---
marp: true
theme: default
paginate: true
size: 16:9
title: "FlowDec (ICLR 2025) — course project review"
author: "AMLT ITMO 2026"
---

<!--
Marp deck. Render with: npx @marp-team/marp-cli report/slides.md -o report/slides.pdf
Timing target: 10 min demo + 5 min Q&A.
-->

# FlowDec
### A flow-based full-band general audio codec with high perceptual quality
**Welker, Le, Chen, Hsu, Gerkmann, Richard, Wu — ICLR 2025**

Course project · AMLT ITMO 2026

https://github.com/facebookresearch/FlowDec · https://sp-uhh.github.io/FlowDec/

---

## Verdict first

- A **48 kHz full-band** neural codec for **speech + music + ambient** at **4.5–7.5 kbit/s**.
- **No GAN.** A non-adversarial codec (NDAC) + a **conditional-flow-matching (CFM) postfilter**.
- **6 network evaluations (NFE)** instead of ~60 for score-based ScoreDec (~10× cheaper).
- **Best FAD** among compared codecs; listening-test quality **on par** with GAN-based DAC.

> "Not a replacement for FLAC/Opus for storage — a component for very-low-bitrate coding
> and for tokenizing audio for audio-LLMs."

---

## 1. Introduction — the problem

- Classic codecs (Opus, AAC) = hand-designed; modern neural codecs (EnCodec, DAC) learn it all.
- To sound natural at low bitrate, neural codecs almost always use **adversarial (GAN) training**:
  - unstable, many discriminators + loss weights, risk of **mode collapse**.
- **ScoreDec** avoided the GAN with a *score-based diffusion postfilter* — but:
  - **speech only**, **24 kbit/s**, and **~60 NFE** (RTF ≈ 1.7) → impractical.

**FlowDec = keep the postfilter idea, rebuild it on flow matching, generalize it.**

---

## 1. Introduction — what is new

1. **New joint CFM coupling for enhancement.** Noise is centred on the codec output:
   `x0 ~ N(y, Σ_y)` → `x1 = x*`. The flow is **contractive** (reaches the target), stable,
   and accurate at **very few steps**. **One** hyperparameter, not four.
2. **Frequency-dependent σ_y(f).** Audio power falls with frequency → avoids high-freq
   over-smoothing; estimated from data.
3. **Non-adversarial codec (NDAC).** DAC minus all discriminators, plus **CQT loss**
   (low-frequency) and **L1 waveform loss** (phase / SI-SDR).
4. **General full-band.** 48 kHz, speech + music + ambient, **4.5–7.5 kbit/s**.

**Not "JPEG for audio":** the decoder *generatively reconstructs* lost detail instead of
just inverting — so it stays plausible at bitrates where classical codecs fall apart.

---

## 2. Model — overview

```
x*  ─encoder E─►  latent z ─RVQ─►  codes c        (bitrate set by #codebooks nq)
                                    │
                                    └─decoder D0─► coarse y
                                                     │
                              FlowDec postfilter Ω (CFM ODE)  ◄── 6 NFE
                                                     │
                                                     ▼
                                              enhanced x̂ (48 kHz)
```

- **NDAC-75**: 75 Hz feature rate, 48 kHz, hop 640, 10 codebooks × 1024 dims,
  `nq = {10, 8, 6, 4}` → **{7.5, 6.0, 4.5, 3.0} kbit/s**.
- **Postfilter**: modified **NCSN++ 2-D U-Net**, **26 M params** — first two depths 256 ch,
  depth 4, no attention (linear-time), operating on a **compressed complex STFT**
  (1534-sample Hann, hop 384, α = 0.3, phase kept).

---

## 2. Model — algorithm

**Train (joint flow matching):**

    L_JFM = E ‖ v_θ(x_t, t, y) − ( x* − (y + σ_y ε) ) ‖²₂ ,   x_t = t x* + (1−t)(y + σ_y ε)

Regress the *constant velocity* carrying a noisy point from the codec estimate to the target.
Adam, lr 1e-4, 800k iters, 2 s snippets, EMA 0.999.

**Infer:** `X = φ(y) + σ_y(f)·ε`, integrate `dX/dt = v_θ(X, t, Y)` with the **midpoint**
solver, `N` steps → **NFE = 2N** (default **N=3, NFE=6**), then φ⁻¹ and clip.

**Pros:** no GAN, stable, interpretable; 6 NFE; full-band general audio; best FAD.
**Cons:** worse intrusive SI-SDR than DAC (perception–distortion trade-off); heavy deps;
**non-causal → not streaming**; CC-BY-NC.

---

## 3. My experiments — setup

- **My data (48 kHz mono, 5 s):** 2 speech (macOS TTS), 2 music (synth chords + percussion +
  bass), 2 ambient (1/f noise + chirps). Deterministic generator, `data/custom/`.
- Also pulled the authors' **published demo audio** to validate my pipeline against theirs.
- **Official checkpoints** (`flowdec_75m` + `ndac-75`), official inference API, EMA weights.
- **Metrics (identical for all):** SI-SDR, fwSSNR, logSpecMSE, **FAD×100** (CLAP-laion-audio).
- **Classical baselines (required):** **Opus** via `ffmpeg` (3.0 / 4.5 / 7.5 kbit/s) and
  **Griffin-Lim** from a **rate-matched** (~7.7 kbit/s) mel magnitude spectrogram.

Hardware: Apple M4 Pro, 24 GB. **CPU** = faithful (reproduces authors); MPS = fast but lossy.

---

## 3. Reproduction on the authors' own audio

Our pipeline vs the authors' official FlowDec outputs (their clean references):

| clip | method | SI-SDR | fwSSNR |
|---|---|---:|---:|
| Music | authors FlowDec | 7.70 | 15.75 |
| Music | **ours** | 7.60 | 15.96 |
| Speech | authors FlowDec | 8.60 | 15.20 |
| Speech | **ours** | 8.87 | 15.38 |
| Sound | authors FlowDec | 0.92 | 15.57 |
| Sound | **ours** | 0.65 | 15.20 |

→ **Faithful** (within ~0.5 dB; the postfilter is stochastic). ✔

---

## 3. Results on my own data

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

(FlowDec ×3 bitrates, NDAC-only, Opus ×3, Griffin-Lim; mean over 6 clips.)

---

## 3. FlowDec vs classical baselines (one music clip)

| method | rate | SI-SDR | fwSSNR | FAD×100 |
|---|---:|---:|---:|---:|
| FlowDec | 7.5 | BEST | BEST | BEST |
| FlowDec | 4.5 | | | |
| Opus | 7.5 | | | |
| Opus | 4.5 | | | |
| Griffin-Lim | ~7.7 | | | |

| method | rate (kbit/s) | SI-SDR (dB) | fwSSNR (dB) | logSpecMSE |
|---|---:|---:|---:|---:|
| FlowDec | 7.5 | 14.56 | 20.04 | 97 |
| FlowDec | 4.5 | 11.84 | 16.93 | 105 |
| Opus | 7.5 | 2.33 | 7.79 | 2274 |
| Opus | 4.5 | 0.05 | 6.60 | 2297 |
| Griffin-Lim | ~7.7 | -45.38 | 2.64 | 172 |

Audible: Opus smooths the highs, Griffin-Lim is buzzy/incoherent, FlowDec keeps the "air".

---

## 3. NFE ablation — the efficiency claim

| NFE | SI-SDR (dB) | fwSSNR (dB) | logSpecMSE |
|---:|---:|---:|---:|
| 2 | 13.02 | 14.68 | 430 |
| 4 | 12.86 | 18.11 | 126 |
| 6 | 12.56 | 18.52 | 115 |
| 50 | 12.39 | 18.60 | 112 |

- Quality rises steeply NFE 2 → 6, then plateaus → **NFE = 6** is the sweet spot.
- Contrast: **ScoreDec collapses at NFE=6** (paper: FAD×100 = 145, SI-SDR ≈ −27 dB).
- Spectrograms: `results/plots/custom_spectrograms.png`.

---

## 3. Own finding — a device caveat

- Mac **M4 Pro**: CPU RTF ≈ 15 (slow but faithful).
- **MPS ≈ 6× faster, but degrades quality** (logSpecMSE up ~10 dB, SI-SDR down a few dB).
  NDAC is numerically identical across devices → the gap is in the ODE path (float32 matmuls).
- → I report **CPU** numbers, and document MPS as a fast preview only (not in the paper).

---

## 4. Conclusion

- **Non-adversarial codec + CFM postfilter** matches GAN-based codecs perceptually and
  **beats them in FAD**, at **6 NFE** — a genuinely practical generative codec.
- Reproduced the authors' outputs on their own audio; on **my own** data it clearly beats
  rate-matched **Opus** and **Griffin-Lim** on perception and signal metrics.
- **Key insight:** SI-SDR and FAD disagree — FlowDec trades distortion for perception
  (paper: SI-SDR 7.55 vs DAC 9.23, but FAD 1.62 vs 4.15).
- **Limits:** not streaming (non-causal), CC-BY-NC, heavy deps, MPS not faithful.

---

## Backup — Q&A prep

- **"Why a codec if Opus exists?"** → generative decoder; discrete tokens for audio-LLMs;
  a lower bitrate = shorter token sequence = cheaper generation.
- **"Novelty vs earlier neural codecs?"** → first competitive **non-adversarial** codec, and
  a CFM coupling stable at **6 NFE** where ScoreDec needs ~60.
- **"FAD better but SI-SDR worse?"** → perception–distortion trade-off: plausible ≠ close.
- **"What blocks adoption?"** → CC-BY-NC licence, dependencies, non-streaming, inference cost.
- **"Why does MPS hurt?"** → bandwidth-limited float32 ops; NDAC identical, so it is the solver.
