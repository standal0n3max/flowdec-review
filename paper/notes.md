# FlowDec — reading notes (ICLR 2025)

Source PDF: `paper/flowdec_iclr.pdf` (ICLR proceedings), extracted text `paper/flowdec_iclr.txt`.
OpenReview `uxDFlPGRLX`. Demo page: <https://sp-uhh.github.io/FlowDec/>.

## One-line summary

A 48 kHz full-band **general-audio** neural codec that is **non-adversarial** (no GAN) and adds
a stochastic **conditional-flow-matching postfilter** on top of a deterministic codec, reaching
good perceptual quality at **4.5–7.5 kbit/s** with only **6 DNN evaluations**.

## Method building blocks

- **Postfilter codec view:** decoding is one-to-many, so `D(E(x*)) = x*` is impossible for a
  one-to-one decoder. Instead learn `p(x̂ | c)` and sample from it: `D_s(c) = Ω(D_0(c))`.
- **Joint flow matching (their CFM variant).**
  - Standard FM: `x0 ~ N(0, σ²I)` independent of `x1`; conditional paths are OT but the
    marginal field is not, giving high-variance training / low straightness.
  - FlowDec: couple `x0` to the codec output: `q0(x0|x1,y) = N(y, Σ_y)`, `σ1 = 0`, linear
    interpolation `x_t = t x1 + (1−t) x0`. Loss (Eq. 11):
    `E ‖ v_θ(x_t,t,y) − (x* − (y + σ_y ε)) ‖²₂`.
  - Consequences: **contractive** flow field (numerically stable, converges locally), reaches
    the target exactly, **one** hyperparameter `σ_y`.
- **Frequency-dependent `σ_y(f)`** (Section 3.5): per-STFT-band quantile heuristic (768-point
  curve, Gaussian-smoothed, bandwidth 3) to avoid high-frequency over-smoothing. Global
  `σ_y = 0.66` for the ablation variants.
- **Non-adversarial codec = NDAC** (Section 3.3–3.4): DAC with all adversarial losses removed,
  plus a **multi-scale CQT loss** (nnAudio CQT2010v2, 9 octaves, hop 256, min freq 27.5 Hz,
  {16,32,48,64,80} bins/octave, weight 1 for music else 0) and an **L1 waveform loss**
  (weight 50). Fixes bad low-frequency modeling and SI-SDR ≈ −30 dB of the naive non-adv. DAC.
- **Feature representation** (Appendix A.4): amplitude-compressed complex STFT,
  `Φ(x) = β |X̃|^α exp(i∠X̃)`, 1534-sample Hann window (31.96 ms, F=768), hop 384 (75% overlap),
  **α = 0.3** (vs ScoreDec's 0.5).
- **Backbone** (Appendix A.3): NCSN++ 2-D U-Net reconfigured — first two depths 256 channels,
  depth 4, 1 ResNet block/depth, **all attention removed**, **26 M parameters** (vs original 65 M).
- **Training:** Adam, lr 1e-4, 800 k iterations, 2 s snippets, batch 64, EMA decay 0.999.

## Codec variants (Table 2)

| Name | Bitrates (kbit/s) | fs (kHz) | Hop | #codebooks | latent dim |
|---|---|---|---|---|---|
| DAC (official) | 0.86–7.75 | 44.1 | 512 | 9 | 1024 |
| NDAC-75 | 0.75–7.50 | 48 | 640 | 10 | 1024 |
| NDAC-25 | 0.25–4.00 | 48 | 1920 | 16 | 128 |

FlowDec-75m: NDAC-75 + multi-bitrate postfilter, `nq = {10,8,6,4}` → {7.5,6.0,4.5,3.0} kbit/s.
FlowDec-75s: single-bitrate 7.5 (ablation). FlowDec-25s: NDAC-25, 4.0 kbit/s.

## Key results (paper)

Headline comparison (Table 4, FlowDec-75s vs ScoreDec / constant-σt), **NFE=6**:

| Method | FAD×100 | SI-SDR | fwSSNR |
|---|---:|---:|---:|
| FlowDec | **1.62** | 7.55 | 15.46 |
| ScoreDec | 145.30 | −27.23 | 3.15 |
| σt=0.05 | 28.88 | 9.95 | 5.50 |
| σt=0.66 | 29.83 | 10.10 | 6.55 |

Same at **NFE=50**: FlowDec 1.34 / 7.41 / 15.65; ScoreDec 5.73 / 7.50 / 14.45.
→ FlowDec is stable at low NFE; ScoreDec collapses at NFE=6.

Other numbers:
- DAC-75 at 7.5 kbit/s: **FAD 4.15**, SI-SDR 10.03, fwSSNR 16.57 → FlowDec has better FAD,
  worse SI-SDR (the perception–distortion trade-off).
- **RTF** (A100): NDAC-75 0.0134, NDAC-25 0.0084; postfilter ≈ 0.0358·NFE → FlowDec-75 ≈ 0.229
  at NFE=6, vs **ScoreDec 1.707**.
- Listening tests (MUSHRA): FlowDec **on par** with DAC-75 at equal bitrate; both above Opus.
- Ablations: original NCSN++ and α=0.5 are worse (Table 7).

## Our review's own findings

- Official checkpoints reproduce the authors' published demo outputs (SI-SDR/fwSSNR within
  ~0.5 dB; the postfilter is stochastic).
- Apple-Silicon **MPS** is ~6× faster than CPU but **degrades quality** (logSpecMSE up ~10 dB);
  NDAC is numerically identical across devices, so the gap is in the ODE/postfilter path.
