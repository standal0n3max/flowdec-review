# Copyright (c) 2026 AMLT course project (FlowDec review).
# Single source of truth for the presentation deck: both the HTML and the PPTX
# renderers read the slides from here, so the two artefacts cannot drift apart.
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from beartype import beartype

import fill_report

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "report" / "assets"
DECK_HTML = ROOT / "report" / "flowdec_slides.html"
DECK_PPTX = ROOT / "report" / "flowdec_slides.pptx"

# --------------------------------------------------------------------------- #
# Content blocks.  A slide is an ordered list of blocks; each renderer walks
# the blocks and decides how to draw them.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Bullets:
    """A bullet list.  `ordered=True` renders a numbered list."""

    items: tuple[str, ...]
    ordered: bool = False


@dataclass(frozen=True)
class Para:
    """A plain paragraph."""

    text: str


@dataclass(frozen=True)
class Callout:
    """A highlighted note / takeaway box."""

    text: str


@dataclass(frozen=True)
class Table:
    """A simple table with a header row."""

    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    caption: str = ""


@dataclass(frozen=True)
class Picture:
    """An image, addressed by a path relative to the project root."""

    path: str
    caption: str = ""
    width: float = 0.9


@dataclass(frozen=True)
class Audio:
    """An A/B listening group built from slugs declared in `AUDIO_CLIPS`."""

    clips: tuple[str, ...]
    caption: str = ""


Block = Bullets | Para | Callout | Table | Picture | Audio


@dataclass(frozen=True)
class AudioAsset:
    """A demo clip plus the three derived files the builders create for it."""

    slug: str
    label: str
    source: str
    caption: str

    @property
    def mp3(self) -> str:
        return f"report/assets/audio/{self.slug}.mp3"

    @property
    def mp4(self) -> str:
        return f"report/assets/audio/{self.slug}.mp4"

    @property
    def wave(self) -> str:
        return f"report/assets/audio/{self.slug}_wave.png"


@dataclass(frozen=True)
class Slide:
    """One deck section."""

    sid: str
    nav: str
    eyebrow: str
    title: str
    subtitle: str
    blocks: tuple[Block, ...]
    notes: str = ""
    required: str = ""


# --------------------------------------------------------------------------- #
# Audio clips.  Sources live in the (git-ignored) results/demo directory; the
# derived mp3/mp4/wave assets are committed under report/assets/audio.
# --------------------------------------------------------------------------- #

AUDIO_CLIPS: tuple[AudioAsset, ...] = (
    AudioAsset(
        "music_flowdec45", "FlowDec 4.5 kbit/s", "results/demo/music_music_0_flowdec_4.5kbps.wav",
        "Music — FlowDec, 4.5 kbit/s (NFE 6)",
    ),
    AudioAsset(
        "music_opus45", "Opus 4.5 kbit/s", "results/demo/music_music_0_opus_4.5kbps.wav",
        "Music — Opus, 4.5 kbit/s",
    ),
    AudioAsset(
        "ambient_flowdec45", "FlowDec 4.5 kbit/s", "results/demo/ambient_ambient_0_flowdec_4.5kbps.wav",
        "Ambient — FlowDec, 4.5 kbit/s (NFE 6)",
    ),
    AudioAsset(
        "ambient_opus45", "Opus 4.5 kbit/s", "results/demo/ambient_ambient_0_opus_4.5kbps.wav",
        "Ambient — Opus, 4.5 kbit/s",
    ),
    AudioAsset(
        "speech_flowdec45", "FlowDec 4.5 kbit/s", "results/demo/speech_speech_0_flowdec_4.5kbps.wav",
        "Speech — FlowDec, 4.5 kbit/s (NFE 6)",
    ),
    AudioAsset(
        "speech_opus45", "Opus 4.5 kbit/s", "results/demo/speech_speech_0_opus_4.5kbps.wav",
        "Speech — Opus, 4.5 kbit/s",
    ),
    AudioAsset(
        "music_flowdec75", "FlowDec 7.5 kbit/s", "results/demo/music_music_0_flowdec_7.5kbps.wav",
        "Music — FlowDec, 7.5 kbit/s (NFE 6)",
    ),
)

REFERENCE_CLIPS: dict[str, str] = {
    "music": "data/custom/music/music_0.wav",
    "ambient": "data/custom/ambient/ambient_0.wav",
    "speech": "data/custom/speech/speech_0.wav",
}

AUDIO_BY_SLUG: dict[str, AudioAsset] = {clip.slug: clip for clip in AUDIO_CLIPS}


# --------------------------------------------------------------------------- #
# Markdown-ish inline formatting (bold + code only).
# --------------------------------------------------------------------------- #


def md_to_html(text: str) -> str:
    """Escape `text` and turn **bold** / `code` markers into HTML."""
    out = html.escape(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"`(.+?)`", r"<code>\1</code>", out)
    return out


def strip_md(text: str) -> str:
    """Drop the **bold** / `code` markers, keeping plain text."""
    out = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    out = re.sub(r"`(.+?)`", r"\1", out)
    return out


# --------------------------------------------------------------------------- #
# Numbers: reuse the exact aggregation logic that fills report/report.md so the
# deck and the report can never disagree.
# --------------------------------------------------------------------------- #


def _custom_rows() -> list[dict[str, Any]]:
    rows = fill_report.load("custom_metrics.json")
    if not isinstance(rows, list) or not rows:
        raise SystemExit("results/custom_metrics.json not found; run the pipeline first")
    return cast("list[dict[str, Any]]", rows)


def _fad_map() -> dict[str, Any]:
    fad = fill_report.load("fad_custom.json")
    return cast("dict[str, Any]", fad) if isinstance(fad, dict) else {}


def _mean(rows: list[dict[str, Any]], method: str, bitrate: float | None = None) -> dict[str, float]:
    return fill_report.agg(rows, method, bitrate)


def _num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _fad100(fad: dict[str, Any], key: str, digits: int = 1) -> str:
    value = fad.get(key)
    if not isinstance(value, (int, float)):
        return "—"
    return _num(float(value) * 100.0, digits)


def custom_table() -> Table:
    """Mean metrics over all six custom clips, with FAD×100."""
    rows = _custom_rows()
    fad = _fad_map()
    order: tuple[tuple[str, float | None], ...] = (
        ("flowdec", 7.5), ("flowdec", 4.5), ("flowdec", 3.0),
        ("ndac", 7.5),
        ("opus", 7.5), ("opus", 4.5), ("opus", 3.0),
        ("griffin_lim", None),
    )
    body: list[tuple[str, ...]] = []
    for method, bitrate in order:
        stats = _mean(rows, method, bitrate)
        if not stats:
            continue
        rate = "~7.7" if method == "griffin_lim" else _num(cast(float, bitrate), 1)
        fad_key = "griffin_lim" if bitrate is None else f"{method}_{bitrate}"
        body.append((
            fill_report.METHOD_LABELS[method],
            rate,
            _num(stats["si_sdr"]),
            _num(stats["fwssnr"]),
            _num(stats["logspec_mse"], 0),
            _fad100(fad, fad_key),
        ))
    return Table(
        headers=("method", "rate (kbit/s)", "SI-SDR (dB)", "fwSSNR (dB)", "logSpecMSE", "FAD×100"),
        rows=tuple(body),
        caption="Mean over our six custom 48 kHz clips (lower logSpecMSE and FAD are better).",
    )


def per_category_table() -> Table:
    """FlowDec vs Opus vs Griffin-Lim, split by speech / music / ambient."""
    rows = _custom_rows()
    body: list[tuple[str, ...]] = []
    for category in ("speech", "music", "ambient"):
        subset = [r for r in rows if r.get("category") == category]
        for method, bitrate in (("flowdec", 7.5), ("flowdec", 4.5), ("opus", 4.5), ("griffin_lim", None)):
            stats = _mean(subset, method, bitrate)
            if not stats:
                continue
            rate = "~7.7" if method == "griffin_lim" else _num(cast(float, bitrate), 1)
            body.append((
                category,
                f"{fill_report.METHOD_LABELS[method]} ({rate})",
                _num(stats["si_sdr"]),
                _num(stats["fwssnr"]),
                _num(stats["logspec_mse"], 0),
            ))
    return Table(
        headers=("category", "method (rate, kbit/s)", "SI-SDR (dB)", "fwSSNR (dB)", "logSpecMSE"),
        rows=tuple(body),
        caption="The same picture in every category: FlowDec is far ahead of Opus and Griffin-Lim.",
    )


def reproduction_table() -> Table:
    """Our pipeline vs the authors' own published outputs."""
    rows = fill_report.load("reproduction_authors.json")
    body: list[tuple[str, ...]] = []
    labels = {
        "authors_flowdec": "authors FlowDec",
        "ours_flowdec": "ours FlowDec",
        "authors_dac75": "authors DAC-75",
    }
    if isinstance(rows, list) and rows:
        data = cast("list[dict[str, Any]]", rows)
        for clip in sorted({str(r["clip"]) for r in data}):
            for method in ("authors_flowdec", "ours_flowdec", "authors_dac75"):
                row = next((r for r in data if r["clip"] == clip and r["method"] == method), None)
                if row is None:
                    continue
                body.append((
                    clip,
                    labels[method],
                    _num(float(row["si_sdr"])),
                    _num(float(row["fwssnr"])),
                    _num(float(row["logspec_mse"]), 0),
                ))
    return Table(
        headers=("clip", "method", "SI-SDR (dB)", "fwSSNR (dB)", "logSpecMSE"),
        rows=tuple(body),
        caption="Our pipeline on the authors' clean references vs their official outputs.",
    )


def nfe_table() -> Table:
    """Quality as a function of the number of function evaluations."""
    rows = fill_report.load("nfe_ablation.json")
    body: list[tuple[str, ...]] = []
    if isinstance(rows, list) and rows:
        data = cast("list[dict[str, Any]]", rows)
        for nfe in sorted({int(r["nfe"]) for r in data}):
            subset = [r for r in data if int(r["nfe"]) == nfe]
            body.append((
                str(nfe),
                _num(float(sum(r["si_sdr"] for r in subset) / len(subset))),
                _num(float(sum(r["fwssnr"] for r in subset) / len(subset))),
                _num(float(sum(r["logspec_mse"] for r in subset) / len(subset)), 0),
            ))
    return Table(
        headers=("NFE", "SI-SDR (dB)", "fwSSNR (dB)", "logSpecMSE"),
        rows=tuple(body),
        caption="Quality climbs sharply up to NFE = 6 and then plateaus.",
    )


def one_clip_table() -> Table:
    """The music_0 clip, method by method (same table as report/slides.md)."""
    rows = _custom_rows()
    body: list[tuple[str, ...]] = []
    clip_key = "music_music_0"
    for method, bitrate in (("flowdec", 7.5), ("flowdec", 4.5), ("opus", 7.5), ("opus", 4.5), ("griffin_lim", None)):
        selected = [
            r for r in rows
            if r["clip"] == clip_key and r["method"] == method
            and (bitrate is None or abs(float(r["bitrate"]) - bitrate) < 1e-6)
        ]
        if not selected:
            continue
        row = selected[0]
        rate = "~7.7" if method == "griffin_lim" else _num(cast(float, bitrate), 1)
        body.append((
            fill_report.METHOD_LABELS[method],
            rate,
            _num(float(row["si_sdr"])),
            _num(float(row["fwssnr"])),
            _num(float(row["logspec_mse"]), 0),
        ))
    return Table(
        headers=("method", "rate (kbit/s)", "SI-SDR (dB)", "fwSSNR (dB)", "logSpecMSE"),
        rows=tuple(body),
        caption="A single music clip, so the comparison is audible rather than averaged.",
    )


def data_table() -> Table:
    """Our custom data set."""
    return Table(
        headers=("category", "clips", "source"),
        rows=(
            ("speech", "2", "macOS `say` text-to-speech (Samantha), converted to 48 kHz"),
            ("music", "2", "synthesised polyphonic chords + percussion + bass line"),
            ("ambient / sound", "2", "synthesised 1/f noise beds with swells and chirps"),
        ),
        caption="Six deterministic 5 s mono clips at 48 kHz, generated by scripts/make_custom_data.py.",
    )


# --------------------------------------------------------------------------- #
# Asset paths.
# --------------------------------------------------------------------------- #

EQ_TRAIN = "report/assets/formulas/eq_train.png"
EQ_INFER = "report/assets/formulas/eq_infer.png"
EQ_COUPLING = "report/assets/formulas/eq_coupling.png"
DIAGRAM_SVG = "report/assets/diagram/pipeline.svg"
DIAGRAM_PNG = "report/assets/diagram/pipeline.png"
CODE_CKPT = "report/assets/code/cell_ckpt.png"
CODE_PIPELINE = "report/assets/code/cell_pipeline.png"
CODE_METRICS = "report/assets/code/cell_metrics.png"
CODE_NFE = "report/assets/code/cell_nfe.png"
PLOT_BARS = "results/plots/custom_bars.png"
PLOT_SPEC = "results/plots/custom_spectrograms.png"
PLOT_NFE = "results/plots/nfe_curve.png"


# --------------------------------------------------------------------------- #
# The deck itself.
# --------------------------------------------------------------------------- #


@beartype
def build_slides() -> list[Slide]:
    """Assemble the 15-section deck (14 planned sections + one A/B listening slide)."""
    rows = _custom_rows()
    fad = _fad_map()

    flow45 = _mean(rows, "flowdec", 4.5)
    flow75 = _mean(rows, "flowdec", 7.5)
    ndac75 = _mean(rows, "ndac", 7.5)
    opus45 = _mean(rows, "opus", 4.5)
    gl = _mean(rows, "griffin_lim", None)

    slides: list[Slide] = []

    # ---- 1. Title ---------------------------------------------------------- #
    slides.append(Slide(
        sid="title",
        nav="Title",
        eyebrow="AMLT ITMO 2026 · course project",
        title="FlowDec: a flow-based full-band general audio codec",
        subtitle="Paper review and independent testing on our own 48 kHz data · 10 min + 5 min Q&A",
        blocks=(
            Para(
                "**Paper:** *FlowDec: A flow-based full-band general audio codec with high perceptual "
                "quality* — Welker, Le, Chen, Hsu, Gerkmann, Richard, Wu, **ICLR 2025** (Poster)."
            ),
            Para("OpenReview `uxDFlPGRLX` · arXiv 2503.01485 · code: github.com/facebookresearch/FlowDec (CC-BY-NC core)."),
            Picture(DIAGRAM_PNG, "FlowDec in one picture: a non-adversarial codec plus a conditional-flow-matching postfilter.", 1.0),
            Callout(
                "Every number in this deck is produced by our own pipeline from the official checkpoints, "
                "and is reproducible with `uv run python scripts/run_experiments.py`."
            ),
        ),
        notes="Introduce the paper, the task (review + independent testing), and the timing budget.",
    ))

    # ---- 2. Verdict first -------------------------------------------------- #
    slides.append(Slide(
        sid="verdict",
        nav="Verdict",
        eyebrow="00 / Abstract",
        title="Verdict first",
        subtitle="What the paper claims, and what our own measurements say",
        blocks=(
            Bullets((
                "A **48 kHz full-band** neural codec for **speech + music + ambient** at **4.5–7.5 kbit/s**.",
                "**No GAN.** A non-adversarial codec (**NDAC**) plus a **conditional-flow-matching (CFM)** postfilter.",
                f"**6 network evaluations (NFE)** instead of ~60 for score-based ScoreDec — ~10× cheaper sampling.",
                f"On **our own** data at 4.5 kbit/s: SI-SDR **{_num(flow45['si_sdr'])} dB** "
                f"vs Opus **{_num(opus45['si_sdr'])} dB**; FAD×100 **{_fad100(fad, 'flowdec_4.5')}** "
                f"vs **{_fad100(fad, 'opus_4.5')}**.",
                f"Against the codec-only baseline NDAC-75 the picture **reverses**: NDAC wins on SI-SDR "
                f"({_num(ndac75['si_sdr'])} vs {_num(flow75['si_sdr'])} dB) but loses on FAD×100 "
                f"({_fad100(fad, 'ndac_7.5')} vs {_fad100(fad, 'flowdec_7.5')}).",
            )),
            Callout(
                "Positioning (authors): not a replacement for FLAC/Opus for storage — a component for "
                "**very-low-bitrate** coding and for **tokenizing audio** for generative audio-language models."
            ),
        ),
        notes="Answer the whole question up front, then walk through evidence. Leave the SI-SDR/FAD reversal for later.",
    ))

    # ---- 3. Introduction — problem ---------------------------------------- #
    slides.append(Slide(
        sid="problem",
        nav="Problem",
        eyebrow="01 / Introduction",
        title="The problem: natural-sounding audio at very low bitrates",
        subtitle="Why almost every neural codec reaches for a GAN — and why that is a problem",
        blocks=(
            Bullets((
                "An **audio codec** compresses a waveform into a compact, quantized representation and "
                "reconstructs the waveform from it.",
                "Legacy codecs (**Opus, AAC, EVS**) use hand-crafted transform/prediction designs; modern "
                "**end-to-end neural codecs** (SoundStream, EnCodec, DAC) learn the whole pipeline.",
                "To sound natural at very low bitrate, almost all neural codecs use **adversarial (GAN) "
                "training**: a discriminator pushes the decoder towards natural audio — but GAN training is "
                "unstable, needs several hand-tuned discriminators and loss weights, and can mode-collapse.",
                "**ScoreDec** avoided the GAN with a *score-based diffusion* postfilter — but it worked for "
                "**speech only**, only at **24 kbit/s**, and needed ~**60 NFE** (RTF ≈ 1.7) → impractical.",
            )),
            Callout(
                "**FlowDec keeps the postfilter idea, rebuilds it on conditional flow matching, and generalises "
                "it** to full-band speech, music and ambient at 4.5–7.5 kbit/s."
            ),
        ),
        notes="The gap is: no adversarial training, but also no expensive sampling. FlowDec closes both.",
    ))

    # ---- 4. Introduction — novelty (required criterion) -------------------- #
    slides.append(Slide(
        sid="novelty",
        nav="Novelty",
        eyebrow="02 / Introduction",
        title="What is new here",
        subtitle="Four concrete contributions (novelty — required criterion)",
        blocks=(
            Bullets((
                "**New joint CFM coupling for signal enhancement.** The noising distribution is centred on "
                "the codec output: `x0 ~ N(y, Σ_y)` interpolates to `x1 = x*`. The flow is therefore "
                "**contractive** (trajectories actually reach the target), numerically stable, and accurate "
                "at very few steps. **One** hyperparameter (`σ_y`) instead of ScoreDec's four.",
                "**Frequency-dependent `σ_y(f)`.** Audio power drops with frequency, so a single scalar "
                "over-smooths the highs; a per-band `σ_y` estimated from the data (quantile heuristic) "
                "visibly improves low-NFE quality.",
                "**Non-adversarial codec training (NDAC).** DAC with *all* discriminators removed, plus two "
                "losses a non-adversarial codec needs: a **multi-scale constant-Q (CQT) loss** for low "
                "frequencies and an **L1 waveform loss** for phase / SI-SDR. Distribution matching is handed "
                "to the CFM postfilter.",
                "**General full-band audio.** 48 kHz (≤20 kHz), **speech + music + ambient**, at "
                "**4.5–7.5 kbit/s** — versus ScoreDec's speech-only 24 kbit/s.",
            )),
            Callout(
                "**Novelty (required criterion).** The technical core is the *joint* CFM coupling "
                "`x0 ~ N(y, Σ_y)` with `σ1 = 0`, which is what makes a generative postfilter stable at "
                "NFE = 6 where ScoreDec needs ~60."
            ),
        ),
        notes="Spell out the four contributions; the committee explicitly asks for novelty.",
        required="Novelty",
    ))

    # ---- 5. Model overview ------------------------------------------------- #
    slides.append(Slide(
        sid="model-overview",
        nav="Model",
        eyebrow="03 / Model",
        title="Model overview: two-stage postfilter codec",
        subtitle="A non-adversarial codec produces a coarse estimate; a CFM postfilter regenerates the detail",
        blocks=(
            Picture(DIAGRAM_PNG, "The full pipeline: waveform → NDAC encoder → RVQ codes → decoder → y → FlowDec Ω → x̂.", 1.0),
            Bullets((
                "**Underlying codec — NDAC-75.** 75 Hz feature rate, 48 kHz, hop 640, 10 codebooks × 1024 dims. "
                "The bitrate is chosen **at inference** by how many codebooks are used: "
                "`nq = 10 / 8 / 6 / 4 → 7.5 / 6.0 / 4.5 / 3.0 kbit/s`.",
                "The postfilter is a modified **NCSN++ 2-D U-Net** (23.7 M trainable parameters by our own "
                "count; the paper quotes ~26 M): first two depths widened to 256 channels (suppresses "
                "high-frequency artefacts on music), depth reduced to 4, and **all attention removed** so "
                "runtime is linear in duration.",
                "It works in a **compressed complex STFT** domain: 1534-sample Hann window (31.96 ms), "
                "hop 384 (75 % overlap), amplitude compression `α = 0.3`, phase preserved. This invertible "
                "transform φ is applied before the flow and inverted afterwards.",
            )),
            Picture(EQ_COUPLING, "The joint coupling that makes the flow contractive and stable at few steps.", 0.62),
        ),
        notes="Walk the diagram left to right. Emphasise that the bitrate is a runtime choice, not a separate model.",
    ))

    # ---- 6. Model algorithm ------------------------------------------------ #
    slides.append(Slide(
        sid="algorithm",
        nav="Algorithm",
        eyebrow="04 / Model",
        title="The algorithm: joint flow matching + midpoint ODE",
        subtitle="Train to regress a constant velocity; sample by integrating a short ODE",
        blocks=(
            Picture(EQ_TRAIN, "", 0.78),
            Picture(EQ_INFER, "", 0.78),
            Bullets((
                "**Training.** Regress the *constant velocity* that carries a point from its noisy, "
                "codec-centred start to the clean target. Adam, lr 1e-4, 800 k iterations, 2 s snippets, "
                "batch 64, EMA 0.999.",
                "**Inference.** Start from `X = φ(y) + σ_y(f)·ε`, integrate `dX/dt = v_θ(X, t, Y)` with the "
                "**midpoint** solver for `N` steps (**NFE = 2N**; default `N = 3` → **NFE = 6**), then map "
                "back with φ⁻¹ and clip.",
            )),
            Callout(
                "**Pros.** No GAN (stable, interpretable, no mode collapse) · 6 NFE vs ~60 for ScoreDec "
                "(RTF ≈ 0.23 vs 1.71 on an A100) · full-band general audio at 4.5–7.5 kbit/s · one "
                "hyperparameter · best FAD among compared codecs."
            ),
            Callout(
                "**Cons.** Worse **intrusive** metrics than adversarial DAC (the rate–distortion–perception "
                "trade-off) · still needs a strong codec *and* a generative DNN · **non-causal → not "
                "streaming-capable** · licence **CC-BY-NC** and a heavy dependency stack."
            ),
        ),
        notes="Keep this slide short: two equations, one training sentence, one inference sentence, pros/cons.",
    ))

    # ---- 7. Experiments — setup -------------------------------------------- #
    slides.append(Slide(
        sid="setup",
        nav="Setup",
        eyebrow="05 / Experiments",
        title="Experimental setup on our own data",
        subtitle="Official checkpoints, our own 48 kHz clips, and classical baselines",
        blocks=(
            Bullets((
                "We deliberately did **not** reuse the paper's test set: a small, deterministic, 48 kHz mono "
                "set of 5 s clips (`scripts/make_custom_data.py`, fixed seed 1234).",
                "Everything runs through the **official checkpoints** (`flowdec_75m` + `ndac-75`) and the "
                "official inference API — `dac.encode(...) → model.enhance(..., N, solver)` — with EMA weights "
                "and the default midpoint solver.",
                "Metrics are identical for every method, as in the paper: **SI-SDR**, **fwSSNR**, "
                "**logSpecMSE**, and **FAD×100** with CLAP-laion-audio embeddings (perception).",
                "**Classical baselines (required):** **Opus** through `ffmpeg` (CBR, 60 ms frames) at "
                "3.0 / 4.5 / 7.5 kbit/s, and **Griffin-Lim** from a rate-matched (~7.7 kbit/s) mel magnitude "
                "spectrogram — the textbook phase-recovery approach that discards phase.",
                "Hardware: Apple **M4 Pro, 24 GB**. All reported numbers are **CPU** (the faithful path); "
                "MPS is discussed later as a fast-but-lossy preview.",
            )),
            data_table(),
        ),
        notes="Stress that this is our own data and our own baseline implementations, not the paper's tables.",
    ))

    # ---- 8. Reproduction on the authors' audio (required) ------------------ #
    slides.append(Slide(
        sid="reproduction",
        nav="Reproduction",
        eyebrow="06 / Experiments",
        title="Reproduction on the authors' own audio",
        subtitle="Sanity-check our pipeline before trusting our own numbers (reproduction — required criterion)",
        blocks=(
            Para(
                "We downloaded the authors' published demo audio (clean reference plus their official FlowDec "
                "and DAC-75 outputs) and ran **our** pipeline on their clean references."
            ),
            reproduction_table(),
            Bullets((
                "Our FlowDec output matches the authors' own within a fraction of a dB in SI-SDR / fwSSNR — "
                "**faithful**, as expected, since the postfilter is stochastic and exact sample equality is "
                "not expected. Their **DAC-75** output keeps the better SI-SDR, which is the trade-off we "
                "return to in the next two slides.",
            )),
            Picture(CODE_CKPT, "The official inference API in practice: load the checkpoint pair, then run the pipeline.", 0.98),
            Callout(
                "**Reproduction (required criterion):** our pipeline reproduces the authors' published outputs "
                "from the official checkpoints, on their own audio."
            ),
        ),
        notes="This slide is the credibility anchor: everything after it is our own measurement, not a citation.",
        required="Reproduction",
    ))

    # ---- 9. Results on our own data ---------------------------------------- #
    slides.append(Slide(
        sid="results",
        nav="Results",
        eyebrow="07 / Experiments",
        title="Results on our own 48 kHz data",
        subtitle="FlowDec across bitrates, versus the codec-only baseline and the classical codecs",
        blocks=(
            Picture(PLOT_BARS, "Mean metrics over our six clips (results/plots/custom_bars.png).", 0.94),
            custom_table(),
            Para(
                "The whole inference path is two calls on the official API — `dac.encode(...)` and "
                "`model.enhance(..., N=3, solver=\"midpoint\")`; see the code-cell screenshots in "
                "`notebooks/flowdec_demo.ipynb`."
            ),
        ),
        notes="Walk the table row by row: FlowDec beats Opus/GL everywhere; NDAC-only wins on SI-SDR but loses on FAD.",
    ))

    # ---- 10. vs classical baselines (required) ----------------------------- #
    slides.append(Slide(
        sid="baselines",
        nav="Baselines",
        eyebrow="08 / Experiments",
        title="FlowDec vs classical baselines",
        subtitle="Opus and Griffin-Lim at a matched low rate (comparison — required criterion)",
        blocks=(
            Picture(PLOT_SPEC, "Spectrograms of one music clip: clean, FlowDec, Opus, Griffin-Lim.", 0.98),
            one_clip_table(),
            Bullets((
                "At 4.5–7.5 kbit/s Opus is severely degraded, and unusable at 3 kbit/s "
                f"({_num(_mean(rows, 'opus', 3.0)['si_sdr'])} dB SI-SDR).",
                f"**Griffin-Lim** is dominated because it discards phase entirely "
                f"({_num(gl['si_sdr'])} dB SI-SDR, FAD×100 {_fad100(fad, 'griffin_lim')}).",
                "FlowDec is ahead on **every** metric, and its **FAD** advantage is large — i.e. it *sounds* "
                "much closer to the reference than either classical baseline.",
                "In the spectrograms FlowDec reconstructs high-frequency structure that Opus smooths away, "
                "while Griffin-Lim produces a buzzy, phase-incoherent texture.",
            )),
            Callout(
                "**Comparison with classical baselines (required criterion):** rate-matched Opus and "
                "Griffin-Lim lose to FlowDec on perception *and* on signal metrics."
            ),
            Para(
                "Raw Python behind this table (`notebooks/flowdec_demo.ipynb`, re-run against the official "
                "checkpoints): `scripts/build_slide_assets.py` screenshots the executed cells into "
                "`report/assets/code/`."
            ),
        ),
        notes="Make the audible point here or on the next slide; the numbers alone are not the argument.",
        required="Comparison with classical baselines",
    ))

    # ---- 11. A/B listening ------------------------------------------------- #
    slides.append(Slide(
        sid="listen",
        nav="Listen",
        eyebrow="09 / Experiments",
        title="Listen: FlowDec vs Opus at 4.5 kbit/s",
        subtitle="Same clip, same bitrate — play them in this order and compare by ear",
        blocks=(
            Para(
                "Same clip, same bitrate. For each category we play **FlowDec 4.5 kbit/s**, then "
                "**Opus 4.5 kbit/s**; the matching clean reference is the first clip of the deck if you want "
                "the original."
            ),
            Audio(
                ("music_flowdec45", "music_opus45"),
                "Music — FlowDec, then Opus, both at 4.5 kbit/s",
            ),
            Audio(
                ("ambient_flowdec45", "ambient_opus45"),
                "Ambient — FlowDec, then Opus, both at 4.5 kbit/s",
            ),
            Callout(
                "Opus sounds dull and smeared; FlowDec keeps harmonics and the high-frequency **\"air\"** — "
                "which is exactly what the FAD numbers on the previous slide measure. The same pair at "
                "7.5 kbit/s is in the backup section."
            ),
        ),
        notes="Play at most one pair live if time is short; the rest is backup material in the HTML deck.",
    ))

    # ---- 12. NFE ablation -------------------------------------------------- #
    slides.append(Slide(
        sid="nfe",
        nav="NFE",
        eyebrow="10 / Experiments",
        title="NFE ablation: the efficiency claim",
        subtitle="How many ODE steps does the postfilter really need?",
        blocks=(
            Picture(PLOT_NFE, "Quality vs the number of function evaluations (results/plots/nfe_curve.png).", 0.8),
            nfe_table(),
            Bullets((
                "Quality climbs steeply from **NFE = 2 to NFE = 6** and then plateaus → **NFE = 6** is a good "
                "operating point, which is the paper's central efficiency claim.",
                "Contrast: **ScoreDec collapses at NFE = 6** — the paper reports FAD×100 = 145 and "
                "SI-SDR ≈ −27 dB at that budget.",
            )),
            Audio(("music_flowdec75",), "FlowDec at 7.5 kbit/s with NFE = 6 — the recommended operating point."),
            Picture(CODE_NFE, "The NFE sweep used for the table above.", 0.86),
        ),
        notes="The claim is not only quality but cost: 6 NFE instead of ~60 is the whole practical argument.",
    ))

    # ---- 13. Device caveat (our own finding) ------------------------------- #
    slides.append(Slide(
        sid="device",
        nav="Device caveat",
        eyebrow="11 / Experiments",
        title="Our own finding: a device caveat",
        subtitle="A result that is not in the paper — reported because it affects reproducibility",
        blocks=(
            Bullets((
                "On this Mac (**M4 Pro, 24 GB**), **CPU** inference is faithful but slow: RTF ≈ 15.",
                "Apple-Silicon **MPS** is ~6× faster but **degrades quality**: logSpecMSE grows by roughly an "
                "order of magnitude and SI-SDR drops by a few dB.",
                "The **NDAC codec itself is numerically identical** across devices, so the gap comes from the "
                "**ODE / postfilter path** (bandwidth-limited `float32` matrix products).",
                "Therefore every number in this deck is a **CPU** number, and MPS is documented only as a fast "
                "preview (`scripts/diagnose_mps.py`).",
            )),
            Callout(
                "This is **our own contribution**: a silent numerical-fidelity trap on Apple Silicon that is "
                "easy to fall into when reproducing the paper."
            ),
        ),
        notes="Lead with 'this is not in the paper' — it is the clearest original result we have.",
    ))

    # ---- 14. Conclusion ---------------------------------------------------- #
    slides.append(Slide(
        sid="conclusion",
        nav="Conclusion",
        eyebrow="12 / Conclusion",
        title="Conclusion",
        subtitle="What we verified, what surprised us, and what limits adoption",
        blocks=(
            Bullets((
                "A **non-adversarial** codec plus a **conditional-flow-matching postfilter** matches GAN-based "
                "codecs perceptually and beats them in **FAD**, while sampling in only **6 NFE**.",
                "We **reproduced** the authors' outputs on their own audio; on **our own** data FlowDec clearly "
                "beats rate-matched **Opus** and **Griffin-Lim** on perception and on signal metrics, and stays "
                "robust from NFE = 4.",
                f"**Key insight — SI-SDR and FAD disagree.** We reproduce the paper's exact reversal against the "
                f"codec-only baseline: NDAC-75 wins on SI-SDR ({_num(ndac75['si_sdr'])} vs "
                f"{_num(flow75['si_sdr'])} dB) but loses on FAD×100 "
                f"({_fad100(fad, 'ndac_7.5')} vs {_fad100(fad, 'flowdec_7.5')}). A plausible waveform is not "
                "the same thing as a close waveform.",
                "**Limitations.** Not streaming-capable (non-causal DNNs) · **CC-BY-NC** licence and heavy "
                "dependencies · intrusive metrics can look worse than adversarial DAC · on Apple Silicon the "
                "fast MPS path is not numerically faithful.",
                "**Where it fits.** Not a replacement for FLAC/Opus for lossless or storage use, but a strong "
                "component for very-low-bitrate general-audio coding and for tokenizing audio for "
                "audio-language models.",
            )),
            Callout(
                "Practical summary: FlowDec makes a **generative** postfilter cheap enough (6 NFE) to be a "
                "real engineering option for very-low-bitrate, full-band, general audio."
            ),
        ),
        notes="Mirror the verdict slide: claim → evidence → caveat. Then stop and invite questions.",
    ))

    # ---- 15. Backup — Q&A prep --------------------------------------------- #
    slides.append(Slide(
        sid="backup",
        nav="Backup Q&A",
        eyebrow="13 / Backup",
        title="Backup — Q&A preparation",
        subtitle="The questions we expect, and the short answers",
        blocks=(
            Bullets((
                "**\"Why a codec if Opus exists?\"** → FlowDec's decoder *generatively reconstructs* lost "
                "detail, and it produces **discrete tokens**, which is exactly what audio-LLM / TTS systems "
                "need; a lower bitrate means a shorter token sequence and cheaper language-model generation.",
                "**\"Novelty compared with earlier neural codecs?\"** → the first competitive "
                "*non-adversarial* codec, and a CFM coupling that stays stable at **6 NFE** where ScoreDec "
                "needs ~60.",
                "**\"FAD is better but SI-SDR is worse than DAC — is that not a contradiction?\"** → no: it is "
                "the perception–distortion trade-off. Plausible ≠ close. The stochastic postfilter "
                "deliberately trades waveform fidelity for perceptual plausibility.",
                "**\"What blocks real adoption?\"** → the CC-BY-NC licence, the heavy dependency stack (audiotools, "
                "hydra, DAC), the non-causal (non-streaming) architecture, and inference cost versus a "
                "classical codec.",
                "**\"Why does MPS hurt quality?\"** → bandwidth-limited `float32` operations in the solver; "
                "NDAC is numerically identical on both devices, so the gap is entirely in the postfilter path.",
                "**\"How did you build the data?\"** → a deterministic, seeded generator "
                "(`scripts/make_custom_data.py`) so the results are reproducible, not cherry-picked.",
            )),
            per_category_table(),
            Audio(
                ("speech_flowdec45", "speech_opus45", "music_flowdec75"),
                "Speech at 4.5 kbit/s (FlowDec vs Opus), then music at 7.5 kbit/s",
            ),
            Callout(
                "Full details, including the reproducibility appendix and the licence/attribution notes, are in "
                "`report/report.md`."
            ),
        ),
        notes="Backup only. Keep it at the end of the deck and jump here only if the question warrants it.",
    ))

    return slides
