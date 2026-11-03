# Copyright (c) 2026 AMLT course project (FlowDec review).
# Main experiment driver.
#
# Stages:
#   reproduce : run our pipeline on the authors' published demo audio and compare
#               with their official FlowDec / DAC-75 outputs (sanity/reproduction).
#   custom    : run FlowDec (multi-bitrate) + Opus + Griffin-Lim on our own 48 kHz
#               data, compute identical metrics, save demo assets, plots, JSON.
#   nfe       : FlowDec quality vs the number of solver steps (NFE).
#
# Everything is deterministic (fixed seed) and writes into `results/`.
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Any, cast

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from baselines import (  # noqa: E402
    griffin_lim_mel_kbps,
    griffin_lim_roundtrip,
    opus_actual_kbps,
    opus_roundtrip,
)
from flowdec_lib import (  # noqa: E402
    NQ_TO_KBPS_75M,
    count_parameters,
    full_flowdec,
    load_flowdec,
    load_wav48k,
    save_wav48k,
    to_numpy_mono,
)
from metrics import FAD, mel_spectrogram, metrics_for_pair, write_json  # noqa: E402

RESULTS = ROOT / "results"
DEMO_DIR = RESULTS / "demo"
CACHE_DIR = RESULTS / "cache"
CUSTOM = ROOT / "data" / "custom"
AUTHORS = ROOT / "data" / "authors"

SEED = 1234
SR = 48000
CLIP_SECONDS = 5.0
NQ_LIST = [10, 6, 4]  # 7.5, 4.5, 3.0 kbit/s for flowdec_75m
OPUS_KBPS = [3.0, 4.5, 7.5]
SOLVER = "midpoint"
N_STEPS = 3  # NFE = 6


def flowdec_cached(bundle: Any, src: torch.Tensor, clip_key: str, nq: int, n_steps: int, seed: int) -> dict[str, torch.Tensor]:
    """`full_flowdec` with an on-disk cache (FlowDec inference is the slow part)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = CACHE_DIR / f"{bundle.variant}_{clip_key}_{src.shape[-1]}_nq{nq}_N{n_steps}_s{seed}.pt"
    if key.is_file():
        return cast(dict[str, torch.Tensor], torch.load(key, weights_only=False))
    out = full_flowdec(bundle, src, nq=nq, n_steps=n_steps, solver=SOLVER, seed=seed)
    torch.save(out, key)
    return out


def clip_seconds(wav: torch.Tensor, seconds: float) -> torch.Tensor:
    n = int(seconds * SR)
    return wav[:, :n]


def collect_custom() -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for category in ("speech", "music", "ambient"):
        for path in sorted((CUSTOM / category).glob("*.wav")):
            items.append((category, path))
    return items


def plot_metric_bars(rows: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted({r["method"] for r in rows})
    bitrates = sorted({r["bitrate"] for r in rows if r["method"] == "flowdec"})
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, metric, title in zip(axes, ("si_sdr", "fwssnr", "logspec_mse"), ("SI-SDR (dB)", "fwSSNR (dB)", "logSpecMSE (↓)")):
        x = np.arange(len(bitrates))
        width = 0.8 / len(methods)
        for mi, method in enumerate(methods):
            vals, label = [], method
            if method == "flowdec":
                for br in bitrates:
                    sel = [r[metric] for r in rows if r["method"] == method and r["bitrate"] == br]
                    vals.append(np.mean(sel) if sel else np.nan)
                label = "FlowDec (NFE=6)"
            else:
                br = bitrates[0] if bitrates else 0.0
                sel = [r[metric] for r in rows if r["method"] == method]
                flat = np.mean(sel) if sel else np.nan
                vals = [flat for _ in bitrates]
                label = f"{method} ({flat:.4g})" if method != "flowdec" else method
            ax.bar(x + mi * width, vals, width, label=label, color=colors[mi % len(colors)])
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels([f"{b}" for b in bitrates])
        ax.set_xlabel("bitrate (kbit/s)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("FlowDec vs classical baselines on our custom data (mean over clips)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_nfe_curve(rows: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title in zip(axes, ("si_sdr", "fwssnr", "logspec_mse"), ("SI-SDR (dB)", "fwSSNR (dB)", "logSpecMSE (↓)")):
        for clip in sorted({r["clip"] for r in rows}):
            sel = sorted([r for r in rows if r["clip"] == clip], key=lambda r: r["nfe"])
            ax.plot([r["nfe"] for r in sel], [r[metric] for r in sel], marker="o", label=clip)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({r["nfe"] for r in rows}))
        ax.get_xaxis().set_major_formatter(lambda v, _: f"{int(v)}")
        ax.set_xlabel("NFE (midpoint solver)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("FlowDec quality vs number of ODE steps on our own data (7.5 kbit/s)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_spectrograms(items: list[tuple[str, str, NDArray[np.floating[Any]]]], out_path: Path) -> None:
    """items: list of (row_label, col_label, waveform)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted({r for r, _, _ in items})
    cols = sorted({c for _, c, _ in items})
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.2 * len(cols), 2.4 * len(rows)), squeeze=False)
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            wav = next(w for r, c, w in items if r == row and c == col)
            mel = mel_spectrogram(wav, sr=SR)
            ax = axes[i][j]
            ax.imshow(mel, origin="lower", aspect="auto", cmap="magma",
                      extent=[0, wav.shape[0] / SR, 0, SR / 2000])
            ax.set_ylabel("kHz" if j == 0 else "")
            ax.set_xlabel("s")
            if i == 0:
                ax.set_title(col, fontsize=9)
            if j == 0:
                ax.annotate(row, xy=(-0.35, 0.5), xycoords="axes fraction", rotation=90,
                            va="center", ha="center", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def stage_reproduce(bundle: Any, fad: FAD | None) -> None:
    """Run our pipeline on the authors' demo audio and compare with their outputs."""
    rows: list[dict[str, Any]] = []
    for stem in ("Music_Licensed_1", "Speech_EARS_1", "Sound_AudioSet_1"):
        clean = load_wav48k(AUTHORS / f"{stem}_Clean.wav")
        n = min(int(CLIP_SECONDS * SR), clean.shape[-1])
        src = clean[:, :n]
        ref = to_numpy_mono(src)
        for tag, suffix in (("authors_flowdec", "FlowDec_75m_NFE_6"), ("authors_dac75", "DAC_75_48_kHz")):
            est = to_numpy_mono(load_wav48k(AUTHORS / f"{stem}_{suffix}.wav"))
            m = min(len(ref), len(est))
            row = {"clip": stem, "method": tag, "bitrate": 7.5, **metrics_for_pair(ref[:m], est[:m])}
            rows.append(row)
        out = flowdec_cached(bundle, src, stem, NQ_LIST[0], N_STEPS, SEED)
        est = to_numpy_mono(out["flowdec"])
        m = min(len(ref), len(est))
        rows.append({"clip": stem, "method": "ours_flowdec", "bitrate": 7.5, **metrics_for_pair(ref[:m], est[:m])})
        est_n = to_numpy_mono(out["ndac"])
        m2 = min(len(ref), len(est_n))
        rows.append({"clip": stem, "method": "ours_ndac", "bitrate": 7.5, **metrics_for_pair(ref[:m2], est_n[:m2])})
        save_wav48k(DEMO_DIR / f"authors_{stem}_ours.wav", out["flowdec"])
    write_json(RESULTS / "reproduction_authors.json", rows)
    print("[reproduce] wrote results/reproduction_authors.json")


def stage_custom(bundle: Any, fad: FAD | None) -> None:
    items = collect_custom()
    rows: list[dict[str, Any]] = []
    spectrogram_items: list[tuple[str, str, NDArray[np.floating[Any]]]] = []
    # Cache every method's output per clip so metrics and FAD reuse them.
    outputs: dict[str, list[NDArray[np.floating[Any]]]] = {m: [] for m in
                                             (["flowdec_7.5", "flowdec_4.5", "flowdec_3.0",
                                               "ndac_7.5", "opus_7.5", "opus_4.5", "opus_3.0", "griffin_lim"])}
    ref_wavs: list[NDArray[np.floating[Any]]] = []

    for category, path in items:
        clean = load_wav48k(path)
        n = min(int(CLIP_SECONDS * SR), clean.shape[-1])
        src = clean[:, :n]
        ref = to_numpy_mono(src)
        stem = f"{category}_{path.stem}"
        ref_wavs.append(ref)

        # FlowDec at each bitrate; the top bitrate also gives us the NDAC-only output.
        flow_by_nq: dict[int, NDArray[np.floating[Any]]] = {}
        ndac = None
        for nq in NQ_LIST:
            out = flowdec_cached(bundle, src, stem, nq, N_STEPS, SEED)
            est = to_numpy_mono(out["flowdec"])
            flow_by_nq[nq] = est
            if nq == NQ_LIST[0]:
                ndac = to_numpy_mono(out["ndac"])
            m = min(len(ref), len(est))
            kbps = NQ_TO_KBPS_75M[nq]
            rows.append({"clip": stem, "category": category, "method": "flowdec",
                         "bitrate": kbps, "nq": nq, "nfe": N_STEPS * 2,
                         **metrics_for_pair(ref[:m], est[:m])})
            save_wav48k(DEMO_DIR / f"{stem}_flowdec_{kbps}kbps.wav", out["flowdec"])
            outputs[f"flowdec_{kbps}"].append(est)

        assert ndac is not None
        m = min(len(ref), len(ndac))
        rows.append({"clip": stem, "category": category, "method": "ndac",
                     "bitrate": NQ_TO_KBPS_75M[NQ_LIST[0]], "nq": NQ_LIST[0], "nfe": 0,
                     **metrics_for_pair(ref[:m], ndac[:m])})
        outputs["ndac_7.5"].append(ndac)

        save_wav48k(DEMO_DIR / f"{stem}_input.wav", src)

        for kbps in OPUS_KBPS:
            est = opus_roundtrip(src, kbps)
            m = min(len(ref), len(est))
            rows.append({"clip": stem, "category": category, "method": "opus", "bitrate": kbps,
                         **metrics_for_pair(ref[:m], est[:m])})
            outputs[f"opus_{kbps}"].append(est)
            if kbps == 4.5:
                save_wav48k(DEMO_DIR / f"{stem}_opus_{kbps}kbps.wav", torch.from_numpy(est).unsqueeze(0))

        gl = griffin_lim_roundtrip(src)
        m = min(len(ref), len(gl))
        rows.append({"clip": stem, "category": category, "method": "griffin_lim",
                     "bitrate": griffin_lim_mel_kbps(), **metrics_for_pair(ref[:m], gl[:m])})
        outputs["griffin_lim"].append(gl)

        if category == "music" and path.stem == "music_0":
            spectrogram_items += [
                ("music", "clean", ref),
                ("music", "NDAC 7.5k", ndac),
                ("music", f"FlowDec {NQ_TO_KBPS_75M[NQ_LIST[0]]}k", flow_by_nq[NQ_LIST[0]]),
                ("music", f"FlowDec {NQ_TO_KBPS_75M[NQ_LIST[1]]}k", flow_by_nq[NQ_LIST[1]]),
                ("music", "Opus 4.5k", outputs["opus_4.5"][-1]),
                ("music", "Griffin-Lim", gl),
            ]

    print("[custom] Opus actual bitrates:",
          {b: round(opus_actual_kbps(clip_seconds(load_wav48k(items[0][1]), CLIP_SECONDS), b), 2) for b in OPUS_KBPS})

    # Persist metrics immediately so a later (slower) stage cannot lose the run.
    write_json(RESULTS / "custom_metrics.json", rows)

    if fad is not None:
        ref_emb = fad.embed(ref_wavs)
        fad_scores: dict[str, float] = {}
        for method, wavs in outputs.items():
            if not wavs:
                continue
            est_emb = fad.embed(wavs)
            fad_scores[method] = fad.score_from_embeddings(ref_emb, est_emb)
        write_json(RESULTS / "fad_custom.json", fad_scores)
        print("[custom] FAD:", {k: round(v, 3) for k, v in fad_scores.items()})

    plot_metric_bars(rows, RESULTS / "plots" / "custom_bars.png")
    if spectrogram_items:
        plot_spectrograms(spectrogram_items, RESULTS / "plots" / "custom_spectrograms.png")
    print("[custom] wrote results/custom_metrics.json and plots")


def stage_nfe(bundle: Any) -> None:
    rows: list[dict[str, Any]] = []
    picks = [p for p in collect_custom() if p[1].stem in ("music_0", "speech_0")]
    for category, path in picks:
        clean = load_wav48k(path)
        n = min(int(4.0 * SR), clean.shape[-1])
        src = clean[:, :n]
        ref = to_numpy_mono(src)
        stem = f"{category}_{path.stem}"
        for n_steps, nfe in ((1, 2), (2, 4), (3, 6), (25, 50)):
            out = flowdec_cached(bundle, src, stem, NQ_LIST[0], n_steps, SEED)
            est = to_numpy_mono(out["flowdec"])
            m = min(len(ref), len(est))
            rows.append({"clip": stem, "nfe": nfe, "bitrate": 7.5, **metrics_for_pair(ref[:m], est[:m])})
            if nfe == 6:
                save_wav48k(DEMO_DIR / f"{stem}_flowdec_75kbps_nfe6.wav", out["flowdec"])
    write_json(RESULTS / "nfe_ablation.json", rows)
    plot_nfe_curve(rows, RESULTS / "plots" / "nfe_curve.png")
    print("[nfe] wrote results/nfe_ablation.json and plots/nfe_curve.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["reproduce", "custom", "nfe", "all"], default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--variant", default="flowdec_75m")
    parser.add_argument("--no-fad", action="store_true", help="skip CLAP-based FAD (slow download)")
    args = parser.parse_args()

    t0 = time.time()
    bundle = load_flowdec(args.variant, torch.device(args.device))
    print(f"[setup] loaded {args.variant} on {args.device}; "
          f"FlowDec params={count_parameters(bundle.model)}")

    fad = None if args.no_fad else FAD(sample_rate=SR)

    if args.stage in ("reproduce", "all"):
        stage_reproduce(bundle, fad)
    if args.stage in ("custom", "all"):
        stage_custom(bundle, fad)
    if args.stage in ("nfe", "all"):
        stage_nfe(bundle)

    print(f"[done] total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()