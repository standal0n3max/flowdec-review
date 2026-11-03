# Copyright (c) 2026 AMLT course project (FlowDec review).
# Fill the report/slides tables from the results JSON written by run_experiments.py.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from beartype import beartype

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

METHOD_LABELS = {
    "flowdec": "FlowDec",
    "ndac": "NDAC (codec only)",
    "opus": "Opus",
    "griffin_lim": "Griffin-Lim",
}

FAD_KEY_LABELS = {
    "flowdec_7.5": ("FlowDec", 7.5),
    "flowdec_6.0": ("FlowDec", 6.0),
    "flowdec_4.5": ("FlowDec", 4.5),
    "flowdec_3.0": ("FlowDec", 3.0),
    "ndac_7.5": ("NDAC (codec only)", 7.5),
    "opus_7.5": ("Opus", 7.5),
    "opus_4.5": ("Opus", 4.5),
    "opus_3.0": ("Opus", 3.0),
    "griffin_lim": ("Griffin-Lim", 7.7),
}


@beartype
def load(name: str) -> Any:
    path = RESULTS / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@beartype
def agg(rows: list[dict[str, Any]], method: str, bitrate: float | None = None) -> dict[str, float]:
    sel = [r for r in rows if r["method"] == method and (bitrate is None or abs(r["bitrate"] - bitrate) < 1e-6)]
    if not sel:
        return {}
    return {
        "si_sdr": float(np.mean([r["si_sdr"] for r in sel])),
        "fwssnr": float(np.mean([r["fwssnr"] for r in sel])),
        "logspec_mse": float(np.mean([r["logspec_mse"] for r in sel])),
        "n": len(sel),
    }


def fmt(v: float | np.floating[Any] | None, digits: int = 2) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def main() -> None:
    custom = load("custom_metrics.json")
    fad = load("fad_custom.json") or {}
    nfe = load("nfe_ablation.json")
    repro = load("reproduction_authors.json")
    if custom is None:
        raise SystemExit("custom_metrics.json not found; run the pipeline first")

    # ---------- Custom table (mean over all clips) ----------
    order = [
        ("flowdec", 7.5), ("flowdec", 4.5), ("flowdec", 3.0),
        ("ndac", 7.5),
        ("opus", 7.5), ("opus", 4.5), ("opus", 3.0),
        ("griffin_lim", None),
    ]
    lines = ["| method | rate (kbit/s) | SI-SDR (dB) | fwSSNR (dB) | logSpecMSE | FAD×100 |",
             "|---|---:|---:|---:|---:|---:|"]
    for method, br in order:
        a = agg(custom, method, br)
        if not a:
            continue
        rate = "~7.7" if method == "griffin_lim" else fmt(br, 1)
        fad_key = f"{method}_{br}" if br is not None else "griffin_lim"
        fad_val = fad.get(fad_key)
        fad_txt = fmt(fad_val * 100, 1) if isinstance(fad_val, (int, float)) else "—"
        lines.append(
            f"| {METHOD_LABELS[method]} | {rate} | {fmt(a['si_sdr'])} | {fmt(a['fwssnr'])} | "
            f"{fmt(a['logspec_mse'], 0)} | {fad_txt} |"
        )
    custom_table = "\n".join(lines)

    # ---------- Per-category table (FlowDec 7.5/4.5 vs Opus 7.5/4.5 vs GL) ----------
    cat_lines = ["| category | method (rate) | SI-SDR (dB) | fwSSNR (dB) | logSpecMSE |",
                 "|---|---|---:|---:|---:|"]
    for category in ("speech", "music", "ambient"):
        rows_cat = [r for r in custom if r.get("category") == category]
        for method, br in (("flowdec", 7.5), ("flowdec", 4.5), ("opus", 4.5), ("griffin_lim", None)):
            a = agg(rows_cat, method, br)
            if not a:
                continue
            rate = "~7.7" if method == "griffin_lim" else fmt(br, 1)
            cat_lines.append(
                f"| {category} | {METHOD_LABELS[method]} ({rate}) | {fmt(a['si_sdr'])} | "
                f"{fmt(a['fwssnr'])} | {fmt(a['logspec_mse'], 0)} |"
            )
    per_category_table = "\n".join(cat_lines)

    # ---------- Reproduction table ----------
    repro_lines = ["| clip | method | SI-SDR (dB) | fwSSNR (dB) | logSpecMSE |",
                   "|---|---|---:|---:|---:|"]
    if repro:
        for clip in sorted({r["clip"] for r in repro}):
            for method, label in (("authors_flowdec", "authors FlowDec"), ("ours_flowdec", "ours FlowDec"),
                                  ("authors_dac75", "authors DAC-75")):
                row = next((r for r in repro if r["clip"] == clip and r["method"] == method), None)
                if row:
                    repro_lines.append(
                        f"| {clip} | {label} | {fmt(row['si_sdr'])} | {fmt(row['fwssnr'])} | "
                        f"{fmt(row['logspec_mse'], 0)} |"
                    )
    repro_table = "\n".join(repro_lines)

    # ---------- NFE table ----------
    nfe_lines = ["| NFE | SI-SDR (dB) | fwSSNR (dB) | logSpecMSE |", "|---:|---:|---:|---:|"]
    if nfe:
        for nfe_val in sorted({r["nfe"] for r in nfe}):
            sel = [r for r in nfe if r["nfe"] == nfe_val]
            nfe_lines.append(
                f"| {nfe_val} | {fmt(np.mean([r['si_sdr'] for r in sel]))} | "
                f"{fmt(np.mean([r['fwssnr'] for r in sel]))} | {fmt(np.mean([r['logspec_mse'] for r in sel]), 0)} |"
            )
    nfe_table = "\n".join(nfe_lines)

    # ---------- One-clip (music_0) table for the slide ----------
    clip_key = "music_music_0"
    clip_lines = ["| method | rate (kbit/s) | SI-SDR (dB) | fwSSNR (dB) | logSpecMSE |",
                  "|---|---:|---:|---:|---:|"]
    for method, br in (("flowdec", 7.5), ("flowdec", 4.5), ("opus", 7.5), ("opus", 4.5), ("griffin_lim", None)):
        sel = [r for r in custom if r["clip"] == clip_key and r["method"] == method
               and (br is None or abs(r["bitrate"] - br) < 1e-6)]
        if not sel:
            continue
        rate = "~7.7" if method == "griffin_lim" else fmt(br, 1)
        clip_lines.append(
            f"| {METHOD_LABELS[method]} | {rate} | {fmt(sel[0]['si_sdr'])} | {fmt(sel[0]['fwssnr'])} | "
            f"{fmt(sel[0]['logspec_mse'], 0)} |"
        )
    clip_table = "\n".join(clip_lines)

    # ---------- Apply to files ----------
    replacements = {
        "REPRO_TABLE_PLACEHOLDER": repro_table,
        "CUSTOM_TABLE_PLACEHOLDER": custom_table,
        "PER_CATEGORY_PLACEHOLDER": per_category_table,
        "NFE_TABLE_PLACEHOLDER": nfe_table,
        "FILL_FROM_RESULTS": clip_table,
    }
    for rel in ("report/report.md", "report/slides.md"):
        path = ROOT / rel
        text = path.read_text(encoding="utf-8")
        for key, value in replacements.items():
            text = text.replace(key, value)
        path.write_text(text, encoding="utf-8")
        print(f"filled {rel}")

    print("\n=== custom table ===\n" + custom_table)
    print("\n=== nfe table ===\n" + nfe_table)
    print("\n=== FAD ===\n" + json.dumps({k: round(v * 100, 1) for k, v in fad.items()}, indent=2))


if __name__ == "__main__":
    main()