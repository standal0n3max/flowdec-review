# Copyright (c) 2026 AMLT course project (FlowDec review).
# Diagnostic: check that our inference reproduces the authors' demo outputs and
# understand the CPU vs MPS discrepancy of the stochastic postfilter.
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from flowdec_lib import (  # noqa: E402
    full_flowdec,
    load_flowdec,
    load_wav48k,
    to_numpy_mono,
)
from metrics import fw_ssnr, log_spec_mse, si_sdr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AUTHORS = ROOT / "data" / "authors"
SECONDS = 6.0
SR = 48000


def clip(seconds: float, sr: int = SR) -> int:
    return int(seconds * sr)


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "flowdec_75m"
    nq = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    for stem in ("Music_Licensed_1", "Speech_EARS_1", "Sound_AudioSet_1"):
        clean = to_numpy_mono(load_wav48k(AUTHORS / f"{stem}_Clean.wav"))
        auth_flow = to_numpy_mono(load_wav48k(AUTHORS / f"{stem}_FlowDec_75m_NFE_6.wav"))
        auth_dac = to_numpy_mono(load_wav48k(AUTHORS / f"{stem}_DAC_75_48_kHz.wav"))
        n = min(clip(SECONDS), clean.shape[0], auth_flow.shape[0])
        ref = clean[:n]

        print(f"\n=== {stem} (first {SECONDS:.0f}s) ===")
        print(
            f"  authors  FlowDec: SI-SDR={si_sdr(ref, auth_flow[:n]):6.2f} "
            f"fwSSNR={fw_ssnr(ref, auth_flow[:n]):6.2f} logSpec={log_spec_mse(ref, auth_flow[:n]):7.1f}"
        )
        print(
            f"  authors  DAC-75 : SI-SDR={si_sdr(ref, auth_dac[:n]):6.2f} "
            f"fwSSNR={fw_ssnr(ref, auth_dac[:n]):6.2f} logSpec={log_spec_mse(ref, auth_dac[:n]):7.1f}"
        )

        src = load_wav48k(AUTHORS / f"{stem}_Clean.wav")[:, :n]
        for dev in ("cpu", "mps"):
            bundle = load_flowdec(model, torch.device(dev))
            out = full_flowdec(bundle, src, nq=nq, n_steps=3, seed=0)
            est = to_numpy_mono(out["flowdec"])
            m = min(n, est.shape[0])
            rel = np.abs(est[:m] - auth_flow[:m]).max() / (np.abs(auth_flow[:m]).max() + 1e-8)
            print(
                f"  ours({dev:>3}): SI-SDR={si_sdr(ref[:m], est[:m]):6.2f} "
                f"fwSSNR={fw_ssnr(ref[:m], est[:m]):6.2f} logSpec={log_spec_mse(ref[:m], est[:m]):7.1f} "
                f"maxdiff-vs-authors={rel:.3f}"
            )


if __name__ == "__main__":
    main()