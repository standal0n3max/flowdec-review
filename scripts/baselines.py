# Copyright (c) 2026 AMLT course project (FlowDec review).
# Classical baselines on the SAME task and bitrates: Opus and Griffin-Lim.
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from typing import Any
from numpy.typing import NDArray
import torch
from beartype import beartype

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from flowdec_lib import save_wav48k, to_numpy_mono  # noqa: E402

SR = 48000


@beartype
def opus_roundtrip(wav48k: torch.Tensor, kbps: float, sr: int = SR) -> NDArray[np.floating[Any]]:
    """
    Encode and decode with Opus at a fixed bitrate via ffmpeg.

    Opus is a transform codec driven by the same low-bitrate goal as FlowDec, so
    it is the natural classical baseline. We pass the audio through an .ogg file
    with `-b:a` set to the requested bitrate and read the decoded samples back.
    """
    mono = to_numpy_mono(wav48k)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        enc = Path(tmp) / "out.opus"
        dst = Path(tmp) / "dec.wav"
        save_wav48k(src, torch.from_numpy(mono).unsqueeze(0))
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(src),
                "-c:a", "libopus",
                "-b:a", f"{int(round(kbps * 1000))}",
                "-vbr", "off",
                "-application", "audio",
                "-frame_duration", "60",
                str(enc),
            ],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(enc), "-ac", "1", "-ar", str(sr), str(dst)],
            check=True, capture_output=True,
        )
        decoded, _ = torchaudio_load_mono(dst)
    return decoded


@beartype
def torchaudio_load_mono(path: str | Path) -> tuple[NDArray[np.floating[Any]], int]:
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    arr = to_numpy_mono(wav)
    return arr, sr


@beartype
def opus_actual_kbps(wav48k: torch.Tensor, kbps: float) -> float:
    """Measure the real bitrate of an Opus stream produced for the given target."""
    mono = to_numpy_mono(wav48k)
    n_samples = mono.shape[0]
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        enc = Path(tmp) / "out.opus"
        save_wav48k(src, torch.from_numpy(mono).unsqueeze(0))
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                "-c:a", "libopus", "-b:a", f"{int(round(kbps * 1000))}",
                "-vbr", "off", "-application", "audio", "-frame_duration", "60", str(enc),
            ],
            check=True, capture_output=True,
        )
        size_bits = enc.stat().st_size * 8
    return float(size_bits / (n_samples / SR) / 1000.0)


@beartype
def griffin_lim_roundtrip(
    wav48k: torch.Tensor,
    n_mels: int = 20,
    n_fft: int = 2048,
    hop: int = 1000,
    n_iter: int = 64,
    sr: int = SR,
) -> NDArray[np.floating[Any]]:
    """
    Griffin-Lim baseline: keep only a low-rate log-mel magnitude spectrum and
    recover the phase iteratively. This is a classical, phase-blind codec-style
    baseline: it transmits a mel spectrogram and invents the phase from the
    magnitude. Default config is rate-matched to ~7.7 kbit/s (20 mel bands at
    48 frames/s, 8-bit coefficients).
    """
    import librosa

    x = to_numpy_mono(wav48k)
    mel = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
    # mel power -> linear magnitude STFT (best least-squares fit), then Griffin-Lim.
    inv_mel = librosa.feature.inverse.mel_to_stft(mel, sr=sr, n_fft=n_fft)
    y = librosa.griffinlim(inv_mel, n_iter=n_iter, hop_length=hop, n_fft=n_fft, window="hann")
    if y.shape[0] < x.shape[0]:
        y = np.pad(y, (0, x.shape[0] - y.shape[0]))
    return y[: x.shape[0]].astype(np.float32)


@beartype
def griffin_lim_mel_kbps(n_mels: int = 20, sr: int = SR, hop: int = 1000, bits: int = 8) -> float:
    """
    Effective rate of the Griffin-Lim baseline in kbit/s.

    The "code" is a mel magnitude spectrogram quantized to `bits` per coefficient
    at `sr / hop` frames per second, i.e. a genuine low-rate representation.
    """
    frame_rate = sr / hop
    return n_mels * bits * frame_rate / 1000.0