# Copyright (c) 2026 AMLT course project (FlowDec review).
# Objective metrics used by FlowDec (SI-SDR, fwSSNR, logSpecMSE) plus FAD.
#
# The waveform metrics mirror the official implementations in
# `code/FlowDec/flowdec/eval/metrics.py` so our numbers are comparable to the
# paper. FAD follows Gui et al. (2024): Frechet distance between CLAP embeddings.
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import torch
from beartype import beartype

# pysepm prints a deprecation warning on import; keep notebooks clean.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from pysepm.qualityMeasures import fwSNRseg

EPS = 1e-8


@beartype
def si_sdr(reference: NDArray[np.floating[Any]], estimate: NDArray[np.floating[Any]]) -> float:
    """
    Scale-invariant SDR in dB, with an optional global sign flip.

    Mirrors `flowdec.eval.metrics.SISXR`: we allow the estimate to be inverted
    (`y + x` instead of `y - x`) to correct a global phase flip before scoring.
    """
    x = np.asarray(reference, dtype=np.float64).reshape(-1)
    y = np.asarray(estimate, dtype=np.float64).reshape(-1)
    n = min(x.shape[0], y.shape[0])
    x, y = x[:n], y[:n]

    if np.linalg.norm(y + x) < np.linalg.norm(y - x):
        target = -x
    else:
        target = x

    # Closed-form optimal scaling of `target` towards `y`.
    alpha = float(np.dot(y, target) / (np.dot(target, target) + EPS))
    s_target = alpha * target
    e_res = y - s_target
    return float(10.0 * np.log10((np.linalg.norm(s_target) ** 2) / (np.linalg.norm(e_res) ** 2 + EPS)))


@beartype
def fw_ssnr(reference: NDArray[np.floating[Any]], estimate: NDArray[np.floating[Any]], sr: int = 48000) -> float:
    """Frequency-weighted segmental SNR (dB), same call as the official wrapper."""
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    est = np.asarray(estimate, dtype=np.float64).reshape(-1)
    n = min(ref.shape[0], est.shape[0])
    return float(
        fwSNRseg(
            ref[:n],
            est[:n],
            fs=sr,
            frameLen=0.03,
            overlap=0.75,
        )
    )


@beartype
def log_spec_mse(
    reference: NDArray[np.floating[Any]],
    estimate: NDArray[np.floating[Any]],
    sr: int = 48000,
    win_dur: float = 32e-3,
    hop_dur: float = 8e-3,
    eps: float = 1e-8,
) -> float:
    """
    logSpecMSE: MSE of dB log-magnitude spectrograms.

    Exact re-implementation of `flowdec.eval.metrics.LogSpecMSE` so our numbers are
    comparable to the paper: power spectrogram (|STFT|^2), clamped to a floor of
    `eps` (i.e. -80 dB), then `10*log10`. The floor choice dominates the value, so
    matching it is essential.
    """
    x = torch.as_tensor(np.asarray(reference, dtype=np.float32)).reshape(-1)
    y = torch.as_tensor(np.asarray(estimate, dtype=np.float32)).reshape(-1)
    n_fft = int(win_dur * sr)
    hop = int(hop_dur * sr)
    window = torch.hann_window(n_fft)

    def _logspec(sig: torch.Tensor) -> torch.Tensor:
        stft = torch.stft(sig, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                          window=window, center=True, return_complex=True)
        power = stft.real.pow(2) + stft.imag.pow(2)
        return 10.0 * torch.log10(torch.clamp(power, min=eps))

    s_ref = _logspec(x)
    s_est = _logspec(y)
    t = min(s_ref.shape[-1], s_est.shape[-1])
    f = min(s_ref.shape[0], s_est.shape[0])
    return float(torch.mean(torch.square(s_ref[:f, :t] - s_est[:f, :t])).item())


@beartype
def mel_spectrogram(
    wav: NDArray[np.floating[Any]],
    sr: int = 48000,
    n_fft: int = 2048,
    hop: int = 512,
    n_mels: int = 128,
) -> NDArray[np.floating[Any]]:
    """Log-mel spectrogram in dB for visualization."""
    import librosa

    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    mel = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
    return cast(NDArray[np.floating[Any]], librosa.power_to_db(mel, ref=np.max))


class FAD:
    """
    Frechet Audio Distance using laion-clap embeddings (CLAP-laion-audio).

    FAD = ||mu_r - mu_e||^2 + Tr(C_r + C_e - 2 * (C_r^{1/2} C_e C_r^{1/2})^{1/2})

    Reference and estimate are given as lists of 48 kHz mono waveforms. With few
    clips the covariance estimate is ill-conditioned, so we shrink the covariance
    towards a diagonal by `shrink` (a standard, documented regularizer) and fall
    back to a diagonal-only Frechet distance when Cholesky fails.
    """

    def __init__(self, sample_rate: int = 48000, shrink: float = 1e-2, device: str = "cpu") -> None:
        self.sample_rate = sample_rate
        self.shrink = shrink
        self.device = device
        self._embedder = None

    def _load(self) -> None:
        import laion_clap

        # `630k-audioset-best.pt` is the CLAP-laion-audio checkpoint; it matches
        # the package's default HTSAT-tiny backbone, so we keep the default amodel.
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt(model_id=1, verbose=False)
        model.eval()
        self._embedder = model

    def embed(self, wavs: list[NDArray[np.floating[Any]]]) -> NDArray[np.floating[Any]]:
        """Audio embeddings of shape (N, 512) for a list of mono waveforms."""
        if self._embedder is None:
            self._load()
        assert self._embedder is not None
        # CLAP's HTSAT front-end expects a fixed length per batch; pad to the
        # longest clip in the batch (zero-padding at the end).
        max_len = max(np.asarray(w, dtype=np.float32).reshape(-1).shape[0] for w in wavs)
        tensors = []
        for wav in wavs:
            x = np.asarray(wav, dtype=np.float32).reshape(-1)
            if x.shape[0] < max_len:
                x = np.pad(x, (0, max_len - x.shape[0]))
            tensors.append(torch.from_numpy(x))
        batch = torch.stack(tensors, dim=0)
        with torch.inference_mode():
            emb = self._embedder.get_audio_embedding_from_data(x=batch, use_tensor=True)
        return emb.detach().cpu().numpy().astype(np.float64)

    def _cov(self, x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Regularized covariance with a small diagonal shrink for stability."""
        c = np.cov(x, rowvar=False)
        c = np.atleast_2d(c)
        d = np.mean(np.diag(c)) if c.size else 1.0
        out: NDArray[np.floating[Any]] = c + self.shrink * d * np.eye(c.shape[0])
        return out

    def score_from_embeddings(self, ref: NDArray[np.floating[Any]], est: NDArray[np.floating[Any]]) -> float:
        mu1, mu2 = ref.mean(0), est.mean(0)
        c1, c2 = self._cov(ref), self._cov(est)
        diff = float(np.sum((mu1 - mu2) ** 2))
        try:
            from scipy import linalg

            covmean = linalg.sqrtm(c1.astype(np.float64) @ c2.astype(np.float64))
            covmean = cast(NDArray[np.float64], covmean)
            if np.iscomplexobj(covmean):
                covmean = np.real(covmean)
            trace = float(np.trace(c1 + c2 - 2.0 * covmean))
        except Exception:
            trace = float(np.trace(c1) + np.trace(c2) - 2.0 * np.sum(np.sqrt(np.diag(c1) * np.diag(c2))))
        return diff + trace

    def score(self, ref_wavs: list[NDArray[np.floating[Any]]], est_wavs: list[NDArray[np.floating[Any]]]) -> float:
        """Fréchet Audio Distance between two sets of waveforms."""
        ref_emb = self.embed(ref_wavs)
        est_emb = self.embed(est_wavs)
        return self.score_from_embeddings(ref_emb, est_emb)

    def pairwise_scores(self, ref_wavs: list[NDArray[np.floating[Any]]], est_wavs: list[NDArray[np.floating[Any]]]) -> list[float]:
        """
        Indicative per-clip FAD: distance from the single-clip estimate embedding to
        the reference distribution. Useful when the estimate set is too small for a
        stable covariance, as in a hand-recorded custom set.
        """
        ref_emb = self.embed(ref_wavs)
        est_emb = self.embed(est_wavs)
        mu1 = ref_emb.mean(0)
        c1 = self._cov(ref_emb)
        out = []
        for row in est_emb:
            out.append(float(np.sum((row - mu1) ** 2)) + float(np.trace(c1)))
        return out


@beartype
def metrics_for_pair(reference: NDArray[np.floating[Any]], estimate: NDArray[np.floating[Any]], sr: int = 48000) -> dict[str, float]:
    """All waveform metrics for one (reference, estimate) pair."""
    return {
        "si_sdr": si_sdr(reference, estimate),
        "fwssnr": fw_ssnr(reference, estimate, sr=sr),
        "logspec_mse": log_spec_mse(reference, estimate, sr=sr),
    }


@beartype
def write_json(path: str | Path, data: object) -> None:
    import json

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)