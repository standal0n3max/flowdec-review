# Copyright (c) 2026 AMLT course project (FlowDec review).
# Shared helpers: device selection, checkpoint loading, codec/postfilter inference.
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import torch
import torchaudio
from beartype import beartype
from dac import DAC
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

# Make the cloned official repository importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
FLOWDEC_REPO = REPO_ROOT / "code" / "FlowDec"
if str(FLOWDEC_REPO) not in sys.path:
    sys.path.insert(0, str(FLOWDEC_REPO))

CKPT_DIR = REPO_ROOT / "paper" / "checkpoints"

# FlowDec-75m was trained with these codebook counts; each maps to a bitrate.
NQ_TO_KBPS_75M: dict[int, float] = {10: 7.5, 8: 6.0, 6: 4.5, 4: 3.0}
NQ_TO_KBPS_25S: dict[int, float] = {16: 4.0}

MODEL_TO_NDAC: dict[str, str] = {"flowdec_75m": "ndac-75", "flowdec_25s": "ndac-25"}


@beartype
def pick_device(prefer: str | None = None) -> torch.device:
    """Pick the best available device: cuda > mps > cpu, unless overridden."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class FlowDecBundle:
    """A loaded (NDAC codec, FlowDec postfilter) pair plus inference metadata."""

    dac: DAC
    model: torch.nn.Module
    device: torch.device
    variant: str


@beartype
def load_flowdec(
    variant: str = "flowdec_75m",
    device: torch.device | None = None,
    use_ema_weights: bool = True,
) -> FlowDecBundle:
    """Load an official NDAC + FlowDec checkpoint pair and put it into eval mode."""
    dev = device or pick_device()
    ndac_name = MODEL_TO_NDAC[variant]

    dac_weights = CKPT_DIR / "ndac" / ndac_name / "800k" / "dac" / "weights.pth"
    flow_ckpt = CKPT_DIR / "flowdec" / variant / "step=800000.ckpt"
    if not dac_weights.is_file():
        raise FileNotFoundError(f"NDAC weights not found: {dac_weights}")
    if not flow_ckpt.is_file():
        raise FileNotFoundError(f"FlowDec checkpoint not found: {flow_ckpt}")

    # `dac` is the descript-audio-codec package; the FlowDec repo lives next to it.
    sys.path.insert(0, str(FLOWDEC_REPO))
    dac = DAC.load(str(dac_weights))
    dac.to(dev)
    dac.eval()

    try:
        initialize_config_dir(config_dir=str(FLOWDEC_REPO / "config"), version_base="1.3")
    except ValueError:
        pass
    conf = compose(config_name=variant)
    ckpt = torch.load(flow_ckpt, map_location="cpu", weights_only=False)
    key = "_pl_ema_state_dict" if use_ema_weights else "state_dict"
    model = instantiate(conf["model"])
    model.load_state_dict(ckpt[key])
    # `sigma_y` (and some buffers) are float64; MPS has no float64 support, so we
    # cast everything to float32 (matches the weights, which are already fp32).
    model = model.float()
    model.to(dev)
    model.eval()

    return FlowDecBundle(dac=dac, model=model, device=dev, variant=variant)


@beartype
def load_wav48k(path: str | Path) -> torch.Tensor:
    """Load a mono waveform resampled to 48 kHz, shape (1, T)."""
    wav, _sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if _sr != 48000:
        wav = torchaudio.functional.resample(wav, _sr, 48000)
    out: torch.Tensor = wav
    return out


@beartype
def save_wav48k(path: str | Path, wav: torch.Tensor) -> None:
    """Save a mono 48 kHz waveform, clipping to [-1, 1] and creating parent dirs."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach().cpu()
    while wav.dim() > 2:
        wav = wav.squeeze(0)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    peak = wav.abs().max()
    if peak > 1.0:
        wav = wav / peak
    torchaudio.save(str(path), wav, 48000)


@beartype
def prepare_input(bundle: FlowDecBundle, wav48k: torch.Tensor) -> torch.Tensor:
    """
    Resample to the codec sample rate, move to the device and shape as (B, 1, T).

    The NDAC/DAC encoder expects `[batch, channels, time]`; a bare `[channels,
    time]` tensor would be misread as a huge channel count.
    """
    x = wav48k
    if x.dim() == 1:
        x = x.unsqueeze(0)  # (T,) -> (1, T)
    if x.dim() == 2:
        x = x.unsqueeze(0) if x.shape[0] == 1 else x.unsqueeze(1)
    if x.shape[1] > 1:
        x = x.mean(dim=1, keepdim=True)
    if 48000 != bundle.dac.sample_rate:
        x = torchaudio.functional.resample(x, 48000, bundle.dac.sample_rate)
    return x.to(bundle.device)


@beartype
def ndac_reconstruct(bundle: FlowDecBundle, x: torch.Tensor, nq: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the non-adversarial codec only.

    Returns a 2-tuple of (waveform at the codec output, quantized latent).
    """
    dac = bundle.dac
    with torch.inference_mode():
        xin = dac.preprocess(x, dac.sample_rate)
        _z, codes, _latents, _, _ = dac.encode(xin, n_quantizers=nq)
        zq, _, _ = dac.quantizer.from_codes(codes)
        zq_t = cast(torch.Tensor, zq)
        xhat_ndac = cast(torch.Tensor, dac.decode(zq_t))
    return xhat_ndac, zq_t


@beartype
def flowdec_enhance(
    bundle: FlowDecBundle,
    xhat_ndac: torch.Tensor,
    n_steps: int = 3,
    solver: str = "midpoint",
) -> torch.Tensor:
    """Apply the FlowDec stochastic postfilter (N solver steps; midpoint => NFE = 2N)."""
    with torch.inference_mode():
        enhance = cast(Any, bundle.model).enhance
        xhat = cast(torch.Tensor, enhance(xhat_ndac, N=n_steps, solver=solver))
        if xhat.abs().max() > 1.0:
            xhat = xhat / xhat.abs().max()
    return xhat


@beartype
def set_seed(seed: int = 0) -> None:
    """Seed torch (and numpy) so the stochastic postfilter is reproducible."""
    np.random.seed(seed)
    torch.manual_seed(seed)


@beartype
def full_flowdec(
    bundle: FlowDecBundle,
    wav48k: torch.Tensor,
    nq: int,
    n_steps: int = 3,
    solver: str = "midpoint",
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """
    End-to-end: codec encode -> quantize -> codec decode -> FlowDec postfilter.

    Returns a dict with keys 'ndac', 'flowdec' (48 kHz waveforms) and 'zq' (latent).
    Pass `seed` to make the stochastic postfilter deterministic.
    """
    if seed is not None:
        set_seed(seed)
    x = prepare_input(bundle, wav48k)
    xhat_ndac, zq = ndac_reconstruct(bundle, x, nq)
    xhat_flow = flowdec_enhance(bundle, xhat_ndac, n_steps=n_steps, solver=solver)
    if bundle.dac.sample_rate != 48000:
        xhat_ndac = torchaudio.functional.resample(xhat_ndac, bundle.dac.sample_rate, 48000)
        xhat_flow = torchaudio.functional.resample(xhat_flow, bundle.dac.sample_rate, 48000)
    return {"ndac": xhat_ndac.cpu(), "flowdec": xhat_flow.cpu(), "zq": zq.cpu()}


@beartype
def count_parameters(module: torch.nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


@beartype
def to_numpy_mono(wav: torch.Tensor) -> NDArray[np.floating[Any]]:
    """Flatten any (.., C, T), (C, T), (1, 1, T) or (T,) tensor to a 1-D float32 array."""
    arr = wav.detach().cpu().numpy()
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return arr.reshape(1).astype(np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=0)
    return arr.reshape(-1).astype(np.float32)