# Copyright (c) 2026 AMLT course project (FlowDec review).
# Build a small custom 48 kHz evaluation set (speech / music / ambient).
#
# Everything here is generated or recorded by us, not taken from the authors'
# test set, and is fully deterministic so the demo artifacts are reproducible.
# Drop your own recordings into `data/custom/raw/` to use real audio instead.
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from beartype import beartype

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from flowdec_lib import save_wav48k  # noqa: E402

SR = 48000
OUT = REPO_ROOT / "data" / "custom"
CLIP_SECONDS = 8.0

SPEECH_TEXTS = [
    "Aurora borealis shimmers over the frozen lake, painting the sky in violet and jade.",
    "Neural audio codecs reconstruct speech, music and ambient sound at very low bitrates.",
]


def _voice() -> str:
    """Pick the first available macOS voice from a preference list."""
    try:
        listing = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=False).stdout
    except FileNotFoundError:
        return ""
    available = {line.split()[0] for line in listing.splitlines() if line.strip()}
    for candidate in ("Samantha", "Ava", "Allison", "Alex", "Daniel"):
        if candidate in available:
            return candidate
    return ""


@beartype
def synth_speech() -> list[Path]:
    """Two TTS utterances converted to 48 kHz mono WAV via ffmpeg/afconvert."""
    out_dir = OUT / "speech"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    voice = _voice()
    for i, text in enumerate(SPEECH_TEXTS):
        aiff = out_dir / f"speech_{i}.aiff"
        wav = out_dir / f"speech_{i}.wav"
        cmd = ["say"]
        if voice:
            cmd += ["-v", voice]
        cmd += ["-o", str(aiff), text]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(aiff), "-ac", "1", "-ar", str(SR), str(wav)],
                check=True,
                capture_output=True,
            )
            aiff.unlink(missing_ok=True)
            paths.append(wav)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"[warn] speech synthesis failed ({exc}); skipping clip {i}")
    return paths


@beartype
def synth_music(seed: int = 0) -> list[Path]:
    """
    Two short polyphonic chord progressions with harmonics, percussive
    transients, broadband content and vibrato: enough structure to stress both
    low (bass) and high (cymbals/air) frequency reconstruction.
    """
    rng = np.random.default_rng(seed)
    out_dir = OUT / "music"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    progressions = [
        [220.0, 261.63, 329.63, 392.0],
        [196.0, 246.94, 293.66, 349.23],
    ]
    for pi, chord in enumerate(progressions):
        n = int(CLIP_SECONDS * SR)
        t = np.arange(n) / SR
        sig = np.zeros(n, dtype=np.float64)
        # Arpeggiated notes, four per second, each with pitch-dependent decay.
        step = SR // 4
        for k in range(0, n, step):
            freq = chord[(k // step) % len(chord)]
            seg_len = min(step * 2, n - k)
            ts = np.arange(seg_len) / SR
            env = np.exp(-3.0 * ts)
            partials = 1.0 * np.sin(2 * np.pi * freq * ts)
            partials += 0.5 * np.sin(2 * np.pi * 2 * freq * ts + 0.1)
            partials += 0.25 * np.sin(2 * np.pi * 3 * freq * ts + 0.2)
            partials += 0.12 * np.sin(2 * np.pi * 4 * freq * ts + 0.3)
            partials += 0.06 * np.sin(2 * np.pi * 6 * freq * ts + 0.4)
            partials += 0.03 * np.sin(2 * np.pi * freq * 9 * ts)
            vib = 1.0 + 0.004 * np.sin(2 * np.pi * 5.5 * ts)
            sig[k : k + seg_len] += env * partials * vib
        # Percussive hits (broadband transients) on every beat pair.
        for k in range(0, n, step * 2):
            dur = int(0.06 * SR)
            ts = np.arange(min(dur, n - k)) / SR
            burst = np.exp(-40.0 * ts) * rng.standard_normal(ts.shape[0])
            burst += 0.2 * np.exp(-30.0 * ts) * np.sin(2 * np.pi * 90.0 * ts)
            sig[k : k + ts.shape[0]] += 0.35 * burst
        # Bass line an octave below, plus a gentle noise bed.
        bass = 0.35 * np.sin(2 * np.pi * chord[0] / 2 * t)
        sig = sig + bass + 0.01 * rng.standard_normal(n)
        sig = sig / (np.max(np.abs(sig)) + 1e-8) * 0.9
        path = out_dir / f"music_{pi}.wav"
        save_wav48k(path, __import__("torch").from_numpy(sig.astype(np.float32)).unsqueeze(0))
        paths.append(path)
    return paths


@beartype
def synth_ambient(seed: int = 0) -> list[Path]:
    """Two ambient beds: pink-ish noise with slow swells and a few chirps."""
    rng = np.random.default_rng(seed)
    out_dir = OUT / "ambient"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ci in range(2):
        n = int(CLIP_SECONDS * SR)
        t = np.arange(n) / SR
        # Simple 1/f noise: cumulative sum of white noise gives a brownian slope.
        white = rng.standard_normal(n)
        pink = np.cumsum(white)
        pink = pink - np.mean(pink)
        pink = pink / (np.max(np.abs(pink)) + 1e-8)
        swells = 0.6 + 0.4 * np.sin(2 * np.pi * 0.12 * t + ci)
        sig = pink * swells
        # A couple of tonal chirps (birds / distant tones).
        for f0, f1, start in ((1800.0, 2600.0, 1.2), (1200.0, 900.0, 4.0)):
            s = int(start * SR)
            dur = int(0.35 * SR)
            ts = np.arange(dur) / SR
            sweep = np.cumsum(f0 + (f1 - f0) * ts / dur) / SR
            chirp = 0.25 * np.sin(2 * np.pi * sweep) * np.hanning(dur)
            sig[s : s + dur] += chirp
        sig = sig / (np.max(np.abs(sig)) + 1e-8) * 0.85
        path = out_dir / f"ambient_{ci}.wav"
        save_wav48k(path, __import__("torch").from_numpy(sig.astype(np.float32)).unsqueeze(0))
        paths.append(path)
    return paths


def main() -> None:
    made: list[Path] = []
    made += synth_speech()
    made += synth_music()
    made += synth_ambient()

    # Copy any user-supplied recordings so the pipeline can pick them up.
    raw_dir = OUT / "raw"
    if raw_dir.is_dir():
        for raw in sorted(raw_dir.glob("*.wav")):
            made.append(raw)

    print(f"Generated {len(made)} clips under {OUT}:")
    for p in made:
        print(" -", p.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()