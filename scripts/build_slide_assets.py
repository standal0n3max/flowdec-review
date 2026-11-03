# Copyright (c) 2026 AMLT course project (FlowDec review).
# Builds every media asset the deck needs: equation images, the pipeline diagram,
# code-cell screenshots, and the audio files (mp3 for HTML, mp4 for PowerPoint).
from __future__ import annotations

import html
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageChops, ImageOps  # noqa: E402
from beartype import beartype
from pygments import highlight  # noqa: E402
from pygments.formatters import HtmlFormatter  # noqa: E402
from pygments.lexers import PythonLexer  # noqa: E402

import slides_content as sc

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "report" / "assets"

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CHROME_TIMEOUT_S = 60
SCALE = 2  # device pixel ratio for the screenshots -> crisp on a projector


# --------------------------------------------------------------------------- #
# Small shell helper.
# --------------------------------------------------------------------------- #


@beartype
def run(argv: list[str], timeout: int = CHROME_TIMEOUT_S) -> None:
    """Run a command, raising with a readable message on a non-zero exit."""
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip()[-600:]
        raise RuntimeError(f"command failed ({result.returncode}): {argv[0]}\n{tail}")


@beartype
def ffmpeg(args: list[str]) -> None:
    """Run ffmpeg quietly."""
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args])


@beartype
def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Formulas (matplotlib mathtext -> identical pixels in HTML and PPTX).
# --------------------------------------------------------------------------- #

EQUATIONS: tuple[tuple[str, str, int], ...] = (
    (
        "eq_train.png",
        r"$\mathcal{L}_{\mathrm{JFM}} = \mathbb{E}\left\| v_\theta(x_t, t, y)"
        r" - \left(x^{*} - (y + \sigma_y \varepsilon)\right) \right\|_2^2$",
        22,
    ),
    (
        "eq_infer.png",
        r"$X = \varphi(y) + \sigma_y(f)\,\varepsilon \qquad"
        r"\frac{dX}{dt} = v_\theta(X, t, Y) \qquad"
        r"t \in [0, 1] \qquad \mathrm{NFE} = 2N$",
        22,
    ),
    (
        "eq_coupling.png",
        r"$x_0 \sim \mathcal{N}(y, \Sigma_y) \;\longrightarrow\; x_1 = x^{*}$",
        28,
    ),
)


@beartype
def build_formulas() -> list[str]:
    """Render the equations; returns the written asset paths (relative to the root)."""
    out_dir = ASSETS / "formulas"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for name, tex, size in EQUATIONS:
        fig = plt.figure(figsize=(14, 1.4))
        fig.text(0.5, 0.5, tex, ha="center", va="center", fontsize=size, color="#102345")
        fig.savefig(out_dir / name, dpi=200, bbox_inches="tight", transparent=False,
                    facecolor="white", pad_inches=0.25)
        plt.close(fig)
        written.append(f"formulas/{name}")
    return written


# --------------------------------------------------------------------------- #
# Pipeline diagram: a hand-written SVG (single source) plus a PNG via Chrome.
# --------------------------------------------------------------------------- #

PIPELINE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 360" font-family="Inter, Helvetica, Arial, sans-serif">
  <defs>
    <marker id="arw" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0 0 L10 5 L0 10 z" fill="#5188ca"/>
    </marker>
  </defs>
  <rect x="0" y="0" width="1200" height="360" fill="#ffffff"/>
  <rect x="24" y="96" width="1152" height="150" rx="18" fill="#f7faff" stroke="#dce7f5"/>

  <!-- waveform in -->
  <g>
    <rect x="44" y="140" width="150" height="62" rx="12" fill="#ffffff" stroke="#dce7f5" stroke-width="1.5"/>
    <text x="119" y="166" text-anchor="middle" font-size="17" font-weight="700" fill="#102345">waveform x*</text>
    <text x="119" y="188" text-anchor="middle" font-size="12" fill="#647992">48 kHz mono, 5 s</text>
  </g>

  <path d="M196 171 H246" stroke="#5188ca" stroke-width="2" marker-end="url(#arw)"/>
  <text x="221" y="160" text-anchor="middle" font-size="11" fill="#647992">E</text>

  <!-- latent z -->
  <g>
    <rect x="248" y="140" width="132" height="62" rx="12" fill="#ffffff" stroke="#dce7f5" stroke-width="1.5"/>
    <text x="314" y="166" text-anchor="middle" font-size="17" font-weight="700" fill="#102345">latent z</text>
    <text x="314" y="188" text-anchor="middle" font-size="12" fill="#647992">encoder E (NDAC)</text>
  </g>

  <path d="M382 171 H432" stroke="#5188ca" stroke-width="2" marker-end="url(#arw)"/>
  <text x="407" y="160" text-anchor="middle" font-size="11" fill="#647992">RVQ</text>

  <!-- codes -->
  <g>
    <rect x="434" y="140" width="150" height="62" rx="12" fill="#eff6ff" stroke="#9fc7ff" stroke-width="1.8"/>
    <text x="509" y="166" text-anchor="middle" font-size="17" font-weight="700" fill="#1676f3">discrete codes c</text>
    <text x="509" y="188" text-anchor="middle" font-size="12" fill="#4c729d">the bitstream · nq codebooks</text>
  </g>

  <path d="M586 171 H636" stroke="#5188ca" stroke-width="2" marker-end="url(#arw)"/>
  <text x="611" y="160" text-anchor="middle" font-size="11" fill="#647992">D0</text>

  <!-- coarse y -->
  <g>
    <rect x="638" y="140" width="150" height="62" rx="12" fill="#ffffff" stroke="#dce7f5" stroke-width="1.5"/>
    <text x="713" y="166" text-anchor="middle" font-size="17" font-weight="700" fill="#102345">coarse y = D0(c)</text>
    <text x="713" y="188" text-anchor="middle" font-size="12" fill="#647992">non-adversarial decoder</text>
  </g>

  <path d="M790 171 H840" stroke="#5188ca" stroke-width="2" marker-end="url(#arw)"/>

  <!-- FlowDec -->
  <g>
    <rect x="842" y="126" width="176" height="90" rx="14" fill="#f4efff" stroke="#cdb5f6" stroke-width="1.8"/>
    <text x="930" y="158" text-anchor="middle" font-size="18" font-weight="700" fill="#7441bc">FlowDec postfilter</text>
    <text x="930" y="180" text-anchor="middle" font-size="13" fill="#7441bc">CFM ODE · midpoint solver</text>
    <text x="930" y="200" text-anchor="middle" font-size="13" fill="#7441bc">6 NFE · 23.7 M parameters</text>
  </g>

  <path d="M1020 171 H1070" stroke="#5188ca" stroke-width="2" marker-end="url(#arw)"/>

  <!-- output -->
  <g>
    <rect x="1072" y="140" width="90" height="62" rx="12" fill="#ffffff" stroke="#dce7f5" stroke-width="1.5"/>
    <text x="1117" y="166" text-anchor="middle" font-size="17" font-weight="700" fill="#102345">x&#770;</text>
    <text x="1117" y="188" text-anchor="middle" font-size="12" fill="#647992">enhanced</text>
  </g>

  <!-- bitrate note -->
  <text x="509" y="70" text-anchor="middle" font-size="13" fill="#4c729d">
    nq = 10 / 8 / 6 / 4 codebooks -&gt; 7.5 / 6.0 / 4.5 / 3.0 kbit/s (chosen at inference)
  </text>
  <text x="930" y="272" text-anchor="middle" font-size="13" fill="#4c729d">
    works in a compressed complex STFT domain (1534-sample Hann, hop 384, alpha = 0.3, phase kept)
  </text>
  <text x="600" y="326" text-anchor="middle" font-size="13" fill="#647992">
    Two stages: a non-adversarial codec gives a coarse estimate; a conditional-flow-matching postfilter
    generatively reconstructs the missing full-band detail.
  </text>
</svg>
"""


@beartype
def build_diagram() -> list[str]:
    """Write `diagram/pipeline.svg` and rasterise it to `diagram/pipeline.png`."""
    out_dir = ASSETS / "diagram"
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / "pipeline.svg"
    png_path = out_dir / "pipeline.png"
    write_text(svg_path, PIPELINE_SVG)

    page = (
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        "html,body{margin:0;padding:0;background:#fff}"
        "svg{display:block;width:1200px;height:360px}"
        "</style></head><body>" + PIPELINE_SVG + "</body></html>"
    )
    with tempfile.TemporaryDirectory() as tmp:
        page_path = Path(tmp) / "diagram.html"
        write_text(page_path, page)
        run([
            CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
            "--disable-extensions", f"--force-device-scale-factor={SCALE}",
            "--window-size=1200,360", f"--screenshot={png_path}", f"file://{page_path}",
        ])
    return ["diagram/pipeline.svg", "diagram/pipeline.png"]


# --------------------------------------------------------------------------- #
# Code-cell screenshots: rendered from the executed notebook, rasterised with
# Chrome.  Standalone HTML keeps the output readable on a projector.
# --------------------------------------------------------------------------- #

CODE_CELLS: tuple[tuple[str, str, str], ...] = (
    (
        "cell_ckpt.png",
        "cell-ckpt",
        "Loading the official checkpoints",
    ),
    (
        "cell_pipeline.png",
        "cell-pipeline",
        "End-to-end FlowDec on a 5 s music clip",
    ),
    (
        "cell_metrics.png",
        "cell-metrics",
        "Metrics against the classical baselines",
    ),
    (
        "cell_nfe.png",
        "cell-nfe",
        "NFE sweep",
    ),
)

CODE_PAGE_CSS = """
html,body{margin:0;padding:0;background:#fff}
body{width:1000px;font-family:Inter,"Helvetica Neue",Arial,sans-serif}
.card{margin:0;border:1px solid #dce7f5;border-radius:14px;overflow:hidden}
.head{background:#f7faff;border-bottom:1px solid #dce7f5;padding:16px 24px}
.head .t{font-size:17px;font-weight:700;color:#102345}
.head .s{font-size:11px;color:#647992;margin-top:4px}
pre{margin:0;padding:20px 24px;background:#ffffff;font:13.5px/1.6 "SFMono-Regular",Menlo,Consolas,monospace;
    color:#1f2d3d;white-space:pre-wrap;overflow-wrap:anywhere}
.cout{border-top:1px solid #dce7f5;background:#fbfdff;padding:18px 24px;
      font:13.5px/1.6 "SFMono-Regular",Menlo,Consolas,monospace;color:#1f2d3d;
      white-space:pre-wrap;overflow-wrap:anywhere}
.cout .lbl{display:block;font:10px/1 "SFMono-Regular",Menlo,monospace;letter-spacing:1px;
           text-transform:uppercase;color:#647992;margin-bottom:10px}
"""


@beartype
def _cell_html(title: str, subtitle: str, code_html: str, out_text: str) -> str:
    out_block = ""
    if out_text.strip():
        out_block = f'<div class="cout"><span class="lbl">output</span>{html.escape(out_text)}</div>'
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CODE_PAGE_CSS}{HtmlFormatter(style='friendly').get_style_defs('pre')}</style>"
        "</head><body><div class='card'>"
        f"<div class='head'><div class='t'>{html.escape(title)}</div>"
        f"<div class='s'>{html.escape(subtitle)}</div></div>"
        f"{code_html}{out_block}</div></body></html>"
    )


@beartype
def _cell_sources() -> dict[str, tuple[str, str]]:
    import json

    nb = json.loads((ROOT / "notebooks" / "flowdec_demo.ipynb").read_text(encoding="utf-8"))
    cells = nb["cells"]

    def source(index: int) -> str:
        return "".join(cells[index]["source"])

    def streams(index: int) -> str:
        chunks: list[str] = []
        for out in cells[index].get("outputs", []):
            if out.get("output_type") == "stream":
                chunks.append("".join(out.get("text", [])))
        text = "".join(chunks)
        lines = [ln for ln in text.splitlines() if "nn.Parameters" not in ln]
        return "\n".join(lines).strip()

    return {
        "cell-ckpt": (source(4), streams(4)),
        "cell-pipeline": (source(10), "Figure: waveform + log-mel spectrogram, clean vs FlowDec (shown separately)."),
        "cell-metrics": (source(12), streams(12)),
        "cell-nfe": (source(16), streams(16)),
    }


@beartype
def build_code_cells() -> list[str]:
    """Screenshot four notebook cells into `code/cell_*.png`."""
    out_dir = ASSETS / "code"
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = _cell_sources()
    formatter = HtmlFormatter(style="friendly")
    written: list[str] = []

    for name, key, title in CODE_CELLS:
        code, output = sources[key]
        code_html = highlight(code, PythonLexer(), formatter)
        subtitle = "Cell from notebooks/flowdec_demo.ipynb (executed once, frozen)."
        page = _cell_html(title, subtitle, code_html, output)

        with tempfile.TemporaryDirectory() as tmp:
            page_path = Path(tmp) / f"{key}.html"
            write_text(page_path, page)
            raw = Path(tmp) / "raw.png"
            run([
                CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                "--disable-extensions", f"--force-device-scale-factor={SCALE}",
                "--window-size=1000,1800", f"--screenshot={raw}", f"file://{page_path}",
            ])
            trim_and_save(raw, out_dir / name)
        written.append(f"code/{name}")
    return written


# --------------------------------------------------------------------------- #
# Image trimming (drop the blank area below the screenshot content).
# --------------------------------------------------------------------------- #


@beartype
def trim_and_save(raw: Path, out: Path, border: int = 0) -> None:
    """Crop the uniform white margins produced by a full-page Chrome screenshot."""
    image = Image.open(raw).convert("RGB")
    background = Image.new("RGB", image.size, 0xFFFFFF)
    bbox = ImageChops.difference(image, background).getbbox()
    if bbox is None:
        image.save(out)
        return
    cropped = image.crop(bbox)
    if border:
        cropped = ImageOps.expand(cropped, border=border, fill=0xFFFFFF)
    cropped.save(out)


# --------------------------------------------------------------------------- #
# Audio: mp3 for the HTML deck, mp4 (poster waveform + sound) for PowerPoint.
# --------------------------------------------------------------------------- #

WAVE_COLORS = "#1676f3"


@beartype
def build_audio() -> list[str]:
    """For every clip write `<slug>.mp3`, `<slug>.mp4` and `<slug>_wave.png`."""
    out_dir = ASSETS / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = [c for c in sc.AUDIO_CLIPS if not (ROOT / c.source).is_file()]
    if missing:
        raise SystemExit(
            "missing demo audio (run scripts/run_experiments.py first):\n  "
            + "\n  ".join(c.source for c in missing)
        )
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")

    written: list[str] = []
    for clip in sc.AUDIO_CLIPS:
        source = str(ROOT / clip.source)
        mp3 = out_dir / f"{clip.slug}.mp3"
        wave = out_dir / f"{clip.slug}_wave.png"
        mp4 = out_dir / f"{clip.slug}.mp4"

        ffmpeg(["-i", source, "-ac", "1", "-codec:a", "libmp3lame", "-b:a", "80k", str(mp3)])
        ffmpeg([
            "-i", source,
            "-filter_complex", f"showwavespic=s=1000x260:colors=0x{WAVE_COLORS.lstrip('#')}",
            "-frames:v", "1", str(wave),
        ])
        ffmpeg([
            "-loop", "1", "-i", str(wave), "-i", source,
            "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
            "-r", "5", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
            "-shortest", str(mp4),
        ])
        written += [f"audio/{clip.slug}.mp3", f"audio/{clip.slug}.mp4", f"audio/{clip.slug}_wave.png"]
    return written


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #


@beartype
def main() -> None:
    if not Path(CHROME).is_file():
        raise SystemExit(f"Google Chrome not found at {CHROME}")

    print("formulas:", ", ".join(build_formulas()))
    print("diagram: ", ", ".join(build_diagram()))
    print("code:    ", ", ".join(build_code_cells()))
    print("audio:   ", ", ".join(build_audio()))

    total = sum(p.stat().st_size for p in ASSETS.rglob("*") if p.is_file())
    print(f"\nassets written to {ASSETS} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
