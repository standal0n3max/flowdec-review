# Copyright (c) 2026 AMLT course project (FlowDec review).
# Renders report/flowdec_slides.pptx from the same content module the HTML deck
# uses.  Audio has no native shape in python-pptx, so each clip is embedded as a
# short mp4 (waveform poster + sound), which plays reliably in PowerPoint.
from __future__ import annotations

import math
from pathlib import Path
from typing import cast

from PIL import Image
from beartype import beartype
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.presentation import Presentation as PresentationType
from pptx.slide import Slide
from pptx.table import Table
from pptx.text.text import TextFrame
from pptx.util import Inches, Pt

import slides_content as sc

ROOT = Path(__file__).resolve().parents[1]
OUT = sc.DECK_PPTX

# --------------------------------------------------------------------------- #
# Geometry and palette.
# --------------------------------------------------------------------------- #

SLIDE_W = 13.3333
SLIDE_H = 7.5
MARGIN = 0.62
CONTENT_W = SLIDE_W - 2 * MARGIN
BODY_TOP = 1.78
BODY_BOTTOM = 7.02

FONT = "Segoe UI"
MONO = "Consolas"


@beartype
def rgb(hex_triplet: str) -> RGBColor:
    """`RGBColor` ships no type annotations, so wrap it once for mypy."""
    red = int(hex_triplet[0:2], 16)
    green = int(hex_triplet[2:4], 16)
    blue = int(hex_triplet[4:6], 16)
    return RGBColor(red, green, blue)  # type: ignore[no-untyped-call]

NAVY = rgb("102345")
TEXT = rgb("243D61")
MUTED = rgb("647992")
BLUE = rgb("1676F3")
PURPLE = rgb("8954EF")
LINE = rgb("DCE7F5")
PAPER = rgb("F7FAFF")
TINT = rgb("EFF6FF")
WHITE = rgb("FFFFFF")
GREEN = rgb("127A4D")
CALLOUT_LINE = rgb("CCE0FC")

SIZE_BULLET = 13.5
SIZE_PARA = 13.5
SIZE_CALLOUT = 12.5
SIZE_TABLE = 11.5
SIZE_CAPTION = 9.5
SIZE_EYEBROW = 10.5
SIZE_TITLE = 27.0
SIZE_SUBTITLE = 13.5

CHARS_WIDE = 0.505  # average glyph advance, in em, for mixed-case text
LINE_FACTOR = 1.24  # line height, as a multiple of the font size
BULLET_INDENT_IN = 0.28


# --------------------------------------------------------------------------- #
# Measuring: a conservative estimate of how tall a piece of text will be.
# --------------------------------------------------------------------------- #


@beartype
def chars_per_line(size: float, width_in: float) -> int:
    """How many characters fit on one line at this font size and width."""
    return max(10, int(width_in * 72.0 / (CHARS_WIDE * size)))


@beartype
def text_height(text: str, size: float, width_in: float) -> float:
    """Estimated height (inches) of `text` wrapped to `width_in`."""
    per_line = chars_per_line(size, width_in)
    lines = 0
    for paragraph in text.split("\n"):
        lines += max(1, math.ceil(len(paragraph) / per_line))
    return lines * LINE_FACTOR * size / 72.0


@beartype
def image_aspect(root_relative: str) -> float:
    """Height/width ratio of an image asset."""
    with Image.open(ROOT / root_relative) as img:
        width, height = img.size
    return height / width


# --------------------------------------------------------------------------- #
# Low-level shape helpers.
# --------------------------------------------------------------------------- #


@beartype
def add_rect(slide: Slide, left: float, top: float, width: float, height: float,
             fill: RGBColor, outline: RGBColor | None = None, outline_pt: float = 0.75) -> None:
    """Add a filled (and optionally outlined) rectangle."""
    from pptx.enum.shapes import MSO_SHAPE

    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.adjustments[0] = 0.06
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if outline is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = outline
        shape.line.width = Pt(outline_pt)
    shape.shadow.inherit = False
    shape.text_frame.text = ""


@beartype
def add_line(slide: Slide, left: float, top: float, width: float, color: RGBColor,
             height_pt: float = 1.6) -> None:
    """Add a thin horizontal rule."""
    from pptx.enum.shapes import MSO_SHAPE

    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Pt(height_pt)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False


@beartype
def set_font_margins(frame: TextFrame, margin_in: float) -> None:
    """
    Zero (or set) the frame margins.

    `python-pptx` implements these as read/write properties, but the bundled
    stubs declare them read-only, hence the narrow pyright exceptions.
    """
    frame.margin_left = Inches(margin_in)  # pyright: ignore[reportAttributeAccessIssue]
    frame.margin_right = Inches(margin_in)  # pyright: ignore[reportAttributeAccessIssue]
    frame.margin_top = Inches(0)  # pyright: ignore[reportAttributeAccessIssue]
    frame.margin_bottom = Inches(0)  # pyright: ignore[reportAttributeAccessIssue]


@beartype
def add_text(
    slide: Slide,
    left: float,
    top: float,
    width: float,
    height: float,
    text: str,
    size: float,
    color: RGBColor = TEXT,
    bold: bool = False,
    italic: bool = False,
    font: str = FONT,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    spacing: float = 0.0,
    line_spacing: float = 1.06,
) -> None:
    """Add a single-paragraph, word-wrapped text box."""
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    set_font_margins(frame, 0.0)
    frame.vertical_anchor = MSO_ANCHOR.TOP
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.line_spacing = line_spacing
    paragraph.space_after = Pt(spacing)
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = font


@beartype
def add_paragraphs(slide: Slide, left: float, top: float, width: float, height: float,
                   items: list[str], size: float, spacing_pt: float = 5.0,
                   color: RGBColor = TEXT) -> None:
    """Add a multi-paragraph text box (used for bullet lists)."""
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    set_font_margins(frame, 0.0)
    for index, item in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.line_spacing = 1.08
        paragraph.space_after = Pt(spacing_pt)
        run = paragraph.add_run()
        run.text = item
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.name = FONT


# --------------------------------------------------------------------------- #
# Block heights (used to fit a slide without overlapping anything).
# --------------------------------------------------------------------------- #


@beartype
def bullet_lines(block: sc.Bullets, size: float, width: float) -> list[str]:
    """Prefix bullets with a marker, since python-pptx has no bullet API."""
    lines: list[str] = []
    for index, item in enumerate(block.items, start=1):
        marker = f"{index}.  " if block.ordered else "•  "
        lines.append(marker + sc.strip_md(item))
    return lines


@beartype
def bullet_list_height(block: sc.Bullets, size: float, width: float) -> float:
    total = 0.0
    for index, item in enumerate(block.items, start=1):
        prefix = f"{index}. " if block.ordered else "• "
        total += text_height(prefix + sc.strip_md(item), size, width - BULLET_INDENT_IN)
    total += 0.07 * max(0, len(block.items) - 1)
    return total


@beartype
def table_height(block: sc.Table, scale: float) -> float:
    """Table height: a taller header row plus one row per record."""
    rows = (0.38 + 0.245 * len(block.rows)) * scale
    caption = 0.26 if block.caption else 0.0
    return rows + caption


@beartype
def picture_height(block: sc.Picture, width: float, scale: float) -> float:
    caption = 0.28 if block.caption else 0.0
    return width * min(block.width, 1.0) * scale * image_aspect(block.path) + caption


@beartype
def player_height(count: int, scale: float) -> float:
    """Height of an audio group: one compact play-card per clip plus a label."""
    return (0.26 + count * 0.66) * scale


# --------------------------------------------------------------------------- #
# Slide construction.
# --------------------------------------------------------------------------- #


@beartype
def blank_slide(prs: PresentationType) -> Slide:
    return cast(Slide, prs.slides.add_slide(prs.slide_layouts[6]))


@beartype
def add_header(slide: Slide, slide_meta: sc.Slide, index: int, total: int) -> float:
    """Draw the eyebrow / title / subtitle and return the body top (inches)."""
    add_rect(slide, MARGIN, 0.30, 0.26, 0.055, BLUE)
    add_text(slide, MARGIN, 0.38, CONTENT_W - 2.0, 0.26,
             sc.strip_md(slide_meta.eyebrow).upper(), SIZE_EYEBROW, MUTED)
    add_text(slide, SLIDE_W - MARGIN - 1.9, 0.38, 1.9, 0.26,
             f"{index:02d} / {total:02d}", SIZE_EYEBROW, MUTED, font=MONO, align=PP_ALIGN.RIGHT)

    add_text(slide, MARGIN, 0.70, CONTENT_W, 0.52,
             sc.strip_md(slide_meta.title), SIZE_TITLE, NAVY, bold=True, line_spacing=1.0)

    subtitle_top = 1.26
    if slide_meta.required:
        add_text(slide, MARGIN, subtitle_top - 0.02, CONTENT_W, 0.24,
                 "✓  " + slide_meta.required + "  (required criterion)", 10.5, GREEN, bold=True)
        subtitle_top += 0.26

    add_text(slide, MARGIN, subtitle_top, CONTENT_W, 0.4,
             sc.strip_md(slide_meta.subtitle), SIZE_SUBTITLE, MUTED)
    body_top = subtitle_top + text_height(sc.strip_md(slide_meta.subtitle), SIZE_SUBTITLE, CONTENT_W) + 0.20
    add_line(slide, MARGIN, body_top - 0.10, CONTENT_W, LINE)
    return max(body_top, BODY_TOP)


@beartype
def draw_para(slide: Slide, block: sc.Para, left: float, top: float, width: float, size: float) -> float:
    height = text_height(sc.strip_md(block.text), size, width)
    add_text(slide, left, top, width, height, sc.strip_md(block.text), size, TEXT)
    return height


@beartype
def draw_bullets(slide: Slide, block: sc.Bullets, left: float, top: float, width: float,
                 size: float) -> float:
    items = bullet_lines(block, size, width)
    height = bullet_list_height(block, size, width)
    add_paragraphs(slide, left, top, width, height, items, size, spacing_pt=5.0)
    return height


@beartype
def draw_callout(slide: Slide, block: sc.Callout, left: float, top: float, width: float,
                 size: float) -> float:
    inner = width - 0.60
    body = text_height(sc.strip_md(block.text), size, inner)
    height = body + 0.42
    add_rect(slide, left, top, width, height, TINT, outline=CALLOUT_LINE)
    add_rect(slide, left + 0.01, top + 0.10, 0.05, height - 0.20, BLUE)
    add_text(slide, left + 0.30, top + 0.21, inner, body, sc.strip_md(block.text), size, TEXT)
    return height


@beartype
def draw_picture(slide: Slide, block: sc.Picture, left: float, top: float, width: float,
                 height: float) -> float:
    """Place an image, scaled to fit the box while keeping its aspect ratio."""
    aspect = image_aspect(block.path)
    caption_h = 0.28 if block.caption else 0.0
    limit = width * min(block.width, 1.0)
    box_h = max(0.2, height - caption_h)
    draw_w = min(limit, box_h / aspect)
    draw_h = draw_w * aspect
    if draw_h > box_h:
        draw_h = box_h
        draw_w = draw_h / aspect
    x = left + (width - draw_w) / 2.0
    slide.shapes.add_picture(block.path if Path(block.path).is_absolute() else str(ROOT / block.path),
                             Inches(x), Inches(top), Inches(draw_w), Inches(draw_h))
    used = draw_h
    if block.caption:
        caption_top = top + draw_h + 0.05
        add_text(slide, left, caption_top, width, 0.22, sc.strip_md(block.caption),
                 SIZE_CAPTION, MUTED, align=PP_ALIGN.CENTER)
        used += caption_h
    return used


@beartype
def apply_column_widths(table: object, widths: list[float]) -> None:
    """
    Set each column width.

    `python-pptx` has no typed accessor for the column collection, so this small
    wrapper is the single place that touches it.
    """
    columns = table.columns  # type: ignore[attr-defined]
    for index, width in enumerate(widths):
        columns[index].width = Inches(width)


@beartype
def draw_table(slide: Slide, block: sc.Table, left: float, top: float, width: float,
               scale: float) -> float:
    headers = list(block.headers)
    rows = [list(row) for row in block.rows]
    header_h = 0.38 * scale
    body_h = 0.245 * scale
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(left), Inches(top),
                                   Inches(width), Inches(header_h + body_h * len(rows)))
    table = shape.table
    table.first_row = True
    table.horz_banding = False

    # Give the first column a bit more room and spread the rest evenly.
    first = max(1.55, width * 0.20)
    other = (width - first) / max(1, len(headers) - 1)
    apply_column_widths(table, [first if i == 0 else other for i in range(len(headers))])

    table.rows[0].height = Inches(header_h)
    for r in range(1, len(rows) + 1):
        table.rows[r].height = Inches(body_h)

    size = SIZE_TABLE * scale
    for column, header in enumerate(headers):
        set_cell(table, 0, column, header, size - 1.5, MUTED, bold=True, fill=PAPER)
    for r, row in enumerate(rows, start=1):
        for column, text in enumerate(row):
            bold = column == 0
            color = NAVY if column == 0 else TEXT
            set_cell(table, r, column, text, size, color, bold=bold, fill=WHITE)

    used = header_h + body_h * len(rows)
    if block.caption:
        add_text(slide, left, top + used + 0.05, width, 0.22, sc.strip_md(block.caption),
                 SIZE_CAPTION, MUTED)
        used += 0.26
    return used


@beartype
def set_cell(table: Table, row: int, column: int, text: str, size: float, color: RGBColor,
             bold: bool, fill: RGBColor) -> None:
    """Style one table cell (python-pptx exposes no high-level helper for this)."""
    cell = table.cell(row, column)
    frame = cell.text_frame
    frame.word_wrap = True
    frame.margin_left = Inches(0.08)  # pyright: ignore[reportAttributeAccessIssue]
    frame.margin_right = Inches(0.06)  # pyright: ignore[reportAttributeAccessIssue]
    frame.margin_top = Inches(0.02)  # pyright: ignore[reportAttributeAccessIssue]
    frame.margin_bottom = Inches(0.02)  # pyright: ignore[reportAttributeAccessIssue]
    paragraph = frame.paragraphs[0]
    paragraph.line_spacing = 1.0
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = FONT
    cell.fill.solid()  # type: ignore[no-untyped-call]
    cell.fill.fore_color.rgb = fill


@beartype
def draw_audio(slide: Slide, block: sc.Audio, left: float, top: float, width: float,
               height: float) -> float:
    """Embed each clip as an mp4 (waveform poster + sound) inside a card."""
    count = len(block.clips)
    label_h = 0.24 if block.caption else 0.0
    row_h = (height - label_h) / max(1, count)
    wave_h = max(0.28, row_h - 0.24)  # leave room for the label line inside the card
    card_w = width  # cards span the full body width, so nothing overflows sideways
    x = left

    if block.caption:
        add_text(slide, left, top, width, label_h, sc.strip_md(block.caption),
                 11.5, NAVY, bold=True)

    y = top + label_h
    for slug in block.clips:
        clip = sc.AUDIO_BY_SLUG[slug]
        add_rect(slide, x, y, card_w, row_h - 0.06, WHITE, outline=LINE)
        slide.shapes.add_movie(
            str(ROOT / clip.mp4),
            Inches(x + 0.08), Inches(y + 0.04), Inches(card_w - 0.16), Inches(wave_h),
            poster_frame_image=str(ROOT / clip.wave),
            mime_type="video/mp4",
        )
        add_text(slide, x + 0.16, y + wave_h + 0.03, card_w - 0.32, 0.18,
                 f"{clip.label} — {sc.strip_md(clip.caption)}", SIZE_CAPTION, MUTED)
        y += row_h
    return height


# --------------------------------------------------------------------------- #
# Fitting the blocks into the body area.
# --------------------------------------------------------------------------- #


@beartype
def natural_height(block: sc.Block, width: float, scale: float) -> float:
    """Desired height of a block at the given scale."""
    if isinstance(block, sc.Para):
        return text_height(sc.strip_md(block.text), SIZE_PARA * scale, width) + 0.14
    if isinstance(block, sc.Bullets):
        return bullet_list_height(block, SIZE_BULLET * scale, width) + 0.16
    if isinstance(block, sc.Callout):
        return text_height(sc.strip_md(block.text), SIZE_CALLOUT * scale, width - 0.60) + 0.56
    if isinstance(block, sc.Table):
        return table_height(block, scale) + 0.16
    if isinstance(block, sc.Picture):
        return picture_height(block, width, scale) + 0.30
    # `sc.Block` has no members left, so `block` is an `Audio` here.
    return player_height(len(block.clips), scale) + 0.18


@beartype
def is_visual(block: sc.Block) -> bool:
    return isinstance(block, (sc.Picture, sc.Audio))


@beartype
def layout(
    blocks: list[sc.Block], width: float, available: float
) -> tuple[list[tuple[sc.Block, float, float]], float]:
    """
    Fit the blocks, shrinking everything uniformly until they all fit.

    Returns `(rows, scale)` where each row is `(block, height, gap_above)`; the
    scale is reused by the draw calls so the rendered sizes match the fit.
    """
    if not blocks:
        return [], 1.0

    gap_unit = 0.16
    gaps = gap_unit * max(0, len(blocks) - 1)
    minimum = 0.30
    scale = 1.0

    for _ in range(60):
        total = sum(natural_height(b, width, scale) for b in blocks)
        if total + gaps <= available or scale <= minimum:
            break
        # Solve for the next scale, then damp it a little to avoid oscillation.
        ratio = (total + gaps) / max(available, 0.1)
        scale = max(minimum, scale * (1.0 / ratio) ** 0.98)

    heights = [natural_height(b, width, scale) for b in blocks]
    used = sum(heights) + gaps
    slack = max(0.0, available - used)
    extra_gap = min(0.28, slack / max(1, len(blocks)))

    rows: list[tuple[sc.Block, float, float]] = []
    for index, block in enumerate(blocks):
        gap = 0.0 if index == 0 else gap_unit + extra_gap
        rows.append((block, heights[index], gap))
    return rows, scale


@beartype
def render_slide(prs: PresentationType, slide_meta: sc.Slide, index: int, total: int) -> None:
    slide = blank_slide(prs)
    add_rect(slide, 0.0, 0.0, SLIDE_W, SLIDE_H, WHITE)
    body_top = add_header(slide, slide_meta, index, total)

    blocks = list(slide_meta.blocks)
    available = BODY_BOTTOM - body_top
    fitted, scale = layout(blocks, CONTENT_W, available)

    y = body_top
    for block, height, gap in fitted:
        y += gap
        if isinstance(block, sc.Para):
            used = draw_para(slide, block, MARGIN, y, CONTENT_W, SIZE_PARA * scale)
        elif isinstance(block, sc.Bullets):
            used = draw_bullets(slide, block, MARGIN, y, CONTENT_W, SIZE_BULLET * scale)
        elif isinstance(block, sc.Callout):
            used = draw_callout(slide, block, MARGIN, y, CONTENT_W, SIZE_CALLOUT * scale)
        elif isinstance(block, sc.Table):
            used = draw_table(slide, block, MARGIN, y, CONTENT_W, scale)
        elif isinstance(block, sc.Picture):
            used = draw_picture(slide, block, MARGIN, y, CONTENT_W, height)
        else:
            # `sc.Block` has no members left, so `block` is an `Audio` here.
            used = draw_audio(slide, block, MARGIN, y, CONTENT_W, height)
        y += max(used, 0.1)

    add_line(slide, MARGIN, SLIDE_H - 0.34, CONTENT_W, LINE)
    add_text(slide, MARGIN, SLIDE_H - 0.27, CONTENT_W, 0.22,
             "FlowDec · Welker et al., ICLR 2025 · AMLT ITMO 2026 course project", SIZE_CAPTION, MUTED)
    if slide_meta.notes:
        notes_frame = slide.notes_slide.notes_text_frame
        if notes_frame is not None:
            notes_frame.text = sc.strip_md(slide_meta.notes)


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #


@beartype
def build() -> PresentationType:
    slides = sc.build_slides()
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    for index, slide_meta in enumerate(slides, start=1):
        render_slide(prs, slide_meta, index, len(slides))
    return prs


@beartype
def verify() -> None:
    """Fail loudly if a referenced asset is missing."""
    missing: list[str] = []
    for slide_meta in sc.build_slides():
        for block in slide_meta.blocks:
            if isinstance(block, sc.Picture) and not (ROOT / block.path).is_file():
                missing.append(block.path)
            if isinstance(block, sc.Audio):
                for slug in block.clips:
                    clip = sc.AUDIO_BY_SLUG[slug]
                    for path in (clip.mp4, clip.wave):
                        if not (ROOT / path).is_file():
                            missing.append(path)
    if missing:
        raise SystemExit(
            "missing assets (run scripts/build_slide_assets.py first):\n  " + "\n  ".join(sorted(set(missing)))
        )


@beartype
def main() -> None:
    verify()
    prs = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    slide_count = len(prs.slides._sldIdLst)  # pyright: ignore[reportPrivateUsage]
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {slide_count} slides)")


if __name__ == "__main__":
    main()
