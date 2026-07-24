"""Shared text -> image rendering for the optical compression POC.

Renders text into square page images using a monospace font, paginating
as needed. Capacity math (chars per line / lines per page) assumes a
FIXED-WIDTH font. A proportional font will silently break the estimate
(see plan.md "Known gotchas").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# DejaVu Sans Mono ships on most Linux systems (fonts-dejavu-core).
DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

MARGIN_PX = 24
LINE_SPACING = 1.15


@dataclass(frozen=True)
class PageLayout:
    page_px: int
    font_size: int
    char_w: float
    line_h: float
    cols: int
    rows: int
    chars_per_page: int


def measure_layout(page_px: int, font_size: int, font_path: str = DEFAULT_FONT_PATH) -> PageLayout:
    font = ImageFont.truetype(font_path, font_size)
    # Monospace: measure a wide run of 'M' and divide, rather than trusting
    # a single-glyph bbox (hinting can round single chars unevenly).
    probe = "M" * 40
    bbox = font.getbbox(probe)
    char_w = (bbox[2] - bbox[0]) / len(probe)
    line_h = font.size * LINE_SPACING

    usable = page_px - 2 * MARGIN_PX
    cols = max(1, int(usable / char_w))
    rows = max(1, int(usable / line_h))
    return PageLayout(
        page_px=page_px,
        font_size=font_size,
        char_w=char_w,
        line_h=line_h,
        cols=cols,
        rows=rows,
        chars_per_page=cols * rows,
    )


def paginate(text: str, layout: PageLayout) -> list[str]:
    """Greedy word-wrap into fixed-width lines, then slice into pages."""
    import textwrap

    wrapped: list[str] = []
    for raw_line in text.split("\n"):
        if raw_line == "":
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(raw_line, width=layout.cols, break_long_words=True) or [""])

    pages = []
    for i in range(0, len(wrapped), layout.rows):
        pages.append("\n".join(wrapped[i : i + layout.rows]))
    return pages or [""]


def render_page(page_text: str, layout: PageLayout, font_path: str = DEFAULT_FONT_PATH) -> Image.Image:
    img = Image.new("L", (layout.page_px, layout.page_px), color=255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, layout.font_size)
    y = MARGIN_PX
    for line in page_text.split("\n"):
        draw.text((MARGIN_PX, y), line, font=font, fill=0)
        y += layout.line_h
    return img


def render_text_to_pages(
    text: str,
    page_px: int,
    font_size: int,
    font_path: str = DEFAULT_FONT_PATH,
) -> tuple[list[Image.Image], PageLayout]:
    layout = measure_layout(page_px, font_size, font_path)
    page_texts = paginate(text, layout)
    images = [render_page(pt, layout, font_path) for pt in page_texts]
    return images, layout


def save_pages(images: list[Image.Image], out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, img in enumerate(images):
        p = out_dir / f"{stem}_p{i:02d}.png"
        img.save(p)
        paths.append(p)
    return paths
