"""
Font resolution for multilingual PDF exports (Legal English + Devanagari / Hindi).

Uses system fonts when available (Windows Nirmala / Noto on Linux).
Override with LEXFORGE_PDF_FONT and LEXFORGE_PDF_FONT_BOLD environment variables (paths to .ttf).
"""

from __future__ import annotations

import os

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

_LF_REGISTERED = False
FONT_BODY = "Helvetica"
FONT_BOLD = "Helvetica-Bold"


def _exists(p: str | None) -> bool:
    return bool(p and os.path.isfile(p))


def register_lexforge_pdf_fonts() -> None:
    """Register LexForgeSans / LexForgeSans-Bold for UTF-8 body text."""
    global _LF_REGISTERED, FONT_BODY, FONT_BOLD
    if _LF_REGISTERED:
        return

    env_r = os.environ.get("LEXFORGE_PDF_FONT")
    env_b = os.environ.get("LEXFORGE_PDF_FONT_BOLD")

    paths_regular: list[str | None] = [env_r]
    paths_bold: list[str | None] = [env_b]

    windir = os.environ.get("WINDIR") or os.environ.get("SystemRoot") or r"C:\Windows"
    fonts_dir = os.path.join(windir, "Fonts")

    paths_regular.extend([
        os.path.join(fonts_dir, "Nirmala.ttf"),
        os.path.join(fonts_dir, "NirmalaUI.ttf"),
        os.path.join(fonts_dir, "Mangal.ttf"),
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ])
    paths_bold.extend([
        os.path.join(fonts_dir, "NirmalaB.ttf"),
        os.path.join(fonts_dir, "NirmalaUI-Bold.ttf"),
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ])

    reg = next((p for p in paths_regular if _exists(p)), None)
    bld = next((p for p in paths_bold if _exists(p)), None)

    if reg:
        pdfmetrics.registerFont(TTFont("LexForgeSans", reg))
        FONT_BODY = "LexForgeSans"
        if bld:
            pdfmetrics.registerFont(TTFont("LexForgeSans-Bold", bld))
            FONT_BOLD = "LexForgeSans-Bold"
        else:
            FONT_BOLD = FONT_BODY

    _LF_REGISTERED = True
