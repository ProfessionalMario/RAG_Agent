"""
Render terminal-style PNG screenshots from real CLI output.

Run:  python docs/render_demos.py

Outputs go to `docs/img/`. The script invokes the actual CLI, captures the
output, and rasterises it onto a dark "terminal" canvas via Pillow so the
README has authentic, deterministic screenshots that always match what the
user will see when they run the project themselves.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "docs" / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Authentic "terminal" palette
BG = (24, 26, 30)
FG = (220, 223, 228)
ACCENT = (130, 200, 255)
TITLE_BAR = (40, 44, 52)
DOT_RED = (255, 95, 86)
DOT_YEL = (255, 189, 46)
DOT_GRN = (39, 201, 63)


def _font(size: int = 14) -> ImageFont.FreeTypeFont:
    """Try a monospace font, fall back to default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/nix/store/*/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    import glob
    for pat in candidates:
        for p in glob.glob(pat):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _render(text: str, out_path: Path, title: str = "rag_agent",
            width: int = 980, padding: int = 18) -> None:
    """Rasterise `text` onto a dark terminal canvas and save to `out_path`."""
    font = _font(14)
    lines = text.expandtabs(4).splitlines() or [""]

    # Measure
    bbox = font.getbbox("Mg")
    line_h = (bbox[3] - bbox[1]) + 4
    title_h = 28
    height = title_h + padding * 2 + line_h * len(lines)

    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    # Title bar
    draw.rectangle([(0, 0), (width, title_h)], fill=TITLE_BAR)
    cy = title_h // 2
    for i, color in enumerate((DOT_RED, DOT_YEL, DOT_GRN)):
        draw.ellipse([(14 + i * 22, cy - 7), (14 + i * 22 + 14, cy + 7)],
                     fill=color)
    draw.text((width // 2 - 60, cy - 8), title, fill=FG, font=font)

    # Body
    y = title_h + padding
    for line in lines:
        color = ACCENT if line.lstrip().startswith(("◈", "═", "║", "╔", "╚", ">")) else FG
        draw.text((padding, y), line, fill=color, font=font)
        y += line_h

    img.save(out_path)
    print(f"  -> {out_path.relative_to(ROOT)}")


def _capture(*args: str) -> str:
    """Run a CLI command (offline) and capture its stdout."""
    env = {**os.environ, "LD_PRELOAD": "/lib/x86_64-linux-gnu/libstdc++.so.6"}
    proc = subprocess.run(
        [sys.executable, "cli.py", *args],
        cwd=ROOT,
        env=env,
        capture_output=True, text=True, timeout=120,
    )
    return proc.stdout.strip()


def render_pipeline_demo() -> None:
    print("Rendering pipeline demo...")
    out = _capture("pipeline", "data/reports/sample_report.txt")
    # Trim spinner artefacts
    cleaned = "\n".join(
        l for l in out.splitlines()
        if not any(s in l for s in ("Processing", "Running RAG"))
    )
    _render(cleaned, IMG_DIR / "pipeline_demo.png",
            title="$ python cli.py pipeline ...")


def render_query_demo() -> None:
    print("Rendering query demo...")
    out = _capture("query", "How should I scale a numeric feature with outliers?")
    text = (
        "$ python cli.py query \"How should I scale a numeric feature with outliers?\"\n\n"
        + out
    )
    _render(text, IMG_DIR / "query_demo.png",
            title="$ python cli.py query ...")


def render_menu_demo() -> None:
    print("Rendering menu demo (synthetic)...")
    text = (
        "$ python cli.py\n\n"
        "████████████████████████████████████████\n"
        "   RAG-ML PREPROCESSING ENGINE   \n"
        "████████████████████████████████████████\n\n"
        "[1] Run pipeline (analyze EDA report)\n"
        "[2] Live query (ask the knowledge base)\n"
        "[3] Exit\n\n"
        "Selection: _"
    )
    _render(text, IMG_DIR / "menu_demo.png",
            title="$ python cli.py")


if __name__ == "__main__":
    render_menu_demo()
    render_pipeline_demo()
    render_query_demo()
    print("Done.")
