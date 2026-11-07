"""Export every chart in the deck to PNG, for the PowerPoint and the Word document.

  ...python.exe -I -X utf8 scripts\\viz\\build_figures.py

Each <svg> in deliverables/deck_2026-10-19/project/slides/*.html is written out as an SVG (canvas widened on the right so the value
labels are not cut off) and converted to PNG with LibreOffice headless. Writes deliverables/presentation/figs/<slide>_<n>.png.
"""
import glob
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SL = os.path.join(ROOT, "deliverables", "deck_2026-10-19", "project", "slides")
FIG = os.path.join(ROOT, "deliverables", "presentation", "figs")
SOFFICE = r"C:\Program Files\LibreOffice\program\soffice.exe"
EXTRA = 120  # px added to the right of every chart, for the value labels


def widen(svg):
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    if not m:
        return svg
    w, h = float(m.group(1)), float(m.group(2))
    svg = svg.replace(m.group(0), f'viewBox="0 0 {w + EXTRA:.0f} {h:.0f}"', 1)
    svg = re.sub(r'<svg([^>]*?)width="[\d.]+"', lambda mm: f'<svg{mm.group(1)}width="{w + EXTRA:.0f}"', svg, count=1)
    if "xmlns=" not in svg[:200]:
        svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    return svg


def main():
    os.makedirs(FIG, exist_ok=True)
    for f in glob.glob(os.path.join(FIG, "*")):
        os.remove(f)
    made = []
    for path in sorted(glob.glob(os.path.join(SL, "*.html"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        src = open(path, encoding="utf-8").read()
        for i, svg in enumerate(re.findall(r"<svg.*?</svg>", src, re.S), start=1):
            name = f"{stem}_{i}"
            svg_path = os.path.join(FIG, name + ".svg")
            open(svg_path, "w", encoding="utf-8", newline="\n").write(widen(svg))
            subprocess.run([SOFFICE, "--headless", "--convert-to", "png", "--outdir", FIG, svg_path],
                           capture_output=True, text=True, timeout=300)
            png = os.path.join(FIG, name + ".png")
            if os.path.exists(png):
                made.append(name)
            else:
                print("FAILED", name)
    print("wrote", len(made), "PNG charts:", ", ".join(made))


if __name__ == "__main__":
    main()
