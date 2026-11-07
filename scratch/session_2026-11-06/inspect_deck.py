"""Read-only inspection of the existing short deck: slide size, layouts, shapes, text, theme fonts/colours."""
import sys, zipfile, re
from pathlib import Path
from pptx import Presentation
from pptx.util import Emu

SRC = Path(r"C:/Users/13144/Downloads/CS2 Win-Probability Short Version.pptx")
prs = Presentation(str(SRC))
print("slide size (in):", round(Emu(prs.slide_width).inches, 2), "x", round(Emu(prs.slide_height).inches, 2))
print("slides:", len(prs.slides))
print("layouts:")
for i, lay in enumerate(prs.slide_layouts):
    print(f"  [{i}] {lay.name} placeholders={[ph.placeholder_format.type for ph in lay.placeholders]}")

for n, slide in enumerate(prs.slides, 1):
    print(f"\n=== slide {n} layout={slide.slide_layout.name}")
    for shp in slide.shapes:
        kind = shp.shape_type
        pos = f"x={Emu(shp.left).inches:.2f} y={Emu(shp.top).inches:.2f} w={Emu(shp.width).inches:.2f} h={Emu(shp.height).inches:.2f}" if shp.left is not None else ""
        line = f"  - {shp.name} type={kind} {pos}"
        if shp.has_text_frame and shp.text_frame.text.strip():
            fonts = set()
            for p in shp.text_frame.paragraphs:
                for r in p.runs:
                    fonts.add((r.font.name, r.font.size.pt if r.font.size else None, r.font.bold))
            line += f"\n      text: {shp.text_frame.text[:300]!r}\n      runs: {sorted(fonts, key=str)[:6]}"
        if getattr(shp, "has_chart", False) and shp.has_chart:
            line += f"\n      chart: {shp.chart.chart_type}"
        if getattr(shp, "has_table", False) and shp.has_table:
            line += f"\n      table {len(shp.table.rows)}x{len(shp.table.columns)}"
        print(line)
    if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
        print("  notes:", slide.notes_slide.notes_text_frame.text[:300])

# theme colours and fonts
with zipfile.ZipFile(SRC) as z:
    names = z.namelist()
    themes = [n for n in names if n.startswith("ppt/theme/") and n.endswith(".xml")]
    print("\nthemes:", themes)
    if themes:
        x = z.read(themes[0]).decode("utf8", "ignore")
        print("colours:", re.findall(r'<a:(dk1|lt1|dk2|lt2|accent\d|hlink)>.*?val="([0-9A-Fa-f]{6}|\w+)"', x)[:12])
        print("fonts:", re.findall(r'<a:(majorFont|minorFont)><a:latin typeface="([^"]*)"', x))
    print("media:", [n for n in names if n.startswith("ppt/media/")][:20])
