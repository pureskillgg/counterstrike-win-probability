"""PowerPoint version of the deck: the same slides, in the deck order, with the text, tables and charts; each slide's aside text goes in the
speaker notes.

  ...py -3.10 -I -X utf8 scripts\\viz\\build_pptx.py      (python-pptx is installed for Python 3.10)

Reads deliverables/deck_2026-10-19/project/slides/*.html (the source of the deck) and the chart PNGs from build_figures.py.
Writes deliverables/presentation/CS2_win_probability_presentation_2026-10-19.pptx.
"""
import json
import os
import struct
import sys
from html.parser import HTMLParser

sys.stdout.reconfigure(encoding="utf-8")
from pptx import Presentation  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SL = os.path.join(ROOT, "deliverables", "deck_2026-10-19", "project", "slides")
FIG = os.path.join(ROOT, "deliverables", "presentation", "figs")
OUT = os.path.join(ROOT, "deliverables", "presentation", "CS2_win_probability_presentation_2026-10-19.pptx")
DATE = "2026-10-19"


class SlideParser(HTMLParser):
    BLOCK = {"h1", "h2", "h3", "p", "li", "aside"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks = []
        self.kind = None
        self.buf = []
        self.in_svg = 0
        self.svg_n = 0
        self.table = None
        self.row = None
        self.cell = None

    def flush(self):
        if self.kind:
            text = " ".join("".join(self.buf).split())
            if text:
                self.blocks.append((self.kind, text))
        self.kind = None
        self.buf = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "svg":
            if self.in_svg == 0:
                self.flush()
                self.svg_n += 1
                self.blocks.append(("img", self.svg_n))
            self.in_svg += 1
            return
        if self.in_svg:
            return
        if tag == "table":
            self.flush()
            self.table = []
            return
        if tag == "tr" and self.table is not None:
            self.row = []
            return
        if tag in ("td", "th") and self.row is not None:
            self.cell = []
            return
        if tag in self.BLOCK:
            self.flush()
            kind = {"h1": "title", "h2": "title", "h3": "head", "aside": "note", "li": "item"}.get(tag, "para")
            if tag == "p" and "position:absolute" in (a.get("style") or "").replace(" ", ""):
                kind = "footer"
            self.kind = kind

    def handle_endtag(self, tag):
        if tag == "svg":
            self.in_svg -= 1
            return
        if self.in_svg:
            return
        if tag == "table" and self.table is not None:
            self.blocks.append(("table", self.table))
            self.table = None
            return
        if tag == "tr" and self.row is not None:
            self.table.append(self.row)
            self.row = None
            return
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
            return
        if tag in self.BLOCK:
            self.flush()

    def handle_data(self, data):
        if self.in_svg:
            return
        if self.cell is not None:
            self.cell.append(data)
            return
        if self.kind:
            self.buf.append(data)

    def close(self):
        super().close()
        self.flush()


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


def main():
    order = json.load(open(os.path.join(SL, "..", "deck.json"), encoding="utf-8"))["order"]
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    n_images = 0
    for i, sid in enumerate(order, start=1):
        parser = SlideParser()
        parser.feed(open(os.path.join(SL, sid + ".html"), encoding="utf-8").read())
        parser.close()
        blocks = parser.blocks
        slide = prs.slides.add_slide(blank)
        title = next((t for k, t in blocks if k == "title"), sid)
        tb = slide.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12.1), Inches(0.9))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.text = title
        tf.paragraphs[0].runs[0].font.size = Pt(30)
        tf.paragraphs[0].runs[0].font.bold = True

        images = [k for k, _ in blocks if k == "img"]
        image_ids = [v for k, v in blocks if k == "img"]
        has_img = bool(image_ids)
        text_w = 6.4 if has_img else 12.1
        lines = []
        for kind, text in blocks:
            if kind in ("para", "item", "head"):
                lines.append(("• " if kind == "item" else "") + text)
        body = slide.shapes.add_textbox(Inches(0.6), Inches(1.35), Inches(text_w), Inches(5.6))
        btf = body.text_frame
        btf.word_wrap = True
        for j, line in enumerate(lines):
            p = btf.paragraphs[0] if j == 0 else btf.add_paragraph()
            p.text = line
            p.space_after = Pt(6)
            for r in p.runs:
                r.font.size = Pt(16 if len(lines) < 9 else 14)
        chars_per_line = 62 if has_img else 120
        est_lines = sum(1 + len(s) // chars_per_line for s in lines)
        table_y = 1.4 + min(5.0, 0.42 * est_lines + 0.2)
        for kind, data in blocks:
            if kind != "table":
                continue
            rows, cols = len(data), max(len(r) for r in data)
            width = text_w if has_img else 12.1
            shape = slide.shapes.add_table(rows, cols, Inches(0.6), Inches(table_y), Inches(width), Inches(0.3 * rows))
            for r_i, row in enumerate(data):
                for c_i in range(cols):
                    cell = shape.table.cell(r_i, c_i)
                    cell.text = row[c_i] if c_i < len(row) else ""
                    for p in cell.text_frame.paragraphs:
                        for run in p.runs:
                            run.font.size = Pt(11)
                            run.font.bold = r_i == 0
            table_y += 0.3 * rows + 0.2
        y = 1.4
        for n in image_ids:
            png = os.path.join(FIG, f"{sid}_{n}.png")
            if not os.path.exists(png):
                continue
            w_px, h_px = png_size(png)
            width_in = 5.6
            height_in = width_in * h_px / w_px
            if height_in > 5.3:
                height_in = 5.3
                width_in = height_in * w_px / h_px
            slide.shapes.add_picture(png, Inches(12.9 - width_in), Inches(y), Inches(width_in), Inches(height_in))
            y += height_in + 0.2
            n_images += 1
        footer = next((t for k, t in blocks if k == "footer"), "")
        if footer:
            fb = slide.shapes.add_textbox(Inches(0.6), Inches(6.95), Inches(12.1), Inches(0.4))
            fb.text_frame.text = footer
            fb.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
        note = next((t for k, t in blocks if k == "note"), "")
        if note:
            slide.notes_slide.notes_text_frame.text = note
    prs.save(OUT)
    print("wrote", OUT, "| slides", len(order), "| chart images", n_images)


if __name__ == "__main__":
    main()
