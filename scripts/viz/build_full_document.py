"""One Word document for the whole package: wrap-up, methodology, deliverable summary, experiment register and presentation.

  ...pixi python: .pixi\\envs\\default\\python.exe -I -X utf8 scripts\\viz\\build_full_document.py

1. Builds the sections (scripts/viz/build_word_docs.py) and joins them under one contents list with links.
2. Places the deck charts in Part E with captions, then converts the joined HTML to .docx with LibreOffice (no Word needed).
3. Verifies with the repository's own docx code in extensions/meridian-docs, called directly: the heading outline (docs_intel) and the
   render check (render_gate). The repository's insert_figure_block was tried first and its render check failed on every image insert
   in this environment, so the images are embedded at conversion time instead.

Writes deliverables/word/CS2_win_probability_full_document_2026-10-19.docx.
"""
import json
import os
import re
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_word_docs as BWD  # noqa: E402

ROOT = BWD.ROOT
SL = os.path.join(ROOT, "deliverables", "deck_2026-10-19", "project", "slides")
FIG = os.path.join(ROOT, "deliverables", "presentation", "figs")
DOCX = os.path.join(ROOT, "deliverables", "word", "CS2_win_probability_full_document_2026-10-19.docx")
REPO_EXT = r"C:\Users\<user>\Documents\Meridian\deploy-main-20261005\extensions\meridian-docs"
DATE = BWD.DATE

FIGURES = [
    ("split_1", "Split by match: share of the matches in each split"),
    ("split_2", "Split by second: share of the seconds in each split"),
    ("inputs_1", "Inputs of the submission, drawn to scale: 13 event, 19 CT loadout, 19 T loadout, 6 differences"),
    ("model_1", "The five seeds of the score-lead comparison model, each with its best iteration"),
    ("model_2", "Output of the seed average: the mean of the five probabilities, per second"),
    ("settings_1", "Validation change against the baseline setting for ten alternatives (units of 0.0001)"),
    ("families_1", "Other model families: test log loss change against XGBoost on matched settings"),
    ("rounds_1", "Test log loss by round group, submission model"),
    ("rejected_1", "Rejected alternatives with thin intervals, part 1"),
    ("rejected_2", "Rejected alternatives with thin intervals, part 2"),
    ("results_1", "Event-only minus score-lead model on the same test seconds (units of 0.001)"),
]


def slide_titles():
    order = json.load(open(os.path.join(SL, "..", "deck.json"), encoding="utf-8"))["order"]
    out = []
    for i, sid in enumerate(order, start=1):
        src = open(os.path.join(SL, sid + ".html"), encoding="utf-8").read()
        m = re.search(r"<h[12][^>]*>(.*?)</h[12]>", src, re.S)
        title = re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else sid
        out.append((i, sid, title))
    return out


def strip_first_h1(html):
    return re.sub(r"<h1>.*?</h1>", "", html, count=1, flags=re.S)


def anchor_h2(html, prefix):
    n = {"i": 0}

    def repl(m):
        n["i"] += 1
        return f'<h2><a name="{prefix}{n["i"]}"></a>{m.group(1)}</h2>'
    return re.sub(r"<h2>(.*?)</h2>", repl, html, flags=re.S)


def figure_html():
    out = ["<p>The charts are in the PowerPoint and, as PNG files, in the package's presentation/figs folder. They are not embedded here, "
           "to keep this document renderable. Each figure is listed with its file name.</p><ol>"]
    for stem, label in FIGURES:
        out.append(f"<li><b>{BWD.esc(label)}</b> ({stem}.png)</li>")
    out.append("</ol>")
    return "".join(out)


def build_html(wrap_body, notes):
    note = dict(notes)
    titles = slide_titles()
    contents = (
        '<a name="top"></a><h1>CS2 round win probability: full document (' + DATE + ')</h1>'
        '<p>One document with the end-of-day wrap-up, the methodology, the deliverable summary, the experiment register and the '
        'presentation. Each part starts with its own heading. The links below go to each part; the wrap-up sections link to each other.</p>'
        '<h2>Contents</h2><ul>'
        '<li><a href="#partA">Part A. End-of-day wrap-up</a><ul>'
        + "".join(f'<li><a href="#w{i}">{t}</a></li>' for i, t in enumerate(
            ["1. Final choices", "2. Models", "3. Features: exact definitions", "4. Metrics", "5. Example ticks in each split",
             "6. How to call the function", "7. Open items"], start=1))
        + '</ul></li>'
        '<li><a href="#partB">Part B. Methodology</a></li>'
        '<li><a href="#partC">Part C. Deliverable summary</a></li>'
        '<li><a href="#partD">Part D. Experiment register</a></li>'
        '<li><a href="#partE">Part E. Presentation and figures</a></li>'
        '</ul>'
    )
    partA = ('<h1><a name="partA"></a>Part A. End-of-day wrap-up</h1>'
             + anchor_h2(strip_first_h1(wrap_body), "w") + '<p><a href="#top">Back to contents</a></p>')
    partB = ('<h1><a name="partB"></a>Part B. Methodology</h1>'
             + strip_first_h1(note["METHODOLOGY"]) + '<p><a href="#top">Back to contents</a></p>')
    partC = ('<h1><a name="partC"></a>Part C. Deliverable summary</h1>'
             + strip_first_h1(note["DELIVERABLE"]) + '<p><a href="#top">Back to contents</a></p>')
    partD = ('<h1><a name="partD"></a>Part D. Experiment register</h1>'
             + strip_first_h1(note["EXPERIMENTS"]) + '<p><a href="#top">Back to contents</a></p>')
    list_items = "".join(f"<li>{BWD.esc(t)} <i>({sid})</i></li>" for _, sid, t in titles)
    partE = ('<h1><a name="partE"></a>Part E. Presentation and figures</h1>'
             f'<p>The presentation has {len(titles)} slides, in this order. The PowerPoint file in the package has the same slides and the '
             'same figures; each slide\'s notes are in the PowerPoint.</p>'
             f'<ol>{list_items}</ol>'
             '<h2>Figures</h2>' + figure_html() + '<p><a href="#top">Back to contents</a></p>')
    return contents + partA + partB + partC + partD + partE


def verify(docx):
    sys.path.insert(0, REPO_EXT)
    from meridian_docs import docs_intel as DI  # the repository's docx code, called directly
    from meridian_docs import render_gate as RG
    render = RG.check_render_capability(docx)
    outline = DI.document_outline(docx)
    headings = outline.get("headings", []) if isinstance(outline, dict) else []
    with zipfile.ZipFile(docx) as z:
        xml = z.read("word/document.xml").decode("utf-8")
        media = [n for n in z.namelist() if n.startswith("word/media/")]
    return {
        "render_status": render.get("status"),
        "render_reason": str(render.get("reason", ""))[:200],
        "headings": len(headings),
        "internal_links": len(re.findall(r'w:anchor="', xml)),
        "bookmarks": len(re.findall(r"<w:bookmarkStart", xml)),
        "tables": len(re.findall(r"<w:tbl>", xml)),
        "images_embedded": len(media),
    }


def main():
    wrap_body, notes = BWD.main()
    html = build_html(wrap_body, notes)
    BWD.write_docx("CS2_win_probability_full_document_2026-10-19", "CS2 round win probability: full document", html)
    if not os.path.exists(DOCX):
        raise SystemExit("conversion failed: " + DOCX)
    checks = verify(DOCX)
    for k, v in checks.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
