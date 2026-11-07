"""Precise, in-place edits to the combined Word report, using the repository's docx code (extensions/meridian-docs), called directly.

  ...pixi python: .pixi\\envs\\default\\python.exe -I -X utf8 scripts\\viz\\apply_docx_edits.py

Each edit changes only the matched text inside its paragraph; the runs around it keep their bold and code formatting. The document is
loaded and saved with the repository's stdlib docx functions (_load_docx_xml_stdlib, _save_docx_xml_stdlib), written to a temporary copy
and then swapped in. Every edit must match exactly once, or the script stops and changes nothing.

Edits: the scoring function's default is the 57 submission (decided 19 October), not the 58 comparison, and the 13-input wording.
"""
import os
import sys
import xml.etree.ElementTree as ET

sys.stdout.reconfigure(encoding="utf-8")
REPO_EXT = r"C:\Users\<user>\Documents\Meridian\deploy-main-20261005\extensions\meridian-docs"
sys.path.insert(0, REPO_EXT)
from meridian_docs import docs_intel as DI  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOCX = os.path.join(ROOT, "deliverables", "word", "CS2_win_probability_full_document_2026-10-19.docx")
TMP = DOCX + ".edit_tmp.docx"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

# (kind, old text, new text). kind: "replace" = substring inside one paragraph; "exact" = whole paragraph text equals old, replace text;
# "delete" = remove the whole paragraph that contains old.
EDITS = [
    ("replace", "which is the served default. It is not the submission.",
     "Until 19 October it was the served default; since then the scoring function's default is the submission (57 inputs), and the comparison is passed by name."),
    ("replace", "the default serving model needs the status channel,",
     "the default serving model (the submission) needs the status channel,"),
    ("replace", "It is the model the serving function uses by default.",
     "It was the model the serving function used by default until 19 October; the function's default is now the 57-input submission."),
    ("replace", "Serving: scripts/serving/win_probability.py (default).",
     "Serving: scripts/serving/win_probability.py. The check below was run before the default changed on 19 October."),
    ("replace", "Production: five-seed ensemble", "Score-lead comparison: five-seed ensemble"),
    ("exact", "served by default", "served by default until 19 October"),
    ("replace", "Production model served by default (",
     "Production model served by default until 19 October; the default is now winprob_v6_drop_heavy (see E63) ("),
    ("replace", "Not set as served (served: false). Serving parity checked on 294 October test matches (below).",
     "The default of win_probability.py. Its metadata says served: false. Serving parity checked on 294 October test matches (below)."),
    ("replace", "The default in win_probability.py.", 'Passed by name: model_name="winprob_v8_ensemble".'),
    ("replace", "The default model_name is still winprob_v8_ensemble, the 58-input comparison; pass \"winprob_v6_drop_heavy\" for the submission.",
     "The default model_name is winprob_v6_drop_heavy, the submission; pass \"winprob_v8_ensemble\" for the 58-input comparison."),
    ("delete", "The default model in the function is still the 58-input comparison.", None),
    ("replace", "The default model is the production ensemble. It needs the per-tick player_status channel in the tome.",
     "The default model is the submission, winprob_v6_drop_heavy (57 inputs). It needs the per-tick player_status channel in the tome."),
    ("replace", "the production metadata says",
     "the score-lead comparison, winprob_v8_ensemble, is passed by name and its metadata says"),
]


def para_text(p):
    return "".join(t.text or "" for t in p.iter(W + "t"))


def runs(p):
    return [c for c in p if c.tag == W + "r"]


def run_text(r):
    return "".join(t.text or "" for t in r.findall(W + "t"))


def set_run_text(r, text):
    ts = r.findall(W + "t")
    for t in ts[1:]:
        r.remove(t)
    if ts:
        t = ts[0]
    else:
        t = ET.SubElement(r, W + "t")
    t.text = text
    t.set(XML_SPACE, "preserve")


def replace_in_para(p, old, new):
    rs = runs(p)
    texts = [run_text(r) for r in rs]
    full = "".join(texts)
    if full.count(old) != 1:
        raise ValueError(f"'{old[:60]}' matches {full.count(old)} times in a paragraph")
    start = full.index(old)
    end = start + len(old)
    segs, pos = [], 0
    for i, tx in enumerate(texts):
        a, b = pos, pos + len(tx)
        if b > start and a < end:
            segs.append((i, max(start, a) - a, min(end, b) - a))
        pos = b
    first_i, last_i = segs[0][0], segs[-1][0]
    first_run, last_run = rs[first_i], rs[last_i]
    new_elems = []
    prefix = texts[first_i][:segs[0][1]]
    suffix = texts[last_i][segs[-1][2]:]
    if prefix:
        r = ET.fromstring(ET.tostring(first_run))
        set_run_text(r, prefix)
        new_elems.append(r)
    nr = ET.fromstring(ET.tostring(first_run))
    set_run_text(nr, new)
    new_elems.append(nr)
    if suffix:
        r = ET.fromstring(ET.tostring(last_run))
        set_run_text(r, suffix)
        new_elems.append(r)
    idx = list(p).index(first_run)
    for r in rs[first_i:last_i + 1]:
        p.remove(r)
    for k, e in enumerate(new_elems):
        p.insert(idx + k, e)


def find_paras(root, kind, old):
    out = []
    for p in root.iter(W + "p"):
        t = para_text(p)
        if (kind == "exact" and t == old) or (kind != "exact" and old in t):
            out.append(p)
    if len(out) != 1:
        raise ValueError(f"{kind} '{old[:60]}': {len(out)} matching paragraphs (need exactly 1)")
    return out[0]


def main():
    raw, root = DI._load_docx_xml_stdlib(DOCX)
    before = {"bold_runs": len(root.findall(".//" + W + "b"))}
    parent_of = {c: p for p in root.iter() for c in p}
    for kind, old, new in EDITS:
        if kind == "delete":
            p = find_paras(root, "replace", old)
            parent_of[p].remove(p)
        elif kind == "exact":
            p = find_paras(root, "exact", old)
            replace_in_para(p, old, new)
        else:
            p = find_paras(root, "replace", old)
            replace_in_para(p, old, new)
        print("ok:", kind, "|", old[:70].replace("\n", " "))
    res = DI._save_docx_xml_stdlib(raw, root, TMP)
    os.replace(TMP, DOCX)
    texts = " ".join(r["text"] for r in DI.parse_docx(DOCX))
    missing = [new[:50] for _, _, new in EDITS if new and new[:40] not in texts]
    still = [old[:50] for kind, old, _ in EDITS if kind == "replace" and old in texts]
    print("paragraphs:", len(DI.parse_docx(DOCX)), "| bold runs before:", before["bold_runs"],
          "| after:", len(ET.fromstring(zipfile_read(DOCX)).findall(".//" + W + "b")))
    print("new text missing:", missing or "none")
    print("old text still present:", still or "none")
    print("save result keys:", sorted(res)[:6] if isinstance(res, dict) else type(res))


def zipfile_read(path):
    import zipfile
    with zipfile.ZipFile(path) as z:
        return z.read("word/document.xml")


if __name__ == "__main__":
    main()
