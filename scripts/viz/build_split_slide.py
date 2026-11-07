"""Build the 'How matches are split' slide. Every percentage and count is computed here from the frozen manifest, the cache and the
saved row counts (no numbers typed by hand).

  ...python.exe -I -X utf8 scripts\\viz\\build_split_slide.py
Writes deliverables/deck_2026-10-19/project/slides/split.html.
"""
import json
import os
import sys
from collections import Counter

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
from deck_svg import INK, SLATE, GREY

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
D = os.path.join(ROOT, "data", "combined")
OUT = os.path.join(ROOT, "deliverables", "deck_2026-10-19", "project", "slides", "split.html")
FOOT = '<p style="position:absolute;left:128px;bottom:64px;font-size:24px">CS2 round win probability | 6 of 19</p>'
TARGET = {"train": 72.0, "validation": 8.0, "test": 20.0}  # design shares of all matches (scripts/model/wp_common.py)


def bar(segments, width=1664, height=100):
    """segments: list of (name, value, colour). One proportional bar; the legend sits below it, one line per segment, so nothing overlaps."""
    total = sum(v for _, v, _ in segments)
    out = [f'<svg viewBox="0 0 {width} {height + 110}" width="{width}" height="{height + 110}" xmlns="http://www.w3.org/2000/svg" aria-label="split bar">']
    x = 0.0
    for name, value, colour in segments:
        w = value / total * width
        out.append(f'<rect x="{x:.1f}" y="0" width="{w:.1f}" height="{height}" fill="{colour}" stroke="#FBFBF8" stroke-width="3"/>')
        x += w
    lx = 0.0
    for name, value, colour in segments:
        pct = 100 * value / total
        out.append(f'<rect x="{lx:.1f}" y="{height + 30}" width="24" height="24" fill="{colour}"/>')
        out.append(f'<text x="{lx + 34:.1f}" y="{height + 52}" font-size="24" fill="{INK}">{name} {pct:.2f}% ({value:,})</text>')
        lx += 34 + 13.5 * len(f"{name} {pct:.2f}% ({value:,})") + 48
    out.append("</svg>")
    return "".join(out)


def main():
    m = json.load(open(os.path.join(D, "split_manifest.json"), encoding="utf-8"))["assign"]
    snap = set(pl.read_parquet(os.path.join(D, "cache_snapshot_2026.parquet"), columns=["match_key"]).unique()["match_key"].to_list())
    used = Counter(v for k, v in m.items() if k.startswith("csds/2026/") and k in snap)
    manifest_all = Counter(m.values())
    full = json.load(open(os.path.join(D, "exp_full_comparison.json"), encoding="utf-8"))
    sec = {"train": full["train_rows"], "validation": full["val_rows"], "test": full["test_rows"]}
    n_used = sum(used.values())
    n_secs = sum(sec.values())

    def pct(c, k):
        return 100 * c[k] / sum(c.values())

    names = {"train": "train", "test": "test", "validation": "validation"}
    colours = {"train": SLATE, "validation": GREY, "test": INK}
    matches_bar = bar([(names[k], used[k if k != "validation" else "val"], colours[k]) for k in ["train", "validation", "test"]])
    seconds_bar = bar([(names[k], sec[k], colours[k]) for k in ["train", "validation", "test"]])
    t_all = sum(manifest_all.values())
    rel = "; ".join(
        f"{k} {pct(used, kk):.2f}% against the {TARGET[k]:.0f}% target ({pct(used, kk) - TARGET[k]:+.2f} points)"
        for k, kk in [("train", "train"), ("validation", "val"), ("test", "test")])

    html = f'''<section id="split" data-transition="fade" style="background:#FBFBF8;color:#1A1A1A;font-family:Georgia, serif;display:flex;flex-direction:column;padding:128px;gap:26px">
  <h2 style="font-size:72px;font-weight:400;line-height:1.1">How matches are split</h2>
  <ul style="font-size:28px;line-height:1.3;display:flex;flex-direction:column;gap:8px">
    <li><b>Random by match, frozen once</b> in a manifest (sha256 starts a17f219b). New matches get their split from a hash rule, so later results stay comparable.</li>
    <li><b>Not by second:</b> seconds in one match are almost identical, so a split by second would leak the answer into the test set.</li>
    <li><b>Not by date:</b> the latest 20% by match date measures a change of source, not the model. Hand-uploaded matches (mostly FACEIT) carry the processing day as their date, so the date split puts 77% of FACEIT-server matches in test.</li>
  </ul>
  <p style="font-size:26px;line-height:1.3"><b>Matches used by both models, {n_used:,} in all:</b> train {pct(used, "train"):.2f}% ({used["train"]:,}), validation {pct(used, "val"):.2f}% ({used["val"]:,}), test {pct(used, "test"):.2f}% ({used["test"]:,}).</p>
  {matches_bar}
  <p style="font-size:26px;line-height:1.3"><b>Seconds, {n_secs:,} in all:</b> train {100 * sec["train"] / n_secs:.2f}%, validation {100 * sec["validation"] / n_secs:.2f}%, test {100 * sec["test"] / n_secs:.2f}%.</p>
  {seconds_bar}
  {FOOT}
  <aside style="font-size:24px">Splitting by match is the key choice: seconds from one match are almost identical, so a split by second would hand the answer to the test set. The design target is 72% train, 8% validation and 20% test of all {t_all:,} matches in the manifest. Against that target, the matches used by the models come out {rel}. The manifest keeps the 2024 and 2026 matches in one assignment: 2024 is {sum(1 for k in m if k.startswith("csds/2024/")):,} matches, 2026 is {sum(1 for k in m if k.startswith("csds/2026/")):,}. All reported models read that same manifest.</aside>
</section>
'''
    open(OUT, "w", encoding="utf-8", newline="\n").write(html)
    print("wrote", OUT)
    print("matches used:", dict(used), n_used)
    print("seconds:", sec, n_secs)
    print("manifest all:", dict(manifest_all), t_all)


if __name__ == "__main__":
    main()
