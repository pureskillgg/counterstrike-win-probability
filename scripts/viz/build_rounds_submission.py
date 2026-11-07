"""Build the 'Round groups' slide for the submission model (57 inputs, no score lead), from the saved numbers.

  ...python.exe -I -X utf8 scripts\\viz\\build_rounds_submission.py
Writes deliverables/deck_2026-10-19/project/slides/rounds.html.
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
from deck_svg import bars, RUST, INK

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
D = os.path.join(ROOT, "data", "combined")
OUT = os.path.join(ROOT, "deliverables", "deck_2026-10-19", "project", "slides", "rounds.html")
GROUPS = [("Pistol rounds (1, 13)", "pistol (1, 13)", RUST),
          ("Rounds after pistol (2, 14)", "after pistol (2, 14)", INK),
          ("All other rounds", "other rounds", INK)]


def main():
    s = json.load(open(os.path.join(D, "exp_submission_57.json"), encoding="utf-8"))
    t = s["splits"]["test"]
    g = t["groups"]
    items = []
    for name, key, colour in GROUPS:
        x = g[key]
        lo, hi = x["ci95"]
        items.append({"label": name, "value": x["logloss"], "lo": lo, "hi": hi, "color": colour,
                      "text": f"{x['logloss']:.3f} [{lo:.3f}, {hi:.3f}]"})
    lo, hi = t["ci95"]
    items.append({"label": "All test rounds", "value": t["logloss"], "lo": lo, "hi": hi, "color": INK,
                  "text": f"{t['logloss']:.3f} [{lo:.3f}, {hi:.3f}]"})
    chart = bars(items, vmax=0.6, label_w=620, value_w=520, ticks=[0, 0.2, 0.4, 0.6], tick_fmt="{:.1f}",
                 axis_title="per-second test log loss, submission model (57 inputs, no score lead); bar = level, line = 95% match interval")
    pistol = g["pistol (1, 13)"]["logloss"]
    after = g["after pistol (2, 14)"]["logloss"]
    other = g["other rounds"]["logloss"]
    html = f'''<section id="rounds" data-transition="fade" style="background:#FBFBF8;color:#1A1A1A;font-family:Georgia, serif;display:flex;flex-direction:column;padding:128px;gap:24px">
  <h2 style="font-size:72px;font-weight:400;line-height:1.1">Log loss differs by round group</h2>
  <p style="font-size:26px;line-height:1.3">Test log loss by round group, for the submission model (57 inputs, no score lead, one seed). Pistol rounds have the highest log loss and rounds right after the pistol the lowest.</p>
  {chart}
  <div style="display:flex;flex-direction:column;gap:8px;border:2px solid {RUST};border-radius:8px;padding:16px 24px">
    <p style="font-size:24px;line-height:1.4"><b style="color:{RUST}">Open:</b> separate models for the pistol and after-pistol groups were tried in an earlier experiment whose reference was the 58-input seed-0 model (exp_families_pistol.json), not the 59-input set. They have not been tested on the submission model, and the pistol and after-pistol groups are not yet compared like-for-like with normal rounds.</p>
  </div>
  <p style="position:absolute;left:128px;bottom:64px;font-size:24px">CS2 round win probability | 13 of 19</p>
  <aside style="font-size:24px">Test log loss: pistol {pistol:.3f}, after pistol {after:.3f}, other rounds {other:.3f}. The intervals resample whole matches, not seconds, because seconds inside one match are correlated. The group numbers describe where the model's log loss is high or low; they do not say whether a round type is harder to predict on its own.</aside>
</section>
'''
    open(OUT, "w", encoding="utf-8", newline="\n").write(html)
    print("wrote", OUT)
    for it in items:
        print(f"{it['label']}: {it['text']}")


if __name__ == "__main__":
    main()
