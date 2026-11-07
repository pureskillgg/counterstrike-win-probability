"""Renumber the slide footers for the 18-slide deck and write the deck index (project/deck.json) with the new order.

  ...python.exe -I -X utf8 scripts\\viz\\finalize_expanded_deck.py

Reads the live index copy (deliverables/deck_2026-10-19/live_copy/project/deck.json) so that title, faces and other keys are kept as read;
only `order` and `sections` change. Writes deliverables/deck_2026-10-19/project/deck.json.
"""
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DECK = os.path.join(ROOT, "deliverables", "deck_2026-10-19")
SL = os.path.join(DECK, "project", "slides")
ORDER = ["cover", "question", "pool", "data", "roster", "split", "split_rationale", "ticks", "inputs", "model", "settings",
         "families", "rounds", "rejected", "results", "submission", "compare", "limits", "open"]
N = len(ORDER)
FOOT = re.compile(r"CS2 round win probability \| [^<]*")

for i, sid in enumerate(ORDER, start=1):
    path = os.path.join(SL, sid + ".html")
    s = open(path, encoding="utf-8").read()
    if sid == "cover":
        assert "CS2 round win probability |" not in s, "cover should carry no footer"
        continue
    s2, n = FOOT.subn(f"CS2 round win probability | {i} of {N}", s)
    assert n == 1, (sid, n)
    open(path, "w", encoding="utf-8", newline="\n").write(s2)
    print(f"{sid}: footer {i} of {N}")

live = json.load(open(os.path.join(DECK, "live_copy", "project", "deck.json"), encoding="utf-8"))
deck = dict(live)
deck["order"] = ORDER
deck["sections"] = {
    "s1": {"description": "What the model is and what data it uses", "start": "cover"},
    "s2": {"description": "Scope, data and how matches and seconds are chosen", "start": "pool"},
    "s3": {"description": "The model, how its settings were chosen, and the alternatives", "start": "model"},
    "s4": {"description": "Results, limits and open items", "start": "results"},
}
os.makedirs(os.path.join(DECK, "project"), exist_ok=True)
json.dump(deck, open(os.path.join(DECK, "project", "deck.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("wrote project/deck.json with", N, "slides")
