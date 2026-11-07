"""Assemble the publish folder for the live Slides deck from the generated slide files.

  ...python.exe -I -X utf8 scripts\\viz\\assemble_deck.py --slides <dir with generated *.html> --live-deck-json <saved copy of the live project/deck.json> --root <publish root>

Writes <root>/project/deck.json (every key of the live index kept as read; only title-neutral `order` and `sections` change) and copies each generated slide to
<root>/project/slides/<id>.html. The old first-batch slides stay in the deck as an appendix (they are listed in the order after the appendix divider), so no slide file
has to be deleted. Prints the Artifact call arguments (file_path + files) to use.
"""
import argparse
import json
import os
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")
MAIN = ["cover", "brief", "beforeafter", "m_data", "faceit", "eval", "m_headline", "m_calib", "m_positions", "graph", "m_robust", "viewer", "decisions", "next"]
# appendix, in this order: the other results of this deck, then the first-batch slides of 13 October (not regenerated)
APPENDIX_NEW = ["m_benchmark", "m_ticks", "m_enough", "m_maps", "m_scope", "m_transfer", "m_families", "m_foundation", "m_distill", "m_stack", "m_hparams", "m_gpucpu",
                "incremental", "lit", "questions", "m_temporal"]
APPENDIX_HEAD = ["appendix"]
SECTIONS = {"context": ("Brief, before and after, data and how we evaluate.", "cover"),
            "results": ("The final model, calibration and what player positions add.", "m_headline"),
            "deploy": ("Tests, and the replay viewer.", "m_robust"),
            "next": ("Decisions and next steps.", "decisions"),
            "appendix": ("More results, then the first-batch exploration of 1 to 8 October, kept for the record.", "appendix")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", required=True)
    ap.add_argument("--live-deck-json", required=True)
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    live = json.load(open(a.live_deck_json, encoding="utf-8"))
    old_order = live["order"]
    have = {f[:-5] for f in os.listdir(a.slides) if f.endswith(".html")}
    main_ids = [i for i in MAIN if i in have]
    missing = [i for i in MAIN if i not in have]
    app_new = [i for i in APPENDIX_NEW if i in have]
    new_set = set(main_ids) | set(APPENDIX_HEAD) | set(app_new)
    # old slides that are not regenerated go to the appendix, in their old order; regenerated ids that are not in MAIN (none expected) are not duplicated
    appendix_old = [i for i in old_order if i not in new_set]
    order = main_ids + [i for i in APPENDIX_HEAD if i in have] + app_new + appendix_old
    assert len(order) == len(set(order)), "duplicate ids in the order"
    deck = dict(live)
    deck["order"] = order
    deck["sections"] = {k: {"description": d, "start": s} for k, (d, s) in SECTIONS.items() if s in order}
    os.makedirs(os.path.join(a.root, "project", "slides"), exist_ok=True)
    for i in sorted(have):
        shutil.copy2(os.path.join(a.slides, i + ".html"), os.path.join(a.root, "project", "slides", i + ".html"))
    with open(os.path.join(a.root, "project", "deck.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(deck, f, indent=2, ensure_ascii=False)
    print(f"slides in main: {len(main_ids)} (missing: {missing}); appendix: {len(appendix_old)} old slides after the divider; files written: {len(have)} + deck.json")
    print("files:", {f"project/slides/{i}.html": f"project/slides/{i}.html" for i in sorted(have)})


if __name__ == "__main__":
    main()
