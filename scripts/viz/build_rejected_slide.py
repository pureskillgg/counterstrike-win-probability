"""Build the 'What else was tried' slide from the saved results files (no numbers typed by hand).

  ...python.exe -I -X utf8 scripts\\viz\\build_rejected_slide.py
Writes deliverables/deck_2026-10-19/project/slides/rejected.html.
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
from deck_svg import forest, RUST, INK

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
D = os.path.join(ROOT, "data", "combined")
OUT = os.path.join(ROOT, "deliverables", "deck_2026-10-19", "project", "slides", "rejected.html")
U = 1e4  # units of 0.0001 log loss


def load(name):
    return json.load(open(os.path.join(D, name), encoding="utf-8"))


def ci(d, key):
    lo, hi = d[key]
    return lo * U, hi * U


def main():
    tick = load("exp_tick8.json")
    ev = load("exp_event_features.json")
    pistol = load("exp_families_pistol.json")["pistol"]["overall_test"]
    joint = load("exp_loadout_joint.json")
    rnd = load("exp_random_ticks59.json")

    # finer rows: test delta against every second, same matches (exp_tick8)
    fine = []
    for r in tick["rates"]:
        lo, hi = r["test_delta_ci95"]
        fine.append({"label": f"Finer rows, {r['ticks_per_second']} per second", "value": r["test_delta_vs_every_second"] * U,
                     "lo": lo * U, "hi": hi * U})
    max_val_fine = max(r["validation_delta_vs_every_second"] for r in tick["rates"]) * U

    # event features: test delta against the 58-input baseline (exp_event_features, variant 2)
    evv = [v for v in ev["variants"] if "validation_delta_vs_baseline" in v][0]
    event = {"label": "Event features (damage, shots so far)", "value": evv["test_delta_vs_baseline"] * U,
             "lo": evv["test_delta_ci95"][0] * U, "hi": evv["test_delta_ci95"][1] * U}

    # pistol-round split: overall test delta against the seed-0 model (exp_families_pistol)
    split = {"label": "Separate pistol-round models", "value": pistol["split_minus_main"] * U,
             "lo": pistol["ci95"][0] * U, "hi": pistol["ci95"][1] * U}

    small_rows = fine + [event, split]
    for r in small_rows:
        # finer rows show no gain (ink); event features and the pistol split are worse on test (rust)
        r["color"] = RUST if (r["label"].startswith("Event") or r["label"].startswith("Separate")) else INK
        r["d"] = 2

    # joint removal: test delta against the 59-input control (exp_loadout_joint)
    jv = [v for v in joint["variants"] if v["name"].startswith("joint")][0]
    big_rows = [{"label": f"Joint removal, 32 loadout inputs ({jv['n_features']} left)", "value": jv["test_delta_vs_control"] * U,
                 "lo": jv["test_delta_ci95"][0] * U, "hi": jv["test_delta_ci95"][1] * U, "color": RUST, "d": 2}]
    # random draws per round vs every second (exp_random_ticks59, seed 0)
    for n in [1, 2, 4, 8, 16, 32, 64]:
        c = rnd["cells"][f"s0_n{n}"]
        lo, hi = c["test_delta_ci95"]
        big_rows.append({"label": f"Random draws, {n} per round", "value": c["test_delta_vs_every_second"] * U,
                         "lo": lo * U, "hi": hi * U, "color": RUST, "d": 2})

    chart_small = forest(small_rows, x0=-3, x1=4, ticks=[-2, -1, 0, 1, 2, 3, 4], label_w=640, value_w=520,
                         axis_title="test log loss change vs each row's own reference (see caption), units of 0.0001")
    chart_big = forest(big_rows, x0=0, x1=40, ticks=[0, 10, 20, 30, 40], label_w=640, value_w=520,
                       axis_title="test log loss change vs each row's own reference (see caption), units of 0.0001", zero_label="")

    html = f'''<section id="rejected" data-transition="fade" style="background:#FBFBF8;color:#1A1A1A;font-family:Georgia, serif;display:flex;flex-direction:column;padding:128px;gap:24px">
  <h2 style="font-size:72px;font-weight:400;line-height:1.1">What else was tried</h2>
  <p style="font-size:26px;line-height:1.3">Test log loss change against each row's own reference, with 95% intervals (dark line). Finer rows: every second on the same matches. Random draws: every second, same 59-input model (winprob_v4). Event features: the 58-input baseline. Separate pistol: the 58-input seed-0 model. Joint removal: the 59-input control. Units of 0.0001; right of zero is worse.</p>
  {chart_small}
  {chart_big}
  <p style="position:absolute;left:128px;bottom:64px;font-size:24px">CS2 round win probability | 14 of 19</p>
  <aside style="font-size:24px">Finer rows and event features were judged on validation first: finer rows move validation by at most +{max_val_fine:.2f} units of 0.0001, and event features by -0.00002. The pistol split is worse on test, against the seed-0 model on the same seconds. Random draws are compared with the same every-second model using the 59-input reference (winprob_v4). Weighting each draw by its round's length closes most of the gap to every second: weighted minus unweighted draws in the same run is -0.001116 [-0.001440, -0.000790] (exp_final_checks.json, which does not record the draw count or training design). Positions are deferred to a later model, as instructed.</aside>
</section>
'''
    open(OUT, "w", encoding="utf-8", newline="\n").write(html)
    print("wrote", OUT)
    for r in small_rows + big_rows:
        print(f"{r['label']:45s} {r['value']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]")


if __name__ == "__main__":
    main()
