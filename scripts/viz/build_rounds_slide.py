"""Build the 'Round types differ in difficulty' slide from the saved production predictions (no numbers typed by hand).

Per-second log loss of the five-seed production ensemble by round group, with a match-cluster 95% interval (1,000 resamples), recomputed from
data/combined/export_winprob_v8_ensemble_test_predictions.parquet joined to the labels in data/combined/cache_snapshot_2026.parquet.

  ...python.exe -I -X utf8 scripts\\viz\\build_rounds_slide.py
Writes deliverables/deck_2026-10-19/project/slides/rounds.html and data/combined/exp_rounds_groups.json.
"""
import json
import os
import sys

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
from deck_svg import bars, RUST, INK

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
D = os.path.join(ROOT, "data", "combined")
OUT = os.path.join(ROOT, "deliverables", "deck_2026-10-19", "project", "slides", "rounds.html")
N_BOOT = 1000


def logloss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def match_bootstrap(keys, d, n=N_BOOT, seed=0):
    _, inv = np.unique(np.asarray(keys), return_inverse=True)
    s = np.bincount(inv, weights=np.asarray(d, dtype=np.float64))
    c = np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n, len(s)))
    means = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return float(s.sum() / c.sum()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ex = pl.read_parquet(os.path.join(D, "export_winprob_v8_ensemble_test_predictions.parquet"))
    lab = pl.read_parquet(os.path.join(D, "cache_snapshot_2026.parquet"), columns=["match_key", "round", "tick", "label_ct_win"])
    j = ex.join(lab, on=["match_key", "round", "tick"], how="left")
    assert j.filter(pl.col("label_ct_win").is_null()).height == 0, "labels did not join one-to-one"
    y = j["label_ct_win"].to_numpy().astype(np.float64)
    p = j["p"].to_numpy().astype(np.float64)
    keys = j["match_key"].to_numpy()
    rnd = j["round"].to_numpy()
    loss = logloss(y, p)
    groups = [("Pistol rounds (1, 13)", np.isin(rnd, [1, 13])), ("Rounds after pistol (2, 14)", np.isin(rnd, [2, 14])),
              ("All other rounds", ~np.isin(rnd, [1, 2, 13, 14]))]
    out = {"model": "winprob_v8_ensemble", "test_seconds": int(j.height), "groups": {}}
    items = []
    for name, m in groups:
        # the level of a group's loss: bootstrap the matches of the group, with the group's own per-second losses
        mean, lo, hi = match_bootstrap(keys[m], loss[m])
        out["groups"][name] = {"seconds": int(m.sum()), "matches": int(len(np.unique(keys[m]))), "mean": mean, "ci95": [lo, hi]}
        items.append({"label": f"{name}", "value": mean, "lo": lo, "hi": hi, "color": RUST if "Pistol" in name else INK,
                      "text": f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"})
        print(f"{name}: {m.sum():,} seconds in {len(np.unique(keys[m]))} matches; {mean:.6f} [{lo:.6f}, {hi:.6f}]", flush=True)
    mean, lo, hi = match_bootstrap(keys, loss)
    out["all_rounds"] = {"mean": mean, "ci95": [lo, hi]}
    items.append({"label": "All test rounds", "value": mean, "lo": lo, "hi": hi, "color": INK, "text": f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"})
    print(f"all rounds: {mean:.6f} [{lo:.6f}, {hi:.6f}]", flush=True)
    W = json.load(open(os.path.join(D, "exp_families_pistol.json"), encoding="utf-8"))["pistol"]
    W_pistol = W["pistol rounds (1, 13)"]
    W_after = W["round after pistol (2, 14)"]
    W_all = W["overall_test"]
    json.dump(out, open(os.path.join(D, "exp_rounds_groups.json"), "w", encoding="utf-8"), indent=1)

    chart = bars(items, vmax=0.6, label_w=620, value_w=520, ticks=[0, 0.2, 0.4, 0.6], tick_fmt="{:.1f}",
                 axis_title="per-second test log loss of the production ensemble (lower is better); bar = level, line = 95% match interval")
    html = f'''<section id="rounds" data-transition="fade" style="background:#FBFBF8;color:#1A1A1A;font-family:Georgia, serif;display:flex;flex-direction:column;padding:128px;gap:24px">
  <h2 style="font-size:72px;font-weight:400;line-height:1.1">Round types differ in difficulty</h2>
  <p style="font-size:26px;line-height:1.3">Test log loss by round group, five-seed production model. Pistol rounds are the hardest for the model, and rounds right after the pistol are the easiest.</p>
  {chart}
  <div style="display:flex;flex-direction:column;gap:8px;border:2px solid #9A3412;border-radius:8px;padding:16px 24px">
    <p style="font-size:24px;line-height:1.4"><b style="color:#9A3412">Separate models per group: rejected.</b> Pistol rounds {W_pistol['dedicated_minus_main']*1e3:+.2f} x10^-3 [{W_pistol['ci95'][0]*1e3:+.2f}, {W_pistol['ci95'][1]*1e3:+.2f}]; after pistol {W_after['dedicated_minus_main']*1e3:+.2f} x10^-3 [{W_after['ci95'][0]*1e3:+.2f}, {W_after['ci95'][1]*1e3:+.2f}]; overall {W_all['split_minus_main']*1e3:+.2f} x10^-3 [{W_all['ci95'][0]*1e3:+.2f}, {W_all['ci95'][1]*1e3:+.2f}]. These are against the single seed-0 model on the same test seconds, not the five-seed average.</p>
  </div>
  <p style="position:absolute;left:128px;bottom:64px;font-size:24px">CS2 round win probability | 10 of 14</p>
  <aside style="font-size:24px">The round groups differ a lot in difficulty: pistol rounds are the hardest for the model, rounds right after the pistol are the easiest, and the rest sit in between. Each group's interval resamples whole matches, because seconds inside one round move together. Splitting the model by round group was tested with dedicated models trained only on each group's seconds. Each group has far fewer training seconds than the main model, and the dedicated models did worse on the same test seconds, so the single model stays.</aside>
</section>
'''
    open(OUT, "w", encoding="utf-8", newline="\n").write(html)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
