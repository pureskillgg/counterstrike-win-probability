"""A8 check 7: sensitivity of the cross-period 2024 log loss to (a) the 2024 round label defect and
(b) round-end-tick states. Same model as cross_period.py model A (trained on 2026 train + test,
early stopping on the 2026 primary validation slice), same 2024 relaxed kept set (cross_period.run_2024).

Label rule (ASSUMPTION, not verified): where a round has bomb_exploded (and not bomb_defused), the
true winner is T (2); where it has bomb_defused, the true winner is CT (3). A round is "contradicted"
when winner_team_code disagrees with that rule. For contradicted rounds the corrected label is the
opposite of the recorded label (the rule is binary). Rounds without a bomb outcome are not changed.
The bomb events themselves are not checked here.
Run from the project root:
  .pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A8_cross_period_sensitivity.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
T, CT = 2, 3


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ll(y, p):
    return float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]))


def main():
    cp = load("cross_period", ROOT / "scripts" / "model" / "sufficiency" / "cross_period.py")
    bk, fv = cp.bk, cp.fv
    print("== A8 check 7: cross-period 2024 sensitivity ==")
    A = cp.run_2026()
    B = cp.run_2024()
    Xtr_a, ytr_a = bk.xy(A["S_trA"])
    Xva_a, yva_a = bk.xy(A["S_vaA"])
    mA, dev, biA, _ = cp.fit_xgb(Xtr_a, ytr_a, Xva_a, yva_a)
    print(f"model A: best_iteration {biA}; device {dev}")
    S24 = B["S"]
    X24, y24 = bk.xy(S24)
    p24 = cp.predict(mA, X24, biA)
    print(f"2024 states {S24.height}; rounds {S24.select(['match_key', 'round']).unique().height}")
    print(f"LL all 2024 states (recorded labels) {ll(y24, p24):.5f}  (cross_period.md: 0.50441)")
    print(f"AUC all 2024 states {roc_auc_score(y24, p24):.4f}  (cross_period.md: 0.8401)")

    # 2024 round_end and bomb events (same tome, same suffix as cross_period.run_2024)
    old = (fv.TOME, fv.SUFFIX)
    fv.TOME, fv.SUFFIX = cp.TOME_2024, cp.SUFFIX_2024
    try:
        RE = fv.read("round_end", ["match_key", "round", "tick", "winner_team_code"])
        BS = fv.read("bomb_state", ["match_key", "round", "tick", "event_type"])
    finally:
        fv.TOME, fv.SUFFIX = old
    kr = S24.select(["match_key", "round"]).unique()
    RE = RE.with_columns(pl.col("round").cast(pl.Int64)).join(kr.with_columns(pl.col("round").cast(pl.Int64)),
                                                              on=["match_key", "round"], how="inner")
    evs = (BS.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
           .with_columns(pl.col("round").cast(pl.Int64))
           .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
    ev_map = {(m, int(r)): set(e) for m, r, e in evs.select(["match_key", "round", "evs"]).iter_rows()}
    end_tick = {(m, int(r)): int(t) for m, r, t in RE.select(["match_key", "round", "tick"]).iter_rows()}
    winner = {(m, int(r)): int(w) for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows()}

    contradicted = set()
    testable = 0
    for k, w in winner.items():
        e = ev_map.get(k)
        if not e:
            continue
        if "bomb_exploded" in e and "bomb_defused" in e:
            continue
        testable += 1
        expect = T if "bomb_exploded" in e else CT
        if w != expect:
            contradicted.add(k)
    r13 = {k for k in contradicted if k[1] == 13}
    print(f"\n2024 kept rounds with a testable bomb outcome: {testable}; contradicted by winner code: "
          f"{len(contradicted)} (round 13: {len(r13)})")

    keys = list(zip(S24["match_key"].to_list(), [int(r) for r in S24["round"].to_list()]))
    in_c = np.array([k in contradicted for k in keys])
    at_end = np.array([int(t) == end_tick.get(k, -1) for k, t in zip(keys, S24["tick"].to_list())])
    print(f"states in contradicted rounds: {int(in_c.sum())} ({100 * in_c.mean():.2f}%); "
          f"states at round_end tick: {int(at_end.sum())} ({100 * at_end.mean():.2f}%)")

    y_corr = y24.copy()
    y_corr[in_c] = 1 - y_corr[in_c]
    print(f"\nLL all states, recorded labels           {ll(y24, p24):.5f}")
    print(f"LL all states, contradicted rounds fixed  {ll(y_corr, p24):.5f}  "
          f"(change {ll(y_corr, p24) - ll(y24, p24):+.5f})")
    keep = ~in_c
    print(f"LL excluding contradicted rounds          {ll(y24[keep], p24[keep]):.5f} (n={int(keep.sum())})")
    print(f"LL contradicted-round states, recorded    {ll(y24[in_c], p24[in_c]):.5f}; "
          f"after fix {ll(y_corr[in_c], p24[in_c]):.5f}")
    ne = ~at_end
    print(f"\nLL excluding round_end-tick states (recorded labels) {ll(y24[ne], p24[ne]):.5f} (n={int(ne.sum())}); "
          f"AUC {roc_auc_score(y24[ne], p24[ne]):.4f}")
    both = ne & keep
    print(f"LL excluding round_end-tick and contradicted states  {ll(y24[both], p24[both]):.5f} (n={int(both.sum())}); "
          f"AUC {roc_auc_score(y24[both], p24[both]):.4f}")
    fixed_ne = ne
    print(f"LL excluding round_end-tick states, contradicted fixed {ll(y_corr[fixed_ne], p24[fixed_ne]):.5f}")
    print("\nReading: ASSUMPTION that the bomb outcome is the true winner in testable rounds (not verified).")


if __name__ == "__main__":
    main()
