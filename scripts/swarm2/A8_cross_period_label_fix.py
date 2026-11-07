"""A8 check 10: cross-period 2024 log loss with the round-13 label defect corrected from the score fields.

Model A (cross_period.py: trained on 2026 train + test, early stopping on 2026 primary validation),
2024 relaxed kept set (cross_period.run_2024). Label corrections, applied to the recorded labels:
  (S) score-implied winner: for round 13 with a freeze row in round 14 whose score changes by exactly one
      point on one field, the observed winner is the side whose field rose. No side swap is scheduled after
      round 13 (MR16 and MR24), so the field identity is unchanged. Rounds where the recorded winner_team_code
      differs from this observed winner are corrected.
  (B) bomb rule (as in A8_cross_period_sensitivity.py): rounds with bomb_exploded -> T, bomb_defused -> CT.
Also reports the like-for-like gap against 2026 test non-round-end log loss (model B, 0.48242) and against
the reported all-state gap. ASSUMPTION: the score fields and bomb events are the truth where they agree with
each other; this is supported for round 13 by 799 of 799 bomb-testable transitions agreeing with the score
field (A8_round13_score_check.py), but it is not proven.
Run from the project root:
  .pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A8_cross_period_label_fix.py
"""

import glob
import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
T, CT = 2, 3
TOME_2024 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ll(y, p):
    return float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]))


def read_channel(root, ch, cols):
    folders = sorted(glob.glob(str(root / f"basic_{ch}.*")))
    files = sorted(glob.glob(str(Path(folders[0]) / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def main():
    cp = load("cross_period", ROOT / "scripts" / "model" / "sufficiency" / "cross_period.py")
    bk = cp.bk
    print("== A8 check 10: cross-period label fix ==")
    A = cp.run_2026()
    B = cp.run_2024()
    Xtr_a, ytr_a = bk.xy(A["S_trA"])
    Xva_a, yva_a = bk.xy(A["S_vaA"])
    mA, _, biA, _ = cp.fit_xgb(Xtr_a, ytr_a, Xva_a, yva_a)
    S24 = B["S"]
    X24, y24 = bk.xy(S24)
    p24 = cp.predict(mA, X24, biA)
    print(f"model A best_iteration {biA}; 2024 states {S24.height}; LL recorded {ll(y24, p24):.5f}")

    RE = read_channel(TOME_2024, "round_end", ["match_key", "round", "tick", "winner_team_code"])
    RS = read_channel(TOME_2024, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    BS = read_channel(TOME_2024, "bomb_state", ["match_key", "round", "tick", "event_type"])
    fr = (RS.filter(pl.col("event_type") == "round_freeze_end")
          .with_columns(pl.col("round").cast(pl.Int64))
          .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
    fmap = {(m, int(r)): (t, c) for m, r, t, c in fr.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
    ev = (BS.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
          .with_columns(pl.col("round").cast(pl.Int64))
          .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
    evm = {(m, int(r)): set(e) for m, r, e in ev.select(["match_key", "round", "evs"]).iter_rows()}
    winner = {(m, int(r)): int(w) for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows()}
    end_tick = {(m, int(r)): int(t) for m, r, t in RE.select(["match_key", "round", "tick"]).iter_rows()}

    keys = list(zip(S24["match_key"].to_list(), [int(r) for r in S24["round"].to_list()]))
    ticks = [int(t) for t in S24["tick"].to_list()]
    at_end = np.array([t == end_tick.get(k, -1) for k, t in zip(keys, ticks)])

    score_fix = set()
    score_obs = 0
    score_unknown = 0
    for k in set(keys):
        m, r = k
        if r != 13:
            continue
        f0, f1 = fmap.get((m, 13)), fmap.get((m, 14))
        if f0 is None or f1 is None or f0[0] is None or f1[0] is None:
            score_unknown += 1
            continue
        dt, dc = f1[0] - f0[0], f1[1] - f0[1]
        if (dt, dc) == (1, 0):
            obs = T
        elif (dt, dc) == (0, 1):
            obs = CT
        else:
            score_unknown += 1
            continue
        score_obs += 1
        if obs != winner[k]:
            score_fix.add(k)
    print(f"round 13 rounds in 2024 kept set: {sum(1 for k in set(keys) if k[1] == 13)}; with an observable "
          f"score change: {score_obs}; unobservable: {score_unknown}; recorded winner != score winner: {len(score_fix)}")

    bomb_fix = set()
    for k, w in winner.items():
        e = evm.get(k)
        if not e or ("bomb_exploded" in e and "bomb_defused" in e):
            continue
        expect = T if "bomb_exploded" in e else CT
        if w != expect:
            bomb_fix.add(k)
    in_s = np.array([k in score_fix for k in keys])
    in_b = np.array([k in bomb_fix for k in keys])
    in_both = in_s | in_b
    print(f"states in score-corrected rounds: {int(in_s.sum())} ({100 * in_s.mean():.2f}% of 2024 states); "
          f"bomb-corrected: {int(in_b.sum())}; union: {int(in_both.sum())}")

    def flipped(mask):
        y = y24.copy()
        y[mask] = 1 - y[mask]
        return y

    ne = ~at_end
    y_s, y_b, y_u = flipped(in_s), flipped(in_b), flipped(in_both)
    print("\n-- all 2024 kept states --")
    print(f"recorded labels                 LL {ll(y24, p24):.5f}  AUC {roc_auc_score(y24, p24):.4f}")
    print(f"round-13 score-corrected        LL {ll(y_s, p24):.5f}  AUC {roc_auc_score(y_s, p24):.4f}")
    print(f"bomb-corrected                  LL {ll(y_b, p24):.5f}")
    print(f"union corrected                 LL {ll(y_u, p24):.5f}  AUC {roc_auc_score(y_u, p24):.4f}")
    print("\n-- 2024 non-round-end states --")
    print(f"recorded labels                 LL {ll(y24[ne], p24[ne]):.5f}  AUC {roc_auc_score(y24[ne], p24[ne]):.4f} (n={int(ne.sum())})")
    print(f"union corrected                 LL {ll(y_u[ne], p24[ne]):.5f}  AUC {roc_auc_score(y_u[ne], p24[ne]):.4f}")
    print("\n-- 2026 test, model B, non-round-end states (cross_period.py model B) --")
    print("reference from A8_variants.py seed 0 V0 non-round-end: LL 0.48242, AUC 0.8415")
    ref26 = 0.48242
    # same-model comparison: model B (primary train only, as the in-period 2026 test row) on 2024
    Xtr_b, ytr_b = bk.xy(A["S_trB"])
    Xva_b, yva_b = bk.xy(A["S_vaB"])
    Xte_b, yte_b = bk.xy(A["S_teB"])
    mB, _, biB, _ = cp.fit_xgb(Xtr_b, ytr_b, Xva_b, yva_b)
    pte_b = cp.predict(mB, Xte_b, biB)
    p24_b = cp.predict(mB, X24, biB)
    te_keys = list(zip(A["S_teB"]["match_key"].to_list(), [int(r) for r in A["S_teB"]["round"].to_list()],
                       [int(t) for t in A["S_teB"]["tick"].to_list()]))
    E26 = cp.fv.read("round_end", ["match_key", "round", "tick"])  # cp.fv.TOME is the 2026 tome here
    end26 = {(m, int(r)): int(t) for m, r, t in E26.iter_rows()}
    te_ne = np.array([t != end26.get((m, r), -1) for m, r, t in te_keys])
    yte_ne = yte_b[te_ne]
    pte_ne = pte_b[te_ne]
    ref26_b = ll(yte_ne, pte_ne)
    print(f"\n-- same model B (primary train only), 2026 test non-round-end LL {ref26_b:.5f} "
          f"(n={int(te_ne.sum())}); 2026 test all-state LL {ll(yte_b, pte_b):.5f} --")
    print(f"model B on 2024, recorded labels, all states          LL {ll(y24, p24_b):.5f}")
    print(f"model B on 2024, union-corrected, all states          LL {ll(y_u, p24_b):.5f}")
    print(f"model B on 2024, recorded labels, non-round-end       LL {ll(y24[ne], p24_b[ne]):.5f}")
    print(f"model B on 2024, union-corrected, non-round-end       LL {ll(y_u[ne], p24_b[ne]):.5f}")
    print(f"model B on 2024, round-13 score-corrected only, non-round-end LL {ll(y_s[ne], p24_b[ne]):.5f}")
    print("\n-- gaps (same model B where stated) --")
    print(f"reported all-state gap (0.50441 - 0.43660), model A on 2024 vs model B on 2026 {0.50441 - 0.43660:+.5f}")
    print(f"model B: all-state gap, recorded 2024 labels   {ll(y24, p24_b) - ll(yte_b, pte_b):+.5f}")
    print(f"model B: all-state gap, union-corrected labels {ll(y_u, p24_b) - ll(yte_b, pte_b):+.5f}")
    print(f"model B: non-round-end gap, recorded labels    {ll(y24[ne], p24_b[ne]) - ref26_b:+.5f}")
    print(f"model B: non-round-end gap, union-corrected    {ll(y_u[ne], p24_b[ne]) - ref26_b:+.5f}")
    print("\nReading: the cross-period verdict threshold in cross_period.md is a 0.02 gap.")


if __name__ == "__main__":
    main()
