"""verify_D2_7 step 2: 2026 -> 2024 cross-period log loss under recorded and corrected 2024 labels.

Reused unchanged from scripts/model/sufficiency/cross_period.py: run_2026, run_2024 (state features,
splits, kept set), fit_xgb, predict. Model A is trained on 2026 with 2026 labels (verified clean).
NOT reused: the 2024 evaluation labels. Those are recomputed in verify_D2_7_01_truth.py from raw
channels (side truth). Evaluation-only correction, as in the D2-7 proposed fix; no retraining.
Prints only. Nothing is written to disk.
"""

import contextlib
import importlib.util
import io
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)

ROOT = Path(r"G:\pureskill\csds-cs2")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cp = load("cross_period", ROOT / "scripts" / "model" / "sufficiency" / "cross_period.py")
t1 = load("verify_D2_7_01_truth", ROOT / "scripts" / "swarm2" / "verify_D2_7_01_truth.py")
bk = cp.bk


def ll(y, p):
    return float(log_loss(np.asarray(y), np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0), labels=[0, 1]))


def auc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(p, dtype=np.float64)))


def main():
    t0 = time.time()
    print("== step A: reproduce cross_period.py models (reused pipeline, recorded labels) ==")
    A = cp.run_2026()
    B = cp.run_2024()
    print(f"kept 2024 (cross_period kept set): matches={B['counts']['kept_matches']} "
          f"rounds={B['counts']['kept_rounds']} states={B['counts']['states']}")

    Xtr_a, ytr_a = bk.xy(A["S_trA"])
    Xva_a, yva_a = bk.xy(A["S_vaA"])
    mA, devA, biA, _ = cp.fit_xgb(Xtr_a, ytr_a, Xva_a, yva_a)
    X24, y24 = bk.xy(B["S"])
    p24 = cp.predict(mA, X24, biA)

    Xtr_b, ytr_b = bk.xy(A["S_trB"])
    Xva_b, yva_b = bk.xy(A["S_vaB"])
    Xte_b, yte_b = bk.xy(A["S_teB"])
    mB, devB, biB, _ = cp.fit_xgb(Xtr_b, ytr_b, Xva_b, yva_b)
    pte_b = cp.predict(mB, Xte_b, biB)
    p24_b = cp.predict(mB, X24, biB)

    print(f"model A best_iteration={biA} ({devA}); model B best_iteration={biB} ({devB})")
    print(f"in-period reference: model B on 2026 primary test, all states: "
          f"LL={ll(yte_b, pte_b):.5f} (claimed 0.43660)")
    print(f"cross-period: model A on 2024 kept set, recorded labels, all states: "
          f"LL={ll(y24, p24):.5f} (claimed 0.50441) n={len(y24)}")
    print(f"model B on 2024 kept set, recorded labels, all states: LL={ll(y24, p24_b):.5f}")

    print("\n== step B: independent 2024 side truth (verify_D2_7_01_truth.build, raw channels) ==")
    with contextlib.redirect_stdout(io.StringIO()):
        tab24, _ = t1.build("2024", t1.TOME24, t1.SUF24)

    # max_rounds rule as in cross_period.py: 2 x (round of halftime announcement) + 2, else 24
    rs_ev = t1.read(t1.TOME24, t1.SUF24, "round_state", ["match_key", "round", "event_type"])
    half = (rs_ev.filter(pl.col("event_type") == "round_announce_last_round_half")
            .group_by("match_key").agg(pl.col("round").min().alias("half_round")))
    rnd = (tab24.join(half, on="match_key", how="left")
           .with_columns(pl.when(pl.col("half_round").is_not_null())
                         .then(pl.col("half_round") * 2 + 2).otherwise(24).alias("max_rounds")))

    # swap at the halftime round should be 1 (membership-based swap vs max_rounds rule)
    sw_chk = (rnd.filter(pl.col("round") == pl.col("max_rounds") // 2)
              .group_by("max_rounds")
              .agg(pl.len().alias("n_halftime_rounds"), (pl.col("swap") == 1).sum().alias("swap_is_1"),
                   pl.col("swap").is_null().sum().alias("swap_undecided"))
              .sort("max_rounds"))
    print("halftime-round swap check (membership swap at round max_rounds//2):")
    print(sw_chk)

    S = (B["S"].join(rnd.select(["match_key", "round", "truth_ct", "R_ct", "flag", "grp", "day",
                                 "max_rounds", "bomb_ct", "elim_ct", "player_count"]),
                     on=["match_key", "round"], how="left"))
    print(f"\nstates joined to 2024 round table: {S.height}; unmatched: {S.filter(pl.col('R_ct').is_null()).height}")
    mism_rec = S.filter(pl.col("label_ct_win") != pl.col("R_ct")).height
    print(f"states where cross_period label != round_end winner (R) recomputed here: {mism_rec}")

    y_rec = S["label_ct_win"].to_numpy().astype(np.int64)
    y_corr = (S.select(pl.when(pl.col("truth_ct").is_not_null()).then(pl.col("truth_ct"))
                       .otherwise(pl.col("label_ct_win")).alias("y"))["y"].to_numpy().astype(np.int64))
    has_truth = S["truth_ct"].is_not_null().to_numpy()
    flagged = (S["flag"].fill_null(0) == 1).to_numpy()
    p = p24

    def rep(name, mask):
        n = int(mask.sum())
        if n == 0:
            print(f"{name:<70} n_states=0")
            return
        print(f"{name:<70} n_states={n:>7}  LL recorded={ll(y_rec[mask], p[mask]):.5f}  "
              f"LL corrected={ll(y_corr[mask], p[mask]):.5f}  AUC rec={auc(y_rec[mask], p[mask]):.4f}  "
              f"AUC corr={auc(y_corr[mask], p[mask]):.4f}")

    print("\n-- LL by subset (model A, same predictions; only labels differ) --")
    allm = np.ones(len(y_rec), dtype=bool)
    rep("all states (corrected = score truth where known, else recorded)", allm)
    rep("states in rounds with score truth", has_truth)
    rep("states in rounds with NO score truth (kept recorded label)", ~has_truth)
    rep("states in flagged rounds (score truth != recorded)", flagged)
    rep("states in rounds with truth and recorded == truth (unflagged)", has_truth & ~flagged)

    grp = S["grp"].to_numpy()
    print("\n-- by round group (flag = score truth != recorded) --")
    for g in ["r01-12", "r13", "r14-24", "r25", "r28", "r29+"]:
        rep(f"group {g}", (grp == g))

    maxr = S["max_rounds"].to_numpy()
    rnd_np = S["round"].to_numpy()
    print("\n-- flagged rounds by max_rounds --")
    rep("flagged, max_rounds 24", flagged & (maxr == 24))
    rep("flagged, max_rounds 16", flagged & (maxr == 16))
    rep("flagged, round 9 in max_rounds 16 matches", flagged & (rnd_np == 9) & (maxr == 16))
    rep("flagged, rounds 13/25/28 (MR12 defect rounds)", flagged & np.isin(rnd_np, [13, 25, 28]))

    print("\n-- gap analysis --")
    lr_all, lc_all = ll(y_rec, p), ll(y_corr, p)
    print(f"LL recorded all states:  {lr_all:.5f}")
    print(f"LL corrected all states: {lc_all:.5f}  (change {lc_all - lr_all:+.5f})")
    print(f"gap vs in-period 0.43660: recorded {lr_all - 0.43660:+.5f}, corrected {lc_all - 0.43660:+.5f} "
          f"(rule GAP_LIMIT 0.02)")
    unfl = has_truth & ~flagged
    print(f"LL on states in unflagged rounds with truth (recorded == corrected): {ll(y_rec[unfl], p[unfl]):.5f} "
          f"(n={int(unfl.sum())}); this bounds the non-label part of the gap")

    print("\n-- kept-set round counts --")
    kept_rounds = S.select(["match_key", "round"]).unique()
    fl_rounds = S.filter(pl.col("flag") == 1).select(["match_key", "round"]).unique()
    truth_rounds = S.filter(pl.col("truth_ct").is_not_null()).select(["match_key", "round"]).unique()
    print(f"kept rounds {kept_rounds.height}; with score truth {truth_rounds.height}; flagged {fl_rounds.height} "
          f"({100 * fl_rounds.height / kept_rounds.height:.2f}% of kept rounds)")
    fr = S.filter(pl.col("flag") == 1).select(["match_key", "round", "grp", "max_rounds", "bomb_ct", "elim_ct",
                                              "truth_ct", "R_ct", "day"]).unique(subset=["match_key", "round"])
    print("flagged kept rounds by group and max_rounds:")
    print(fr.group_by(["grp", "max_rounds"]).len().sort(["grp", "max_rounds"]))
    fb = fr.filter(pl.col("bomb_ct").is_not_null())
    fe = fr.filter(pl.col("elim_ct").is_not_null())
    print(f"flagged kept rounds with bomb outcome: {fb.height}; bomb == score truth: "
          f"{int((fb['bomb_ct'] == fb['truth_ct']).sum())}; bomb == recorded R: "
          f"{int((fb['bomb_ct'] == fb['R_ct']).sum())}")
    print(f"flagged kept rounds with elimination outcome: {fe.height}; elim == score truth: "
          f"{int((fe['elim_ct'] == fe['truth_ct']).sum())}; elim == recorded R: "
          f"{int((fe['elim_ct'] == fe['R_ct']).sum())}")
    print(f"flagged kept rounds with NO bomb and NO elimination outcome (score-only): "
          f"{fr.filter(pl.col('bomb_ct').is_null() & pl.col('elim_ct').is_null()).height}")

    print(f"\nelapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
