"""D2-7 independent verifier (read-only on the project; writes only its own stdout to logs/swarm2/verify_D2_7.log).

Claim under test: 2024 round_end winner_team_code is wrong for 2,117 rounds (round 13: 1,888; round 25: 114;
round 28: 115; 2.7% of 78,595 rounds), and the 2026-to-2024 cross-period log loss (0.50441 vs 0.43660 in
period) uses those labels. The batch-holdout check is said to be possibly run on 2024 rounds.

Part 1 (label defect). Truth per (match, round) from two sources that do not read round_end.winner_team_code:
  bomb  : bomb_state event. bomb_exploded -> T (2) wins, bomb_defused -> CT (3) wins.  ASSUMPTION: CS rule.
  score : freeze-end score of round k+1 minus freeze-end score of round k (round_state).
          No-swap reading (T score +1 or CT score +1), or the swap reading when the bake-off schedule
          (bakeoff.swap_expected rule, max_rounds = 2 x halftime round + 2, default 24 in 2024) swaps sides
          after round k.  ASSUMPTION: the score fields are side-based, as the bake-off assumes.
  deaths: distinct player deaths per side per round (player_death), a fourth source for reason 8 / 9 rounds.
          ASSUMPTION: the death feed is complete enough for an elimination test.
  control: the same truth on the 2026 tome (claim: 0 mismatches in 7,427 rounds).

Part 2 (cross-period). Reuses cross_period.py's own pipeline (run_2026, run_2024, fit_xgb, predict, and the
bake-off builders it imports). Rebuilds Model A and the 2024 states, reproduces the reported 0.50441, reproduces
Model B for the in-period 0.43660, then recomputes the 2024 log loss with corrected 2024 labels (same states,
same model). The corrected labels here are this verifier's labels from Part 1, not the D2-4 labels, which were
not available. Shared code (state builder, kept-set filters) is not re-verified here.
"""

import glob
import importlib.util
import re
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(24)
pl.Config.set_fmt_str_lengths(40)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF24 = ".2023-12-28,2024-02-25"
TOME26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUF26 = ".2026-09-18,2026-10-08"
T, CT = 2, 3
REPORTED_IN_PERIOD = 0.43660
REPORTED_CROSS = 0.50441
REPORTED_FLIP = {"round13": 1888, "round25": 114, "round28": 115, "total": 2117, "denominator": 78595}
GAP_LIMIT = 0.02  # verdict rule from cross_period.py, fixed there before its run
BOOT_REPS = 200  # ASSUMPTION: 200 match-level resamples for speed (batch_holdout used 1000)
T0 = time.time()

spec = importlib.util.spec_from_file_location("cp_verify_d27", ROOT / "scripts" / "model" / "sufficiency" / "cross_period.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)
bk = cp.bk


def read_folder(folder, cols):
    fs = sorted(glob.glob(str(folder / "dataframe_*")))
    if not fs:
        raise SystemExit(f"no parquet files in {folder}")
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def swap_after(r, mx):
    """bakeoff.swap_expected, as a polars expression."""
    return (r == (mx // 2)) | ((r > mx) & (((r - mx - 3) % 6) == 0))


def build_truth(tome, suf, year):
    print(f"\n== truth table {year}: {tome} ==")
    RE = read_folder(tome / f"basic_round_end{suf}",
                     ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
    RS = read_folder(tome / f"basic_round_state{suf}",
                     ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    BS = read_folder(tome / f"basic_bomb_state{suf}", ["match_key", "round", "tick", "event_type"])
    n_keys = RE.select(["match_key", "round"]).unique().height
    print(f"round_end rows {RE.height}; distinct (match, round) {n_keys}")
    RE1 = RE.unique(["match_key", "round"], keep="last")  # cross_period.py keeps the last row too
    FZ = (RS.filter(pl.col("event_type") == "round_freeze_end")
          .unique(["match_key", "round"], keep="first")
          .select(["match_key", "round", "t_score", "ct_score"]))
    FZn = (FZ.with_columns((pl.col("round") - 1).alias("round"))
           .select(["match_key", "round", pl.col("t_score").alias("t_next"), pl.col("ct_score").alias("ct_next")]))
    half = (RS.filter(pl.col("event_type") == "round_announce_last_round_half")
            .group_by("match_key").agg(pl.col("round").min().alias("half_r")))
    keys = RE1.select("match_key").unique()
    if year == 2024:
        # cross_period.py rule: 2 x (halftime announcement round) + 2; default 24 (no max_rounds in this header)
        mx = (keys.join(half, on="match_key", how="left")
              .with_columns(pl.when(pl.col("half_r").is_not_null()).then(pl.col("half_r") * 2 + 2)
                            .otherwise(pl.lit(24)).alias("maxr")))
    else:
        hdr = read_folder(tome / "header", ["key", "max_rounds"]).rename({"key": "match_key"})
        mx = keys.join(hdr.unique("match_key", keep="first"), on="match_key", how="left") \
                 .rename({"max_rounds": "maxr"})
        print(f"2026 header max_rounds distribution: {dict(hdr['max_rounds'].value_counts().rows())}")
    bomb = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
            .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("ev")))
    n_both = bomb.filter(pl.col("ev").list.contains("bomb_defused") & pl.col("ev").list.contains("bomb_exploded")).height
    print(f"rounds with bomb outcome events: {bomb.height}; with both defused and exploded: {n_both}")
    bomb = bomb.with_columns(
        pl.when(pl.col("ev").list.contains("bomb_defused") & ~pl.col("ev").list.contains("bomb_exploded"))
        .then(pl.lit(CT))
        .when(pl.col("ev").list.contains("bomb_exploded") & ~pl.col("ev").list.contains("bomb_defused"))
        .then(pl.lit(T)).otherwise(pl.lit(None, dtype=pl.Int64)).alias("bomb_side"))

    base = (RE1.join(FZ, on=["match_key", "round"], how="left")
            .join(FZn, on=["match_key", "round"], how="left")
            .join(bomb.select(["match_key", "round", "bomb_side"]), on=["match_key", "round"], how="left")
            .join(mx.select(["match_key", "maxr"]), on="match_key", how="left"))
    sT_nosw = (pl.col("t_next") - pl.col("t_score") == 1) & (pl.col("ct_next") - pl.col("ct_score") == 0)
    sCT_nosw = (pl.col("ct_next") - pl.col("ct_score") == 1) & (pl.col("t_next") - pl.col("t_score") == 0)
    # swap reading: after a side swap the team that was T is now on the CT side, so T won means
    # ct_next - t_score == 1 and t_next - ct_score == 0 (checked on match m_161472e7, round 12)
    sT_sw = (pl.col("ct_next") - pl.col("t_score") == 1) & (pl.col("t_next") - pl.col("ct_score") == 0)
    sCT_sw = (pl.col("t_next") - pl.col("ct_score") == 1) & (pl.col("ct_next") - pl.col("t_score") == 0)
    sw = swap_after(pl.col("round"), pl.col("maxr"))
    base = base.with_columns(
        pl.when(sw & sT_sw).then(pl.lit(T)).when(sw & sCT_sw).then(pl.lit(CT))
        .when(~sw & sT_nosw).then(pl.lit(T)).when(~sw & sCT_nosw).then(pl.lit(CT))
        .otherwise(pl.lit(None, dtype=pl.Int64)).alias("score_side"),
        swap_after(pl.col("round"), pl.col("maxr")).alias("sched_swap_after"),
    )
    base = base.with_columns(
        (pl.col("winner_team_code") == CT).cast(pl.Int8).alias("w_ct"),
        pl.when(pl.col("bomb_side") == CT).then(pl.lit(1)).when(pl.col("bomb_side") == T).then(pl.lit(0))
        .otherwise(pl.lit(-1)).cast(pl.Int8).alias("bomb_ct"),
        pl.when(pl.col("score_side") == CT).then(pl.lit(1)).when(pl.col("score_side") == T).then(pl.lit(0))
        .otherwise(pl.lit(-1)).cast(pl.Int8).alias("score_ct"),
        pl.when(pl.col("round") == (pl.col("maxr") // 2) + 1).then(pl.lit("first_after_halftime"))
        .when(pl.col("round") == pl.col("maxr") + 1).then(pl.lit("first_OT_round"))
        .when(pl.col("round") == pl.col("maxr") + 4).then(pl.lit("first_OT_2nd_half"))
        .otherwise(pl.lit("other")).alias("rbin"),
        pl.col("match_key").str.extract(r"csds/20\d\d/\d\d/(\d\d)/", 1).alias("day"),
    )
    print(f"schedule: {base.filter(pl.col('sched_swap_after')).height} rounds followed by a scheduled swap")
    return base, mx, n_keys


def summarize(base, title):
    df = base.with_columns(
        pl.when(pl.col("day").cast(pl.Int64) <= 8).then(pl.lit("days01-08"))
        .otherwise(pl.lit("days12-25")).alias("grp"))
    agg = df.group_by(["grp", "rbin"]).agg(
        pl.len().alias("rounds"),
        (pl.col("bomb_ct") >= 0).sum().alias("bomb_known"),
        ((pl.col("bomb_ct") >= 0) & (pl.col("bomb_ct") != pl.col("w_ct"))).sum().alias("winner_ne_bomb"),
        (pl.col("score_ct") >= 0).sum().alias("score_known"),
        ((pl.col("score_ct") >= 0) & (pl.col("score_ct") != pl.col("w_ct"))).sum().alias("winner_ne_score"),
        ((pl.col("bomb_ct") >= 0) & (pl.col("score_ct") >= 0)).sum().alias("both_known"),
        ((pl.col("bomb_ct") >= 0) & (pl.col("score_ct") >= 0) & (pl.col("bomb_ct") != pl.col("score_ct")))
        .sum().alias("bomb_ne_score"),
    ).sort(["grp", "rbin"])
    print(f"\n-- {title}: by round bin --")
    print(agg)
    byr = df.filter(pl.col("round").is_in([9, 13, 25, 28])).group_by(["grp", "round"]).agg(
        pl.len().alias("rounds"),
        (pl.col("bomb_ct") >= 0).sum().alias("bomb_known"),
        ((pl.col("bomb_ct") >= 0) & (pl.col("bomb_ct") != pl.col("w_ct"))).sum().alias("winner_ne_bomb"),
        ((pl.col("score_ct") >= 0) & (pl.col("score_ct") != pl.col("w_ct"))).sum().alias("winner_ne_score"),
        (pl.col("score_ct") >= 0).sum().alias("score_known"),
    ).sort(["grp", "round"])
    print(f"-- {title}: rounds 9, 13, 25, 28 --")
    print(byr)
    tot = df.select(
        pl.len().alias("rounds"),
        (pl.col("bomb_ct") >= 0).sum().alias("bomb_known"),
        ((pl.col("bomb_ct") >= 0) & (pl.col("bomb_ct") != pl.col("w_ct"))).sum().alias("winner_ne_bomb"),
        (pl.col("score_ct") >= 0).sum().alias("score_known"),
        ((pl.col("score_ct") >= 0) & (pl.col("score_ct") != pl.col("w_ct"))).sum().alias("winner_ne_score"),
        ((pl.col("bomb_ct") >= 0) & (pl.col("score_ct") >= 0)).sum().alias("both_known"),
        ((pl.col("bomb_ct") >= 0) & (pl.col("score_ct") >= 0) & (pl.col("bomb_ct") != pl.col("score_ct")))
        .sum().alias("bomb_ne_score"),
        ((pl.col("bomb_ct") >= 0) | (pl.col("score_ct") >= 0)).sum().alias("any_truth_known"),
        (((pl.col("bomb_ct") >= 0) & (pl.col("bomb_ct") != pl.col("w_ct")))
         | ((pl.col("score_ct") >= 0) & (pl.col("score_ct") != pl.col("w_ct")))).sum().alias("winner_flagged_any"),
    )
    print(f"-- {title}: all rounds in this tome --")
    print(tot)
    return df


def death_check(base, tome, suf):
    print("\n== fourth source: distinct deaths per side per round (2024 tome) ==")
    DE = read_folder(tome / f"basic_player_death{suf}", ["match_key", "round", "player_id", "player_team_code"])
    PI = read_folder(tome / f"basic_player_info{suf}", ["match_key", "round", "player_id", "team_code"])
    pi1 = PI.unique(["match_key", "round", "player_id"], keep="first")
    d = (DE.join(pi1, on=["match_key", "round", "player_id"], how="left")
         .with_columns(pl.coalesce(["player_team_code", "team_code"]).alias("side"))
         .filter(pl.col("side").is_in([T, CT]))
         .unique(["match_key", "round", "player_id"], keep="first"))
    t_d = d.filter(pl.col("side") == T).group_by(["match_key", "round"]).len().rename({"len": "t_deaths"})
    c_d = d.filter(pl.col("side") == CT).group_by(["match_key", "round"]).len().rename({"len": "ct_deaths"})
    j = (base.join(t_d, on=["match_key", "round"], how="left").join(c_d, on=["match_key", "round"], how="left")
         .with_columns(pl.col("t_deaths").fill_null(0), pl.col("ct_deaths").fill_null(0),
                       (pl.col("day").cast(pl.Int64) >= 12).alias("late")))
    j = j.with_columns(
        (pl.col("ct_deaths") == 5).alias("all_ct_dead"), (pl.col("t_deaths") == 5).alias("all_t_dead"))
    r9 = j.filter(pl.col("win_reason_code") == 9)
    r8 = j.filter(pl.col("win_reason_code") == 8)
    print("reason 9 (round_end says T won by eliminating CT): share with all 5 CT dead")
    print(r9.group_by(["rbin", "late"]).agg(pl.len().alias("n"), pl.col("all_ct_dead").mean().alias("share"))
          .sort(["rbin", "late"]))
    print("reason 8 (round_end says CT won): share with all 5 T dead")
    print(r8.group_by(["rbin", "late"]).agg(pl.len().alias("n"), pl.col("all_t_dead").mean().alias("share"))
          .sort(["rbin", "late"]))
    fl = j.filter((pl.col("rbin") == "first_after_halftime") & pl.col("late") & (pl.col("bomb_ct") >= 0)
                  & (pl.col("bomb_ct") != pl.col("w_ct")))
    ok = j.filter((pl.col("rbin") == "first_after_halftime") & pl.col("late") & (pl.col("bomb_ct") >= 0)
                  & (pl.col("bomb_ct") == pl.col("w_ct")))
    print("first-after-halftime rounds, late days, bomb known. flagged = winner contradicts bomb; agree = winner matches bomb")
    print(fl.group_by(["winner_team_code", "win_reason_code", "bomb_side"]).agg(
        pl.len().alias("n_flagged"), pl.col("ct_deaths").mean().alias("mean_ct_deaths"),
        pl.col("t_deaths").mean().alias("mean_t_deaths"), pl.col("all_ct_dead").sum().alias("n_all_ct_dead")))
    print(ok.group_by(["winner_team_code", "win_reason_code", "bomb_side"]).agg(
        pl.len().alias("n_agree"), pl.col("ct_deaths").mean().alias("mean_ct_deaths"),
        pl.col("t_deaths").mean().alias("mean_t_deaths"), pl.col("all_ct_dead").sum().alias("n_all_ct_dead")))


def ll_rows(y, p):
    pc = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(pc) + (1 - y) * np.log(1 - pc))


def part1():
    base24, mx24, n_keys24 = build_truth(TOME24, SUF24, 2024)
    print(f"\n2024 maxr distribution (all round_end matches): {dict(mx24['maxr'].value_counts().rows())}")
    df24 = summarize(base24, "2024 (all round_end rounds)")
    print("\n-- claim check: 2024 round-13 bomb rows, day 12-25 vs 01-07 (bomb known only) --")
    late = base24.filter((pl.col("round") == 13) & (pl.col("bomb_ct") >= 0))
    print(late.with_columns((pl.col("day").cast(pl.Int64) >= 12).alias("late"))
          .group_by("late").agg(pl.len().alias("n_bomb"),
                                ((pl.col("bomb_ct") != pl.col("w_ct"))).sum().alias("mismatch")))

    print("\n-- denominators available in the 2024 outputs --")
    print("round_end distinct (match, round):", n_keys24, "| rounds in cross_period kept set: 53816 (from cross_period.json)")

    # the claim's round set vs the score flags, all 2024 rounds
    flags = df24.filter(pl.col("score_ct") >= 0).filter(pl.col("score_ct") != pl.col("w_ct"))
    print("\n-- score-flagged rounds (winner != score truth), all 2024 rounds, by round --")
    print(flags.group_by("round").len().sort("len", descending=True).head(12))

    death_check(base24, TOME24, SUF24)

    base26, mx26, n_keys26 = build_truth(TOME26, SUF26, 2026)
    df26 = summarize(base26, "2026 control (same method)")
    print("\n2026 claim: 0 mismatches in 7,427 rounds. Our 2026 rounds with bomb or score truth:",
          df26.filter((pl.col("bomb_ct") >= 0) | (pl.col("score_ct") >= 0)).height)
    return base24, base26


def part2(base24):
    print("\n========== Part 2: cross-period recomputation ==========")
    A = cp.run_2026()
    B = cp.run_2024()
    S = B["S"]
    print(f"\n2024 states: {S.height}; matches {S['match_key'].n_unique()}; "
          f"rounds {S.select(['match_key', 'round']).unique().height}; label CT share {S['label_ct_win'].mean():.5f}")

    Xtr_a, ytr_a = bk.xy(A["S_trA"])
    Xva_a, yva_a = bk.xy(A["S_vaA"])
    mA, devA, biA, errA = cp.fit_xgb(Xtr_a, ytr_a, Xva_a, yva_a)
    X24, y24_orig = bk.xy(S)
    p24 = cp.predict(mA, X24, biA)
    Xtr_b, ytr_b = bk.xy(A["S_trB"])
    Xva_b, yva_b = bk.xy(A["S_vaB"])
    Xte_b, yte_b = bk.xy(A["S_teB"])
    mB, devB, biB, errB = cp.fit_xgb(Xtr_b, ytr_b, Xva_b, yva_b)
    pte_b = cp.predict(mB, Xte_b, biB)
    ll_B = bk.metrics(yte_b, pte_b)["logloss"]
    print(f"Model A device {devA} best_iter {biA}; Model B device {devB} best_iter {biB}")
    print(f"In-period Model B test log loss (recomputed): {ll_B:.5f}   (reported {REPORTED_IN_PERIOD:.5f})")

    ci = bk.FEATS.index("ct_alive_minus_t")
    rate_a = float(ytr_a.mean())
    p24_const = np.full(len(y24_orig), rate_a)
    lrA = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr_a[:, [ci]], ytr_a)
    p24_lr = lrA.predict_proba(X24[:, [ci]])[:, 1]
    lrB = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr_b[:, [ci]], ytr_b)
    pte_lr = lrB.predict_proba(Xte_b[:, [ci]])[:, 1]
    ll_lrB = bk.metrics(yte_b, pte_lr)["logloss"]
    ll_orig = bk.metrics(y24_orig, p24)["logloss"]
    print(f"2024 cross-period log loss, original labels (recomputed): {ll_orig:.5f}   (reported {REPORTED_CROSS:.5f})")
    print(f"In-period LR control (B) recomputed: {ll_lrB:.5f} (cross_period.json: 0.460934)")

    # truth labels joined onto the 2024 states (row order preserved)
    tr = base24.select(["match_key", "round", "w_ct", "bomb_ct", "score_ct", "rbin"])
    S_idx = S.select(["match_key", "round", "label_ct_win", "kind"]).with_row_index("ridx")
    J = S_idx.join(tr, on=["match_key", "round"], how="left").sort("ridx")
    assert J.height == S.height, "join changed the number of states"
    n_no_truth = J.filter(pl.col("w_ct").is_null()).height
    mism_w = J.filter(pl.col("w_ct").is_not_null() & (pl.col("w_ct") != pl.col("label_ct_win"))).height
    print(f"states with no round_end truth row: {n_no_truth}; states whose label != round_end winner recomputed: {mism_w}")
    bomb = J["bomb_ct"].fill_null(-1).to_numpy().astype(np.int64)
    score = J["score_ct"].fill_null(-1).to_numpy().astype(np.int64)
    y_orig = J["label_ct_win"].to_numpy().astype(np.int64)
    y_bomb = np.where(bomb >= 0, bomb, y_orig)
    y_score = np.where(score >= 0, score, y_orig)
    y_both = np.where(bomb >= 0, bomb, np.where(score >= 0, score, y_orig))
    live = (J["kind"] == "live_start").to_numpy()

    variants = [("original (round_end winner)", y_orig),
                ("bomb-only correction", y_bomb),
                ("score-only correction", y_score),
                ("bomb then score correction (primary)", y_both)]
    print("\n-- cross-period 2024, XGBoost Model A (same model, same states), label variants --")
    print(f"{'label variant':<40}{'flip rounds':>12}{'flip states':>12}{'LL all':>10}{'LL live':>10}"
          f"{'gap vs 0.43660':>16}{'LR LL':>10}{'const LL':>10}{'CT share':>10}")
    results = {}
    for name, y in variants:
        chg = (y != y_orig)
        flip_rounds = J.with_columns(pl.Series("yv", y)).filter(pl.col("yv") != pl.col("label_ct_win")) \
            .select(["match_key", "round"]).unique().height
        ll_all = bk.metrics(y, p24)["logloss"]
        ll_live = bk.metrics(y[live], p24[live])["logloss"]
        ll_lr = bk.metrics(y, p24_lr)["logloss"]
        ll_c = bk.metrics(y, p24_const)["logloss"]
        results[name] = (ll_all, ll_live, ll_lr, ll_c, flip_rounds, int(chg.sum()))
        print(f"{name:<40}{flip_rounds:>12}{int(chg.sum()):>12}{ll_all:>10.5f}{ll_live:>10.5f}"
              f"{ll_all - ll_B:>+16.5f}{ll_lr:>10.5f}{ll_c:>10.5f}{y.mean():>10.4f}")

    # attribution: how much of the original log loss sits on states of flipped rounds
    l_orig = ll_rows(y_orig.astype(float), p24)
    flip_mask = (y_both != y_orig)
    print(f"\nattribution (primary variant): flipped states {int(flip_mask.sum())} of {len(y_orig)}; "
          f"their mean loss {l_orig[flip_mask].mean():.4f} vs unflipped {l_orig[~flip_mask].mean():.4f}; "
          f"share of total loss from flipped states {l_orig[flip_mask].sum() / l_orig.sum():.4f}")
    ll_corr_only_unflip = l_orig[~flip_mask].sum() / len(y_orig)
    print(f"log loss over unflipped states only (original label, all states denominator): {ll_corr_only_unflip:.5f}")
    print("per-bin mean loss, original labels (flipped rounds vs others):")
    tbl = J.with_columns(pl.Series("loss", l_orig), pl.Series("flip", flip_mask)).group_by(["rbin", "flip"]) \
        .agg(pl.len().alias("states"), pl.col("loss").mean().alias("mean_loss")).sort(["rbin", "flip"])
    print(tbl)

    # match-level bootstrap of the primary variant and of the paired difference
    rng = np.random.default_rng(0)
    mi = S.select(pl.col("match_key").cast(pl.Categorical).to_physical()).to_numpy().ravel()
    n_m = int(mi.max()) + 1
    l_corr = ll_rows(y_both.astype(float), p24)
    bo, bc = [], []
    for _ in range(BOOT_REPS):
        idx = rng.integers(0, n_m, size=n_m)
        w = np.bincount(idx, minlength=n_m)[mi].astype(float)
        bo.append((w * l_orig).sum() / w.sum())
        bc.append((w * l_corr).sum() / w.sum())
    bo, bc = np.array(bo), np.array(bc)
    d = bc - bo
    print(f"\nmatch bootstrap ({BOOT_REPS} reps, seed 0; ASSUMPTION): original LL 95% CI "
          f"[{np.percentile(bo, 2.5):.5f}, {np.percentile(bo, 97.5):.5f}]; primary-corrected LL 95% CI "
          f"[{np.percentile(bc, 2.5):.5f}, {np.percentile(bc, 97.5):.5f}]; paired (corrected - original) "
          f"[{np.percentile(d, 2.5):+.5f}, {np.percentile(d, 97.5):+.5f}]")

    ll_c_primary = results["bomb then score correction (primary)"][0]
    gap_c = ll_c_primary - ll_B
    print(f"\nVERDICT RULE (cross_period.py, fixed before its run): transfers if corrected 2024 gap <= {GAP_LIMIT}")
    print(f"original gap {ll_orig - ll_B:+.5f}; corrected (primary) gap {gap_c:+.5f}")
    return {"ll_orig": ll_orig, "ll_B": ll_B, "ll_corr": ll_c_primary, "ll_lrB": ll_lrB,
            "variants": results}


def batch_holdout_check():
    print("\n========== Batch holdout: was it run on 2024 rounds? ==========")
    src = (ROOT / "scripts" / "model" / "sufficiency" / "batch_holdout.py").read_text(encoding="utf-8")
    hits = [m.start() for m in re.finditer(r"2024|tomes|TOME|2023-12-28", src)]
    lines = sorted({src[:h].count("\n") + 1 for h in hits})
    print(f"batch_holdout.py mentions of 2024 / tomes / TOME / 2023-12-28: lines {lines or 'none'}")
    print("batch_holdout.py reads 2026 only via ex.bakeoff_kept() (extract_state_1hz) -> fv.TOME is the 2026 tome")
    import json
    bh = json.loads((ROOT / "notes" / "model" / "sufficiency" / "batch_holdout.json").read_text(encoding="utf-8"))
    print(f"batch_holdout.json batch_definition: {bh['batch_definition']}; header_rows {bh['header_rows']}; "
          f"header_matches_by_day keys {list(bh['header_matches_by_day'].keys())[:3]}... ; "
          f"kept_matches {bh['bakeoff_kept_check']['got']['kept_matches']}")
    cp_src = (ROOT / "scripts" / "model" / "sufficiency" / "cross_period.py").read_text(encoding="utf-8").splitlines()
    print(f"cross_period.py line 211: {cp_src[210].strip()}")


def main():
    print("D2-7 verifier (independent). Interpreter: project pixi env, -I -X utf8.")
    batch_holdout_check()
    base24, base26 = part1()
    res = part2(base24)
    print(f"\nTotal wall time: {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
