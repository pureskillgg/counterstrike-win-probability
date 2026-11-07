"""D2 final counts. Reads D2_truth_{2024,2026}.parquet written by D2_truth.py.

Truth T = side-attached winner (round_state freeze-end score deltas, team-tracked; cross-checked with bomb
outcome and elimination). Flipped row = recorded winner_team_code R != T.

Counts:
  1. flipped rows by round group, and whether R equals the static team identity (2 if team X won, 3 if Y won)
     or the lag-1 team code (winning team's side in round r-1), and the same for unflipped rows.
  2. win_reason side (T reasons 1, 9, 13, 17; CT reasons 7, 8, 11, 12, 18) vs R and vs T in flipped rows.
  3. elimination side vs T and R in flipped rows.
  4. match-level co-occurrence of flips at rounds 13, 25, 28 (matches that reach overtime).
  5. winner-derived cumulative score lead (sum of recorded winners, CT +1 / T -1) vs round_state lead
     (ct_score - t_score at round_freeze_end). Rounds where they differ.
  6. round_start vs round_end tick order (window caveat).
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\D2_final.py
"""
import glob
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(40)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
T_REASON = [1, 9, 13, 17]       # ASSUMPTION
CT_REASON = [7, 8, 11, 12, 18]  # ASSUMPTION
STATES_PER_ROUND = 234265 / 28683  # ASSUMPTION: bake-off ratio of kept states to kept rounds


def read(label, ch, cols):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))[0]
    return pl.scan_parquet(sorted(glob.glob(fd + "/dataframe_*"))).select(cols).collect()


def grp(expr_round):
    return (pl.when(pl.col("round") == 13).then(pl.lit("r13"))
            .when(pl.col("round") == 25).then(pl.lit("r25"))
            .when(pl.col("round") == 28).then(pl.lit("r28"))
            .when(pl.col("round") <= 12).then(pl.lit("r01_12"))
            .when(pl.col("round") <= 24).then(pl.lit("r14_24"))
            .otherwise(pl.lit("r29plus")).alias("grp"))


def main():
    tabs = {}
    for label in ["2024", "2026"]:
        t = pl.read_parquet(ROOT / "logs" / "swarm2" / f"D2_truth_{label}.parquet")
        if "s_bomb_right" in t.columns:
            t = t.with_columns(pl.col("s_bomb_right").alias("s_bomb_in"))
        else:
            t = t.with_columns(pl.col("s_bomb").alias("s_bomb_in"))
        t = t.with_columns(
            pl.when(pl.col("win_reason_code").is_in(T_REASON)).then(pl.lit(2))
            .when(pl.col("win_reason_code").is_in(CT_REASON)).then(pl.lit(3))
            .otherwise(pl.lit(None)).cast(pl.Int64).alias("s_reason2"))
        t = t.with_columns(grp(pl.col("round")))
        t = t.with_columns(
            pl.when(pl.col("X_won") == 1).then(pl.lit(2)).otherwise(pl.lit(3)).cast(pl.Int64).alias("pred_static"),
            pl.when(pl.col("X_won") == 1).then(pl.col("sx_prev")).otherwise(5 - pl.col("sx_prev")).cast(pl.Int64).alias("pred_lag"))
        tabs[label] = t
    print("columns 2024 truth table:", tabs["2024"].columns)

    print("\n== 1. Recorded winner vs truth, by round group (rows with truth T and sx known)")
    for label in ["2024", "2026"]:
        t = tabs[label].filter(pl.col("T").is_not_null())
        print(f"\n-- {label}: rows with T = {t.height}")
        print(t.group_by("grp").agg(
            pl.len().alias("n_T"),
            pl.col("flip").sum().alias("flipped"),
            pl.col("flip").mean().round(4).alias("flip_rate"),
        ).sort("grp"))
        fl = t.filter(pl.col("flip") & pl.col("sx").is_not_null() & pl.col("sx_prev").is_not_null())
        un = t.filter(~pl.col("flip") & pl.col("sx").is_not_null() & pl.col("sx_prev").is_not_null())
        print("  flipped rows with sx, sx_prev: ", fl.height)
        print("    R == static identity code:", fl.filter(pl.col("R") == pl.col("pred_static")).height,
              "  R == lag-1 team code:", fl.filter(pl.col("R") == pl.col("pred_lag")).height)
        print("    by group, flipped: R==static / R==lag1 / n:")
        print(fl.group_by("grp").agg(
            (pl.col("R") == pl.col("pred_static")).sum().alias("R==static"),
            (pl.col("R") == pl.col("pred_lag")).sum().alias("R==lag1"),
            pl.len().alias("n")).sort("grp"))
        print("  unflipped rows with sx, sx_prev:", un.height)
        print("    R == static identity code:", un.filter(pl.col("R") == pl.col("pred_static")).height,
              "  R == lag-1 team code:", un.filter(pl.col("R") == pl.col("pred_lag")).height)
        print(un.filter(pl.col("sx") != pl.col("sx_prev")).group_by("grp").agg(
            (pl.col("R") == pl.col("pred_static")).sum().alias("unflipped_swap:R==static"),
            (pl.col("R") == pl.col("pred_lag")).sum().alias("unflipped_swap:R==lag1"),
            pl.len().alias("n_swap_unflipped")).sort("grp"))

    print("\n== 2. win_reason side vs R and vs T")
    for label in ["2024", "2026"]:
        t = tabs[label].filter(pl.col("T").is_not_null() & pl.col("s_reason2").is_not_null())
        print(f"-- {label}: rows {t.height}; reason==R {t.filter(pl.col('s_reason2') == pl.col('R')).height}; "
              f"reason==T {t.filter(pl.col('s_reason2') == pl.col('T')).height}")
        f = t.filter(pl.col("flip"))
        print(f"   flipped rows {f.height}: reason==R {f.filter(pl.col('s_reason2') == pl.col('R')).height}, "
              f"reason==T {f.filter(pl.col('s_reason2') == pl.col('T')).height}")

    print("\n== 3. elimination side in flipped rows")
    for label in ["2024", "2026"]:
        t = tabs[label].filter(pl.col("T").is_not_null() & pl.col("s_elim").is_not_null())
        f = t.filter(pl.col("flip"))
        print(f"-- {label}: rows with elim and T {t.height}; elim==T {t.filter(pl.col('s_elim') == pl.col('T')).height}; "
              f"flipped rows with elim {f.height}: elim==T {f.filter(pl.col('s_elim') == pl.col('T')).height}, "
              f"elim==R {f.filter(pl.col('s_elim') == pl.col('R')).height}")

    print("\n== 3b. bomb outcome vs T and R (in-window bomb event)")
    for label in ["2024", "2026"]:
        t = tabs[label].filter(pl.col("s_bomb_in").is_not_null() & pl.col("T").is_not_null())
        f = t.filter(pl.col("flip"))
        print(f"-- {label}: bomb rows with T {t.height}; bomb==T {t.filter(pl.col('s_bomb_in') == pl.col('T')).height}; "
              f"bomb==R {t.filter(pl.col('s_bomb_in') == pl.col('R')).height}; flipped bomb rows {f.height}: "
              f"bomb==T {f.filter(pl.col('s_bomb_in') == pl.col('T')).height}")

    print("\n== 4. match-level co-occurrence of flips (2024)")
    t = tabs["2024"].filter(pl.col("T").is_not_null())
    f13 = t.filter(pl.col("round") == 13).select(["match_key", pl.col("flip").alias("f13")])
    f25 = t.filter(pl.col("round") == 25).select(["match_key", pl.col("flip").alias("f25")])
    f28 = t.filter(pl.col("round") == 28).select(["match_key", pl.col("flip").alias("f28")])
    print("matches with truth at 13:", f13.height, " flipped at 13:", f13.filter(pl.col("f13")).height)
    ot = f13.join(f25, on="match_key", how="inner").join(f28, on="match_key", how="inner")
    print("matches with truth at 13, 25 and 28 (reached OT):", ot.height)
    print(ot.group_by(["f13", "f25", "f28"]).len().sort(["f13", "f25", "f28"]))
    print("matches with flips at 13 but truth missing at 25 or 28 (non-OT or missing):",
          f13.filter(pl.col("f13")).join(f25, on="match_key", how="anti").height)
    miss = t.filter(pl.col("flip") & ~pl.col("round").is_in([13, 25, 28]))
    print("flipped rows outside rounds 13, 25, 28 (2024):", miss.height)
    print(miss.select(["match_key", "round", "R", "T", "s_elim", "s_bomb_in", "s_reason2", "sx", "sx_prev"]).sort(["match_key", "round"]))

    print("\n== 5. winner-derived cumulative lead vs round_state lead (ct_score - t_score at round_freeze_end)")
    for label in ["2024", "2026"]:
        t = tabs[label].filter(pl.col("t0").is_not_null())
        recs = defaultdict(dict)
        for row in t.select(["match_key", "round", "winner", "T", "t0", "ct0", "flip"]).iter_rows():
            mk, rd, R, T, t0, ct0, fl = row
            recs[mk][rd] = (R, T, ct0 - t0)
        n_rounds = n_bad = n_states_chk = 0
        bad_matches = set()
        n_cumT_bad = 0
        n_matches_complete = 0
        for mk, byr in recs.items():
            rmax = max(byr)
            if rmax != len(byr) or min(byr) != 1:
                continue  # need contiguous rounds from 1 for a cumulative sum
            n_matches_complete += 1
            cumR = 0
            cumT = 0
            t_complete = True
            for rd in range(1, rmax + 1):
                R, T, lead_fe = byr[rd]
                n_rounds += 1
                if cumR != lead_fe:
                    n_bad += 1
                    bad_matches.add(mk)
                if T is None:
                    t_complete = False
                elif t_complete and cumT != lead_fe:
                    n_cumT_bad += 1
                # update cumulative sums after the round
                cumR += 1 if R == 3 else -1
                if T is None:
                    t_complete = False
                else:
                    cumT += 1 if T == 3 else -1
        print(f"-- {label}: matches with contiguous rounds from 1: {n_matches_complete}; rounds checked: {n_rounds}; "
              f"rounds where winner-derived lead != round_state lead: {n_bad} in {len(bad_matches)} matches; "
              f"rounds where truth-derived lead != round_state lead (with full truth history): {n_cumT_bad}")
        print(f"   ASSUMPTION states affected approx = rounds x {STATES_PER_ROUND:.2f}: {n_bad * STATES_PER_ROUND:,.0f}")

    print("\n== 6. round_start vs round_end tick order (window caveat)")
    for label in ["2024", "2026"]:
        rs = read(label, "round_start", ["match_key", "round", "tick"]).sort("tick").unique(["match_key", "round"], keep="first")
        re_ = read(label, "round_end", ["match_key", "round", "tick"]).sort("tick").unique(["match_key", "round"], keep="first")
        j = rs.join(re_, on=["match_key", "round"], suffix="_end")
        print(f"-- {label}: rounds with both: {j.height}; round_start tick > round_end tick: "
              f"{j.filter(pl.col('tick') > pl.col('tick_end')).height}; "
              f"median (start - end) ticks: {j.select((pl.col('tick') - pl.col('tick_end')).median()).item()}")


if __name__ == "__main__":
    main()
