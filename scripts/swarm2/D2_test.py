"""D2 test: is round_end.winner_team_code attached to team identity (not side)?

Hypothesis D2: winner_team_code in round_end follows a team identity rather than the side.

Per (match, round) the script builds
  sx        side (2 = T, 3 = CT) of team X, from player_info.team_code. Team X = the five players on
            code 2 in the first valid round of the match (valid = 10 players, 5 on code 2, 5 on code 3).
  sx_prev / sx_next   the same for rounds r-1 and r+1 (swap detection).
  s_reason  winning side from round_end.win_reason_code (T reasons 1, 9, 13, 17; CT reasons 7, 8, 11, 12, 18).
  s_bomb    winning side from bomb outcome (exploded -> 2, defused -> 3), when a bomb outcome exists.
  s_score   winning side from round_state freeze-end scores, following each team across the swap
            (ASSUMPTION: t_score and ct_score are side-attached).
  winner    recorded round_end.winner_team_code (the field under test).

Readings compared with the recorded winner:
  side          prediction = s_reason
  static team   team X keeps identity code 2 (its round-1 side) for the whole match:
                prediction = 2 if X won (X side == winning side) else 3
  lag-1 team    the winning team's code is the side it held in round r-1 (one-round-stale mapping):
                prediction = side of the winning team in round r-1

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\D2_test.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(40)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
PC = {"2024": 20, "2026": 10}  # ASSUMPTION: round_end.player_count of a full 5v5 round (e2 check uses 20 for 2024; bake-off uses 10 for 2026)
T_REASON = [1, 9, 13, 17]       # ASSUMPTION: CS round-end reason codes won by T
CT_REASON = [7, 8, 11, 12, 18]  # ASSUMPTION: CS round-end reason codes won by CT


def read(label, ch, cols):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))[0]
    return pl.scan_parquet(sorted(glob.glob(fd + "/dataframe_*"))).select(cols).collect()


def bin_expr():
    return (pl.when(pl.col("round") == 13).then(pl.lit("r13"))
            .when(pl.col("round") == 25).then(pl.lit("r25"))
            .when(pl.col("round") <= 12).then(pl.lit("r01_12"))
            .when(pl.col("round") <= 24).then(pl.lit("r14_24"))
            .otherwise(pl.lit("r26plus")).alias("bin"))


def build(label):
    pc = PC[label]
    stats = {}
    re_all = read(label, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
    stats["round_end rows"] = re_all.height
    re_ = re_all.filter(pl.col("player_count") == pc)
    stats["round_end rows with player_count == %d" % pc] = re_.height
    stats["(match,round) pairs with >1 round_end row"] = re_.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1).height
    re_ = re_.sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first", maintain_order=True)

    pi = (read(label, "player_info", ["match_key", "round", "player_id", "team_code"])
          .unique(["match_key", "round", "player_id"], keep="first", maintain_order=True))
    cnt = pi.group_by(["match_key", "round"]).agg(
        pl.len().alias("np"), (pl.col("team_code") == 2).sum().alias("nt"), (pl.col("team_code") == 3).sum().alias("nct"))
    valid = cnt.filter((pl.col("np") == 10) & (pl.col("nt") == 5) & (pl.col("nct") == 5)).select(["match_key", "round"])
    stats["player_info (match,round) keys"] = cnt.height
    stats["valid 5v5 (match,round) keys"] = valid.height
    pv = pi.join(valid, on=["match_key", "round"], how="inner")
    first = valid.group_by("match_key").agg(pl.col("round").min().alias("F"))
    stats["matches with a valid round"] = first.height
    stats["first valid round F: (F, matches) for F<=3"] = first.group_by("F").len().sort("F").filter(pl.col("F") <= 3).rows()
    xmap = (pv.join(first, on="match_key", how="inner")
            .filter((pl.col("round") == pl.col("F")) & (pl.col("team_code") == 2))
            .select(["match_key", "player_id"]).unique()
            .with_columns(pl.lit(True).alias("isX")))
    pv = pv.join(xmap, on=["match_key", "player_id"], how="left").with_columns(pl.col("isX").fill_null(False))
    agg = pv.group_by(["match_key", "round"]).agg(
        ((pl.col("team_code") == 2) & pl.col("isX")).sum().alias("x_t"),
        pl.col("isX").sum().alias("n_x"),
        ((pl.col("team_code") == 2) & (~pl.col("isX"))).sum().alias("y_t"),
        (~pl.col("isX")).sum().alias("n_y"))
    side = agg.with_columns(
        pl.when((pl.col("x_t") == pl.col("n_x")) & (pl.col("y_t") == 0)).then(pl.lit(2))
        .when((pl.col("x_t") == 0) & (pl.col("y_t") == pl.col("n_y"))).then(pl.lit(3))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("sx")).select(["match_key", "round", "sx"])
    sx_prev = side.select(["match_key", (pl.col("round") + 1).alias("round"), pl.col("sx").alias("sx_prev")])
    sx_next = side.select(["match_key", (pl.col("round") - 1).alias("round"), pl.col("sx").alias("sx_next")])

    bs = read(label, "bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
    bo = (bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
          .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
    bo = bo.with_columns(
        pl.when(pl.col("evs").list.contains("bomb_exploded")).then(pl.lit(2))
        .when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("s_bomb")).select(["match_key", "round", "s_bomb"])
    planter = (bs.filter(pl.col("event_type") == "bomb_planted").sort("tick")
               .group_by(["match_key", "round"]).agg(pl.col("player_team_code").first().alias("planter")))

    rs = read(label, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    fe = (rs.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
          .unique(["match_key", "round"], keep="first", maintain_order=True)
          .select(["match_key", "round", "t_score", "ct_score"]))
    fe_next = fe.select(["match_key", (pl.col("round") - 1).alias("round"),
                         pl.col("t_score").alias("t1"), pl.col("ct_score").alias("ct1")])

    base = re_.join(side, on=["match_key", "round"], how="left")
    base = base.join(sx_prev, on=["match_key", "round"], how="left")
    base = base.join(sx_next, on=["match_key", "round"], how="left")
    base = base.join(bo, on=["match_key", "round"], how="left")
    base = base.join(planter, on=["match_key", "round"], how="left")
    base = base.join(fe.rename({"t_score": "t0", "ct_score": "ct0"}), on=["match_key", "round"], how="left")
    base = base.join(fe_next, on=["match_key", "round"], how="left")
    base = base.with_columns(
        pl.when(pl.col("win_reason_code").is_in(T_REASON)).then(pl.lit(2))
        .when(pl.col("win_reason_code").is_in(CT_REASON)).then(pl.lit(3))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("s_reason"),
        pl.col("winner_team_code").cast(pl.Int64).alias("winner"),
        pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day"))
    # team-tracked score winner (side-attached score assumption)
    base = base.with_columns(
        pl.when(pl.col("sx") == 2).then(pl.col("t0")).otherwise(pl.col("ct0")).alias("x0"),
        pl.when(pl.col("sx_next") == 2).then(pl.col("t1")).otherwise(pl.col("ct1")).alias("x1"),
        pl.when(pl.col("sx") == 3).then(pl.col("t0")).otherwise(pl.col("ct0")).alias("y0"),
        pl.when(pl.col("sx_next") == 3).then(pl.col("t1")).otherwise(pl.col("ct1")).alias("y1"))
    base = base.with_columns(
        (pl.col("x1") - pl.col("x0")).alias("dX"), (pl.col("y1") - pl.col("y0")).alias("dY"))
    base = base.with_columns(
        pl.when((pl.col("dX") == 1) & (pl.col("dY") == 0)).then(pl.col("sx"))
        .when((pl.col("dY") == 1) & (pl.col("dX") == 0)).then(5 - pl.col("sx"))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("s_score"))
    base = base.with_columns(
        (pl.col("sx") != pl.col("sx_prev")).alias("swap"),
        bin_expr())
    # predictions under the team-attached readings (team X identity code = 2)
    xw = pl.col("sx") == pl.col("s_reason")
    base = base.with_columns(
        pl.when(xw).then(pl.lit(2)).otherwise(pl.lit(3)).cast(pl.Int64).alias("pred_static"),
        pl.when(xw).then(pl.col("sx_prev")).otherwise(5 - pl.col("sx_prev")).cast(pl.Int64).alias("pred_lag"),
        (pl.col("winner") != pl.col("s_reason")).alias("mm_reason"))
    base = base.with_columns(pl.lit(label).alias("year"))
    return base, stats


def main():
    tables = {}
    for label in ["2024", "2026"]:
        base, stats = build(label)
        print(f"\n===== {label}: build stats")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        tables[label] = base
        base.write_parquet(ROOT / "logs" / "swarm2" / f"D2_rounds_{label}.parquet")
        print(f"  rounds in table: {base.height}; matches: {base['match_key'].n_unique()}")
        known = base.filter(pl.col("s_reason").is_not_null() & pl.col("sx").is_not_null())
        print(f"  rounds with s_reason and sx: {known.height}; with sx_prev too: {known.filter(pl.col('sx_prev').is_not_null()).height}")
        print(f"  reason codes not mapped: {base.filter(pl.col('s_reason').is_null()).height}")

    for label in ["2024", "2026"]:
        b = tables[label]
        k = b.filter(pl.col("s_reason").is_not_null() & pl.col("sx").is_not_null())
        print(f"\n\n########## {label} ##########")
        print("A. Recorded winner vs side reading (s_reason), by round bin:")
        print(k.group_by("bin").agg(
            pl.len().alias("n"), pl.col("mm_reason").sum().alias("mismatch"),
            pl.col("mm_reason").mean().round(5).alias("rate")).sort("bin"))
        print("   by swap flag (sx != sx_prev):")
        print(k.filter(pl.col("sx_prev").is_not_null()).group_by("swap").agg(
            pl.len().alias("n"), pl.col("mm_reason").sum().alias("mismatch"),
            pl.col("mm_reason").mean().round(5).alias("rate")).sort("swap"))
        print("   mismatches by round number (round, n, mismatch) where mismatch > 0:")
        mm_by_round = k.group_by("round").agg(pl.len().alias("n"), pl.col("mm_reason").sum().alias("mm")).filter(pl.col("mm") > 0).sort("round")
        print(mm_by_round.rows())

        print("B. Swap structure (sx changes between round r-1 and r), count of matches by round:")
        sw = b.filter(pl.col("swap") & pl.col("sx_prev").is_not_null()).group_by("round").agg(pl.len().alias("n")).sort("round")
        print(sw.rows())

        print("C. Static team reading (identity fixed at round 1; pred = 2 if team X won):")
        print(k.filter(pl.col("pred_static").is_not_null()).group_by("bin").agg(
            pl.len().alias("n"),
            (pl.col("winner") == pl.col("pred_static")).sum().alias("winner==static"),
            (pl.col("winner") == pl.col("s_reason")).sum().alias("winner==side"),
            (pl.col("pred_static") != pl.col("s_reason")).sum().alias("static_predicts_mismatch")).sort("bin"))

        print("D. Lag-1 team reading (winning team's code = its side in round r-1):")
        kl = k.filter(pl.col("pred_lag").is_not_null())
        print(kl.group_by("bin").agg(
            pl.len().alias("n"),
            (pl.col("winner") == pl.col("pred_lag")).sum().alias("winner==lag1"),
            (pl.col("winner") == pl.col("s_reason")).sum().alias("winner==side"),
            (pl.col("pred_lag") != pl.col("s_reason")).sum().alias("lag1_predicts_mismatch")).sort("bin"))
        mmk = kl.filter(pl.col("mm_reason"))
        print(f"   rows where winner != side: {mmk.height}; of these winner == lag1: {mmk.filter(pl.col('winner') == pl.col('pred_lag')).height}; "
              f"winner == static: {mmk.filter(pl.col('winner') == pl.col('pred_static')).height}")
        print(f"   rows where winner == side but lag1 != side: "
              f"{kl.filter((pl.col('winner') == pl.col('s_reason')) & (pl.col('pred_lag') != pl.col('s_reason'))).height}")

        print("E. Other winner references:")
        e = b.filter(pl.col("s_reason").is_not_null())
        print("   s_bomb known:", e.filter(pl.col("s_bomb").is_not_null()).height,
              " s_bomb == s_reason:", e.filter(pl.col("s_bomb") == pl.col("s_reason")).height,
              " s_bomb != s_reason:", e.filter(pl.col("s_bomb").is_not_null() & (pl.col("s_bomb") != pl.col("s_reason"))).height,
              " s_bomb == winner:", e.filter(pl.col("s_bomb") == pl.col("winner")).height)
        print("   s_score known:", e.filter(pl.col("s_score").is_not_null()).height,
              " s_score == s_reason:", e.filter(pl.col("s_score") == pl.col("s_reason")).height,
              " s_score != s_reason:", e.filter(pl.col("s_score").is_not_null() & (pl.col("s_score") != pl.col("s_reason"))).height,
              " s_score == winner:", e.filter(pl.col("s_score") == pl.col("winner")).height)
        print("   s_score != s_reason by bin:")
        print(e.filter(pl.col("s_score").is_not_null() & (pl.col("s_score") != pl.col("s_reason")))
              .group_by("bin").len().sort("bin"))
        print("   bomb-outcome rows: s_bomb vs recorded winner, by bin:")
        bb = e.filter(pl.col("s_bomb").is_not_null())
        print(bb.group_by("bin").agg(pl.len().alias("n"),
                                     (pl.col("s_bomb") != pl.col("winner")).sum().alias("bomb_ne_winner"),
                                     (pl.col("s_bomb") != pl.col("s_reason")).sum().alias("bomb_ne_reason")).sort("bin"))
        print("   planter side on reason-mismatch rounds, by bin:")
        print(k.filter(pl.col("mm_reason") & pl.col("planter").is_not_null()).group_by(["bin", "planter"]).len().sort(["bin", "planter"]))

    b = tables["2024"]
    k = b.filter(pl.col("s_reason").is_not_null() & pl.col("sx").is_not_null())
    print("\n\n########## 2024 round 13 by match day ##########")
    r13 = k.filter(pl.col("round") == 13)
    print(r13.group_by("day").agg(
        pl.len().alias("n"), pl.col("mm_reason").sum().alias("mm_reason"),
        pl.col("mm_reason").mean().round(3).alias("rate_reason"),
        (pl.col("s_bomb").is_not_null()).sum().alias("n_bomb"),
        ((pl.col("s_bomb").is_not_null()) & (pl.col("s_bomb") != pl.col("winner"))).sum().alias("bomb_ne_winner"),
        (pl.col("s_score").is_not_null()).sum().alias("n_score"),
        ((pl.col("s_score").is_not_null()) & (pl.col("s_score") != pl.col("s_reason"))).sum().alias("score_ne_reason"),
        ((pl.col("s_score").is_not_null()) & (pl.col("s_score") == pl.col("winner"))).sum().alias("score_eq_winner"),
    ).sort("day"))

    print("\n== 2024 matches: round-13 mismatch vs other rounds")
    m13 = r13.select(["match_key", pl.col("mm_reason").alias("mm13"), pl.col("day")])
    m25 = k.filter(pl.col("round") == 25).select(["match_key", pl.col("mm_reason").alias("mm25")])
    other = k.filter(pl.col("round") != 13).group_by("match_key").agg(
        pl.col("mm_reason").sum().alias("mm_other_rounds"), pl.len().alias("n_other"))
    pm = m13.join(m25, on="match_key", how="left").join(other, on="match_key", how="left")
    print("matches with a round-13 row:", pm.height)
    print(pm.group_by(["mm13"]).agg(pl.len().alias("matches"),
                                     pl.col("mm25").fill_null(False).sum().alias("r25_mismatch_matches"),
                                     pl.col("mm25").is_not_null().sum().alias("has_r25"),
                                     pl.col("mm_other_rounds").sum().alias("mm_other_sum"),
                                     pl.col("n_other").sum().alias("other_rounds")))

    for label in ["2024", "2026"]:
        b = tables[label]
        k = b.filter(pl.col("s_reason").is_not_null() & pl.col("sx").is_not_null())
        r13b = k.filter(pl.col("round") == 13).filter(pl.col("sx_prev").is_not_null())
        print(f"\n== {label} round-13 rows: reason mismatch x swap, with static and lag-1 agreement")
        print(r13b.group_by(["mm_reason", "swap"]).agg(
            pl.len().alias("n"),
            (pl.col("winner") == pl.col("pred_static")).sum().alias("winner==static"),
            (pl.col("winner") == pl.col("pred_lag")).sum().alias("winner==lag1")).sort(["mm_reason", "swap"]))


if __name__ == "__main__":
    main()
