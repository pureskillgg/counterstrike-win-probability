"""D2 truth-based test: is the recorded winner_team_code team-attached (identity) or side-attached?

Problem with D2_test.py: win_reason_code is derived from the same stale winner as winner_team_code in
affected rounds (both agree and both disagree with the bomb, the deaths and the scores), so it cannot be
used as the side truth. This script uses side-attached references instead:

  s_bomb   bomb outcome (exploded -> 2 = T, defused -> 3 = CT), from bomb_state in the round window
  s_elim   elimination: the side with five deaths loses (victim side from player_info membership)
  s_score  round_state freeze-end scores, team-tracked across the swap, assuming side-attached scores
           (checked against s_bomb and s_elim below)
  T        truth = s_score if known, else s_bomb, else s_elim (agreement among sources is reported)

Round window = [round_freeze_end tick, round_end tick] (round_start is not a window boundary in this tome).

Readings of the recorded winner R (winner_team_code), with sx = side of team X (X = players on T in the
first valid round), sx_prev = side of X in round r-1, and identity X = code 2:
  side_reading   R == T
  flipped        R == 5 - T   (the recorded code is the other side)
  static TA      recorded code = identity of the winning team: side_TA = sx if R == 2 else 5 - sx
  lag-1 TA       recorded code = winning team's side in round r-1: side_lag = sx if R == sx_prev else 5 - sx

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\D2_truth.py
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


def read(label, ch, cols):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))[0]
    return pl.scan_parquet(sorted(glob.glob(fd + "/dataframe_*"))).select(cols).collect()


def build(label):
    base = pl.read_parquet(ROOT / "logs" / "swarm2" / f"D2_rounds_{label}.parquet")
    # round window from freeze_end to round_end
    fe = (read(label, "round_state", ["match_key", "round", "tick", "event_type"])
          .filter(pl.col("event_type") == "round_freeze_end").sort("tick")
          .unique(["match_key", "round"], keep="first").select(["match_key", "round", pl.col("tick").alias("t_fe")]))
    re_ = (read(label, "round_end", ["match_key", "round", "tick"]).sort("tick")
           .unique(["match_key", "round"], keep="first").select(["match_key", "round", pl.col("tick").alias("t_end")]))
    bs = read(label, "bomb_state", ["match_key", "round", "tick", "event_type"])
    bo = (bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
          .join(fe, on=["match_key", "round"], how="left").join(re_, on=["match_key", "round"], how="left")
          .filter((pl.col("tick") >= pl.col("t_fe")) & (pl.col("tick") <= pl.col("t_end") + 40)))
    bo = (bo.sort("tick").group_by(["match_key", "round"]).agg(
        pl.col("event_type").first().alias("bo_first"), pl.len().alias("n_bo"),
        pl.col("event_type").n_unique().alias("n_bo_types")))
    bo = bo.with_columns(
        pl.when(pl.col("bo_first") == "bomb_exploded").then(pl.lit(2)).otherwise(pl.lit(3)).cast(pl.Int64).alias("s_bomb"))
    base = base.join(bo.select(["match_key", "round", "s_bomb", "n_bo", "n_bo_types"]), on=["match_key", "round"], how="left")

    # team-tracked score winner (side-attached score fields assumed)
    rs = read(label, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    fe_sc = (rs.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
             .unique(["match_key", "round"], keep="first").select(["match_key", "round", "t_score", "ct_score"]))
    fe_next = fe_sc.select(["match_key", (pl.col("round") - 1).alias("round"),
                            pl.col("t_score").alias("t1"), pl.col("ct_score").alias("ct1")])
    sx_next = base.select(["match_key", (pl.col("round") - 1).alias("round"), pl.col("sx").alias("sx_nx")])
    b2 = base.join(fe_sc.rename({"t_score": "t0", "ct_score": "ct0"}), on=["match_key", "round"], how="left")
    b2 = b2.join(fe_next, on=["match_key", "round"], how="left").join(sx_next, on=["match_key", "round"], how="left")
    b2 = b2.with_columns(
        pl.when(pl.col("sx") == 2).then(pl.col("t0")).otherwise(pl.col("ct0")).alias("x0"),
        pl.when(pl.col("sx_nx") == 2).then(pl.col("t1")).otherwise(pl.col("ct1")).alias("x1"),
        pl.when(pl.col("sx") == 3).then(pl.col("t0")).otherwise(pl.col("ct0")).alias("y0"),
        pl.when(pl.col("sx_nx") == 3).then(pl.col("t1")).otherwise(pl.col("ct1")).alias("y1"))
    b2 = b2.with_columns(
        (pl.col("x1") - pl.col("x0")).alias("dX"), (pl.col("y1") - pl.col("y0")).alias("dY"))
    b2 = b2.with_columns(
        pl.when((pl.col("dX") == 1) & (pl.col("dY") == 0)).then(pl.col("sx"))
        .when((pl.col("dY") == 1) & (pl.col("dX") == 0)).then(5 - pl.col("sx"))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("s_score"))

    # elimination side (victim side from membership of team X at round r)
    pi = (read(label, "player_info", ["match_key", "round", "player_id", "team_code"])
          .unique(["match_key", "round", "player_id"], keep="first"))
    firstv = base.filter(pl.col("sx").is_not_null()).group_by("match_key").agg(pl.col("round").min().alias("F"))
    pim = pi.join(firstv, on="match_key").filter(pl.col("round") == pl.col("F"))
    mem = pim.select(["match_key", "player_id", (pl.col("team_code") == 2).alias("isX")]).unique(["match_key", "player_id"])
    dth = read(label, "player_death", ["match_key", "round", "tick", "player_id"]).join(mem, on=["match_key", "player_id"], how="left")
    dth = dth.join(base.select(["match_key", "round", "sx"]), on=["match_key", "round"], how="left")
    dth = dth.with_columns(pl.when(pl.col("isX")).then(pl.col("sx")).otherwise(5 - pl.col("sx")).alias("vside"))
    el = dth.group_by(["match_key", "round"]).agg(
        (pl.col("vside") == 2).sum().alias("t_dead"), (pl.col("vside") == 3).sum().alias("ct_dead"))
    el = el.with_columns(
        pl.when((pl.col("ct_dead") == 5) & (pl.col("t_dead") < 5)).then(pl.lit(2))
        .when((pl.col("t_dead") == 5) & (pl.col("ct_dead") < 5)).then(pl.lit(3))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("s_elim"))
    b2 = b2.join(el.select(["match_key", "round", "s_elim"]), on=["match_key", "round"], how="left")

    # truth
    b2 = b2.with_columns(
        pl.coalesce(["s_score", "s_bomb", "s_elim"]).alias("T"),
        pl.col("winner").alias("R"))
    b2 = b2.with_columns(
        (pl.col("R") != pl.col("T")).alias("flip"),                                  # R disagrees with truth
        pl.when(pl.col("sx") == pl.col("T")).then(pl.lit(1)).otherwise(pl.lit(0)).alias("X_won"),
    )
    b2 = b2.with_columns(
        pl.when(pl.col("R") == 2).then(pl.col("sx")).otherwise(5 - pl.col("sx")).cast(pl.Int64).alias("side_TA"),
        pl.when(pl.col("R") == pl.col("sx_prev")).then(pl.col("sx")).otherwise(5 - pl.col("sx")).cast(pl.Int64).alias("side_lag"),
    )
    b2 = b2.with_columns(pl.lit(label).alias("year"))
    return b2


def main():
    tabs = {}
    for label in ["2024", "2026"]:
        b = build(label)
        tabs[label] = b
        b.write_parquet(ROOT / "logs" / "swarm2" / f"D2_truth_{label}.parquet")

    for label in ["2024", "2026"]:
        b = tabs[label]
        print(f"\n########## {label} ##########")
        print("rows:", b.height, " with sx:", b.filter(pl.col("sx").is_not_null()).height,
              " with truth T:", b.filter(pl.col("T").is_not_null()).height,
              " T from s_score:", b.filter(pl.col("s_score").is_not_null()).height,
              " T only from bomb/elim:", b.filter(pl.col("s_score").is_null() & pl.col("T").is_not_null()).height)

        print("\nSource agreement with each other (rows where both known):")
        for a, c in [("s_score", "s_bomb"), ("s_score", "s_elim"), ("s_bomb", "s_elim"), ("s_score", "R"), ("s_bomb", "R"), ("s_elim", "R")]:
            k = b.filter(pl.col(a).is_not_null() & pl.col(c).is_not_null())
            eq = k.filter(pl.col(a) == pl.col(c)).height
            print(f"  {a} vs {c}: n={k.height} equal={eq} differ={k.height - eq}")

        print("\nRecorded winner R vs truth T, by round bin:")
        k = b.filter(pl.col("T").is_not_null())
        print(k.group_by("bin").agg(
            pl.len().alias("n"), pl.col("flip").sum().alias("R!=T (flipped)"),
            pl.col("flip").mean().round(4).alias("flip_rate")).sort("bin"))

        print("\nRecorded winner R vs truth T, by round bin and swap (sx != sx_prev):")
        ks = k.filter(pl.col("sx_prev").is_not_null())
        print(ks.group_by(["bin", "swap"]).agg(
            pl.len().alias("n"), pl.col("flip").sum().alias("flipped")).sort(["bin", "swap"]))

        print("\nStatic team-attached conversion (side_TA) vs truth, by bin, swap rounds only:")
        print(ks.filter(pl.col("swap")).group_by("bin").agg(
            pl.len().alias("n"), (pl.col("side_TA") == pl.col("T")).sum().alias("side_TA==T"),
            (pl.col("side_TA") != pl.col("T")).sum().alias("side_TA!=T")).sort("bin"))
        print("Lag-1 team-attached conversion (side_lag) vs truth, by bin, swap rounds only:")
        print(ks.filter(pl.col("swap")).group_by("bin").agg(
            pl.len().alias("n"), (pl.col("side_lag") == pl.col("T")).sum().alias("side_lag==T"),
            (pl.col("side_lag") != pl.col("T")).sum().alias("side_lag!=T")).sort("bin"))
        print("Side reading (R as label) vs truth, swap rounds only, by bin:")
        print(ks.filter(pl.col("swap")).group_by("bin").agg(
            pl.len().alias("n"), (pl.col("R") == pl.col("T")).sum().alias("R==T")).sort("bin"))

        print("\nFlipped rows: which team won (X = identity 2) and which side it had at truth:")
        fl = k.filter(pl.col("flip") & pl.col("sx_prev").is_not_null())
        print(fl.group_by(["round", "X_won", "T", "R"]).len().sort(["round", "X_won", "T", "R"]).head(40))

        print("\nFlipped rows by round number (all rounds with a flip):")
        print(k.filter(pl.col("flip")).group_by("round").len().sort("round").rows())

        print("\nRound 13 flip rate by day:")
        r13 = k.filter(pl.col("round") == 13)
        print(r13.group_by("day").agg(pl.len().alias("n"), pl.col("flip").sum().alias("flipped"),
                                      pl.col("flip").mean().round(3).alias("rate")).sort("day"))

        print("\nRound 13 swap rows: flip rate for X won vs Y won (truth):")
        r13s = r13.filter(pl.col("sx_prev").is_not_null() & pl.col("swap"))
        print(r13s.group_by("X_won").agg(pl.len().alias("n"), pl.col("flip").sum().alias("flipped"),
                                          pl.col("flip").mean().round(3).alias("rate")).sort("X_won"))

    # match-level pattern across swaps in 2024 (round 13 vs round 28 flips)
    b = tabs["2024"]
    f13 = b.filter((pl.col("round") == 13) & pl.col("T").is_not_null()).select(["match_key", pl.col("flip").alias("f13")])
    f28 = b.filter((pl.col("round") == 28) & pl.col("T").is_not_null()).select(["match_key", pl.col("flip").alias("f28")])
    j = f13.join(f28, on="match_key", how="inner")
    print("\n2024 matches with both round-13 and round-28 truth: flips at 13 vs 28")
    print(j.group_by(["f13", "f28"]).len().sort(["f13", "f28"]))
    f25 = b.filter((pl.col("round") == 25) & pl.col("T").is_not_null()).select(["match_key", pl.col("flip").alias("f25")])
    j2 = f13.join(f25, on="match_key", how="inner")
    print("2024 matches with round-13 and round-25 truth: flips at 13 vs 25")
    print(j2.group_by(["f13", "f25"]).len().sort(["f13", "f25"]))


if __name__ == "__main__":
    main()
