"""verify D2-5 part 2: rebuild the score-based side truth (independent re-implementation) and recheck
the claim "in flipped rows, reason == R in 2,136 of 2,137; reason == T in 1".

Corrected reason mapping (by engine message name):
  T (code 2) wins: Target_Bombed, Terrorists_Win, Hostages_Not_Rescued, CTs_Surrender
  CT (code 3) wins: Bomb_Defused, CTs_Win, All_Hostages_Rescued, Target_Saved, Terrorists_Surrender
  NOTE: Terrorists_Surrender (17) means the T side surrendered, so CT wins; CTs_Surrender (18) means T wins.
  The D2 mapping (17 -> T, 18 -> CT) is inverted for these two codes.

Score truth (ASSUMPTION: round_state t_score/ct_score are side-attached, checked against bomb outcomes):
  X = five players on code 2 in the first valid 5v5 round of each match (valid = 5 on code 2, 5 on code 3).
  sx = side code of X in round r (from player_info.team_code of X members in that round).
  X's score at r = t_score if sx == 2 else ct_score; at r+1 likewise with sx(r+1).
  Y = the other five. If X's score rose by 1 and Y's did not, X won round r (side sx). Symmetric for Y.
  Scores read at round_freeze_end (first per match-round).

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_5_d_scoretruth.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(40)
pl.Config.set_fmt_str_lengths(50)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
PC = {"2024": 20, "2026": 10}
T_MSG = ["Target_Bombed", "Terrorists_Win", "Hostages_Not_Rescued", "CTs_Surrender"]
CT_MSG = ["Bomb_Defused", "CTs_Win", "All_Hostages_Rescued", "Target_Saved", "Terrorists_Surrender"]


def files(label, ch):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))[0]
    return sorted(glob.glob(fd + "/dataframe_*"))


def load(label, ch, want):
    fs = files(label, ch)
    have = set(pl.read_parquet_schema(fs[0]).keys())
    return pl.scan_parquet(fs).select([c for c in want if c in have]).collect()


def section(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


def build(label):
    re_ = (load(label, "round_end", ["match_key", "round", "tick", "winner_team_code",
                                      "win_reason_code", "win_reason_message", "player_count"])
           .filter(pl.col("player_count") == PC[label]).sort("tick")
           .unique(["match_key", "round"], keep="first"))
    re_ = re_.with_columns(pl.col("win_reason_message").str.replace("#SFUI_Notice_", "").alias("msg"))
    re_ = re_.with_columns(
        pl.when(pl.col("msg").is_in(T_MSG)).then(pl.lit(2))
        .when(pl.col("msg").is_in(CT_MSG)).then(pl.lit(3))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("reason_side"))
    print(f"{label}: full-5v5 rounds {re_.height}; unmapped messages {re_.filter(pl.col('reason_side').is_null()).height}")

    # round_state freeze-end scores
    rs = load(label, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    fe_sc = (rs.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
             .unique(["match_key", "round"], keep="first").select(["match_key", "round", "t_score", "ct_score"]))

    # player sides and team X
    pi = (load(label, "player_info", ["match_key", "round", "player_id", "team_code"])
          .unique(["match_key", "round", "player_id"], keep="first"))
    cnt = pi.group_by(["match_key", "round"]).agg(
        (pl.col("team_code") == 2).sum().alias("n2"), (pl.col("team_code") == 3).sum().alias("n3"))
    valid = cnt.filter((pl.col("n2") == 5) & (pl.col("n3") == 5))
    first = valid.group_by("match_key").agg(pl.col("round").min().alias("F"))
    xset = (pi.join(first, on="match_key").filter((pl.col("round") == pl.col("F")) & (pl.col("team_code") == 2))
            .select(["match_key", "player_id"]).unique())
    print(f"  X-set sizes (should be 5 per match): {xset.group_by('match_key').len()['len'].value_counts().sort('len').rows()}")
    sxt = (pi.join(xset, on=["match_key", "player_id"], how="inner")
           .group_by(["match_key", "round"]).agg(
               (pl.col("team_code") == 2).sum().alias("x2"), (pl.col("team_code") == 3).sum().alias("x3"),
               pl.len().alias("nx")))
    print("  X-member side counts per round (x2, x3, nx) top:",
          sxt.group_by(["x2", "x3"]).len().sort("len", descending=True).head(6).rows())
    sxt = sxt.with_columns(
        pl.when(pl.col("x2") > pl.col("x3")).then(pl.lit(2))
        .when(pl.col("x3") > pl.col("x2")).then(pl.lit(3)).otherwise(pl.lit(None)).cast(pl.Int64).alias("sx"))
    sxt = sxt.select(["match_key", "round", "sx"])

    base = re_.join(sxt, on=["match_key", "round"], how="left")
    nxt = sxt.with_columns((pl.col("round") - 1).alias("round")).rename({"sx": "sx_nx"})
    base = base.join(nxt, on=["match_key", "round"], how="left")
    fe0 = fe_sc.rename({"t_score": "t0", "ct_score": "ct0"})
    fe1 = fe_sc.with_columns((pl.col("round") - 1).alias("round")).rename({"t_score": "t1", "ct_score": "ct1"})
    base = base.join(fe0, on=["match_key", "round"], how="left").join(fe1, on=["match_key", "round"], how="left")
    base = base.with_columns(
        pl.when(pl.col("sx") == 2).then(pl.col("t0")).otherwise(pl.col("ct0")).alias("x0"),
        pl.when(pl.col("sx_nx") == 2).then(pl.col("t1")).otherwise(pl.col("ct1")).alias("x1"),
        pl.when(pl.col("sx") == 3).then(pl.col("t0")).otherwise(pl.col("ct0")).alias("y0"),
        pl.when(pl.col("sx_nx") == 3).then(pl.col("t1")).otherwise(pl.col("ct1")).alias("y1"))
    base = base.with_columns(
        (pl.col("x1") - pl.col("x0")).alias("dX"), (pl.col("y1") - pl.col("y0")).alias("dY"))
    base = base.with_columns(
        pl.when((pl.col("dX") == 1) & (pl.col("dY") == 0)).then(pl.col("sx"))
        .when((pl.col("dY") == 1) & (pl.col("dX") == 0)).then(5 - pl.col("sx"))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("s_score"))

    # bomb outcome (independent of team codes)
    tend = re_.select(["match_key", "round", pl.col("tick").alias("t_end")])
    tfe = (load(label, "round_state", ["match_key", "round", "tick", "event_type"])
           .filter(pl.col("event_type") == "round_freeze_end").sort("tick")
           .unique(["match_key", "round"], keep="first").select(["match_key", "round", pl.col("tick").alias("t_fe")]))
    bs = load(label, "bomb_state", ["match_key", "round", "tick", "event_type"])
    bo = (bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
          .join(tfe, on=["match_key", "round"]).join(tend, on=["match_key", "round"])
          .filter((pl.col("tick") >= pl.col("t_fe")) & (pl.col("tick") <= pl.col("t_end") + 40))
          .sort("tick").group_by(["match_key", "round"]).agg(pl.col("event_type").first().alias("bo")))
    bo = bo.with_columns(pl.when(pl.col("bo") == "bomb_exploded").then(pl.lit(2)).otherwise(pl.lit(3))
                         .cast(pl.Int64).alias("s_bomb"))
    base = base.join(bo.select(["match_key", "round", "s_bomb"]), on=["match_key", "round"], how="left")

    base = base.with_columns(pl.coalesce(["s_score", "s_bomb"]).alias("T"),
                             pl.col("match_key").str.extract(r"csds/\d{4}/\d{2}/(\d{2})/", 1).alias("day"),
                             pl.when(pl.col("round") == 13).then(pl.lit("r13"))
                             .when(pl.col("round") == 25).then(pl.lit("r25"))
                             .when(pl.col("round") == 28).then(pl.lit("r28"))
                             .otherwise(pl.lit("other")).alias("bin"))
    return base


def report(label, b):
    section(f"{label}: score truth vs R vs reason (corrected mapping)")
    n = b.height
    print(f"rows: {n}; sx known: {b.filter(pl.col('sx').is_not_null()).height}; s_score known: {b.filter(pl.col('s_score').is_not_null()).height}; "
          f"s_bomb known: {b.filter(pl.col('s_bomb').is_not_null()).height}; T known: {b.filter(pl.col('T').is_not_null()).height}")
    ss = b.filter(pl.col("s_score").is_not_null() & pl.col("s_bomb").is_not_null())
    print(f"s_score vs s_bomb agreement: {ss.filter(pl.col('s_score') == pl.col('s_bomb')).height} of {ss.height}")
    sc = b.filter(pl.col("s_score").is_not_null())
    print(f"s_score vs R: {sc.filter(pl.col('s_score') == pl.col('winner_team_code')).height} of {sc.height} equal")
    print(f"reason_side vs R (all rows, corrected): {b.filter(pl.col('reason_side') == pl.col('winner_team_code')).height} of {n}")

    k = b.filter(pl.col("T").is_not_null())
    fl = k.filter(pl.col("winner_team_code") != pl.col("T"))
    print(f"\nrows with T: {k.height}; flipped (R != T): {fl.height}")
    print(f"  flipped with reason == R : {fl.filter(pl.col('reason_side') == pl.col('winner_team_code')).height}")
    print(f"  flipped with reason == T : {fl.filter(pl.col('reason_side') == pl.col('T')).height}")
    print(f"  flipped with reason null : {fl.filter(pl.col('reason_side').is_null()).height}")
    print("  flipped by bin (n):", fl.group_by("bin").len().sort("bin").rows())
    print("  flipped by msg:", fl.group_by("msg").len().sort("len", descending=True).rows())
    print("  rows with T where reason != R (all):")
    print(k.filter(pl.col("reason_side") != pl.col("winner_team_code"))
          .select(["match_key", "round", "winner_team_code", "win_reason_code", "msg", "s_score", "s_bomb", "T"]))
    print("  flipped rows where reason == T (exception candidates):")
    exc = fl.filter(pl.col("reason_side") == pl.col("T"))
    print(exc.select(["match_key", "round", "winner_team_code", "win_reason_code", "msg", "sx", "sx_nx",
                      "t0", "ct0", "t1", "ct1", "dX", "dY", "s_score", "s_bomb"]))


def main():
    for label in ["2024", "2026"]:
        b = build(label)
        report(label, b)


if __name__ == "__main__":
    main()
