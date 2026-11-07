r"""D4 check: does the round_state score change across round r reproduce the round winner? (round 13 focus)

For round r with no side swap after it (bakeoff.swap_expected(r) false; A11 for 2026, A12 for 2024),
the score change from round_freeze_end(r) to round_freeze_end(r+1) is the result of round r.
  dC = ct_score(r+1) - ct_score(r), dT = t_score(r+1) - t_score(r)
  score side = CT if dC == 1 and dT == 0; T if dT == 1 and dC == 0; otherwise ambiguous.
Compared with (a) round_end.winner_team_code and (b) the bomb-event side where the round has a bomb outcome.
Scope: all rounds with both freeze_end rows. Assumes side-attached scores (ASSUMPTION A13; the score_lead_fix notes
say attachment varies by match, so disagreements may partly be attachment).
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\D4_score_delta.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(14)
ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
         "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
OUTCOMES = ["bomb_defused", "bomb_exploded"]
T, CT = 2, 3


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def read(base, ch, cols):
    return read_dir(sorted(base.glob(f"basic_{ch}.*"))[0], cols)


def swap_expected(r, mr):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


for era, base in TOMES.items():
    print(f"\n==================== ERA {era} ====================")
    if era == "2026":
        H = read_dir(base / "header", ["key", "max_rounds"])
        mr_of = {r["key"]: int(r["max_rounds"]) for r in H.iter_rows(named=True)}
    else:
        mr_of = None
    rs = read(base, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    fe = (rs.filter(pl.col("event_type") == "round_freeze_end")
          .select(["match_key", "round", "t_score", "ct_score"]).unique(["match_key", "round"], keep="first"))
    nxt = fe.with_columns((pl.col("round") - 1).alias("round")).rename({"t_score": "t_next", "ct_score": "ct_next"})
    D = fe.join(nxt, on=["match_key", "round"], how="inner").with_columns(
        (pl.col("t_next") - pl.col("t_score")).alias("dT"),
        (pl.col("ct_next") - pl.col("ct_score")).alias("dC"),
    ).with_columns(
        pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
        .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("score_side"))
    if mr_of is not None:
        D = D.with_columns(pl.col("match_key").map_elements(lambda k: mr_of.get(k, 24), return_dtype=pl.Int64).alias("mr"))
    else:
        D = D.with_columns(pl.lit(24).alias("mr"))
    D = D.filter(pl.col("round") >= 1)
    D = D.with_columns(pl.struct(["round", "mr"]).map_elements(
        lambda s: not swap_expected(s["round"], s["mr"]), return_dtype=pl.Boolean).alias("valid"))
    D = D.filter(pl.col("valid"))
    re_ = read(base, "round_end", ["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first")
    bs = read(base, "bomb_state", ["match_key", "round", "event_type"]).filter(pl.col("event_type").is_in(OUTCOMES))
    outs = bs.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")).with_columns(
        pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bside"))
    J = D.join(re_, on=["match_key", "round"], how="inner").join(outs.select(["match_key", "round", "bside"]),
                                                              on=["match_key", "round"], how="left")
    J = J.with_columns(
        pl.when(pl.col("round") == 13).then(pl.lit("r13")).otherwise(pl.lit("other")).alias("rbin"),
        (pl.col("score_side") == pl.col("winner_team_code")).alias("score_eq_end"),
        (pl.col("score_side") == pl.col("bside")).alias("score_eq_bomb"),
        (pl.col("score_side") == 0).alias("ambig"),
    )
    print(f"valid rounds with both freeze_end rows: {J.height}; ambiguous score delta: {int(J['ambig'].sum())}")
    print("score side vs round_end winner (ambiguous excluded), by round bin:")
    print(J.filter(~pl.col("ambig")).group_by("rbin").agg(
        pl.len().alias("n"), pl.col("score_eq_end").mean().alias("agree_with_round_end")).sort("rbin"))
    bombJ = J.filter(pl.col("bside").is_not_null() & ~pl.col("ambig"))
    print("bomb rounds (ambiguous excluded): score side vs bomb side and vs round_end winner, by round bin:")
    print(bombJ.group_by("rbin").agg(
        pl.len().alias("n"), pl.col("score_eq_bomb").mean().alias("agree_with_bomb"),
        pl.col("score_eq_end").mean().alias("agree_with_round_end")).sort("rbin"))
    if era == "2024":
        r13 = J.filter((pl.col("round") == 13) & ~pl.col("ambig") & pl.col("bside").is_not_null())
        print("2024 round 13 bomb rounds: score side vs bomb side, crosstab with round_end agreement:")
        print(r13.group_by(["score_eq_bomb", "score_eq_end"]).len().sort(["score_eq_bomb", "score_eq_end"]))

# per-day breakdown of round 13 (valid, unambiguous) for 2024 and 2026
for era, base in TOMES.items():
    rs = read(base, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    fe = (rs.filter(pl.col("event_type") == "round_freeze_end")
          .select(["match_key", "round", "t_score", "ct_score"]).unique(["match_key", "round"], keep="first"))
    nxt = fe.with_columns((pl.col("round") - 1).alias("round")).rename({"t_score": "t_next", "ct_score": "ct_next"})
    D13 = fe.filter(pl.col("round") == 13).join(
        nxt.filter(pl.col("round") == 13), on=["match_key", "round"], how="inner").with_columns(
        (pl.col("ct_next") - pl.col("ct_score")).alias("dC"), (pl.col("t_next") - pl.col("t_score")).alias("dT")).with_columns(
        pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
        .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("score_side"))
    re_ = read(base, "round_end", ["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first")
    J13 = D13.join(re_, on=["match_key", "round"], how="inner").filter(pl.col("score_side") != 0)
    J13 = J13.with_columns(
        (pl.col("score_side") == pl.col("winner_team_code")).alias("agree"),
        pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day"))
    print(f"\n==== {era}: round 13 score-side vs round_end winner, by day (unambiguous rounds) ====")
    print(J13.group_by("day").agg(pl.len().alias("n"), (~pl.col("agree")).sum().alias("disagree"),
                                  pl.col("agree").mean().alias("agree_rate")).sort("day"))
