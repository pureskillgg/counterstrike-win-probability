"""D5-7 verifier, probe 5 (appends to logs/swarm2/verify_D5_7.log; reads only the 2024 tome on G:).

Row-selection artifact check: the round-13 disagreement in build 9961 used the first round_end row by tick.
Does it survive when every round_end row for round 13 is considered (any player_count), and when only
player_count == 20 rows are used? Does any round_end row at round 13 agree with the score change?
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_7_probe5.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
LOG = ROOT / "logs" / "swarm2" / "verify_D5_7.log"
FH = open(LOG, "a", encoding="utf-8", newline="\n")

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)
pl.Config.set_tbl_width_chars(220)


def out(*parts):
    s = " ".join(str(p) for p in parts)
    print(s, flush=True)
    FH.write(s + "\n")
    FH.flush()


def chfiles(ch):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    if not fs:
        raise SystemExit(f"no dataframe files for channel {ch}")
    return fs


def scan(ch):
    return pl.concat([pl.scan_parquet(f) for f in chfiles(ch)], how="vertical_relaxed")


out("\n\n# ===== probe 5 (verify_D5_7_probe5.py): round_end row selection check for round 13 =====")
hfiles = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
H = (pl.concat([pl.read_parquet(f, columns=["key", "build_num"]) for f in hfiles], how="vertical_relaxed")
     .rename({"key": "match_key"}))
RS = (scan("round_state").select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
      .filter(pl.col("event_type").is_in(["round_freeze_end", "round_officially_ended"])).collect())


def first_by_round(et, prefix):
    return (RS.filter(pl.col("event_type") == et).sort(["match_key", "round", "tick"])
            .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias(f"t_{prefix}"),
                                                 pl.col("ct_score").first().alias(f"ct_{prefix}")))


FE = first_by_round("round_freeze_end", "fe")
ROE = first_by_round("round_officially_ended", "roe")
SC = (FE.join(ROE, on=["match_key", "round"], how="inner")
      .with_columns(dT=pl.col("t_roe") - pl.col("t_fe"), dCT=pl.col("ct_roe") - pl.col("ct_fe"))
      .with_columns(score_side=pl.when((pl.col("dT") == 1) & (pl.col("dCT") == 0)).then(2)
                    .when((pl.col("dCT") == 1) & (pl.col("dT") == 0)).then(3).otherwise(pl.lit(None)))
      .filter(pl.col("round") == 13)
      .join(H, on="match_key", how="left"))

RE_all = scan("round_end").select(["match_key", "round", "tick", "winner_team_code", "player_count"]).collect()
R13 = RE_all.filter(pl.col("round") == 13)
out("round_end rows at round 13 (all):", R13.height, " matches with >1 round-13 end row:",
    R13.group_by("match_key").agg(pl.len().alias("n")).filter(pl.col("n") > 1).height)
out("round-13 end-row player_count values:")
out(str(R13.group_by("player_count").agg(pl.len().alias("n")).sort("n", descending=True).head(8)))

# per match: first row (tick order) winner and pc; any row agreeing with score side; any pc==20 row
g = (R13.sort(["match_key", "tick"]).group_by("match_key").agg(
    pl.col("winner_team_code").first().alias("first_w"),
    pl.col("player_count").first().alias("first_pc"),
    pl.col("winner_team_code").alias("all_w"),
    pl.col("player_count").alias("all_pc")))
J = SC.join(g, on="match_key", how="left").with_columns(
    first_eq=pl.col("first_w") == pl.col("score_side"),
    any_eq=pl.col("all_w").list.eval(pl.element() == pl.col("score_side")).list.any() if False else pl.lit(None))
J = J.with_columns(
    any_row_eq=pl.struct(["all_w", "score_side"]).map_elements(
        lambda s: (s["score_side"] in (s["all_w"] or [])) if s["score_side"] is not None else None,
        return_dtype=pl.Boolean),
    pc20_rows=pl.struct(["all_w", "all_pc"]).map_elements(
        lambda s: sum(1 for p in (s["all_pc"] or []) if p == 20), return_dtype=pl.Int64))
out("== 14. round-13 score-change side vs round_end rows, by build (matches with score change) ==")
out(str(J.group_by(["build_num", "first_pc"]).agg(
    pl.len().alias("matches"),
    (pl.col("first_eq") == True).sum().alias("first_row_agrees"),  # noqa: E712
    (pl.col("any_row_eq") == True).sum().alias("any_row_agrees"),  # noqa: E712
    (pl.col("pc20_rows") > 0).sum().alias("has_pc20_row")).sort(["build_num", "first_pc"])))
out("== 14b. build 9961: matches where the first round-13 row disagrees but a different round-13 row agrees ==")
b = J.filter((pl.col("build_num") == 9961) & pl.col("score_side").is_not_null())
out(str(b.group_by(["first_eq", "any_row_eq"]).agg(pl.len().alias("matches")).sort(["first_eq", "any_row_eq"])))

out("\n== end probe 5 ==")
FH.close()
