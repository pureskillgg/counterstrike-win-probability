"""verify D9-10 (c): per-match side-switch schedule from player_info, versus the flip pattern.

For every build 9961 match: boundaries r->r+1 where most common players changed team_code (switch_like).
Groups matches by their switch schedule and shows whether the round-13 / round-9 flips track the schedule.
Metadata from header (map, match_type, points, final scores) and n_rounds from round_end are attached to
check whether the round-9 switch matches are a different match format.
ASSUMPTION: a switch = >=50% of common players change team_code between consecutive rounds (>=3 common players).
ASSUMPTION: round_end rows define the played rounds; the first (match,round) row is kept on duplicates.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(60)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(name, cols):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(files_of(fd))
    have = set(lf.collect_schema().names())
    missing = [c for c in cols if c not in have]
    if missing:
        raise SystemExit(f"[{name}] missing columns {missing}")
    return lf.select(cols).collect()


H = scan("header", ["key", "build_num", "map_name", "match_type", "number_of_points", "tick_rate",
                    "t_starters_score_final", "ct_starters_score_final", "match_date"]).rename({"key": "match_key"})
H9 = H.filter(pl.col("build_num") == 9961)
B = set(H9["match_key"].to_list())

RE = scan("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(
    subset=["match_key", "round"], keep="first").filter(pl.col("match_key").is_in(list(B)))
nr = RE.group_by("match_key").agg(pl.len().alias("n_rounds"), pl.col("round").max().alias("max_round"))

PI = scan("player_info", ["match_key", "round", "player_id", "team_code"]).filter(
    pl.col("match_key").is_in(list(B)))
PIu = PI.unique(subset=["match_key", "round", "player_id"], keep="last")
a = PIu.rename({"round": "r", "team_code": "tA"})
b = PIu.with_columns((pl.col("round") - 1).alias("round")).rename({"round": "r", "team_code": "tB"})
pairs = a.join(b, on=["match_key", "r", "player_id"], how="inner")
pairs = pairs.with_columns((pl.col("tA") != pl.col("tB")).alias("chg"))
bd = (pairs.group_by(["match_key", "r"]).agg(pl.len().alias("ncommon"), pl.col("chg").mean().alias("fch"))
      .filter(pl.col("ncommon") >= 3)
      .with_columns(pl.when(pl.col("fch") >= 0.5).then(1).otherwise(0).alias("sw")))
bd_list = bd.sort(["match_key", "r"]).group_by("match_key", maintain_order=True).agg(
    pl.col("r").alias("rs"), pl.col("sw").alias("sws"), pl.col("fch").alias("fchs"))


def schedule(rs, sws):
    return ",".join(str(r) for r, s in zip(rs, sws) if s == 1) or "none"


sched = bd_list.with_columns(
    pl.struct(["rs", "sws"]).map_elements(lambda s: schedule(s["rs"], s["sws"]), return_dtype=pl.String)
    .alias("switch_sched")).select(["match_key", "switch_sched"])

M = (H9.join(nr, on="match_key", how="left").join(sched, on="match_key", how="left")
     .with_columns(pl.col("switch_sched").fill_null("no_player_info")))
print("build 9961 matches:", M.height, "| with player_info boundaries:", sched.height)

print("\n== switch schedule (boundaries r->r+1 with >=50% team change), top 25 patterns")
top = M.group_by("switch_sched").agg(pl.len().alias("n_matches")).sort("n_matches", descending=True).head(25)
print(top)

print("\n== schedule groups by whether a switch at 12 or 8 occurs")
M = M.with_columns(
    pl.col("switch_sched").str.contains(r"(^|,)12(,|$)").alias("sw12"),
    pl.col("switch_sched").str.contains(r"(^|,)8(,|$)").alias("sw8"),
    pl.col("switch_sched").str.contains(r"(^|,)9(,|$)").alias("sw9"),
    pl.col("switch_sched").str.contains(r"(^|,)13(,|$)").alias("sw13"),
)
print(M.group_by(["sw8", "sw9", "sw12", "sw13"]).agg(pl.len().alias("n")).sort(["sw8", "sw9", "sw12", "sw13"]))

print("\n== metadata by (sw9 without sw13) vs (sw13) vs (neither) groups")
M = M.with_columns(
    pl.when(pl.col("sw9") & ~pl.col("sw13")).then(pl.lit("switch_at_9_only"))
      .when(pl.col("sw13")).then(pl.lit("switch_at_13"))
      .when(pl.col("sw8") & ~pl.col("sw9")).then(pl.lit("switch_at_8_only"))
      .otherwise(pl.lit("other")).alias("grp"))
print(M.group_by("grp").agg(pl.len().alias("n"),
                            pl.col("n_rounds").mean().round(2).alias("mean_rounds"),
                            pl.col("n_rounds").min().alias("min_rounds"),
                            pl.col("n_rounds").max().alias("max_rounds"),
                            (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).mean().round(2)
                            .alias("mean_total_score")).sort("grp"))
print("\nmatch_type by grp:")
print(M.group_by(["grp", "match_type"]).len().sort(["grp", "match_type"]).rows())
print("\nmap_name (top 6) for switch_at_8_only vs other:")
for g in ["switch_at_8_only", "other"]:
    sub = M.filter(pl.col("grp") == g)
    print("  ", g, sub.group_by("map_name").len().sort("len", descending=True).head(6).rows())
