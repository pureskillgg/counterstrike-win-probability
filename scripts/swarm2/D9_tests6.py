"""D9 probe 6: the 10 build-9961 round-9 mismatches, and a Haldane-corrected MH odds ratio for build.

ASSUMPTION: Haldane correction (+0.5 per cell) used only where a stratum cell is zero, for reporting.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D9")
R = pl.read_parquet(SCR / "D9_round.parquet")
M = pl.read_parquet(SCR / "D9_match.parquet")


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(name, want):
    fd = BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(files_of(fd))
    have = set(lf.collect_schema().names())
    return lf.select([c for c in want if c in have]).collect()


DE = scan("player_death", ["match_key", "round", "player_id", "player_team_code"])
DEu = DE.unique(subset=["match_key", "round", "player_id"], keep="first")
dead = (DEu.group_by(["match_key", "round"])
        .agg((pl.col("player_team_code") == 2).sum().alias("T_dead"),
             (pl.col("player_team_code") == 3).sum().alias("CT_dead")))
BS = scan("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
bomb_first = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).sort("tick")
              .group_by(["match_key", "round"]).agg(pl.col("event_type").first().alias("bomb_event"),
                                                     pl.col("player_team_code").first().alias("bomb_team")))

X = (R.join(M.select(["match_key", "build_num", "map_name"]), on="match_key", how="left")
       .join(dead, on=["match_key", "round"], how="left")
       .join(bomb_first, on=["match_key", "round"], how="left")
       .with_columns(pl.col("T_dead").fill_null(0), pl.col("CT_dead").fill_null(0)))
X = X.filter(pl.col("has_outcome") & (pl.col("player_count") == 20) & (pl.col("build_num") == 9961)
             & (pl.col("round") == 9))
print("== build 9961, pc20, round 9, bomb-outcome rounds:", X.height, "mismatch:", int(X["mismatch"].sum()))
print(X.filter(pl.col("mismatch")).select(["match_key", "winner_team_code", "win_reason_message", "bomb_event",
                                           "bomb_team", "bomb_side", "T_dead", "CT_dead", "n_T", "n_CT",
                                           "map_name"]))
print("   mismatch round-9 win messages:", X.filter(pl.col("mismatch")).group_by("win_reason_message").len().rows())
print("   round-9 matched rounds for comparison (build 9961, pc20): n=%d" % X.filter(~pl.col("mismatch")).height)
print("   round-9 mismatch matches with map:", X.filter(pl.col("mismatch")).group_by("map_name").len().rows())
print("   note: the 10 mismatches are in the same tome window as the swap-round defect (day >= 12)")
print("   day distribution of the 10:", X.filter(pl.col("mismatch")).group_by("day").len().sort("day").rows())

print("\n== Haldane-corrected MH odds ratio, build 9961 vs 9842, round 13, pc20 (strata = day)")
Y = (R.join(M.select(["match_key", "build_num"]), on="match_key", how="left")
       .filter(pl.col("has_outcome") & (pl.col("player_count") == 20) & (pl.col("round") == 13)))
num = den = 0.0
rows = []
for s in sorted(Y["day"].unique().to_list()):
    g = Y.filter(pl.col("day") == s)
    a = g.filter((pl.col("build_num") == 9961) & pl.col("mismatch")).height
    b = g.filter((pl.col("build_num") == 9961) & ~pl.col("mismatch")).height
    c = g.filter((pl.col("build_num") == 9842) & pl.col("mismatch")).height
    d = g.filter((pl.col("build_num") == 9842) & ~pl.col("mismatch")).height
    if a + b == 0 or c + d == 0:
        continue
    n = a + b + c + d
    rows.append((s, a, b, c, d))
    num += (a + 0.5) * (d + 0.5) / (n + 1)
    den += (b + 0.5) * (c + 0.5) / (n + 1)
for row in rows:
    print("   day %s: 9961 mm=%d ok=%d | 9842 mm=%d ok=%d" % row)
print("   Haldane-corrected MH OR = %.1f over %d strata" % (num / den, len(rows)))
