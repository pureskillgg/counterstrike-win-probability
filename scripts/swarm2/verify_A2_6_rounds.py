# verify_A2_6 part 3 (read-only): round-number profile of the E2 mismatch, by build_num.
# Same reads and same definition as verify_A2_6_r13.py (player_count==20; bomb_defused->3, bomb_exploded->2).
# Question: is the mismatch confined to switch rounds (13, 25, overtime halves) within build 9961?
import glob
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

BASE = r"G:\pureskill\csds-cs2"
TOME = os.path.join(BASE, "tomes", "2024-02-g", "tome", "csds")
SUF = ".2023-12-28,2024-02-25"
H24 = os.path.join(TOME, "header", "dataframe_00000")


def read(ch, cols):
    fs = sorted(glob.glob(os.path.join(TOME, f"basic_{ch}{SUF}", "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


hdr = pl.read_parquet(H24, columns=["key", "build_num"]).rename({"key": "match_key"})
RE = read("round_end", ["match_key", "round", "winner_team_code", "player_count"]).filter(pl.col("player_count") == 20)
BS = read("bomb_state", ["match_key", "round", "event_type", "player_team_code"])
outcome = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
           .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
j = (RE.join(outcome, on=["match_key", "round"], how="inner")
     .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"))
     .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"))
     .join(hdr, on="match_key", how="left"))

print("winner_team_code values among rounds with outcome:", j["winner_team_code"].unique().sort().to_list())
print("bomb_side values:", j["bomb_side"].unique().sort().to_list())
prof = (j.group_by(["build_num", "round"]).agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"))
        .sort(["build_num", "round"]))
for b in [9842, 9961]:
    sub = prof.filter(pl.col("build_num") == b)
    print(f"\nbuild {b}: rounds with outcome n={int(sub['n'].sum())}, mismatches={int(sub['mm'].sum())}")
    print("round: n/mm (only rounds with a mismatch, plus rounds 12-14, 24-26)")
    rows = [(r["round"], r["n"], r["mm"]) for r in sub.iter_rows(named=True)
            if r["mm"] > 0 or r["round"] in (12, 13, 14, 24, 25, 26)]
    print(rows)
    print("mismatch rounds (distinct):", sorted(set(r["round"] for r in sub.iter_rows(named=True) if r["mm"] > 0)))
print("DONE")
