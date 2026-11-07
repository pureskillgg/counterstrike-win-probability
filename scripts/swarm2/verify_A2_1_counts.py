# verify_A2_1 counts: how many 2024 round labels are affected under the verified switch-round rule, and the
# round-1 completeness by build. Uses the saved round table and small scans only.
import glob
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(20)

SP = r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"
BASE = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
SUF = ".2023-12-28,2024-02-25"
R = pl.read_parquet(os.path.join(SP, "verify_A2_1_round_table.parquet"))
hdr = pl.scan_parquet(os.path.join(BASE, "header", "dataframe_00000")).select(["key", "build_num"]).collect().rename(
    {"key": "match_key"})
fes = (pl.scan_parquet(sorted(glob.glob(os.path.join(BASE, f"basic_round_state{SUF}", "dataframe_*")))[0])
       .select(["match_key", "round", "tick", "event_type", "phase"])
       .filter(pl.col("event_type") == "round_freeze_end").collect()
       .sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first"))
fsec = fes.filter(pl.col("phase") == "second").group_by("match_key").agg(pl.col("round").min().alias("first_second"))
M = hdr.join(fsec, on="match_key", how="left").with_columns(
    pl.when(pl.col("first_second") == 9).then(pl.lit("MR8"))
    .when(pl.col("first_second") == 13).then(pl.lit("MR12"))
    .otherwise(pl.lit("none")).alias("fmt"))
RR = R.join(M.select(["match_key", "fmt"]), on="match_key", how="left")

print("== round-1 completeness by build (matches with round_end r1, freeze_end r1)")
re1 = R.filter(pl.col("round") == 1).select("match_key").unique()
fe1 = fes.filter(pl.col("round") == 1).select("match_key").unique()
allm = hdr.with_columns(pl.col("match_key").is_in(re1["match_key"].to_list()).alias("re1"),
                        pl.col("match_key").is_in(fe1["match_key"].to_list()).alias("fe1"))
print(allm.group_by(["build_num", "re1", "fe1"]).len().sort(["build_num", "re1", "fe1"]).rows(), flush=True)

print("== 9961 switch-round rows (rows at the switch positions; verified rule: MR8 r9, MR12 r13, OT r25 and r28)")
sw = RR.filter((pl.col("build_num") == 9961) & (
    ((pl.col("fmt") == "MR8") & (pl.col("round") == 9)) |
    ((pl.col("fmt") == "MR12") & (pl.col("round") == 13)) |
    (pl.col("round").is_in([25, 28]))))
print("switch rows by fmt/round:", sw.group_by(["fmt", "round"]).len().sort(["fmt", "round"]).rows(), flush=True)
print("switch rows total:", sw.height, flush=True)
print("9961 round_end rows total:", RR.filter(pl.col("build_num") == 9961).height,
      "| 9961 share at switch positions:", round(sw.height / RR.filter(pl.col("build_num") == 9961).height, 4), flush=True)
print("all 2024 round_end rows:", RR.height, "| share of all 2024 rounds:", round(sw.height / RR.height, 4), flush=True)
print("switch rows with bomb outcome:", sw.filter(pl.col("bomb_side").is_not_null()).height,
      "| bomb disagreeing (winner vs bomb):", sw.filter(pl.col("bomb_side").is_not_null() &
                                                         (pl.col("bomb_side") != pl.col("winner_team_code"))).height, flush=True)
print("DONE counts", flush=True)
