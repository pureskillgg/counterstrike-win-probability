"""D7-1 verifier, part B: round-13 bomb-outcome mismatch, recomputed from raw tome columns.
Definition (same as the claim): round with bomb_defused (side 3) or bomb_exploded (side 2); mismatch = side != round_end.winner_team_code.
ASSUMPTION: player_count == 20 subset (as in the claim).
Reads only needed columns; lazy scans; one channel at a time.
"""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(200)
ROOT = Path(r"G:\pureskill\csds-cs2")
SCRATCH = Path("C:/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/e0a18ad2-aa9e-4a3c-9946-29641b2cde82/scratchpad")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))
def scan(ch, cols, pred=None):
    lf = pl.concat([pl.scan_parquet(f) for f in files(BASE / f"basic_{ch}{SUF}")])
    if pred is not None: lf = lf.filter(pred)
    return lf.select(cols).collect()

H = pl.read_parquet(str(SCRATCH / "verify_D7_1_header.parquet"))  # own scratch from part A
RE = scan("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count", "win_reason_code", "win_reason_message"])
print("round_end rows:", RE.height, "| winner codes:", RE["winner_team_code"].value_counts().sort("winner_team_code").rows())
BS = scan("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"],
          pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"]))
outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"), pl.col("tick").min().alias("out_tick")))
plant = (BS.filter(pl.col("event_type") == "bomb_planted").group_by(["match_key", "round"])
         .agg(pl.col("player_team_code").first().alias("planter_side")))
R = (RE.join(outc, on=["match_key", "round"], how="left").join(plant, on=["match_key", "round"], how="left")
       .join(H.rename({"key": "match_key"}), on="match_key", how="left"))
R = R.with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(3)
      .when(pl.col("evs").list.contains("bomb_exploded")).then(2).otherwise(None).cast(pl.Int64).alias("bomb_side"))
R = R.with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm_raw"))
R = R.with_columns(pl.when(pl.col("bomb_side").is_not_null()).then(pl.col("mm_raw")).otherwise(None).alias("mm"))
print("rows with null match header:", R.filter(pl.col("mdt").is_null()).height)
B = R.filter(pl.col("bomb_side").is_not_null() & (pl.col("player_count") == 20))
print("\nB0 player_count==20 bomb-outcome rounds:", B.height, "| mismatches:", int(B["mm"].sum()))
r13 = B.filter(pl.col("round") == 13)
print("B0 round-13 rows:", r13.height, "| unique match_key:", r13["match_key"].n_unique(), "| mismatches:", int(r13["mm"].sum()))
print("B0 round-13 rows by planter side:", r13.group_by(["mm", "planter_side"]).len().sort(["mm", "planter_side"]).rows())
print("B0 other rounds (not 13 or 25) mismatches:", B.filter((~pl.col("round").is_in([13, 25])) & pl.col("mm")).height,
      "of", B.filter(~pl.col("round").is_in([13, 25])).height)
print("B0 round-25 rows:", B.filter(pl.col("round") == 25).height, "| mismatches:", int(B.filter(pl.col("round") == 25)["mm"].sum()))

# B1: time-based counts for round-13 (UTC match_date)
print("\nB1 round-13 counts by UTC match_date:")
def cnt(lbl, cond):
    s = r13.filter(cond); print(f"  {lbl}: n={s.height} mismatches={int(s['mm'].sum())}")
cnt("before 2024-02-07 00:00", pl.col("mdt") < pl.datetime(2024, 2, 7))
cnt("before 2024-02-07 02:39:55 (claimed first mismatch)", pl.col("mdt") < pl.datetime(2024, 2, 7, 2, 39, 55))
cnt("before 2024-02-07 01:01:18 (first build-9961)", pl.col("mdt") < pl.datetime(2024, 2, 7, 1, 1, 18))
cnt("2024-02-06 only", (pl.col("mdt") >= pl.datetime(2024, 2, 6)) & (pl.col("mdt") < pl.datetime(2024, 2, 7)))
cnt("2024-02-07 only", (pl.col("mdt") >= pl.datetime(2024, 2, 7)) & (pl.col("mdt") < pl.datetime(2024, 2, 8)))
cnt("2024-02-08 onward", pl.col("mdt") >= pl.datetime(2024, 2, 8))
print("  gap check: last clean round-13 row before flip zone:",
      r13.filter(~pl.col("mm") & (pl.col("mdt") < pl.datetime(2024, 2, 7, 3)))["mdt"].max())
print("  round-13 rows between 2024-02-06 23:56:33 and 2024-02-07 02:39:54:",
      r13.filter((pl.col("mdt") > pl.datetime(2024, 2, 6, 23, 56, 32)) & (pl.col("mdt") < pl.datetime(2024, 2, 7, 2, 39, 55))).height)

# B2: round-13 build x mismatch crosstab
print("\nB2 round-13 bomb rows: build_num x mismatch:", r13.group_by(["build_num", "mm"]).len().sort(["build_num", "mm"]).rows())
print("B2 round-13 bomb rows: build_num x path key day (mismatch count):")
print(r13.group_by(["key_day", "build_num"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort(["key_day", "build_num"]).rows())

# B3: the clean round-13 rows that sit after the onset
cl = r13.filter((pl.col("mdt") >= pl.datetime(2024, 2, 7)) & (~pl.col("mm")))
print("\nB3 clean round-13 rows at/after 2024-02-07:", cl.height)
print(cl.select(["mdt", "build_num", "map_name", "winner_team_code", "bomb_side", "planter_side", "key_day", "match_key"]).with_columns(pl.col("match_key").str.slice(0, 33)))

# B4: day tables (UTC match day and path day) as in the claim
print("\nB4 round-13 by UTC day (n, mm):", r13.group_by("md_day").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("md_day").rows())
print("B4 round-13 by path day (n, mm) [claim: 0 of 180 in 01-07; 434 of 605 in 12-25]:",
      r13.group_by("key_day").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("key_day").rows())
print("B4 round-13 path days 01..07 total n, mm:", r13.filter(pl.col("key_day") < "2024/02/08").select(pl.len(), pl.col("mm").sum()).rows())
print("B4 round-13 path days 12..25 total n, mm:", r13.filter(pl.col("key_day") >= "2024/02/12").select(pl.len(), pl.col("mm").sum()).rows())

# B5: other rounds mismatch by build and date
oth = B.filter((~pl.col("round").is_in([13, 25])) & pl.col("mm"))
print("\nB5 other-round mismatches (n=%d) by build:" % oth.height, oth.group_by("build_num").len().rows())
print("B5 other-round mismatches by round number:", oth.group_by("round").len().sort("round").rows())
print("B5 other-round mismatches earliest/latest UTC:", oth["mdt"].min(), oth["mdt"].max())
r25 = B.filter(pl.col("round") == 25)
print("B5 round-25 build x mm:", r25.group_by(["build_num", "mm"]).len().sort(["build_num", "mm"]).rows())
R.write_parquet(str(SCRATCH / "verify_D7_1_rounds.parquet"))
print("\nB done: wrote logs/swarm2/verify_D7_1_rounds.parquet (own scratch, all rounds with header join)")
