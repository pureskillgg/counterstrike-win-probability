"""D7-1 verifier, part D: alternative explanations and independent fields.
D1: win_reason_message (separate field) vs winner_team_code vs bomb outcome, round-13 bomb rows.
D2: duplicate-looking matches (same UTC start second and map) among round-13 bomb rows, and mismatch rate with/without dedupe.
D3: round-sequence completeness (maxr vs distinct rounds) by build.
D4: exact build-switch endpoints from header; the single build-9842 match after the switch.
"""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(200)
ROOT = Path(r"G:\pureskill\csds-cs2")
SCRATCH = Path("C:/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/e0a18ad2-aa9e-4a3c-9946-29641b2cde82/scratchpad")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))
R = pl.read_parquet(str(SCRATCH / "verify_D7_1_rounds.parquet"))
H = pl.read_parquet(str(SCRATCH / "verify_D7_1_header.parquet"))
B = R.filter(pl.col("bomb_side").is_not_null() & (pl.col("player_count") == 20))
r13 = B.filter(pl.col("round") == 13)

print("D1 win_reason_message distribution, round-13 bomb rows, by mm:")
print(r13.group_by(["win_reason_message", "bomb_side", "winner_team_code", "mm"]).len().sort(["win_reason_message", "mm"]).rows())
print("D1 all bomb rows, win_reason_message vs bomb_side (no mm filter):")
print(B.group_by(["win_reason_message", "bomb_side"]).len().sort(["win_reason_message", "bomb_side"]).rows())
print("D1 round-13 non-bomb rows (player_count==20): win_reason_message x winner_team_code by build:")
nb = R.filter(pl.col("bomb_side").is_null() & (pl.col("player_count") == 20) & (pl.col("round") == 13) & pl.col("build_num").is_not_null())
print(nb.group_by(["build_num", "win_reason_message", "winner_team_code"]).len().sort(["build_num", "win_reason_message", "winner_team_code"]).rows())

# D2: duplicates
key_dup = H.group_by(["mdt", "map_name"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
dup_keys = H.join(key_dup.select(["mdt", "map_name"]), on=["mdt", "map_name"], how="inner")["key"].to_list()
r13d = r13.with_columns(pl.col("match_key").is_in(dup_keys).alias("in_dup_group"))
print("\nD2 duplicate groups (same UTC second and map): groups", key_dup.height, "| matches", len(dup_keys))
print("D2 round-13 bomb rows in duplicate groups:", r13d.filter(pl.col("in_dup_group")).height, "of", r13d.height,
      "| mismatches among them:", int(r13d.filter(pl.col("in_dup_group"))["mm"].sum()))
print("D2 round-13 rows n,mm by in_dup_group:", r13d.group_by("in_dup_group").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).rows())
# dedupe: keep the first key per (mdt, map)
keep = H.sort("key").group_by(["mdt", "map_name"]).agg(pl.col("key").first().alias("key_keep"))
r13u = r13.join(keep, left_on=["mdt", "map_name"], right_on=["mdt", "map_name"], how="left").filter(pl.col("match_key") == pl.col("key_keep"))
print("D2 deduped round-13 bomb rows:", r13u.height, "| mismatches:", int(r13u["mm"].sum()),
      "| pre-switch rows (mdt<01:01:18):", r13u.filter(pl.col("mdt") < pl.datetime(2024, 2, 7, 1, 1, 18)).height,
      "| their mismatches:", int(r13u.filter(pl.col("mdt") < pl.datetime(2024, 2, 7, 1, 1, 18))["mm"].sum()))
print("D2 unique (mdt,map) groups overall:", H.select(["mdt", "map_name"]).unique().height, "of", H.height)
print("D2 duplicate copies by build:", H.filter(pl.col("key").is_in(dup_keys)).group_by("build_num").len().rows())

# D3: round sequence completeness: distinct rounds vs max round per match
RE = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "build_num"] if False else ["match_key", "round"]) for f in
                files(BASE / f"basic_round_end{SUF}")]).collect()
G = RE.group_by("match_key").agg(pl.col("round").n_unique().alias("nu"), pl.col("round").max().alias("maxr"), pl.len().alias("n"))
G = G.join(H.select(["key", "build_num"]).rename({"key": "match_key"}), on="match_key")
G = G.with_columns((pl.col("maxr") - pl.col("nu")).alias("gaps"))
print("\nD3 matches with a gap in round numbers (maxr - distinct rounds > 0), by build:", G.group_by("build_num").agg((pl.col("gaps") > 0).sum().alias("with_gap"), pl.len().alias("n")).rows())
print("D3 gap sizes (all):", G.group_by("gaps").len().sort("gaps").rows())
print("D3 rows n != nu (duplicate round rows):", G.filter(pl.col("n") != pl.col("nu")).height)

# D4: exact endpoints
b9961_first = H.filter(pl.col("build_num") == 9961)["mdt"].min()
b9842_before = H.filter((pl.col("build_num") == 9842) & (pl.col("mdt") < b9961_first))["mdt"].max()
print("\nD4 last build-9842 match before first build-9961:", b9842_before, "| first build-9961:", b9961_first)
print("D4 build-9842 matches after first build-9961:", H.filter((pl.col("build_num") == 9842) & (pl.col("mdt") > b9961_first)).select(["mdt", "key", "map_name"]).rows())
print("D4 matches (any build) in (00:24:22, 01:01:18]:", H.filter((pl.col("mdt") > pl.datetime(2024, 2, 7, 0, 24, 22)) & (pl.col("mdt") <= pl.datetime(2024, 2, 7, 1, 1, 18))).height)
print("D4 r13 bomb rows of build 9842 after 2024-02-07 00:00:", r13.filter((pl.col("build_num") == 9842) & (pl.col("mdt") >= pl.datetime(2024, 2, 7))).height)
