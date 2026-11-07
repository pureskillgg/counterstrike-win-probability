"""verify D5-6 check 5: match-level structure of the inversion pattern (regime A, late and aff matches).

For each match, list the labels where the round_end winner disagrees with the freeze-end score step (chain, valid
only when chain is a clean +1 step). Then test whether the pattern relates to the map (header map_name) or to the
number of rounds in the match (header final score sum and the largest round_end label).
ASSUMPTION: the chain is an independent truth where it is valid (checked: it equals round_end on every valid row at
labels outside the flagged set, see the first table).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_check5.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(50)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_6")
K = ["match_key", "round"]

T = pl.read_parquet(SCR / "v_rounds.parquet")
M = pl.read_parquet(SCR / "v_matches.parquet")
hdr = (pl.scan_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*")))).select(
    ["key", "t_starters_score_final", "ct_starters_score_final"]).collect()
       .rename({"key": "match_key"}).with_columns(
           final_sum=pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")))
LATE = [f"{d:02d}" for d in range(12, 26)]
M = M.join(hdr.select(["match_key", "final_sum"]), on="match_key", how="left").with_columns(
    grp=pl.when(pl.col("aff_pc20")).then(pl.lit("aff_r13"))
    .when(pl.col("day").is_in(LATE)).then(pl.lit("ctrl_late")).otherwise(pl.lit("other")))
TT = T.filter(pl.col("re_present") & pl.col("rw").is_not_null()).select(K + ["rw", "chain", "implied", "bs_planted"])
TT = TT.with_columns(chain_ok=pl.col("chain").is_not_null() & (pl.col("chain") != 0))

print("== D1: chain vs rw by label over all regime A matches (chain-valid rows only), label 1..30 ==")
regA = M.filter(pl.col("r1_end")).select("match_key")
A = TT.join(regA, on="match_key", how="inner").filter(pl.col("chain_ok"))
A = A.with_columns(flag=(pl.col("chain") != pl.col("rw")).cast(pl.Int8))
print(A.group_by("round").agg(pl.len().alias("n_valid"), pl.col("flag").sum().alias("n_flag"),
                              pl.col("flag").mean().round(3).alias("flag_rate")).sort("round").rows())

# per match flag pattern
pat = (A.filter(pl.col("flag") == 1).group_by("match_key").agg(pl.col("round").sort().alias("flag_labels")))
pat = pat.with_columns(pattern=pl.col("flag_labels").list.eval(pl.element().cast(pl.Utf8)).list.join(","))
allm = regA.select("match_key").join(pat, on="match_key", how="left").with_columns(
    pattern=pl.col("pattern").fill_null("none"))
allm = allm.join(M.select(["match_key", "map_name", "day", "grp", "final_sum", "aff_pc20"]), on="match_key", how="left")
maxlab = T.filter(pl.col("re_present")).group_by("match_key").agg(pl.col("round").max().alias("max_lab"))
allm = allm.join(maxlab, on="match_key", how="left")
print("\n== D2: match-level flag pattern (labels with chain != round_end), regime A matches ==")
print(allm.group_by("pattern").agg(pl.len().alias("matches")).sort("matches", descending=True).head(25))

print("\n== D3: flag pattern vs map_name (regime A, top patterns) ==")
top = ["none", "13", "9", "13,25,28", "9,25,28", "13,28"]
print(allm.filter(pl.col("pattern").is_in(top)).group_by(["pattern", "map_name"]).agg(pl.len().alias("n")).sort(
    ["pattern", "n"], descending=[False, True]).head(60))

print("\n== D4: flag pattern vs header final score sum and max round_end label (regime A) ==")
print(allm.filter(pl.col("pattern").is_in(top)).group_by("pattern").agg(
    pl.len().alias("matches"), pl.col("final_sum").mean().round(2).alias("mean_final_sum"),
    pl.col("final_sum").min().alias("min_final_sum"), pl.col("final_sum").max().alias("max_final_sum"),
    pl.col("max_lab").mean().round(2).alias("mean_max_label")).sort("matches", descending=True))

print("\n== D5: map distribution overall, regime A late matches (for reference) ==")
print(allm.filter(pl.col("grp") == "ctrl_late").group_by("map_name").agg(pl.len().alias("n")).sort("n", descending=True))
