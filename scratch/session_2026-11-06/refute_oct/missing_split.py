import polars as pl, sys
sys.stdout.reconfigure(encoding="utf-8")
R = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/refute_oct/"
S = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct/"
m = pl.read_csv(R + "refute_oct23_31_v4_absent_rounds.csv")
print("absent rounds", m.height, "seconds", int(m["steps"].sum()), "matches", m["match_key"].n_unique())
pm = pl.read_csv(S + "batch_v4_per_match.csv").select(["match_key", "match_split", "date"])
j = m.join(pm, on="match_key", how="left")
print(j.group_by("match_split").agg(pl.len().alias("rounds"), pl.col("steps").sum().alias("secs")).sort("match_split"))
