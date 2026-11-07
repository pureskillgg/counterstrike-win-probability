import json
import polars as pl
S = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/loadout_fix"
mi = pl.read_parquet(S + "/out/match_index.parquet")
print("columns", mi.columns)
ins = mi.filter(pl.col("in_scope"))
target = ins.filter(pl.col("src") == "none").sort("match_key")["match_key"].to_list()
p3_all = ins.filter(pl.col("src") == "local_1hz_2024").sort("match_key")["match_key"].to_list()
p4_all = ins.filter((pl.col("src") == "local_1hz") & (pl.col("date") >= "2026-10-09") & (pl.col("date") <= "2026-10-20")).sort("match_key")["match_key"].to_list()


def even(lst, n):
    if len(lst) <= n:
        return list(lst)
    step = len(lst) / n
    return [lst[int(i * step)] for i in range(n)]


p3 = even(p3_all, 150)
p4 = even(p4_all, 300)
print("target", len(target), "p3 pool", len(p3_all), "p3", len(p3), "p4 pool", len(p4_all), "p4", len(p4))
# target dates range (counts only)
print("target by month", ins.filter(pl.col("src") == "none").group_by("month").agg(pl.len().alias("n")).sort("month").to_dicts())
rule = open(S + "/scripts2/loadout_rule.py", encoding="utf-8").read()
body = open(S + "/scripts2/s3_body.py", encoding="utf-8").read()
body = body.replace("__TARGET__", repr(target)).replace("__P3__", repr(p3)).replace("__P4__", repr(p4))
open(S + "/scripts2/s3_job.py", "w", encoding="utf-8").write(rule + "\n" + body)
json.dump({"target": target, "p3": p3, "p4": p4}, open(S + "/out/s3_key_lists.json", "w"))
print("job written")
