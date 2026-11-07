# A2: tick-based alignment of round_end labels to round_start labels, per match and per day group.
# For each round_end row (label L, tick te), find the latest round_start in the same match with tick <= te (label s*).
# offset d = L - s*. Ideal: d == 0 (label closes the round it started). Also count ends before any start, and rounds never closed.
import os, sys, json, csv, collections
import polars as pl
TOME = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
SCR = sys.argv[1]
def ch_frame(ch, cols):
    d = [x for x in os.listdir(TOME) if x.split(".", 1)[0] == "basic_" + ch]
    dp = os.path.join(TOME, d[0])
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    parts = [pl.scan_parquet(os.path.join(dp, p["dataframe"]["dataframe"])).select(cols).collect() for p in meta["pages"]]
    return pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]
rs = ch_frame("round_start", ["round", "tick", "match_key"]).with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day")).sort("tick")
re_ = ch_frame("round_end", ["round", "tick", "winner_team_code", "win_reason_code", "match_key"]).with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day")).sort("tick")
rs = rs.rename({"round": "s_round", "tick": "s_tick"})
aligned = re_.join_asof(rs.select(["match_key", "s_round", "s_tick"]), left_on="tick", right_on="s_tick", by="match_key", strategy="backward")
aligned = aligned.with_columns((pl.col("round") - pl.col("s_round")).alias("d"))
print("round_end rows:", aligned.height, flush=True)
print("round_end rows with no preceding round_start (pre-start):", aligned.filter(pl.col("s_round").is_null()).height)
pre = aligned.filter(pl.col("s_round").is_null())
print("  pre-start round labels:", pre.group_by("round").len().sort("round").rows())
print("offset d counts overall (round_end label minus round in progress):")
print(aligned.filter(pl.col("s_round").is_not_null()).group_by("d").len().sort("d").rows())
per_day = aligned.filter(pl.col("s_round").is_not_null()).group_by(["day", "d"]).len().sort(["day", "d"])
print("per day (d:count):")
pdd = collections.defaultdict(dict)
for r in per_day.iter_rows(named=True): pdd[r["day"]][r["d"]] = r["len"]
for d in sorted(pdd): print(" ", d, pdd[d])
# per match signature: set of offsets
sig = aligned.filter(pl.col("s_round").is_not_null()).group_by("match_key").agg(pl.col("d").unique().sort().alias("ds"), pl.col("day").first())
sig = sig.with_columns(pl.col("ds").list.eval(pl.element().cast(pl.Utf8)).list.join(",").alias("sig"))
cnt = sig.group_by(["day", "sig"]).len().sort(["day", "len"], descending=[False, True])
print("per-match offset signature by day (sig:count):")
cd = collections.defaultdict(list)
for r in cnt.iter_rows(named=True): cd[r["day"]].append((r["sig"], r["len"]))
for d in sorted(cd): print(" ", d, cd[d])
# matches where round_end labels are not closing the round at the same label (any row d != 0)
mism = sig.filter(pl.col("sig") != "0")
print("matches with any offset != 0:", mism.height, "by day:", mism.group_by("day").len().sort("day").rows())
# rounds never closed: rounds whose start has no round_end in (s_tick, next s_tick)
rs2 = rs.with_columns(pl.col("s_tick").shift(-1).over("match_key").alias("next_tick"))
ends = re_.select(["match_key", "tick"]).rename({"tick": "e_tick"})
j = rs2.join(ends, on="match_key", how="left").filter((pl.col("e_tick") > pl.col("s_tick")) & ((pl.col("next_tick").is_null()) | (pl.col("e_tick") < pl.col("next_tick"))))
cnt_end = j.group_by(["match_key", "s_round", "s_tick"]).len().rename({"len": "n_end"})
print("start rows (rounds) with exactly 1 closing round_end after start:", cnt_end.filter(pl.col("n_end") == 1).height, " with 0:", rs2.height - cnt_end.height, " with >1:", cnt_end.filter(pl.col("n_end") > 1).height)
with open(os.path.join(SCR, "A2_round_alignment_rows.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["day", "match_key", "label", "tick", "winner", "reason", "s_round", "s_tick", "d"])
    for r in aligned.iter_rows(named=True):
        w.writerow([r["day"], r["match_key"], r["round"], r["tick"], r["winner_team_code"], r["win_reason_code"], r["s_round"], r["s_tick"], r["d"]])
print("DONE", flush=True)
