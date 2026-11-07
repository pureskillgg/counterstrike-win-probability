# A2: round-level profile of the 2024-02-g tome, grouped by day (01 = merged; 02-07 and 12-25 = tail copy).
import os, sys, json, collections
import polars as pl
TOME = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
def ch_frame(ch, cols=None):
    d = [x for x in os.listdir(TOME) if x.split(".", 1)[0] == "basic_" + ch]
    dp = os.path.join(TOME, d[0])
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    parts = []
    for p in meta["pages"]:
        f = os.path.join(dp, p["dataframe"]["dataframe"])
        lf = pl.scan_parquet(f)
        if cols: lf = lf.select(cols)
        parts.append(lf.collect())
    return pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]
hdr = pl.read_parquet(os.path.join(TOME, "header", "dataframe_00000"), columns=["key", "match_date", "map_name", "tick_rate", "build_num", "number_of_points", "t_starters_score_final", "ct_starters_score_final", "match_id"])
hdr = hdr.rename({"key": "match_key"}).with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day"))
print("header by day:", hdr.group_by("day").len().sort("day").to_dicts())
rs = ch_frame("round_start", ["round", "tick", "match_key"])
re_ = ch_frame("round_end", ["round", "tick", "winner_team_code", "win_reason_code", "win_reason_message", "legacy_code", "player_count", "match_key"])
print("round_start cols", rs.columns, "rows", rs.height)
print("round_end cols", re_.columns, "rows", re_.height)
for name, df in [("round_start", rs), ("round_end", re_)]:
    df2 = df.with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day"))
    dup = df2.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
    print(f"{name}: (match,round) with >1 row: {dup.height}; rows in those groups: {int(dup['len'].sum()) if dup.height else 0}")
    print(f"{name}: null round={df2['round'].null_count()} null tick={df2['tick'].null_count()}")
    print(f"{name} rows by day:", df2.group_by("day").len().sort("day").rows())
rsD = rs.with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day"))
reD = re_.with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day"))
print("round_end winner_team_code values:", reD.group_by("winner_team_code").len().sort("winner_team_code").rows())
print("round_end win_reason_code values:", reD.group_by("win_reason_code").len().sort("win_reason_code").rows())
print("round_end player_count values:", reD.group_by("player_count").len().sort("player_count").rows())
# per-match round sets
rs_m = rsD.group_by("match_key").agg(pl.col("round").unique().alias("rs_rounds"))
re_m = reD.group_by("match_key").agg(pl.col("round").unique().alias("re_rounds"))
j = hdr.select(["match_key", "day"]).join(rs_m, on="match_key", how="left").join(re_m, on="match_key", how="left")
def rstats(r):
    a = set(r["rs_rounds"] or []); b = set(r["re_rounds"] or [])
    return (len(a), len(b), len(b - a), len(a - b), max(b) if b else None, max(a) if a else None)
rows = []
for r in j.iter_rows(named=True):
    rows.append((r["day"], r["match_key"]) + rstats(r))
import csv
SCR = sys.argv[1]
with open(os.path.join(SCR, "A2_round_sets.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["day", "match_key", "n_rs", "n_re", "re_not_rs", "rs_not_re", "max_re", "max_rs"]); w.writerows(rows)
by = collections.defaultdict(list)
for r in rows: by[r[0]].append(r)
print("per-day: matches, mean n_round_end, min, max, matches with re_not_rs>0, matches with rs_not_re>0, matches with no round_start")
for d in sorted(by):
    rr = by[d]
    ne = [x[3] for x in rr]
    print(d, len(rr), round(sum(ne)/len(ne), 2), min(ne), max(ne),
          sum(1 for x in rr if x[4] > 0), sum(1 for x in rr if x[5] > 0), sum(1 for x in rr if x[2] == 0))
print("matches with no round_start rows:", [r[1] for r in rows if r[2] == 0])
gaps = collections.Counter()
for r in rows:
    b = r[3]
    if r[6] and r[6] != r[3]: gaps[r[0]] += 1
print("matches where max round_end != count of distinct round_end (gap or dup):", dict(gaps))
print("DONE", flush=True)
