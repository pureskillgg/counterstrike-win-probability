# A1: E: source file timestamps and the 90 disagreement counts by key day (2026/10/DD), all 1752 matches.
import os, collections, datetime
import polars as pl
T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
E = r"E:\csds_final\csds\2026\10"
d = pl.read_csv(r"G:\pureskill\csds-cs2\logs\swarm2\A1_header_vs_rounds.csv", infer_schema_length=0).with_columns(
    pl.col("round_match").cast(pl.Utf8))
d = d.with_columns(pl.col("key").str.split("/").list.get(3).alias("day"), pl.col("key").str.split("/").list.get(4).alias("mid"))
stats = collections.defaultdict(lambda: dict(n=0, mt=[], bad=0))
for row in d.iter_rows(named=True):
    s = stats[row["day"]]
    s["n"] += 1
    if row["round_match"] == "false":
        s["bad"] += 1
    p = os.path.join(E, row["day"], row["mid"], "round_end")
    try:
        s["mt"].append(os.stat(p).st_mtime)
    except OSError:
        pass
for day in sorted(stats):
    mt = stats[day]["mt"]
    lo = datetime.datetime.fromtimestamp(min(mt)).isoformat() if mt else None
    hi = datetime.datetime.fromtimestamp(max(mt)).isoformat() if mt else None
    print("day", day, "matches", stats[day]["n"], "disagree", stats[day]["bad"], "round_end mtime min", lo, "max", hi)
print("match_date range:", d.get_column("match_date").min() if "match_date" in d.columns else "n/a")
