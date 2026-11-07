"""D3 secondary: is the round-8 swap group a format (round count) or a map effect? (2024 G: tome, day 12..25).

Joins the swap signature (from D3_swap_signature logic) with round count and header map_name.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\D3_swap_format_map.py
"""
import collections, glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"); SUF = ".2023-12-28,2024-02-25"
def g_read(ch, cols=None):
    if ch == "header":
        folder = TOME / "header"
    else:
        folder = TOME / f"basic_{ch}{SUF}"
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")
RE = g_read("round_end", ["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first")
RS = g_read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
H = g_read("header", ["key", "map_name", "match_date", "number_of_points"])
hdr = {k: (mp, md) for k, mp, md in H.select(["key", "map_name", "match_date"]).iter_rows()}
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
live = {(m, r): (t, c) for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
nrounds = RE.group_by("match_key").agg(pl.len().alias("n"), pl.col("round").max().alias("mx"))
nr = {m: (n, mx) for m, n, mx in nrounds.select(["match_key", "n", "mx"]).iter_rows()}
by_m = collections.defaultdict(list)
for (m, r) in live:
    by_m[m].append(r)
sig = {}
for m, rs in by_m.items():
    swaps = []
    for r in sorted(rs):
        if (m, r + 1) not in live:
            continue
        t1, c1 = live[(m, r)]; t2, c2 = live[(m, r + 1)]
        dt, dc = t2 - t1, c2 - c1
        if (dc == 1 and dt == 0) or (dt == 1 and dc == 0) or (dt == 0 and dc == 0):
            continue
        if dt * dc <= 0:
            swaps.append(r)
    sig[m] = tuple(swaps)
rows = []
for m in by_m:
    day = int(m.split("/")[3])
    if day < 12:
        continue
    mp = hdr.get(m, (None, None))[0]
    n, mx = nr.get(m, (None, None))
    rows.append((m, str(sig[m]), n, mx, mp))
df = pl.DataFrame(rows, schema=["match_key", "sig", "n_rounds", "max_round", "map_name"], orient="row")
df = df.with_columns(pl.col("sig").alias("sig_s"))
print("signature x round count (day 12..25):")
print(df.group_by(["sig_s"]).agg(pl.len().alias("matches"), pl.col("n_rounds").min().alias("min_n"),
                                pl.col("n_rounds").max().alias("max_n"), pl.col("n_rounds").median().alias("med_n"))
      .sort("matches", descending=True))
print("\nmap_name counts by signature (top maps):")
top = df.group_by("map_name").len().sort("len", descending=True).head(12)["map_name"].to_list()
print(df.filter(pl.col("map_name").is_in(top)).group_by(["map_name", "sig_s"]).len().sort(["map_name", "len"], descending=[False, True]))
print("\nmap_name totals (all day 12..25 matches):", df.group_by("map_name").len().sort("len", descending=True).head(12).rows())
print("\nday counts for signature (8,):", df.filter(pl.col("sig_s") == "(8,)").with_columns(
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("d")).group_by("d").len().sort("d").rows())
