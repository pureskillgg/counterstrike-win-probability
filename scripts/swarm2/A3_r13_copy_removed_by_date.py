"""A3: round-13 mismatch (e2 definition) with duplicate copies removed, by match_date and by folder.

Read-only. Copies: exact groups (bakeoff rule, all 2024 matches, lowest key kept) plus the
non-grouped near pairs (>= 8 rounds, same winner, round_end within 64 ticks; lowest key kept).
Same definitions as A3_dedupe_2024_impact.py and A3_r13_by_matchdate.py.
ASSUMPTION: match_date is not verified as the played date; both views are shown.
"""
import collections
import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def read(ch, cols):
    fd = BASE / ("header" if ch == "header" else f"basic_{ch}{SUF}")
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def bucket(d):
    if d < "2024-01-01":
        return "2023-12"
    if d < "2024-02-01":
        return "2024-01"
    if d <= "2024-02-06":
        return "2024-02-01..06"
    if d == "2024-02-07":
        return "2024-02-07"
    if d <= "2024-02-11":
        return "2024-02-08..11"
    return "2024-02-12..25"


def main():
    pl.Config.set_tbl_rows(-1)
    RE = read("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
    BS = read("bomb_state", ["match_key", "round", "event_type"])
    DE = read("player_death", ["match_key", "round", "tick", "player_id"])
    PI = read("player_info", ["match_key", "round", "player_id", "team_code"])
    H = {r["key"]: r["match_date"] for r in read("header", ["key", "match_date"]).iter_rows(named=True)}
    mkeys = sorted(H)

    parts = collections.defaultdict(list)
    for x in RE.iter_rows(named=True):
        parts[x["match_key"]].append(repr(("R", int(x["round"]), int(x["tick"]), int(x["winner_team_code"]))))
    for x in DE.iter_rows(named=True):
        parts[x["match_key"]].append(repr(("D", int(x["round"]), int(x["tick"]), str(x["player_id"]))))
    for x in PI.iter_rows(named=True):
        parts[x["match_key"]].append(repr(("P", int(x["round"]), str(x["player_id"]), str(x["team_code"]))))
    sigs = {m: tuple(sorted(v)) for m, v in parts.items()}
    groups = collections.defaultdict(list)
    for m in mkeys:
        groups[sigs.get(m, ())].append(m)
    drop = set()
    for g in groups.values():
        if len(g) > 1:
            drop.update(sorted(g)[1:])

    per_re = collections.defaultdict(dict)
    for x in RE.iter_rows(named=True):
        per_re[x["match_key"]][int(x["round"])] = (int(x["tick"]), int(x["winner_team_code"]))
    idx = {m: i for i, m in enumerate(mkeys)}
    n = len(mkeys)
    cnt = np.zeros((n, n), dtype=np.int16)
    for r in range(1, int(RE["round"].max()) + 1):
        tk = np.full(n, np.nan, dtype=np.float32)
        wn = np.full(n, -1, dtype=np.int64)
        for m, d in per_re.items():
            if r in d:
                tk[idx[m]], wn[idx[m]] = d[r][0], d[r][1]
        have = ~np.isnan(tk)
        if have.sum() < 2:
            continue
        dd = np.abs(tk[:, None] - tk[None, :])
        sw = (wn[:, None] == wn[None, :]) & have[:, None] & have[None, :]
        cnt += ((dd <= 64) & sw).astype(np.int16)
        del dd, sw
    n_near = 0
    for i, j in zip(*np.where(np.triu(cnt >= 8, k=1))):
        a, b = mkeys[i], mkeys[j]
        if sigs.get(a, ()) != sigs.get(b, ()):
            drop.add(max(a, b))
            n_near += 1
    print(f"copies removed: {len(drop)} (near pairs among them: {n_near}); exact + near")

    outcome = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
               .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
    j = (RE.filter((pl.col("player_count") == 20) & (pl.col("round") == 13))
         .join(outcome, on=["match_key", "round"], how="inner")
         .with_columns(
             pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"))
         .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"),
                       pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("key_day"),
                       pl.col("match_key").is_in(list(drop)).alias("is_copy")))
    j = j.with_columns(pl.col("match_key").map_elements(lambda m: H[m][:10], return_dtype=pl.Utf8).alias("md_date"))
    j = j.with_columns(pl.col("md_date").map_elements(bucket, return_dtype=pl.Utf8).alias("md_bucket"))
    print(f"round 13 rows with outcome: all {j.height} mismatches {int(j['mismatch'].sum())}; "
          f"copies {int(j['is_copy'].sum())} with {int(j.filter(pl.col('is_copy'))['mismatch'].sum())} mismatches")
    k = j.filter(~pl.col("is_copy"))
    print(f"after copy removal: rows {k.height} mismatches {int(k['mismatch'].sum())}")
    for label, col in (("match_date bucket", "md_bucket"), ("key folder day", "key_day")):
        a = j.group_by(col).agg(pl.len().alias("n_all"), pl.col("mismatch").sum().alias("mm_all"))
        b = k.group_by(col).agg(pl.len().alias("n_kept"), pl.col("mismatch").sum().alias("mm_kept"))
        t = a.join(b, on=col, how="left").with_columns(
            pl.col("n_kept").fill_null(0), pl.col("mm_kept").fill_null(0)).sort(col)
        print(f"\nround 13 by {label}: all rows vs copies removed")
        print(t)


if __name__ == "__main__":
    main()
