"""Where do duplicate match copies fall relative to the 2026-10-08 batch? (read-only)

Uses the same event signature as scripts/model/bakeoff.py (round_end, player_death, player_info),
over all header matches, and reports batch (header match_date day) and key-path day for each member,
plus which copy the bakeoff keeps (lowest key).

Run: pixi run python -I scripts\\eda\\dup_by_batch.py
"""
import collections
import glob
from pathlib import Path

import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUFFIX = ".2026-09-18,2026-10-08"


def read(ch, cols=None):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUFFIX}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def batch_of(day):
    if day == "2026-10-08":
        return "new_1008"
    if "2026-10-01" <= day <= "2026-10-07":
        return "other"
    return "outside"


def main():
    hdr = read("header", ["key", "match_date"])
    day = {k: md[:10] for k, md in zip(hdr["key"].to_list(), hdr["match_date"].to_list())}
    RE = read("round_end", ["match_key", "round", "tick", "winner_team_code"])
    DE = read("player_death", ["match_key", "round", "tick", "player_id"])
    PI = read("player_info", ["match_key", "round", "player_id", "team_code"])
    sig_parts = collections.defaultdict(list)
    for x in RE.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("R", int(x["round"]), int(x["tick"]), int(x["winner_team_code"]))))
    for x in DE.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("D", int(x["round"]), int(x["tick"]), str(x["player_id"]))))
    for x in PI.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("P", int(x["round"]), str(x["player_id"]), str(x["team_code"]))))
    groups = collections.defaultdict(list)
    for m in day:
        groups[tuple(sorted(sig_parts.get(m, ())))].append(m)
    dup = [g for g in groups.values() if len(g) > 1]
    members = [m for g in dup for m in g]
    print(f"groups over all matches: {len(dup)}; members {len(members)}")
    print(f"members by batch: {dict(collections.Counter(batch_of(day[m]) for m in members))}")
    print(f"all matches by batch: {dict(collections.Counter(batch_of(d) for d in day.values()))}")
    straddle = 0
    kept_batch = collections.Counter()
    dropped_batch = collections.Counter()
    same_match_date = 0
    key_day_differs = 0
    for g in dup:
        bs = {batch_of(day[m]) for m in g}
        if len(bs) > 1:
            straddle += 1
        g_sorted = sorted(g)
        kept_batch[batch_of(day[g_sorted[0]])] += 1
        for m in g_sorted[1:]:
            dropped_batch[batch_of(day[m])] += 1
        if len({day[m] for m in g}) == 1:
            same_match_date += 1
        if len({"/".join(m.split("/")[1:4]) for m in g}) > 1:
            key_day_differs += 1
    print(f"groups whose members span batches: {straddle}")
    print(f"groups where all members share the same match_date: {same_match_date} of {len(dup)}")
    print(f"groups whose members have different key-path days: {key_day_differs}")
    print(f"kept copy (lowest key) batch: {dict(kept_batch)}; dropped copies batch: {dict(dropped_batch)}")
    # the 2026-10-08 batch: how many of its matches are in a duplicate group
    new_members = [m for m in members if batch_of(day[m]) == "new_1008"]
    print(f"new_1008 matches in a duplicate group: {len(new_members)} of "
          f"{sum(1 for d in day.values() if batch_of(d) == 'new_1008')}")


if __name__ == "__main__":
    main()
