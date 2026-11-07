"""A13: independent recount of duplicate matches in the 2026 tome (read-only, stdout only).

Two definitions, both recomputed from the tome pages (projected columns):
  E) event signature (same as bakeoff.py): round_end (round, tick, winner), player_death (round, tick,
     player_id) and player_info (round, player_id, team_code) per match. Groups of size >= 2.
  F) round fingerprint (same as bakeoff.py section 4): sorted (round, tick, winner) of round_end for
     matches with >= 5 rounds. Groups of size >= 2.
Also reports the header match_date and the key-day distribution, to compare with the docs' claims.
Output: logs/swarm2/A13_dupes.log (captured by the caller).
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path("G:/pureskill/csds-cs2")
TOME = ROOT / "tomes/2026-10-01_to_08/tome/csds"
SUFFIX = ".2026-09-18,2026-10-08"


def read(channel, cols):
    folder = TOME / ("header" if channel == "header" else f"basic_{channel}{SUFFIX}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


hdr = read("header", ["key", "match_date", "max_rounds"])
keys = list(hdr["key"])
print(f"header matches: {len(keys)}")
day_counts = collections.Counter(k.split("/")[3] for k in keys)
print(f"header matches by key day (DD): {dict(sorted(day_counts.items()))}")
print(f"header matches with key day 01-08: {sum(v for d, v in day_counts.items() if d in ('01','02','03','04','05','06','07','08'))}")
dates = hdr["match_date"]
d_eq = collections.Counter(str(d)[:10] for d in dates)
print(f"match_date (UTC date) distinct: {len(d_eq)}; count dated 2026-10-08: {d_eq.get('2026-10-08', 0)} "
      f"({d_eq.get('2026-10-08', 0) / len(keys):.4f})")
match_day_equal = sum(1 for k, d in zip(keys, dates) if str(d)[8:10] == k.split("/")[3])
print(f"match_date day equals key DD: {match_day_equal} of {len(keys)}")

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
sigs = {m: tuple(sorted(v)) for m, v in sig_parts.items()}
groups = collections.defaultdict(list)
for m in keys:
    groups[sigs.get(m, ())].append(m)
E = [g for g in groups.values() if len(g) > 1]
print(f"\n[E] event-signature duplicate groups (all header matches): {len(E)} groups, "
      f"{sum(len(g) for g in E)} matches, copies beyond first: {sum(len(g) - 1 for g in E)}")
print(f"[E] group sizes: {dict(collections.Counter(len(g) for g in E))}")
empty = groups.get((), [])
print(f"[E] matches with an empty signature: {len(empty)}")

per = collections.defaultdict(list)
for x in RE.iter_rows(named=True):
    per[x["match_key"]].append((int(x["round"]), int(x["tick"]), int(x["winner_team_code"])))
fps = collections.defaultdict(list)
for m, lst in per.items():
    if len(lst) >= 5:
        fps[tuple(sorted(lst))].append(m)
F = [v for v in fps.values() if len(v) > 1]
print(f"[F] round-fingerprint groups (>=5 rounds): {len(F)} groups, {sum(len(v) for v in F)} matches, "
      f"copies beyond first: {sum(len(v) - 1 for v in F)}")

# how E and F overlap
E_members = {m for g in E for m in g}
F_members = {m for v in F for m in v}
print(f"[E] members {len(E_members)}; [F] members {len(F_members)}; overlap {len(E_members & F_members)}")
print(f"[F] members not in any E group: {len(F_members - E_members)}")
