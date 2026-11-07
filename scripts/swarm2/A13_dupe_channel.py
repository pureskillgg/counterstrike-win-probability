"""A13: which duplicate-group members does the channel filter remove? (2026 tome, read-only, stdout only)

The bake-off drops a match when a dense channel has zero rows (bomb_state, team_change, player_info
here; all keysets cover the 1,752 headers). This script finds the duplicate-group members (event
signature, all 1,752 headers) that are dropped by that rule, and shows how the group count changes.
Output: logs/swarm2/A13_dupe_channel.log (captured by the caller).
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


hdr = read("header", ["key"])
keys = list(hdr["key"])
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
dup = [g for g in groups.values() if len(g) > 1]
members = {m for g in dup for m in g}
print(f"all headers: {len(dup)} groups, {len(members)} matches")

zero = {}
for ch in ["bomb_state", "team_change", "player_info"]:
    df = read(ch, ["match_key"])
    have = set(df["match_key"].unique().to_list())
    zero[ch] = {m for m in keys if m not in have}
    print(f"dense channel {ch}: zero-row matches {len(zero[ch])}")
dropped = zero["bomb_state"] | zero["team_change"] | zero["player_info"]
print(f"matches dropped by the channel rule (union): {len(dropped)} (bake-off log: 43)")

kept_groups = []
for g in dup:
    kept = [m for m in g if m not in dropped]
    if len(kept) > 1:
        kept_groups.append(kept)
removed_members = [m for m in members if m in dropped]
print(f"duplicate members dropped by the channel rule: {len(removed_members)}")
for m in removed_members:
    reasons = [ch for ch in zero if m in zero[ch]]
    print(f"  {m[:40]} zero rows in {reasons}")
print(f"after channel rule: {len(kept_groups)} groups, {sum(len(g) for g in kept_groups)} matches, "
      f"copies beyond first {sum(len(g) - 1 for g in kept_groups)} (bake-off log: 46 groups, 97 matches, 51 copies)")
single = [g for g in dup if len([m for m in g if m not in dropped]) == 1]
print(f"groups reduced to one kept member (no longer duplicates): {len(single)}")
for g in single:
    print(f"  group of {len(g)}: members {[m[:20] for m in g]}; kept {[m[:20] for m in g if m not in dropped]}")
