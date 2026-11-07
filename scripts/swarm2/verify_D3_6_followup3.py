"""Follow-up 3 for D3-6 verifier. Read-only.

  U. Shared match_date groups in G: 2024 header (232 values, 536 keys in the verifier run): do the
     keys in a group carry identical round_end content (duplicate matches under different uuids)?
  V. Start-at-2 vs start-at-1 (days 12-25) split by header fields: build_num, demo_version_name,
     match_type, map_name, and whether key day equals match_date day.
  W. Match_date month for the 224 claim-set day-01 keys and for all day-01 keys.
  X. Orphan m_225ef2bb (day 16, cs_office, 2024-02-16T04:34:25Z): raw header scores vs G: group members.
"""
import collections
import glob
import os
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
T24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF24 = ".2023-12-28,2024-02-25"
E_TAIL = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds")
E_EXT = Path(r"E:\csds_extracted\csds")


def g_read(folder, cols=None):
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def g24(ch, cols=None):
    return g_read(T24 / ("header" if ch == "header" else f"basic_{ch}{SUF24}"), cols)


def kday(k):
    return int(k.split("/")[3])


H = g24("header", ["key", "match_date", "map_name", "build_num", "demo_version_name", "match_type",
                   "t_starters_score_final", "ct_starters_score_final", "number_of_points"])
Hd = {r["key"]: r for r in H.to_dicts()}
keys = list(Hd)
RE = g24("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
content = collections.defaultdict(list)
for m, r, t, w, pc in RE.select(["match_key", "round", "tick", "winner_team_code", "player_count"]).iter_rows():
    content[m].append((r, t, w, pc))
for m in content:
    content[m] = sorted(content[m])
first_round = {m: min(v)[0] for m, v in content.items()}
start2 = {k for k in keys if k in first_round and first_round[k] == 2}

print("== U. shared match_date groups")
groups = collections.defaultdict(list)
for k in keys:
    groups[Hd[k]["match_date"]].append(k)
shared = {m: ks for m, ks in groups.items() if len(ks) > 1}
print("groups with >1 key:", len(shared), "| keys:", sum(len(v) for v in shared.values()))
size_hist = collections.Counter(len(v) for v in shared.values())
print("group size histogram:", dict(sorted(size_hist.items())))
same_content = collections.Counter()
same_header = collections.Counter()
examples = []
for md, ks in shared.items():
    cts = [tuple(content.get(k, [])) for k in ks]
    same_content["identical_round_end" if len(set(cts)) == 1 else "different_round_end"] += 1
    hdrs = [(Hd[k]["map_name"], Hd[k]["t_starters_score_final"], Hd[k]["ct_starters_score_final"]) for k in ks]
    same_header["identical_map_scores" if len(set(hdrs)) == 1 else "different_map_scores"] += 1
    if len(examples) < 3 and len(set(cts)) > 1:
        examples.append((md, [(k.split('/')[3], len(c)) for k, c in zip(ks, cts)]))
print("group round_end content:", dict(same_content))
print("group header (map, scores):", dict(same_header))
print("examples differing content:", examples)
print("group key days:", dict(collections.Counter(tuple(sorted(kday(k) for k in ks)) for ks in shared.values()).most_common(5)))

print("\n== V. start2 vs start1 (days 12-25) by header fields")
late = [k for k in keys if kday(k) >= 12]
for field in ("build_num", "demo_version_name", "match_type", "map_name"):
    tab = collections.defaultdict(lambda: collections.Counter())
    for k in late:
        cls = "start2" if k in start2 else "start1"
        tab[Hd[k][field]][cls] += 1
    print(f"  {field}:")
    for v, cc in sorted(tab.items(), key=lambda x: -(x[1]["start2"] + x[1]["start1"]))[:8]:
        tot = cc["start2"] + cc["start1"]
        print(f"    {str(v):<40} start2={cc['start2']:<5} start1={cc['start1']:<5} start2_share={cc['start2'] / tot:.2f}")
same_day = collections.Counter()
for k in late:
    cls = "start2" if k in start2 else "start1"
    same = int(Hd[k]["match_date"][8:10]) == kday(k)
    same_day[(cls, "key_day==match_day" if same else "key_day!=match_day")] += 1
print("  key day vs match_date day (days 12-25):", dict(same_day))
mon = collections.defaultdict(collections.Counter)
for k in late:
    cls = "start2" if k in start2 else "start1"
    mon[cls][Hd[k]["match_date"][:7]] += 1
print("  match_date month (days 12-25):", {c: dict(v) for c, v in mon.items()})

print("\n== W. match_date month for day-01 keys")
d1 = [k for k in keys if kday(k) == 1]
print("  all day-01 keys:", dict(collections.Counter(Hd[k]["match_date"][:7] for k in d1)))
print("  day-01 keys with start2:", dict(collections.Counter(Hd[k]["match_date"][:7] for k in d1 if k in start2)))
print("  day-01 keys match_date day 01 exactly:", sum(1 for k in d1 if Hd[k]["match_date"][:10] == "2024-02-01"))

print("\n== X. orphan m_225ef2bb raw header vs G: group")
oh = E_TAIL / "2024" / "02" / "16" / "m_225ef2bb61a4aa191140c10b" / "header"
if oh.is_file():
    r = pl.read_parquet(str(oh), columns=["match_date", "map_name", "build_num", "t_starters_score_final",
                                         "ct_starters_score_final", "number_of_points"]).to_dicts()[0]
    print("  raw orphan:", r)
    for k in groups.get("2024-02-16T04:34:25.000Z", []):
        print("  G:", k.split("/")[3], k.split("/")[4][:8], {x: Hd[k][x] for x in
              ("map_name", "build_num", "t_starters_score_final", "ct_starters_score_final", "number_of_points")},
              "rounds", len(content.get(k, [])))
    ext_h = E_EXT / "2024" / "02" / "16"
    print("  extracted has day 16 folder:", ext_h.is_dir())
print("\nDONE")
