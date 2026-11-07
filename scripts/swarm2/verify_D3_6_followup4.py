"""Follow-up 4 for D3-6 verifier. Read-only.

  Y. build_num over all G: 2024 header keys, crossed with the start-at-2 class and day group.
  Z. Duplicate structure: shared match_date groups whose round_end content is identical (duplicate
     matches under different uuids). Unique-match estimate. ASSUMPTION: identical round_end content
     plus identical map and scores = same match.
  AA. For the 224 claim-set day-01 keys: are they members of duplicate groups, and what is the
     tail-copy and extracted-copy status of their partners?
  AB. Source-side duplicates: for duplicate pairs where both partners have raw round_end in the tail
     copy, do the raw round_end rows agree (is the duplicate present in E:, not created by the tome)?
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
COMP = ["round_end", "round_state", "bomb_state", "bomb_action"]


def g_read(folder, cols=None):
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def g24(ch, cols=None):
    return g_read(T24 / ("header" if ch == "header" else f"basic_{ch}{SUF24}"), cols)


def kday(k):
    return int(k.split("/")[3])


def ep(base, k, ch=""):
    p = k.split("/")
    d = base / p[1] / p[2] / p[3] / p[4]
    return d / ch if ch else d


H = g24("header", ["key", "match_date", "map_name", "build_num", "t_starters_score_final",
                   "ct_starters_score_final", "number_of_points"])
Hd = {r["key"]: r for r in H.to_dicts()}
keys = list(Hd)
RE = g24("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
content = collections.defaultdict(list)
for m, r, t, w, pc in RE.select(["match_key", "round", "tick", "winner_team_code", "player_count"]).iter_rows():
    content[m].append((r, t, w, pc))
for m in content:
    content[m] = tuple(sorted(content[m]))
first_round = {m: min(v)[0] for m, v in content.items()}
start2 = {k for k in keys if first_round.get(k) == 2}

print("== Y. build_num over all keys")
bn = collections.Counter()
for k in keys:
    grp = "d01_07" if kday(k) <= 7 else "d12_25"
    cls = "start2" if k in start2 else "start1"
    bn[(Hd[k]["build_num"], grp, cls)] += 1
for kk, n in sorted(bn.items()):
    print(f"  build={kk[0]} {kk[1]} {kk[2]}: {n}")
b9842 = {k for k in keys if Hd[k]["build_num"] == 9842}
print("start2 set == build 9842 set:", start2 == b9842, "| sizes", len(start2), len(b9842))
print("build_num distribution:", dict(collections.Counter(Hd[k]["build_num"] for k in keys)))

print("\n== Z. duplicate structure")
groups = collections.defaultdict(list)
for k in keys:
    groups[Hd[k]["match_date"]].append(k)
shared = {m: ks for m, ks in groups.items() if len(ks) > 1}
ident = {m: ks for m, ks in shared.items() if len({content.get(k, ()) for k in ks}) == 1
         and len({(Hd[k]["map_name"], Hd[k]["t_starters_score_final"], Hd[k]["ct_starters_score_final"]) for k in ks}) == 1}
surplus = sum(len(ks) - 1 for ks in ident.values())
print("shared-date groups:", len(shared), "| identical content groups:", len(ident),
      "| surplus keys in identical groups:", surplus)
print(f"unique-match estimate (ASSUMPTION): {len(keys) - surplus} = {len(keys)} - {surplus}")
print("identical groups by size:", dict(sorted(collections.Counter(len(v) for v in ident.values()).items())))
print("identical group key-day pairs (top):", collections.Counter(tuple(sorted(kday(k) for k in v)) for v in ident.values()).most_common(6))
print("build_num of identical-group members:", dict(collections.Counter(Hd[k]["build_num"] for v in ident.values() for k in v)))

print("\n== AA. claim-set day-01 keys in duplicate groups")
tail_status = {}
for k in keys:
    d = ep(E_TAIL, k)
    if not d.is_dir():
        tail_status[k] = "missing"
        continue
    names = set(os.listdir(d))
    tail_status[k] = ("empty" if not names else "all4" if all(c in names for c in COMP) else
                      "none" if not any(c in names for c in COMP) else "partial")
ext_status = {}
for k in keys:
    if kday(k) != 1:
        continue
    d = ep(E_EXT, k)
    if not d.is_dir():
        ext_status[k] = "missing"
        continue
    names = set(os.listdir(d))
    ext_status[k] = ("empty" if not names else "all4" if all(c in names for c in COMP) else
                     "none" if not any(c in names for c in COMP) else "partial")
ident_member = {}
for m, ks in ident.items():
    for k in ks:
        ident_member[k] = [x for x in ks if x != k]
claim = [k for k in keys if kday(k) == 1 and tail_status[k] in ("empty", "none")]
print("claim-set size:", len(claim))
print("claim-set keys in an identical duplicate group:", sum(1 for k in claim if k in ident_member))
pc = collections.Counter()
for k in claim:
    partners = ident_member.get(k, [])
    if not partners:
        pc["no_partner"] += 1
        continue
    best = []
    for x in partners:
        best.append(f"tail={tail_status[x]}/ext={ext_status.get(x, 'n.a.')}")
    pc["partner_has_tail_or_ext_compare" if any(
        tail_status[x] == "all4" or ext_status.get(x) == "all4" for x in partners) else "partner_no_compare"] += 1
print("claim-set partner status:", dict(pc))
print("claim-set day-01 keys with ext status all4:", sum(1 for k in claim if ext_status.get(k) == "all4"))

print("\n== AB. source-side duplicates: raw round_end agreement in tail copy for identical pairs")
agree = collections.Counter()
examples = []
for m, ks in ident.items():
    tails = [k for k in ks if tail_status[k] == "all4"]
    if len(tails) < 2:
        agree["fewer_than_two_tail_all4"] += 1
        continue
    raws = []
    for k in tails:
        p = ep(E_TAIL, k, "round_end")
        df = pl.read_parquet(str(p), columns=["round", "tick", "winner_team_code", "player_count"])
        raws.append(tuple(sorted(df.rows())))
    agree["raw_round_end_equal" if len(set(raws)) == 1 else "raw_round_end_DIFF"] += 1
    if len(examples) < 2 and len(set(raws)) > 1:
        examples.append((tails,))
print("tail-copy identical-group checks:", dict(agree))
print("examples differing:", examples)

print("\nDONE")
