"""Verifier D3-5, probe 5: physical side swaps (player_info team_code per round) vs round_state field swaps
vs round_end winner flips. Read only. Prints to stdout.
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(230)
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def g_read(ch, cols=None):
    files = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


PI = g_read("player_info", ["match_key", "round", "player_id", "team_code"])
RE = g_read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).unique(["match_key", "round"], keep="first")
RS = g_read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
fe = RS.filter(pl.col("event_type") == "round_freeze_end").unique(["match_key", "round"], keep="first")
FE = {(m, int(r)): (t, c) for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
A = {(m, int(r)): int(w) for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows()}

# physical sides per (match, round)
side = collections.defaultdict(lambda: {T: set(), CT: set(), "other": set()})
for m, r, pid, tc in PI.select(["match_key", "round", "player_id", "team_code"]).iter_rows():
    key = (m, int(r))
    if tc == T:
        side[key][T].add(pid)
    elif tc == CT:
        side[key][CT].add(pid)
    else:
        side[key]["other"].add(pid)
print("player_info (match,round) keys:", len(side))
sizes = collections.Counter((len(v[T]), len(v[CT])) for v in side.values())
print("(nT, nCT) per round, top:", sizes.most_common(8))


def classify(t, c, t2, c2):
    out = set()
    if (t2, c2) == (t + 1, c):
        out.add((T, 0))
    if (t2, c2) == (t, c + 1):
        out.add((CT, 0))
    if (t2, c2) == (c, t + 1):
        out.add((T, 1))
    if (t2, c2) == (c + 1, t):
        out.add((CT, 1))
    return out


by_m = collections.defaultdict(dict)
for (m, r), v in FE.items():
    by_m[m][r] = v
phys_by_match = {}
field_by_match = {}
phys_tr = collections.Counter()  # r -> physical swap count (transition r -> r+1)
field_tr = collections.Counter()
phys_other = collections.Counter()
for m in by_m:
    rounds = sorted(by_m[m])
    ps, fs = [], []
    for r in rounds:
        if (m, r + 1) not in side or (m, r) not in side:
            continue
        s0, s1 = side[(m, r)], side[(m, r + 1)]
        if not s0[T] or not s1[T]:
            phys_other[r] += 1
            continue
        if s1[T] == s0[T] and s1[CT] == s0[CT]:
            kind = "same"
        elif s1[T] == s0[CT] and s1[CT] == s0[T]:
            kind = "swap"
        else:
            kind = "partial"
        if kind == "swap":
            phys_tr[r] += 1
            ps.append(r)
        if kind == "partial":
            phys_other[r] += 1
    for r in rounds:
        if (r + 1) in by_m[m]:
            t, c = by_m[m][r]
            t2, c2 = by_m[m][r + 1]
            cs = classify(t, c, t2, c2)
            if cs and all(s == 1 for _, s in cs):
                field_tr[r] += 1
                fs.append(r)
    phys_by_match[m] = tuple(ps)
    field_by_match[m] = tuple(fs)

print("\nphysical swap transitions by r (r -> r+1), all matches:", sorted(phys_tr.items()))
print("field-only swap transitions by r:", sorted(field_tr.items()))
print("physical partial/unknown by r:", sorted(phys_other.items())[:20])

# per-match patterns
pat = collections.Counter()
for m in by_m:
    grp = "day12_25" if int(m.split("/")[3]) >= 12 else "day01_07"
    pat[(grp, phys_by_match.get(m, ()), field_by_match.get(m, ()))] += 1
print("\nmatch patterns (group, physical swap rounds, field-only swap rounds): count, top 25:")
for k, v in pat.most_common(25):
    print("  ", k, v)

# A-flip rounds per match vs physical swap
flip_by_match = collections.defaultdict(list)
for (m, r), w in A.items():
    if (m, r) in FE and (m, r + 1) in FE:
        t, c = FE[(m, r)]
        t2, c2 = FE[(m, r + 1)]
        cs = classify(t, c, t2, c2)
        ws = {x for x, _ in cs}
        if len(ws) == 1:
            cw = ws.pop()
            if cw != w:
                flip_by_match[m].append(r)
print("\nmatches with at least one A-flip (A != field winner):", len(flip_by_match))
rel = collections.Counter()
for m, rs in flip_by_match.items():
    ps = phys_by_match.get(m, ())
    for r in rs:
        # position of this flip relative to the nearest physical swap before it
        prev = [p for p in ps if p < r]
        if prev:
            rel[("after_phys_swap_at", max(prev), "flip_round", r, "gap", r - max(prev))] += 1
        else:
            rel[("no_prior_phys_swap", "flip_round", r)] += 1
print("A-flip round relative to physical swaps (top 30):")
for k, v in rel.most_common(30):
    print("  ", k, v)

# matches with a day12-25 physical swap at 8 (no 12)
p8 = [m for m in by_m if 8 in phys_by_match.get(m, ())]
print("\nmatches with physical swap at 8->9:", len(p8), "| of which field swap at 8 too:",
      sum(1 for m in p8 if 8 in field_by_match.get(m, ())))
print("A-flip rounds in those matches:", collections.Counter(r for m in p8 for r in flip_by_match.get(m, [])).most_common(10))
p12 = [m for m in by_m if 12 in phys_by_match.get(m, ())]
print("matches with physical swap at 12->13:", len(p12), "| of which field swap at 12 too:",
      sum(1 for m in p12 if 12 in field_by_match.get(m, ())))
print("A-flip rounds in those matches:", collections.Counter(r for m in p12 for r in flip_by_match.get(m, [])).most_common(10))
