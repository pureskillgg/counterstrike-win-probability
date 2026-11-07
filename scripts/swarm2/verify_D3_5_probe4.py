"""Verifier D3-5, probe 4: concrete matches behind (a) unexpected swap pattern at completed count 7,
(b) round-9 A!=C rows, (c) round-13 A!=C rows. Read only. Prints to stdout.
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
REASON_SIDE = {1: T, 9: T, 13: T, 18: T, 7: CT, 8: CT, 11: CT, 12: CT, 17: CT}


def g_read(ch, cols=None):
    files = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


RE = g_read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).sort("tick")
RS = g_read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).sort("tick")
BS = g_read("bomb_state", ["match_key", "round", "tick", "event_type"])
DE = g_read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code"])

fe = RS.filter(pl.col("event_type") == "round_freeze_end").unique(["match_key", "round"], keep="first")
FE = {(m, r): (tk, t, c) for m, r, tk, t, c in fe.select(["match_key", "round", "tick", "t_score", "ct_score"]).iter_rows()}
A = {(m, r): (w, rc, tk) for m, r, tk, w, rc in RE.unique(["match_key", "round"], keep="first")
     .select(["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).iter_rows()}
BOMB = collections.defaultdict(set)
for m, r, ev in BS.filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"])) \
        .select(["match_key", "round", "event_type"]).iter_rows():
    BOMB[(m, int(r))].add(ev)
DEATHS = collections.defaultdict(lambda: collections.defaultdict(set))
for m, r, pid, team in DE.select(["match_key", "round", "player_id", "player_team_code"]).iter_rows():
    DEATHS[(m, int(r))][team].add(pid)


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

# (a) matches with a swap-pattern transition at completed count 7 (freeze 8 -> 9)
swap7 = []
for m, fes in by_m.items():
    if 8 in fes and 9 in fes:
        _, t, c = fes[8]
        _, t2, c2 = fes[9]
        cands = classify(t, c, t2, c2)
        if (T, 1) in cands or (CT, 1) in cands:
            if len(cands) == 1:
                swap7.append(m)
print("matches with unambiguous swap pattern at 8->9:", len(swap7))
for m in swap7[:3]:
    print("\n" + "=" * 110)
    print("MATCH", m[-45:], "day", m.split("/")[3])
    fes = by_m[m]
    rounds = sorted(fes)
    for r in rounds[:14]:
        tk, t, c = fes[r]
        w = A.get((m, r), (None, None, None))
        cands = classify(*fes[r][1:], *fes[r + 1][1:]) if (r + 1) in fes else set()
        print(f"  freeze r={r:>2} tick={tk:>7} t={t} ct={c} | A={w[0]} reason={w[1]} | bomb={sorted(BOMB.get((m, r), []))} | "
              f"dead T={len(DEATHS[(m, r)].get(T, set()))} CT={len(DEATHS[(m, r)].get(CT, set()))} | next-transition cands={sorted(cands)}")
    # swap evidence: which transitions show swap
    sw = []
    for r in rounds:
        if (r + 1) in fes:
            _, t, c = fes[r]
            _, t2, c2 = fes[r + 1]
            cs = classify(t, c, t2, c2)
            if cs and all(s == 1 for _, s in cs):
                sw.append(r)
    print("  swap-only transitions (round r -> r+1):", sw)

# (b) round-9 A != C rows, the sample of matches
def c_of(m, r):
    if (m, r) not in FE or (m, r + 1) not in FE:
        return None
    _, t, c = FE[(m, r)]
    _, t2, c2 = FE[(m, r + 1)]
    cs = classify(t, c, t2, c2)
    ws = {w for w, _ in cs}
    return ws.pop() if len(ws) == 1 else None


r9 = []
for (m, r), (w, rc, tk) in A.items():
    if r == 9 and m.split("/")[3] >= "12":
        c = c_of(m, 9)
        if c is not None and c != w:
            r9.append(m)
print("\nday12+ round-9 A!=C matches:", len(r9))
both_swap7 = sum(1 for m in r9 if m in set(swap7))
print("  of which also have swap-only pattern at 8->9:", both_swap7)
for m in r9[:2]:
    print("\n" + "=" * 110)
    print("R9 MATCH", m[-45:], "day", m.split("/")[3])
    fes = by_m[m]
    for r in sorted(fes)[:12]:
        tk, t, c = fes[r]
        w = A.get((m, r), (None, None, None))
        print(f"  freeze r={r:>2} tick={tk:>7} t={t} ct={c} | A={w[0]} reason={w[1]} | bomb={sorted(BOMB.get((m, r), []))} | next cands={sorted(classify(*fes[r][1:], *fes[r+1][1:])) if (r+1) in fes else None}")

# (c) round-13 A != C: count by pattern of earlier swap evidence
r13 = []
for (m, r), (w, rc, tk) in A.items():
    if r == 13 and m.split("/")[3] >= "12":
        c = c_of(m, 13)
        if c is not None:
            r13.append((m, w, c))
print("\nday12+ round-13 rows with C:", len(r13), "A!=C:", sum(1 for _, w, c in r13 if w != c))
sw_set = set(swap7)
print("A!=C at 13 among matches with swap at 8->9:", sum(1 for m, w, c in r13 if w != c and m in sw_set),
      "| among matches without:", sum(1 for m, w, c in r13 if w != c and m not in sw_set))
print("A==C at 13 among matches with swap at 8->9:", sum(1 for m, w, c in r13 if w == c and m in sw_set),
      "| among matches without:", sum(1 for m, w, c in r13 if w == c and m not in sw_set))

# (d) the 2024 round-13 flip rate vs. whether the match had a round-13 sides/score anomaly at 11->12 or 12->13
ex = [m for m, w, c in r13 if w != c][:2]
for m in ex:
    print("\n" + "=" * 110)
    print("R13 FLIP MATCH", m[-45:], "day", m.split("/")[3])
    fes = by_m[m]
    for r in sorted(fes)[8:16]:
        tk, t, c = fes[r]
        w = A.get((m, r), (None, None, None))
        print(f"  freeze r={r:>2} tick={tk:>7} t={t} ct={c} | A={w[0]} reason={w[1]} | bomb={sorted(BOMB.get((m, r), []))} | dead T={len(DEATHS[(m, r)].get(T, set()))} CT={len(DEATHS[(m, r)].get(CT, set()))}")
