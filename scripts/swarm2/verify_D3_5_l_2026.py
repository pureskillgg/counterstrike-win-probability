"""D3-5 verifier, step L (2026 bake-off tome): do the bake-off labels (round_end winners) and the bake-off feature
(round_state freeze fields) agree, and do both match the header final totals?

Swap rule = bakeoff.swap_expected(r, max_rounds) as written (copied, not imported): swap after r if r == mr // 2,
or r > mr and (r - mr - 3) % 6 == 0. Witnesses as in step K. Read only.
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(200)
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
T, CT = 2, 3


def rd(ch, cols):
    folders = sorted(glob.glob(str(TOME / f"basic_{ch}.*")))
    if ch == "header":
        folders = sorted(glob.glob(str(TOME / "header")))
    files = []
    for fd in folders:
        files += sorted(glob.glob(str(Path(fd) / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed"), len(files)


def swap_expected(r, mr):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


print("== 2026 bake-off tome: labels, feature and header truth ==")
hdr_df, nf = rd("header", ["key", "max_rounds", "t_starters_score_final", "ct_starters_score_final"])
print("header rows:", hdr_df.height, "files:", nf)
MR = {k: (mr, t, c) for k, mr, t, c in hdr_df.iter_rows()}
print("max_rounds distribution:", collections.Counter(v[0] for v in MR.values()))

RE_df, nre = rd("round_end", ["match_key", "round", "winner_team_code", "player_count"])
print("round_end rows:", RE_df.height, "files:", nre)
RE = RE_df.unique(["match_key", "round"], keep="first")
RS_df, nrs = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"])
print("round_state rows:", RS_df.height, "files:", nrs)
fe = (RS_df.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
BS_df, nbs = rd("bomb_state", ["match_key", "round", "event_type"])
BS = BS_df.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
bomb = BS.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))

W = collections.defaultdict(dict)
for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows():
    W[m][r] = w
LIVE = collections.defaultdict(dict)
for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    LIVE[m][r] = (t, c)
BOMB = {}
for m, r, evs in bomb.iter_rows():
    BOMB[(m, r)] = CT if "bomb_defused" in evs else T

# contiguity
cls = collections.Counter()
for m, d in W.items():
    rs = sorted(d)
    cls["contig" if rs == list(range(1, len(rs) + 1)) else ("start2" if rs and rs[0] == 2 else "other")] += 1
print("\n[A] round-1 presence in 2026 round_end:", dict(cls))

# per round witness table (bakeoff swap rule with the match's max_rounds)
tab = collections.defaultdict(collections.Counter)
swapped_rounds = collections.Counter()
for m in W:
    if m not in MR:
        tab["_"]["no_header"] += 1
        continue
    mr = MR[m][0]
    for r in sorted(W[m]):
        if r not in LIVE[m] or (r + 1) not in LIVE[m]:
            continue
        t1, c1 = LIVE[m][r]
        t2, c2 = LIVE[m][r + 1]
        dd = (t2 - t1, c2 - c1)
        if swap_expected(r, mr):
            tab[r]["swap_transition_excluded"] += 1
            continue
        if dd == (1, 0):
            D = T
        elif dd == (0, 1):
            D = CT
        else:
            tab[r]["unclear"] += 1
            continue
        tab[r]["n"] += 1
        tab[r]["W!=D"] += int(W[m][r] != D)
        if (m, r) in BOMB:
            tab[r]["bomb_n"] += 1
            tab[r]["B!=D"] += int(BOMB[(m, r)] != D)
            tab[r]["B!=W"] += int(BOMB[(m, r)] != W[m][r])
tot = collections.Counter()
for r, c in tab.items():
    if r == "_":
        continue
    for k, v in c.items():
        tot[k] += v
print("\n[B] 2026 witness totals (valid unit-delta transitions, bakeoff swap rule):", dict(tot))
print("    rounds with any W!=D (round: count):", sorted((r, c["W!=D"]) for r, c in tab.items()
                                                     if r != "_" and c["W!=D"] > 0))
print("    rounds with any B!=D (round: count):", sorted((r, c["B!=D"]) for r, c in tab.items()
                                                     if r != "_" and c["B!=D"] > 0))
print("    rounds with any B!=W (round: count):", sorted((r, c["B!=W"]) for r, c in tab.items()
                                                     if r != "_" and c["B!=W"] > 0))

# header truth test (order-free) with the bakeoff swap rule
res = collections.Counter()
exs = []
for m in W:
    if m not in MR or MR[m][1] is None or MR[m][2] is None:
        continue
    mr, ht, hc = MR[m]
    rounds = sorted(W[m])
    M = rounds[-1]
    if rounds != list(range(1, M + 1)):
        res["not_contiguous"] += 1
        continue
    if M != ht + hc:
        res["not_complete"] += 1
        continue
    res["complete"] += 1
    side_a = T
    a = 0
    for r in rounds:
        a += int(W[m][r] == side_a)
        if swap_expected(r, mr):
            side_a = CT if side_a == T else T
    b = M - a
    ok_w = (a, b) in [(ht, hc), (hc, ht)]
    res["winner_ok"] += int(ok_w)
    if M in LIVE[m]:
        t_n, c_n = LIVE[m][M]
        wM = W[m][M]
        tf, cf = t_n + (wM == T), c_n + (wM == CT)
        ok_f = sorted([tf, cf]) == sorted([ht, hc])
        res["field_ok"] += int(ok_f)
        res["field_tested"] += 1
    if not ok_w and len(exs) < 4:
        exs.append((m[-45:], M, mr, ht, hc, a, b))
print("\n[C] 2026 header truth test (contiguous and complete):", dict(res))
print("    winner failures (match, N, max_rounds, hdrT, hdrCT, A, B):", exs)
print("\n== end of 2026 section ==")
