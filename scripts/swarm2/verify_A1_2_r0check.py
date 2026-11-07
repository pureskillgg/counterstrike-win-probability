# -*- coding: utf-8 -*-
"""
verify_A1_2_r0check.py
For the truncated matches whose header labels are feasible only under the swapped convention
(only_B in verify_A1_2_final.py), check whether the starter group was taken from round 1 (r0 == 1)
or from a later round (which would flip the identity if a swap had already happened).
Also print, for the claimed complete matches, the round-1 roster sizes.
"""
import os, glob, collections
import polars as pl

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"


def chan(name):
    folder = [d for d in os.listdir(T) if d.startswith(name + ".")][0]
    return sorted(glob.glob(os.path.join(T, folder, "dataframe_*")))


def scan(name, cols):
    return pl.scan_parquet(chan(name)).select(cols).collect()


h = pl.read_parquet(os.path.join(T, "header", "dataframe_00000"),
                    columns=["key", "max_rounds", "is_wingman", "t_starters_score_final", "ct_starters_score_final"])
hdr = {r[0]: r for r in h.iter_rows()}
pi = scan("basic_player_info", ["match_key", "round", "player_id", "team_code"])
pp = scan("basic_player_personal", ["match_key", "round", "player_id", "steam_id"])
re_ = scan("basic_round_end", ["match_key", "round", "winner_team_code"])
pmap = {(r[0], r[1]): r[2] for r in pp.group_by(["match_key", "player_id"]).agg(pl.col("steam_id").first().alias("sid")).iter_rows()}
side = collections.defaultdict(dict)
for mk, r, pid, tc in pi.iter_rows():
    if tc in (2, 3) and (mk, pid) in pmap:
        side[(mk, r)][pmap[(mk, pid)]] = tc
win = collections.defaultdict(dict)
for mk, r, w in re_.iter_rows():
    win[mk][r] = w

only_B = []
for mk in sorted(hdr):
    if not win.get(mk):
        continue
    S = hdr[mk][3] + hdr[mk][4]
    n = len(win[mk])
    if n >= S:
        continue
    wingman = bool(hdr[mk][2])
    need = 2 if wingman else 5
    # first full-roster round
    r0 = None
    for r in sorted(r for (m, r) in side if m == mk):
        d = side[(mk, r)]
        if sum(1 for v in d.values() if v == 2) == need and sum(1 for v in d.values() if v == 3) == need:
            r0 = r
            break
    if r0 is None:
        continue
    S2 = {s for s, v in side[(mk, r0)].items() if v == 2}
    stT = sum(1 for r in win[mk] if any(side.get((mk, r), {}).get(s) == 2 for s in S2) and win[mk][r] == 2)
    # visible totals (rounds where starter members are observed)
    stT = stCT = 0
    for r in sorted(win[mk]):
        d = side.get((mk, r), {})
        vals = [d[s] for s in S2 if s in d]
        if not vals:
            continue
        if win[mk][r] == vals[0]:
            stT += 1
        else:
            stCT += 1
    hT, hC = hdr[mk][3], hdr[mk][4]
    A = (hT >= stT) and (hC >= stCT)
    B = (hT >= stCT) and (hC >= stT)
    if (not A) and B:
        r1 = side.get((mk, 1), {})
        r1_t = sum(1 for v in r1.values() if v == 2)
        r1_c = sum(1 for v in r1.values() if v == 3)
        only_B.append(mk)
        print(mk.split("/")[3], mk.split("/")[4][:8], "wingman" if wingman else "5v5", "r0=%s" % r0,
              "round1 roster T/CT=%s/%s" % (r1_t, r1_c), "n=%d S=%d" % (n, S),
              "hdr(T,CT)=(%d,%d)" % (hT, hC), "visible start-T=%d start-CT=%d" % (stT, stCT))
print("only_B count:", len(only_B))
