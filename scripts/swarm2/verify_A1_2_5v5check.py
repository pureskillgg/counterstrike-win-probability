# -*- coding: utf-8 -*-
"""
verify_A1_2_5v5check.py
Classification of complete (round_end n == header S) 5v5 matches by day bucket, using the identity from
basic_player_info sides and round_end winners (starter group = side-2 members of round 1), lenient on
missing members (a round is used when at least one starter member is observed).
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
                    columns=["key", "is_wingman", "t_starters_score_final", "ct_starters_score_final"])
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

cls = collections.Counter()
examples = collections.defaultdict(list)
for mk in sorted(hdr):
    if bool(hdr[mk][1]) or not win.get(mk):
        continue
    S = hdr[mk][2] + hdr[mk][3]
    n = len(win[mk])
    day = int(mk.split("/")[3])
    bucket = "d01-03" if day <= 3 else "d04-08"
    if n != S:
        cls[(bucket, "n<S truncated")] += 1
        continue
    r0 = None
    for r in sorted(r for (m, r) in side if m == mk):
        d = side[(mk, r)]
        if sum(1 for v in d.values() if v == 2) == 5 and sum(1 for v in d.values() if v == 3) == 5:
            r0 = r
            break
    if r0 is None:
        cls[(bucket, "no full 5v5 round")] += 1
        continue
    S2 = {s for s, v in side[(mk, r0)].items() if v == 2}
    stT = stCT = 0
    unknown = 0
    for r in sorted(win[mk]):
        d = side.get((mk, r), {})
        vals = [d[s] for s in S2 if s in d]
        if not vals:
            unknown += 1
            continue
        if win[mk][r] == vals[0]:
            stT += 1
        else:
            stCT += 1
    hT, hC = hdr[mk][2], hdr[mk][3]
    if unknown:
        c = "partial (rounds without starter members: %d)" % unknown
    elif stT == hT and stCT == hC:
        c = "header = start-T"
    elif stT == hC and stCT == hT:
        c = "header swapped (exact)"
    else:
        c = "other totals differ"
    cls[(bucket, c)] += 1
    if c not in ("header = start-T",):
        examples[(bucket, c)].append((mk.split("/")[3], mk.split("/")[4][:8], hT, hC, stT, stCT, r0))
print("complete (n==S) 5v5 matches by day bucket:")
for k, v in sorted(cls.items()):
    print("  ", k, v)
print("non-standard examples:")
for k, v in examples.items():
    print("  ", k, v[:12])
