"""verify D7-6 common loader (read-only). Shared by verify_D7_6_alt.py.

Same definitions as verify_D7_6_recompute.py: own min-swap ledger from round_freeze_end scores (never reads
winner_team_code), bomb outcome side (defused -> 3 CT, exploded -> 2 T), header match_date (UTC, first 19 chars).
"""
import glob
import sys
import collections
import datetime as dt
from pathlib import Path

import polars as pl

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
ONSET = dt.datetime(2024, 2, 7)  # ASSUMPTION: claim's pre/post split (UTC)
INF = 10 ** 9


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def read(table, cols, filt_col=None, filt_vals=None):
    lf = pl.concat([pl.scan_parquet(f) for f in files(BASE / f"{table}{SUF}")])
    if filt_col is not None:
        lf = lf.filter(pl.col(filt_col).is_in(filt_vals))
    return lf.select(cols).collect().to_dicts()


def trans(t0, c0, o0, t1, c1, o1):
    A0, B0 = (t0, c0) if o0 == 0 else (c0, t0)
    A1, B1 = (t1, c1) if o1 == 0 else (c1, t1)
    dA, dB = A1 - A0, B1 - B0
    if (dA, dB) == (1, 0):
        return 2 if o0 == 0 else 3
    if (dA, dB) == (0, 1):
        return 2 if o0 == 1 else 3
    return None


def ledger_match(d):
    out = {}
    rs = sorted(d)
    runs, cur = [], [rs[0]]
    for r in rs[1:]:
        if r == cur[-1] + 1:
            cur.append(r)
        else:
            runs.append(cur)
            cur = [r]
    runs.append(cur)
    for run in runs:
        L = len(run)
        if L < 2:
            continue
        T = []
        for i in range(L - 1):
            t0, c0 = d[run[i]]
            t1, c1 = d[run[i + 1]]
            T.append({(o0, o1): trans(t0, c0, o0, t1, c1, o1) for o0 in (0, 1) for o1 in (0, 1)})
        F = [[0, 0]]
        for i in range(L - 1):
            nf = [INF, INF]
            for o0 in (0, 1):
                if F[i][o0] >= INF:
                    continue
                for o1 in (0, 1):
                    if T[i][(o0, o1)] is None:
                        continue
                    nf[o1] = min(nf[o1], F[i][o0] + int(o0 != o1))
            F.append(nf)
        total = min(F[L - 1])
        if total >= INF:
            continue
        G = [None] * L
        G[L - 1] = [0, 0]
        for i in range(L - 2, -1, -1):
            gi = [INF, INF]
            for o0 in (0, 1):
                for o1 in (0, 1):
                    if T[i][(o0, o1)] is None or G[i + 1][o1] >= INF:
                        continue
                    gi[o0] = min(gi[o0], int(o0 != o1) + G[i + 1][o1])
            G[i] = gi
        for i in range(L - 1):
            sides, swaps = set(), set()
            for o0 in (0, 1):
                for o1 in (0, 1):
                    s = T[i][(o0, o1)]
                    if s is None or F[i][o0] >= INF or G[i + 1][o1] >= INF:
                        continue
                    if F[i][o0] + int(o0 != o1) + G[i + 1][o1] == total:
                        sides.add(s)
                        swaps.add(o0 != o1)
            if len(sides) == 1:
                out[run[i]] = (sides.pop(), swaps.pop())
    return out


def load():
    hdr = []
    for f in files(BASE / "header"):
        hdr += pl.read_parquet(f, columns=["key", "match_date", "map_name", "build_num"]).to_dicts()
    MDT, MAP, BLD = {}, {}, {}
    for h in hdr:
        MDT[h["key"]] = dt.datetime.strptime(h["match_date"][:19], "%Y-%m-%dT%H:%M:%S")
        MAP[h["key"]] = h["map_name"]
        BLD[h["key"]] = h["build_num"]
    re_rows = read("basic_round_end", ["match_key", "round", "winner_team_code", "win_reason_message",
                                       "player_count"])
    RE = {}
    for r in re_rows:
        RE.setdefault((r["match_key"], r["round"]), r)
    fe_rows = read("basic_round_state", ["match_key", "round", "tick", "t_score", "ct_score"],
                   "event_type", ["round_freeze_end"])
    fe_rows.sort(key=lambda x: x["tick"])
    FE = {}
    for r in fe_rows:
        FE.setdefault((r["match_key"], r["round"]), (r["t_score"], r["ct_score"]))
    by_match = collections.defaultdict(dict)
    for (k, r), v in FE.items():
        by_match[k][r] = v
    LEDGER = {}
    for k, d in by_match.items():
        for r, v in ledger_match(d).items():
            LEDGER[(k, r)] = v
    bomb = read("basic_bomb_state", ["match_key", "round", "event_type"],
                "event_type", ["bomb_defused", "bomb_exploded"])
    BOMB = collections.defaultdict(set)
    for r in bomb:
        BOMB[(r["match_key"], r["round"])].add(r["event_type"])
    return dict(MDT=MDT, MAP=MAP, BLD=BLD, RE=RE, FE=FE, LEDGER=LEDGER, BOMB=BOMB)


def bomb_side(BOMB, k):
    evs = BOMB.get(k)
    if not evs:
        return None
    return 3 if "bomb_defused" in evs else 2


def rows_for(D):
    """ledger-known rounds (player_count==20, unique ledger) with flags."""
    out = []
    for (k, r), v in D["RE"].items():
        if v["player_count"] != 20:
            continue
        lg = D["LEDGER"].get((k, r))
        if lg is None:
            continue
        prev = D["LEDGER"].get((k, r - 1))
        out.append({
            "k": k, "r": r, "L": lg[0], "w": v["winner_team_code"],
            "fas": bool(prev and prev[1]),
            "period": "pre" if D["MDT"][k] < ONSET else "post",
            "mdt": D["MDT"][k], "map": D["MAP"][k],
            "bs": bomb_side(D["BOMB"], (k, r)),
        })
    return out
