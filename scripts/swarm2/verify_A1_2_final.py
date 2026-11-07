# -*- coding: utf-8 -*-
"""
verify_A1_2_final.py
(1) starter-group round r0 for the claimed matches (identity is read from round r0 onward);
(2) for wingman matches with round_end count n < header round total S (truncated round_end),
    which header labelings are feasible given the visible per-team totals (missing rounds can only add);
(3) wingman classification by day, days 01-03 vs 04-08.
Read-only.
"""
import os, glob, collections
import polars as pl

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
CLAIMED = [
    "csds/2026/10/01/m_36b246426c63e2e81c3165c3/csds",
    "csds/2026/10/01/m_b2289de5d46c89d81317f2ff/csds",
    "csds/2026/10/02/m_fe71dd645987a933c259261f/csds",
    "csds/2026/10/02/m_a76f2712208e50fe59cc1082/csds",
    "csds/2026/10/02/m_ab782810212100e291053f63/csds",
    "csds/2026/10/02/m_7cc561b7c0bd47c32f7550b0/csds",
    "csds/2026/10/02/m_757b2afedab995a6dd2f429d/csds",
    "csds/2026/10/02/m_888882dcf9e1d301f91ad5b4/csds",
    "csds/2026/10/03/m_b0bf8e190e1fcd142f36e859/csds",
    "csds/2026/10/03/m_930de9b06ad0081177ea9537/csds",
]


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


def identity(mk):
    wingman = bool(hdr[mk][2])
    need = 2 if wingman else 5
    r0 = None
    for r in sorted(r for (m, r) in side if m == mk):
        d = side[(mk, r)]
        if sum(1 for v in d.values() if v == 2) == need and sum(1 for v in d.values() if v == 3) == need:
            r0 = r
            break
    if r0 is None:
        return None
    S2 = {s for s, v in side[(mk, r0)].items() if v == 2}
    stT = stCT = 0
    unknown = []
    for r in sorted(win[mk]):
        d = side.get((mk, r), {})
        vals = [d[s] for s in S2 if s in d]
        if not vals:
            unknown.append(r)
            continue
        sT = vals[0]
        if win[mk][r] == sT:
            stT += 1
        else:
            stCT += 1
    return dict(r0=r0, stT=stT, stCT=stCT, unknown=unknown, n=len(win[mk]))


print("== (1) starter group round r0 for claimed matches ==")
for k in CLAIMED:
    a = identity(k)
    print(k.split("/")[3], k.split("/")[4][:8], "r0=", a["r0"] if a else None,
          "| visible stT=%s stCT=%s n=%s unknown_rounds=%s" % (a["stT"], a["stCT"], a["n"], a["unknown"]) if a else "")

print("\n== (2) truncated matches (round_end count n < header S): feasibility of header labelings ==")
print("   (missing rounds can only add to totals; A = header T is start-T total, B = header T is start-CT total)")
feas = collections.Counter()
for mk in sorted(hdr):
    if not win.get(mk):
        continue
    S = hdr[mk][3] + hdr[mk][4]
    n = len(win[mk])
    if n >= S:
        continue
    a = identity(mk)
    if a is None:
        feas["no_roster"] += 1
        continue
    hT, hC = hdr[mk][3], hdr[mk][4]
    A = (hT >= a["stT"]) and (hC >= a["stCT"])
    B = (hT >= a["stCT"]) and (hC >= a["stT"])
    tag = "wingman" if hdr[mk][2] else "5v5"
    day = mk.split("/")[3]
    if A and B:
        cls = "both_feasible"
    elif A:
        cls = "only_A_start_T"
    elif B:
        cls = "only_B_swapped"
    else:
        cls = "neither"
    feas[(tag, day, cls)] += 1
    if tag == "wingman":
        print("   ", day, mk.split("/")[4][:8], "n=%d S=%d" % (n, S), "hdr(T,CT)=(%d,%d)" % (hT, hC),
              "visible start-T=%d start-CT=%d" % (a["stT"], a["stCT"]), "->", cls)
for k, v in sorted(feas.items(), key=lambda x: str(x)):
    print("  ", k, v)

print("\n== (3) wingman classification: days 01-03 vs 04-08 (complete round_end n==S only) ==")
grp = collections.Counter()
for mk in sorted(hdr):
    if not hdr[mk][2] or not win.get(mk):
        continue
    S = hdr[mk][3] + hdr[mk][4]
    n = len(win[mk])
    day = int(mk.split("/")[3])
    bucket = "d01-03" if day <= 3 else "d04-08"
    if n != S:
        grp[(bucket, "n<S (truncated round_end)")] += 1
        continue
    a = identity(mk)
    if a is None:
        grp[(bucket, "no_roster")] += 1
        continue
    if a["unknown"]:
        grp[(bucket, "partial_round_missing_member")] += 1
        continue
    hT, hC = hdr[mk][3], hdr[mk][4]
    if a["stT"] == hT and a["stCT"] == hC:
        grp[(bucket, "header matches start-T")] += 1
    elif a["stT"] == hC and a["stCT"] == hT:
        grp[(bucket, "header swapped (complete)")] += 1
    else:
        grp[(bucket, "other")] += 1
for k, v in sorted(grp.items()):
    print("  ", k, v)
