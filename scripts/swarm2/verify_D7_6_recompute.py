"""verify D7-6 (recompute): independent rebuild of the post-onset winner-vs-ledger disagreement numbers,
plus two side-based checks that do not use the score ledger or winner_team_code.

Inputs (read-only, 2024 tome): header, basic_round_end, basic_round_state (round_freeze_end scores),
basic_bomb_state (bomb_defused / bomb_exploded), basic_round_state (round_end event presence).

Own ledger: min-swap orientation DP over consecutive round_freeze_end (t_score, ct_score). It never reads
winner_team_code. Re-implemented here from the description in scripts/swarm2/D7_8_ledger_dp.py.

ASSUMPTIONS (marked):
  A1 player_count == 20 on round_end (claim convention).
  A2 pre/post split: header match_date (UTC, first 19 chars) < 2024-02-07 00:00 is pre.
  A3 bomb side: bomb_defused -> 3 (CT), else bomb_exploded -> 2 (T); defused wins if both (as D7_1).
  A4 win-reason message -> side table below (game's own round-end text). Validated against
     winner_team_code in the pre period before it is used as an independent check.
  A5 a ledger row exists only for unique (non-ambiguous) min-swap pairs, as in the claim.
"""
import glob
import sys
import collections
import datetime as dt
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
ONSET = dt.datetime(2024, 2, 7)  # A2
INF = 10 ** 9

# A4: side implied by the game's round-end message (2 = T, 3 = CT)
WR_SIDE = {
    "#SFUI_Notice_Target_Bombed": 2,
    "#SFUI_Notice_Terrorists_Win": 2,
    "#SFUI_Notice_CTs_Surrender": 2,
    "#SFUI_Notice_Hostages_Not_Rescued": 2,
    "#SFUI_Notice_Bomb_Defused": 3,
    "#SFUI_Notice_CTs_Win": 3,
    "#SFUI_Notice_Target_Saved": 3,
    "#SFUI_Notice_All_Hostages_Rescued": 3,
    "#SFUI_Notice_Terrorists_Surrender": 3,
}


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def read(table, cols, filt_col=None, filt_vals=None):
    lf = pl.concat([pl.scan_parquet(f) for f in files(BASE / f"{table}{SUF}")])
    if filt_col is not None:
        lf = lf.filter(pl.col(filt_col).is_in(filt_vals))
    return lf.select(cols).collect().to_dicts()


def trans(t0, c0, o0, t1, c1, o1):
    """Winner side (2 T / 3 CT) of the round between freeze-end i and i+1, or None if inconsistent."""
    A0, B0 = (t0, c0) if o0 == 0 else (c0, t0)
    A1, B1 = (t1, c1) if o1 == 0 else (c1, t1)
    dA, dB = A1 - A0, B1 - B0
    if (dA, dB) == (1, 0):
        return 2 if o0 == 0 else 3
    if (dA, dB) == (0, 1):
        return 2 if o0 == 1 else 3
    return None


def ledger_match(d):
    """d: {round: (t, c)} freeze-end scores. Returns {round: (winner_side, swap_between_round_and_next)}
    for pairs whose min-swap optimal paths all agree on the winner side (unique)."""
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


def pct(a, b):
    return f"{100.0 * a / b:.2f}%" if b else "n/a"


# ---------------- header ----------------
hdr = []
for f in files(BASE / "header"):
    hdr += pl.read_parquet(f, columns=["key", "match_date", "map_name", "build_num"]).to_dicts()
MDT, MAP, BLD = {}, {}, {}
bad_dt = 0
for h in hdr:
    try:
        MDT[h["key"]] = dt.datetime.strptime(h["match_date"][:19], "%Y-%m-%dT%H:%M:%S")
    except Exception:
        bad_dt += 1
    MAP[h["key"]] = h["map_name"]
    BLD[h["key"]] = h["build_num"]
print(f"header matches: {len(hdr)} | unique keys: {len(MAP)} | unparseable match_date: {bad_dt}")

# ---------------- round_end (winner_team_code, win reason) ----------------
re_rows = read("basic_round_end", ["match_key", "round", "winner_team_code", "win_reason_code",
                                   "win_reason_message", "player_count"])
RE = {}
for r in re_rows:
    RE.setdefault((r["match_key"], r["round"]), r)
print(f"round_end rows: {len(re_rows)} | unique (match,round): {len(RE)} | "
      f"player_count==20 unique rounds: {sum(1 for v in RE.values() if v['player_count'] == 20)}")

# ---------------- round_freeze_end scores -> own ledger ----------------
fe_rows = read("basic_round_state", ["match_key", "round", "tick", "t_score", "ct_score"],
               "event_type", ["round_freeze_end"])
fe_rows.sort(key=lambda x: x["tick"])
FE = {}
for r in fe_rows:
    FE.setdefault((r["match_key"], r["round"]), (r["t_score"], r["ct_score"]))
print(f"freeze_end rows: {len(fe_rows)} | unique (match,round): {len(FE)}")
by_match = collections.defaultdict(dict)
for (k, r), v in FE.items():
    by_match[k][r] = v
LEDGER = {}
for k, d in by_match.items():
    for r, v in ledger_match(d).items():
        LEDGER[(k, r)] = v
print(f"ledger rows (unique pairs): {len(LEDGER)} | swap-flagged: {sum(1 for v in LEDGER.values() if v[1])}")
print("ledger swap-flagged rounds (round r -> swap between r and r+1), counts by r:",
      sorted(collections.Counter(k[1] for k, v in LEDGER.items() if v[1]).items()))

# ---------------- bomb outcomes (A3) ----------------
bomb = read("basic_bomb_state", ["match_key", "round", "event_type"],
            "event_type", ["bomb_defused", "bomb_exploded"])
BOMB = collections.defaultdict(set)
for r in bomb:
    BOMB[(r["match_key"], r["round"])].add(r["event_type"])
both = sum(1 for v in BOMB.values() if len(v) > 1)
print(f"bomb outcome rounds: {len(BOMB)} | with both defused and exploded: {both}")


def bomb_side(k):
    evs = BOMB.get(k)
    if not evs:
        return None
    return 3 if "bomb_defused" in evs else 2


# round_end event presence in round_state (regime correlate)
re_ev = read("basic_round_state", ["match_key", "round"], "event_type", ["round_end"])
RE_EV_MATCHES = collections.Counter(r["match_key"] for r in re_ev)

# ---------------- analysis table ----------------
rows = []
for (k, r), v in RE.items():
    if v["player_count"] != 20:  # A1
        continue
    lg = LEDGER.get((k, r))
    if lg is None:  # A5
        continue
    Ls, _ = lg
    prev = LEDGER.get((k, r - 1))
    fas = bool(prev and prev[1])
    w = v["winner_team_code"]
    wr = WR_SIDE.get(v["win_reason_message"])
    bs = bomb_side((k, r))
    period = "pre" if MDT.get(k) is not None and MDT[k] < ONSET else "post"
    rows.append({
        "k": k, "r": r, "period": period, "mdt": MDT.get(k), "map": MAP.get(k),
        "build": BLD.get(k), "L": Ls, "w": w, "wr": wr, "bs": bs, "fas": fas,
        "msg": v["win_reason_message"], "wr_code": v["win_reason_code"],
        "w_ok": Ls == w, "wr_ok": (wr == Ls) if wr is not None else None,
        "has_re_ev": RE_EV_MATCHES.get(k, 0) > 0,
    })
print(f"\nledger-known rounds (player_count==20, unique ledger): {len(rows)}")
print("by period:", collections.Counter(x["period"] for x in rows))

# ---------------- A4 validation: win-reason side vs winner_team_code ----------------
print("\n== A4 validation: round_end message vs winner_team_code (all player_count==20 rounds with a winner) ==")
xt = collections.Counter()
for (k, r), v in RE.items():
    if v["player_count"] != 20:
        continue
    per = "pre" if MDT.get(k) is not None and MDT[k] < ONSET else "post"
    xt[(per, v["win_reason_message"], WR_SIDE.get(v["win_reason_message"]), v["winner_team_code"])] += 1
for key in sorted(xt, key=lambda z: (z[0], str(z[1]), z[3])):
    print("  ", key, xt[key])
unmapped = sum(c for (p, m, s, w), c in xt.items() if s is None)
print("  unmapped messages (no WR_SIDE entry):", unmapped)
pre_mis = sum(c for (p, m, s, w), c in xt.items() if p == "pre" and s is not None and s != w)
pre_all = sum(c for (p, m, s, w), c in xt.items() if p == "pre" and s is not None)
print(f"  pre: win-reason side != winner_team_code: {pre_mis} of {pre_all}")

# ---------------- A: recompute claim headline numbers ----------------
post = [x for x in rows if x["period"] == "post"]
pre = [x for x in rows if x["period"] == "pre"]
print("\n== A. headline recompute (ledger vs winner_team_code) ==")
print(f"pre : rows {len(pre)} | flips {sum(1 for x in pre if not x['w_ok'])}   claim: 29623 rows, 51 flips (pre = 29572+51)")
print(f"post: rows {len(post)} | flips {sum(1 for x in post if not x['w_ok'])}   claim: 44944 rows, 2642 flips")
pf = [x for x in post if not x["w_ok"]]
print(f"post flips split by bomb outcome present: bomb={sum(1 for x in pf if x['bs'] is not None)} "
      f"non-bomb={sum(1 for x in pf if x['bs'] is None)}   claim: bomb 505, non-bomb 2137")
print("post flips: winner == ledger(r-1) is not tested here (see recompute of D7_10 (a) below)")

print("\n== A. post by round number (n, flips) for claimed rounds ==")
byr = collections.defaultdict(lambda: [0, 0])
for x in post:
    byr[x["r"]][0] += 1
    byr[x["r"]][1] += int(not x["w_ok"])
for rr in [5, 9, 13, 18, 21, 23, 24, 25, 27, 28]:
    print(f"  r{rr}: n={byr[rr][0]} flips={byr[rr][1]}")
print("  top rounds by flips:", sorted(((v[1], rr, v[0]) for rr, v in byr.items()), reverse=True)[:8])

print("\n== A. first-after-swap (fas) vs other, by period ==")
for per in ["pre", "post"]:
    for fl in [True, False]:
        sub = [x for x in rows if x["period"] == per and x["fas"] == fl]
        f = sum(1 for x in sub if not x["w_ok"])
        print(f"  {per} fas={fl}: n={len(sub)} flips={f} rate={pct(f, len(sub))}")
print("  claim: post fas 2019/2017 (99.9%), post non-fas 42925/625 (1.46%), pre fas 1340/0, pre non-fas 28283/51")

print("\n== A. post round 13 and round 9 split by fas ==")
for rr in [9, 13]:
    for fl in [True, False]:
        sub = [x for x in post if x["r"] == rr and x["fas"] == fl]
        f = sum(1 for x in sub if not x["w_ok"])
        print(f"  post r{rr} fas={fl}: n={len(sub)} flips={f}")
print("  claim: r9 fas 240/240; r13 fas 1744/1742; r13 total 2214/2051")

print("\n== A. matches ==")
post_matches = collections.defaultdict(list)
for x in post:
    post_matches[x["k"]].append(x)
n_flip_m = sum(1 for k, xs in post_matches.items() if any(not x["w_ok"] for x in xs))
print(f"  post matches with >=1 ledger-known round: {len(post_matches)} | with >=1 flip: {n_flip_m} "
      f"({pct(n_flip_m, len(post_matches))})   claim: 2343 of 2478 (94.6%)")
dist = collections.Counter(sum(1 for x in xs if not x["w_ok"]) for xs in post_matches.values())
print("  flips per post match distribution (flips: matches):", sorted(dist.items())[:8])

print("\n== A. map rates (flip rate) ==")
for per in ["post", "pre"]:
    mm = collections.defaultdict(lambda: [0, 0])
    for x in rows:
        if x["period"] != per:
            continue
        mm[x["map"]][0] += 1
        mm[x["map"]][1] += int(not x["w_ok"])
    print(f"  {per}:", sorted(((round(v[1] / v[0], 4), m, v[0], v[1]) for m, v in mm.items() if v[0]), reverse=True))

print("\n== A. post per-day flip rate (UTC match_date day) ==")
pd_ = collections.defaultdict(lambda: [0, 0])
for x in post:
    d = x["mdt"].strftime("%m-%d")
    pd_[d][0] += 1
    pd_[d][1] += int(not x["w_ok"])
print("  ", [(d, v[0], v[1], round(v[1] / v[0], 3)) for d, v in sorted(pd_.items())])

print("\n== A. onset: post fas-9/13 rows in time order around the cut (first 12 flips and agreements) ==")
F = sorted([x for x in post if x["fas"] and x["r"] in (9, 13)], key=lambda x: x["mdt"])
first_flip = next((x for x in F if not x["w_ok"]), None)
print("  first post fas-9/13 flip:", first_flip and (first_flip["mdt"], first_flip["r"], first_flip["map"]))
print("  n fas-9/13 post rows:", len(F), "| flips:", sum(1 for x in F if not x["w_ok"]),
      "| agreements:", sum(1 for x in F if x["w_ok"]))
last_agree_before = max((x["mdt"] for x in F if x["w_ok"] and first_flip and x["mdt"] < first_flip["mdt"]), default=None)
print("  last agreement before first flip:", last_agree_before)
print("  agreements at or after first flip:", sorted(x["mdt"] for x in F if x["w_ok"] and first_flip and x["mdt"] >= first_flip["mdt"]))
pre_fas_all = [x for x in rows if x["period"] == "pre" and x["fas"] and x["r"] in (9, 13)]
print("  pre fas-9/13 rows:", len(pre_fas_all), "| flips:", sum(1 for x in pre_fas_all if not x["w_ok"]))

# ---------------- B. independent checks (not using ledger or winner_team_code) ----------------
print("\n== B. independent side checks on the same rows ==")
print("  wr = side implied by round_end message (A4); bs = bomb outcome side (A3); L = own ledger; w = winner_team_code")


def tally(name, sub):
    n = len(sub)
    if n == 0:
        print(f"  {name}: n=0")
        return
    flips = [x for x in sub if not x["w_ok"]]
    agree = [x for x in sub if x["w_ok"]]
    f_wr_L = sum(1 for x in flips if x["wr"] == x["L"])
    f_wr_w = sum(1 for x in flips if x["wr"] == x["w"])
    f_wr_none = sum(1 for x in flips if x["wr"] is None)
    a_wr_L = sum(1 for x in agree if x["wr"] == x["L"])
    a_wr_none = sum(1 for x in agree if x["wr"] is None)
    print(f"  {name}: n={n} | winner!=ledger {len(flips)}: win-reason sides with LEDGER {f_wr_L}, "
          f"with WINNER_CODE {f_wr_w}, no-msg {f_wr_none} | winner==ledger {len(agree)}: "
          f"win-reason==ledger {a_wr_L}, no-msg {a_wr_none}")


tally("post all", post)
tally("post bomb rounds", [x for x in post if x["bs"] is not None])
tally("post non-bomb rounds", [x for x in post if x["bs"] is None])
tally("post r13 fas", [x for x in post if x["r"] == 13 and x["fas"]])
tally("post r13 non-fas", [x for x in post if x["r"] == 13 and not x["fas"]])
tally("post r9 fas", [x for x in post if x["r"] == 9 and x["fas"]])
tally("post other fas (not 9/13)", [x for x in post if x["fas"] and x["r"] not in (9, 13)])
tally("post non-fas other rounds", [x for x in post if not x["fas"] and x["r"] not in (9, 13)])
tally("pre all", pre)
tally("pre bomb rounds", [x for x in pre if x["bs"] is not None])

print("\n  bomb outcome side vs ledger vs winner_code (rounds with bomb outcome):")
for per in ["pre", "post"]:
    for grp, cond in [("r13", lambda x: x["r"] == 13), ("other", lambda x: x["r"] != 13)]:
        sub = [x for x in rows if x["period"] == per and x["bs"] is not None and cond(x)]
        bL = sum(1 for x in sub if x["bs"] == x["L"])
        bw = sum(1 for x in sub if x["bs"] == x["w"])
        print(f"   {per} {grp}: n={len(sub)} | bomb==ledger {bL} | bomb==winner_code {bw} | "
              f"bomb!=winner_code {len(sub) - bw}")

print("\n== B. claim-recompute of the bomb mismatch (bomb side != winner_team_code), player_count==20 ==")
for lab, cond in [("r13 2024-02-01..07", lambda x: x["r"] == 13 and x["mdt"] < dt.datetime(2024, 2, 8)),
                  ("r13 2024-02-12..25", lambda x: x["r"] == 13 and x["mdt"] >= dt.datetime(2024, 2, 12)),
                  ("other rounds", lambda x: x["r"] != 13)]:
    sub = [x for x in (pre + post) if x["bs"] is not None and cond(x)]
    print(f"  {lab}: rows with bomb outcome={len(sub)} mismatch={sum(1 for x in sub if x['bs'] != x['w'])}")
print("  claim: r13 2024-02-12..25 605 rows / 434 mismatches; 02-01..07 180 / 0; other rounds 14,602 / 41")
print("  NOTE: the 2024-02-01..07 window here is ledger-known rounds only, so counts are a subset of the e2 counts.")

print("\n== C. regime correlates (post) ==")
has = collections.Counter((x["has_re_ev"], not x["w_ok"]) for x in post)
print("  post rows by (match has round_end events in round_state, flip):", sorted(has.items()))
nb = collections.Counter(BLD.get(x["k"]) for x in post)
print("  post build_num counts:", sorted(nb.items(), key=lambda z: -z[1])[:6])
nbp = collections.Counter(BLD.get(x["k"]) for x in pre)
print("  pre build_num counts:", sorted(nbp.items(), key=lambda z: -z[1])[:6])
print("  distinct build_num values across all ledger rows:", len({x["build"] for x in rows}))

print("\n== D. worked example: first post match with a round-13 flip and a bomb outcome ==")
ex = next((x for x in post if x["r"] == 13 and not x["w_ok"] and x["bs"] is not None), None)
if ex:
    k = ex["k"]
    print("  match:", k, "| mdt:", ex["mdt"], "| map:", ex["map"], "| build:", ex["build"])
    for rr in range(11, 16):
        fe = FE.get((k, rr))
        rw = RE.get((k, rr))
        lg = LEDGER.get((k, rr))
        print(f"   r{rr}: freeze(t,c)={fe} ledger={lg and lg[0]} winner_code={rw and rw['winner_team_code']} "
              f"msg={rw and rw['win_reason_message']} bomb={sorted(BOMB.get((k, rr), []))}")
