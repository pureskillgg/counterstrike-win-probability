"""verify_D3_4_a_scores.py: independent recompute of the D3-4 claim (2024 G: tome, folder days 12..25).

Re-derives the swap signature, the first-round-after-swap flip counts and the round-13 bomb-outcome
mismatch directly from parquet (lazy scans, column projection; no code shared with D3_swap_*.py).
Adds arbiters the claim does not use: round_end win_reason_code (bomb / time / elimination reasons)
and the expected post-halftime score-delta pattern.

ASSUMPTIONS (all marked):
  A1 winner_team_code: 2 = T, 3 = CT (claim's convention; cross-checked against win_reason below).
  A2 win_reason_code -> winning side: {1,9,13,17} -> T, {7,8,11,12,18} -> CT (from win_reason_message text).
  A3 bomb-outcome subset = win_reason_code in {1 (Target_Bombed) -> T, 7 (Bomb_Defused) -> CT}.
  A4 freeze row = first round_freeze_end by tick per (match, round); delta(r) = freeze(r+1) - freeze(r).
     freeze(r) is the score before round r, so delta(r) carries the round-r result plus any side swap at r|r+1.
  A5 swap rule identical to the claim: non-clean delta with dt*dc <= 0 and (dt,dc) != (0,0).
  A6 day = folder day (key csds/2024/02/DD/...), as the claim does; match_date day reported as a variant.
  A7 "halftime" expectation for MR12 is a swap at round 12; for an MR8 format it is round 8.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D3_4_a_scores.py
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(220)

TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
T_SIDE, CT_SIDE = 2, 3
SIDE_BY_REASON = {1: 2, 9: 2, 13: 2, 17: 2, 7: 3, 8: 3, 11: 3, 12: 3, 18: 3}
BOMB = {1: 2, 7: 3}


def scan(ch, cols):
    folder = TOME / "header" if ch == "header" else TOME / f"basic_{ch}{SUF}"
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.scan_parquet(files).select(cols).collect()


def day_of(m):
    return int(m.split("/")[3])


def clean(dt, dc):
    return (dc == 1 and dt == 0) or (dt == 1 and dc == 0)


def dside(dt, dc):
    if dt == 1 and dc == 0:
        return T_SIDE
    if dc == 1 and dt == 0:
        return CT_SIDE
    return None


def pred_noswap(w):
    return (1, 0) if w == T_SIDE else (0, 1)


def pred_swap(t, c, w):
    # side-based fields after a halftime swap at r|r+1: new T field = old CT team score (+1 if CT won r)
    return (c - t + (w == CT_SIDE), t - c + (w == T_SIDE))


# ---------- load ----------
RS = scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
fe_rows = RS.filter(pl.col("event_type") == "round_freeze_end")
fe = (fe_rows.sort("tick")
      .group_by(["match_key", "round"], maintain_order=True)
      .agg(pl.col("t_score").first(), pl.col("ct_score").first(), pl.len().alias("n_fe")))
print(f"freeze_end rows={fe_rows.height}  (m,r) pairs={fe.height}  pairs with >1 freeze row={fe.filter(pl.col('n_fe') > 1).height}")
live = {(m, r): (t, c) for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}

RE_all = scan("round_end", ["match_key", "round", "winner_team_code", "win_reason_code"])
dup = RE_all.group_by(["match_key", "round"]).agg(pl.col("winner_team_code").n_unique().alias("nw"),
                                                  pl.len().alias("n")).filter(pl.col("n") > 1)
print(f"round_end rows={RE_all.height}  duplicated (m,r)={dup.height}  of which winner differs={dup.filter(pl.col('nw') > 1).height}")
RE = RE_all.unique(["match_key", "round"], keep="first", maintain_order=True)
win = {(m, r): (w, rc) for m, r, w, rc in RE.select(["match_key", "round", "winner_team_code", "win_reason_code"]).iter_rows()}
n_end = collections.Counter(m for (m, _) in win)

H = scan("header", ["key", "map_name", "match_date", "t_starters_score_final", "ct_starters_score_final"])
hdr = {k: (mp, md, tf, cf) for k, mp, md, tf, cf in H.iter_rows()}

by_m = collections.defaultdict(list)
for (m, r) in live:
    by_m[m].append(r)
matches = [m for m in by_m if day_of(m) >= 12]
print(f"matches with live rows: {len(by_m)}  folder day 12..25: {len(matches)}")

# ---------- A5 swap detection (claim rule) and non-clean category census ----------
def swaps_of(m):
    out = []
    for r in sorted(by_m[m]):
        if (m, r + 1) not in live:
            continue
        t1, c1 = live[(m, r)]
        t2, c2 = live[(m, r + 1)]
        dt, dc = t2 - t1, c2 - c1
        if clean(dt, dc) or (dt, dc) == (0, 0):
            continue
        if dt * dc <= 0:
            out.append(r)
    return tuple(out)


sig = {m: swaps_of(m) for m in matches}
cat = collections.Counter()
for m in matches:
    for r in by_m[m]:
        if (m, r + 1) not in live:
            cat["no_next_freeze"] += 1
            continue
        t1, c1 = live[(m, r)]
        t2, c2 = live[(m, r + 1)]
        dt, dc = t2 - t1, c2 - c1
        if clean(dt, dc):
            cat["clean"] += 1
        elif (dt, dc) == (0, 0):
            cat["zero_delta"] += 1
        elif dt * dc <= 0:
            cat["swap_like"] += 1
        else:
            cat["same_sign_nonclean"] += 1
print("\n== census of consecutive freeze deltas (day 12..25) ==")
print(dict(cat))

print("\n== 1. swap signature reproduction (claim: (12,)2429 ()604 (8,)333 (12,27)70 (27,)15) ==")
sc = collections.Counter(sig.values())
for k, v in sc.most_common():
    print(f"  {str(k):<12} {v}")
print(f"  total {sum(sc.values())} (claim 3451)")

# ---------- 2. flip table reproduction (claim method, winner_team_code as arbiter) ----------
flip_w = collections.Counter()
flip_r = collections.Counter()
for m in matches:
    for s in sig[m]:
        if (m, s + 1) in win and (m, s + 1) in live and (m, s + 2) in live:
            t1, c1 = live[(m, s + 1)]
            t2, c2 = live[(m, s + 2)]
            dt, dc = t2 - t1, c2 - c1
            if clean(dt, dc):
                ds = dside(dt, dc)
                flip_w[(s, ds != win[(m, s + 1)][0])] += 1
                flip_r[(s, ds != SIDE_BY_REASON[win[(m, s + 1)][1]])] += 1
print("\n== 2. flip check, first round after swap (key=(swap round, disagrees?)) ==")
print("  vs winner_team_code (claim: 8F79 8T240 12F719 12T1742 27F20 27T35):", sorted(flip_w.items()))
print("  vs win_reason side  (independent arbiter):                        ", sorted(flip_r.items()))
for s in (8, 12, 27):
    n_t = flip_w[(s, True)]
    n_f = flip_w[(s, False)]
    if n_t + n_f:
        print(f"  swap {s}: disagree {n_t}/{n_t + n_f} = {100*n_t/(n_t+n_f):.1f}%  (claim: 71-75% for 12/8; 27 claimed 71-75% but 35/55 = 63.6%)")

# ---------- 3. round-13 bomb outcome (claim: 605 rows, 434 mismatch; other rounds 14,602 rows, 41 mismatch) ----------
def bomb_stats(rows):
    n = bad = 0
    for m, r in rows:
        w, rc = win[(m, r)]
        if rc in BOMB:
            n += 1
            bad += int(w != BOMB[rc])
    return n, bad


keys_all = list(win.keys())
r13_folder = [(m, r) for (m, r) in keys_all if r == 13 and 12 <= day_of(m) <= 25]
oth_folder = [(m, r) for (m, r) in keys_all if r != 13 and 12 <= day_of(m) <= 25]
print("\n== 3. bomb-outcome mismatch (folder day 12..25) ==")
n13, b13 = bomb_stats(r13_folder)
no, bo = bomb_stats(oth_folder)
print(f"  round 13: {b13}/{n13} mismatch (claim 434/605)")
print(f"  other rounds: {bo}/{no} mismatch (claim 41/14602)")
md_rows = [(m, r) for (m, r) in r13_folder if int(hdr[m][1][8:10]) in range(12, 26)]
nm, bm = bomb_stats(md_rows)
print(f"  round 13, match_date day 12..25 variant: {bm}/{nm} mismatch")
print("  per folder day, round 13 bomb mismatch:")
per_day = collections.defaultdict(lambda: [0, 0])
for m, r in r13_folder:
    w, rc = win[(m, r)]
    if rc in BOMB:
        per_day[day_of(m)][0] += 1
        per_day[day_of(m)][1] += int(w != BOMB[rc])
print("   ", {d: f"{b}/{n}" for d, (n, b) in sorted(per_day.items())})

# ---------- 4. arbiter table by signature group and offset from the swap round ----------
def labels_for(m):
    ss = sig[m]
    if ss:
        return [(f"{ss}@{s}", s) for s in ss]
    return [("()@12", 12)]


tab = collections.defaultdict(collections.Counter)
for m in matches:
    for lab, s in labels_for(m):
        for k in range(-2, 4):
            r = s + k
            if (m, r) not in win:
                continue
            w, rc = win[(m, r)]
            acc = tab[(lab, k)]
            acc["n"] += 1
            if rc in BOMB:
                acc["bomb_n"] += 1
                acc["bomb_mis"] += int(w != BOMB[rc])
            acc["side_n"] += 1
            acc["win_mis_reason"] += int(w != SIDE_BY_REASON[rc])
            if (m, r) in live and (m, r + 1) in live:
                t1, c1 = live[(m, r)]
                t2, c2 = live[(m, r + 1)]
                dt, dc = t2 - t1, c2 - c1
                if clean(dt, dc):
                    acc["dclean"] += 1
                    ds = dside(dt, dc)
                    acc["d_vs_reason_mis"] += int(ds != SIDE_BY_REASON[rc])
                    acc["d_vs_win_mis"] += int(ds != w)
print("\n== 4. arbiter table: label, offset k from swap round s (round r = s+k) ==")
print(f"  {'label':<14}{'k':>3}{'rounds':>8}{'bomb_n':>8}{'bomb_mis':>9}{'win!=reason':>13}{'dclean':>8}{'d!=reason':>11}{'d!=win':>9}")
for (lab, k) in sorted(tab, key=lambda x: (x[0], x[1])):
    a = tab[(lab, k)]
    print(f"  {lab:<14}{k:>3}{a['n']:>8}{a['bomb_n']:>8}{a['bomb_mis']:>9}{a['win_mis_reason']:>13}{a['dclean']:>8}{a['d_vs_reason_mis']:>11}{a['d_vs_win_mis']:>9}")

# ---------- 5. swap-model consistency at the boundary s|s+1 ----------
cons = collections.defaultdict(collections.Counter)
for m in matches:
    for lab, s in labels_for(m):
        if (m, s) not in win or (m, s) not in live or (m, s + 1) not in live:
            continue
        t, c = live[(m, s)]
        t2, c2 = live[(m, s + 1)]
        obs = (t2 - t, c2 - c)
        w, rc = win[(m, s)]
        wr = SIDE_BY_REASON[rc]
        a = cons[lab]
        a["n"] += 1
        a["obs=noswap(w_tc)"] += obs == pred_noswap(w)
        a["obs=swap(w_tc)"] += obs == pred_swap(t, c, w)
        a["obs=swap(w_reason)"] += obs == pred_swap(t, c, wr)
        a["obs=noswap(w_reason)"] += obs == pred_noswap(wr)
        a["swap_pred_clean(hidden)"] += clean(*pred_swap(t, c, wr))
        a["abs_diff<=1"] += abs(c - t) <= 1
print("\n== 5. boundary consistency: observed delta vs no-swap / swap prediction (s = detected swap round) ==")
for lab in sorted(cons):
    a = cons[lab]
    print(f"  {lab:<14} n={a['n']:<6} " + "  ".join(f"{k}={v}" for k, v in a.items() if k != "n"))

# ---------- 6. format check: final starter scores and round totals per signature group ----------
fmt = collections.defaultdict(list)
for m in matches:
    tf, cf = hdr[m][2], hdr[m][3]
    fmt[str(sig[m])].append((max(tf, cf), tf + cf, n_end[m]))
print("\n== 6. format check per signature group ==")
for lab in sorted(fmt, key=lambda x: -len(fmt[x])):
    rows = fmt[lab]
    mx = collections.Counter(x[0] for x in rows)
    sum_eq = sum(1 for x in rows if x[1] == x[2])
    ne = sorted(x[2] for x in rows)
    print(f"  {lab:<12} n={len(rows):<5} max_final_score top={mx.most_common(4)}  "
          f"sum_final==rounds: {sum_eq}/{len(rows)}  rounds min/med/max={ne[0]}/{ne[len(ne)//2]}/{ne[-1]}")

# ---------- 7. declining detection: swap at 12 is only visible when |score diff| >= 2 ----------
det = collections.Counter()
for m in matches:
    if sig[m] == ():
        if (m, 12) in win and (m, 12) in live and (m, 13) in live and n_end[m] >= 13:
            t, c = live[(m, 12)]
            det["()_long_n"] += 1
            det["()_long_abs_diff<=1"] += abs(c - t) <= 1
    if sig[m] == (12,):
        t, c = live[(m, 12)]
        det["(12,)_n"] += 1
        det["(12,)_abs_diff<=1"] += abs(c - t) <= 1
print("\n== 7. halftime score gap: swap visibility ==")
print("  ", dict(det))
