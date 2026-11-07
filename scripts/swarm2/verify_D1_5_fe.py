"""verify_D1_5 fe: per-round side winner from round_state FREEZE-END differences (independent of
round_end and of the within-round window). Handles the halftime / OT swap explicitly:
after a swap round r, the T field at freeze(r+1) is the old CT field plus the CT-side win of round r.

  non-swap r:  dT = t[r+1] - t[r],  dC = c[r+1] - c[r]
  swap r:      dT = c[r+1] - t[r],  dC = t[r+1] - c[r]          (fields exchanged)
  winner side = 2 (T) if dT==1 and dC==0 ; 3 (CT) if dC==1 and dT==0 ; else None (ambiguous)

Then: disagreement of round_end winner vs this derived side, by round number and day group
(d12-25 vs d01-07), and a header-pair test (unordered, A6, A8) using fe-derived winners.
ASSUMPTIONS: A1 side codes, A3 day groups, A5 swap rule (bakeoff), A6 complete match, A8 max_rounds 24.
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
DAYS_HI = {f"{d:02d}" for d in range(12, 26)}
MR = 24


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def fl(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


def rd(ch, cols=None):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fl(ch)], how="vertical_relaxed")


def swap_expected(r, mr=MR):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


def side_a(r, mr=MR):
    n = sum(1 for s in range(1, r) if swap_expected(s, mr))
    return 2 if n % 2 == 0 else 3


RE = rd("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(["match_key", "round"], keep="first")
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
HD0 = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(str(BASE / "header" / "dataframe_*")))], how="vertical_relaxed")
HD = HD0.select([pl.col("key").alias("match_key"), "t_starters_score_final", "ct_starters_score_final"]).unique("match_key")

fe_df = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
         .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("t"), pl.col("ct_score").first().alias("c")))
fe = {(m, int(r)): (int(t), int(c)) for m, r, t, c in fe_df.select(["match_key", "round", "t", "c"]).iter_rows()}
print("freeze_end rows (match,round):", len(fe))

winners = collections.defaultdict(dict)
for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows():
    winners[m][int(r)] = int(w)
hdrs = {m: (int(tf), int(cf)) for m, tf, cf in HD.select(["match_key", "t_starters_score_final",
                                                          "ct_starters_score_final"]).iter_rows()}


def fe_winner(m, r):
    """derived side winner of round r from freeze(r) and freeze(r+1); None if unavailable/ambiguous."""
    a = fe.get((m, r))
    b = fe.get((m, r + 1))
    if a is None or b is None:
        return None
    t_r, c_r = a
    t_n, c_n = b
    if swap_expected(r):
        dT, dC = c_n - t_r, t_n - c_r
    else:
        dT, dC = t_n - t_r, c_n - c_r
    if dT == 1 and dC == 0:
        return 2
    if dC == 1 and dT == 0:
        return 3
    return None


# ------------------------------------------------------------------ F-A round-level agreement
hdr("F-A round_end winner vs freeze-end-derived side winner, by round number and day group")
cnt = collections.Counter()
dis = collections.Counter()
amb = collections.Counter()
for m, rounds in winners.items():
    day = m[len("csds/2024/02/"):len("csds/2024/02/") + 2]
    dg = "d12-25" if day in DAYS_HI else "d01-07"
    for r, w in rounds.items():
        fw = fe_winner(m, r)
        key = (dg, r)
        if fw is None:
            amb[key] += 1
            continue
        cnt[key] += 1
        if fw != w:
            dis[key] += 1
print("rounds 1..27 by dg (n evaluable, disagreements, ambiguous/unavailable):")
for dg in ["d01-07", "d12-25"]:
    print(dg)
    line_r, line_n, line_d, line_a = [], [], [], []
    for r in range(1, 31):
        k = (dg, r)
        line_r.append(f"{r:>3d}")
        line_n.append(f"{cnt.get(k, 0):>4d}")
        line_d.append(f"{dis.get(k, 0):>4d}")
        line_a.append(f"{amb.get(k, 0):>4d}")
    print("  round :", " ".join(line_r))
    print("  n     :", " ".join(line_n))
    print("  disagr:", " ".join(line_d))
    print("  amb   :", " ".join(line_a))

# ------------------------------------------------------------------ F-B bomb-independent round 13 check
hdr("F-B round 13: round_end vs freeze-derived side (d12-25 and d01-07)")
for dg in ["d01-07", "d12-25"]:
    k = (dg, 13)
    print(dg, "n", cnt.get(k, 0), "disagree", dis.get(k, 0), "ambiguous", amb.get(k, 0))

# ------------------------------------------------------------------ F-C header test with fe-derived winners
hdr("F-C header pair test (unordered, A6, A8) with round_end vs fe-derived winners")
complete = []
for m, (tf, cf) in hdrs.items():
    w = winners.get(m)
    if not w:
        continue
    n = len(w)
    if sorted(w) != list(range(1, n + 1)) or n != tf + cf:
        continue
    complete.append((m, tf, cf, n))
print("complete matches:", len(complete))


def totals(wd, n):
    a = sum(1 for r in range(1, n + 1) if wd[r] == side_a(r))
    return a, n - a


def ok(a, b, tf, cf):
    return sorted((a, b)) == sorted((tf, cf))


res = collections.Counter()
for m, tf, cf, n in complete:
    w = dict(winners[m])
    # fe-derived winner for every round where available; else round_end
    wf = {}
    n_fe_missing = 0
    for r in range(1, n + 1):
        fw = fe_winner(m, r)
        if fw is None:
            n_fe_missing += 1
            wf[r] = w[r]
        else:
            wf[r] = fw
    a, b = totals(w, n)
    res["round_end_raw_pass"] += ok(a, b, tf, cf)
    a2, b2 = totals(wf, n)
    res["fe_derived_pass"] += ok(a2, b2, tf, cf)
    res["fe_fully_available"] += (n_fe_missing == 0)
    if n_fe_missing == 0:
        res["fe_fully_avail_roundend_pass"] += ok(a, b, tf, cf)
        res["fe_fully_avail_fe_pass"] += ok(a2, b2, tf, cf)
        res["fe_fully_avail_n"] += 1
print(dict(res))

# ------------------------------------------------------------------ F-D fe-derived vs round_end: per-match disagreements
hdr("F-D per-match: number of rounds where round_end disagrees with fe-derived side (complete matches)")
per = collections.Counter()
for m, tf, cf, n in complete:
    w = winners[m]
    k = 0
    for r in range(1, n + 1):
        fw = fe_winner(m, r)
        if fw is not None and fw != w[r]:
            k += 1
    per[k] += 1
print("matches by count of disagreeing rounds:", dict(sorted(per.items())))
