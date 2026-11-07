"""verify D8-5 (part b): alternatives to 'a per-match halftime regime keyed by the round_state score signature'.

Checks (all independent of scripts/swarm2/D8_*.py):
  T1  detected score signature vs event halftime label (round_announce_last_round_half) over all header matches, days 12-25
  T2  demo build_num x event halftime label x map (header), days 12-25 (no round_end filter)
  T3  complete matches with event halftime 8 (MR8): fit under the MR8 model (swap after 8, round-9 winner flip,
      header orientation 0 or 1 per match) versus the D4 MR12 baseline
  T4  complete matches with event halftime 12 (MR12): fit under the MR12 model (swap after 12, round-13 flip)
  T5  residual MR8 matches not fitted by the MR8 model: single-flip explanations at any round
  T6  round-13 bomb mismatch by build, days 12-25 (build vs day confound for the D8-1 defect)

Conventions (ASSUMPTION): as verify_D8_5_main.py. Halftime after round h = min(label) + 1 (D4-5 convention).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D8_5_b_mr8.py
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def fo(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def ch_files(ch):
    return fo(BASE / f"basic_{ch}{SUF}")


def day_of(key):
    return int(key.split("/")[3])


HDR = pl.concat([pl.read_parquet(f, columns=["key", "map_name", "build_num", "t_starters_score_final",
                                             "ct_starters_score_final"]) for f in fo(BASE / "header")],
                how="vertical_relaxed")
H = {r["key"]: r for r in HDR.iter_rows(named=True)}
print(f"header matches {len(H)}")

RE = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "winner_team_code"]) for f in ch_files("round_end")],
               how="vertical_relaxed").unique(["match_key", "round"], keep="first")
rw_all = collections.defaultdict(dict)
for r in RE.iter_rows(named=True):
    rw_all[r["match_key"]][int(r["round"])] = int(r["winner_team_code"])

BS = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "event_type"])
                .filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])) for f in ch_files("bomb_state")],
               how="vertical_relaxed")
ev = BS.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
bside = {}
for r in ev.iter_rows(named=True):
    bside[(r["match_key"], int(r["round"]))] = CT if "bomb_defused" in r["evs"] else T

HT = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "event_type"])
                .filter(pl.col("event_type") == "round_announce_last_round_half").select(["match_key", "round"])
                for f in ch_files("round_state")], how="vertical_relaxed")
HTL = collections.defaultdict(set)
for r in HT.iter_rows(named=True):
    HTL[r["match_key"]].add(int(r["round"]))


def ev_h(m):
    return (min(HTL[m]) + 1) if HTL.get(m) else None


RSF = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
                 .filter(pl.col("event_type") == "round_freeze_end")
                 .select(["match_key", "round", "tick", "t_score", "ct_score"]) for f in ch_files("round_state")],
                how="vertical_relaxed")
FE = RSF.sort("tick").unique(["match_key", "round"], keep="first")
live = {(m, r): (int(t), int(c)) for m, r, t, c in FE.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
del RSF, FE
by_m = collections.defaultdict(list)
for (m, r) in live:
    by_m[m].append(r)
SIG = {}
for m, rs in by_m.items():
    sw = []
    for r in sorted(rs):
        if (m, r + 1) not in live:
            continue
        t1, c1 = live[(m, r)]
        t2, c2 = live[(m, r + 1)]
        dt, dc = t2 - t1, c2 - c1
        if (dc == 1 and dt == 0) or (dt == 1 and dc == 0) or (dt == 0 and dc == 0):
            continue
        if dt * dc <= 0:
            sw.append(r)
    SIG[m] = tuple(sw)


def side_a(r, swaps):
    k = sum(1 for s in swaps if s < r)
    return T if k % 2 == 0 else CT


def wa_of(rw, n, swaps, flips=()):
    wa = sum(1 for r in range(1, n + 1) if rw[r] == side_a(r, swaps))
    for r in flips:
        wa = wa - 1 if rw[r] == side_a(r, swaps) else wa + 1
    return wa


def fits(rw, n, tot, swaps, orient, flips=()):
    if any(r not in rw for r in flips):
        return False
    tA = tot[0] if orient == 0 else tot[1]
    return wa_of(rw, n, swaps, flips) == tA


def tot_of(m):
    return (int(H[m]["t_starters_score_final"]), int(H[m]["ct_starters_score_final"]))


COMPLETE = []
for m, rw in rw_all.items():
    if m not in H:
        continue
    tA, tB = tot_of(m)
    n = len(rw)
    if n == tA + tB and n <= 24 and sorted(rw) == list(range(1, n + 1)):
        COMPLETE.append(m)
ALLH = [m for m in H if 12 <= day_of(m) <= 25]

# D4 MR12 baseline classes (same rule as the claim)
CLS = {}
for m in COMPLETE:
    rw = rw_all[m]
    n = len(rw)
    tot = tot_of(m)
    mm13 = 13 in rw and (m, 13) in bside and bside[(m, 13)] != rw[13]
    if mm13:
        CLS[m] = "affected"
    elif fits(rw, n, tot, (12,), 0):
        CLS[m] = "fits"
    elif fits(rw, n, tot, (12,), 0, (13,)):
        CLS[m] = "r13_flip_misfit"
    else:
        CLS[m] = "other_misfit"
print(f"complete matches {len(COMPLETE)}; classes {dict(collections.Counter(CLS.values()))}")

# ------------------------------------------------------------------ T1 detector vs event label
print("\n=== T1 detected score signature x event halftime label (all header matches, days 12-25) ===")
t1 = collections.Counter()
for m in ALLH:
    if m in rw_all:
        t1[(str(SIG.get(m, ())), str(ev_h(m)))] += 1
sigs = sorted({k[0] for k in t1})
evs = sorted({k[1] for k in t1})
print(f"{'detected':<12}" + "".join(f"{'event_h=' + e:>14}" for e in evs))
for s in sigs:
    print(f"{s:<12}" + "".join(f"{t1.get((s, e), 0):>14}" for e in evs))
mr8_all = sum(1 for m in ALLH if ev_h(m) == 8 and m in rw_all)
det8_mr8 = sum(1 for m in ALLH if ev_h(m) == 8 and m in rw_all and SIG.get(m) == (8,))
det8_any = sum(1 for m in ALLH if m in rw_all and SIG.get(m) == (8,))
print(f"event MR8 matches (with round_end): {mr8_all}; detected (8,) among them: {det8_mr8} "
      f"(sensitivity {det8_mr8 / mr8_all:.3f}); detected (8,) overall: {det8_any} "
      f"(precision {det8_mr8 / det8_any:.3f})")
c8 = [m for m in COMPLETE if ev_h(m) == 8]
print(f"complete event-MR8 matches: {len(c8)}; of these detected (8,): "
      f"{sum(1 for m in c8 if SIG.get(m) == (8,))}")

# ------------------------------------------------------------------ T2 build x event label x map
print("\n=== T2 build x event halftime label, all header matches days 12-25 ===")
t2 = collections.Counter((H[m]["build_num"], str(ev_h(m))) for m in ALLH)
for k_, v_ in sorted(t2.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
    print(f"  build={k_[0]} halftime_after={k_[1]:<5} n={v_}")
t2m = collections.Counter((H[m]["build_num"], H[m]["map_name"], str(ev_h(m))) for m in ALLH if ev_h(m) == 8)
print("  MR8 (halftime 8) by build x map:", sorted(t2m.items()))
print("  build 9842 days 12-25 maps:",
      sorted(collections.Counter(H[m]["map_name"] for m in ALLH if H[m]["build_num"] == 9842).items()))

# ------------------------------------------------------------------ T3 MR8 model on complete event-MR8 matches
print("\n=== T3 complete event-MR8 matches (n=%d): MR8 model vs MR12 baseline ===" % len(c8))
base_fit = sum(1 for m in c8 if fits(rw_all[m], len(rw_all[m]), tot_of(m), (12,), 0)
               or fits(rw_all[m], len(rw_all[m]), tot_of(m), (12,), 0, (13,)))
print(f"  MR12 baseline (as-is, or r13 flip): {base_fit}/{len(c8)}")
mr8_o0 = sum(1 for m in c8 if fits(rw_all[m], len(rw_all[m]), tot_of(m), (8,), 0, (9,)))
mr8_o1 = sum(1 for m in c8 if fits(rw_all[m], len(rw_all[m]), tot_of(m), (8,), 1, (9,)))
mr8_any = sum(1 for m in c8 if fits(rw_all[m], len(rw_all[m]), tot_of(m), (8,), 0, (9,))
              or fits(rw_all[m], len(rw_all[m]), tot_of(m), (8,), 1, (9,)))
mr8_none_any = sum(1 for m in c8 if fits(rw_all[m], len(rw_all[m]), tot_of(m), (8,), 0)
                   or fits(rw_all[m], len(rw_all[m]), tot_of(m), (8,), 1))
print(f"  MR8 model, round-9 flip: orient0 {mr8_o0}, orient1 {mr8_o1}, either orientation {mr8_any}/{len(c8)}")
print(f"  MR8 model, no flip (physical swap only): either orientation {mr8_none_any}/{len(c8)}")
both = sum(1 for m in c8 if fits(rw_all[m], len(rw_all[m]), tot_of(m), (8,), 0, (9,))
           and fits(rw_all[m], len(rw_all[m]), tot_of(m), (8,), 1, (9,)))
print(f"  both orientations fit with round-9 flip: {both}")
print("  MR8 model (round-9 flip, either orient) by class (MR12 label):",
      sorted(collections.Counter(CLS[m] for m in c8 if (fits(rw_all[m], len(rw_all[m]), tot_of(m), (8,), 0, (9,))
                                                    or fits(rw_all[m], len(rw_all[m]), tot_of(m), (8,), 1, (9,)))).items()))

# orientation pattern: which orientation fits
pat = collections.Counter()
for m in c8:
    n = len(rw_all[m])
    o0 = fits(rw_all[m], n, tot_of(m), (8,), 0, (9,))
    o1 = fits(rw_all[m], n, tot_of(m), (8,), 1, (9,))
    pat[("o0" if o0 else "") + ("o1" if o1 else "") or "none"] += 1
print("  orientation that fits under MR8 model + round-9 flip:", dict(pat))

# ------------------------------------------------------------------ T4 MR12 model on complete event-MR12 matches
c12 = [m for m in COMPLETE if ev_h(m) == 12]
print(f"\n=== T4 complete event-MR12 matches (n={len(c12)}): MR12 model (swap after 12, round-13 flip) ===")
m12 = sum(1 for m in c12 if fits(rw_all[m], len(rw_all[m]), tot_of(m), (12,), 0, (13,))
          or fits(rw_all[m], len(rw_all[m]), tot_of(m), (12,), 0))
print(f"  MR12 model (as-is, with or without r13 flip): {m12}/{len(c12)}")
print(f"  MR12 model with reversed orientation (r13 flip or none): "
      f"{sum(1 for m in c12 if fits(rw_all[m], len(rw_all[m]), tot_of(m), (12,), 1, (13,)) or fits(rw_all[m], len(rw_all[m]), tot_of(m), (12,), 1))}")

# ------------------------------------------------------------------ T5 residual MR8 matches
print("\n=== T5 residual MR8 matches (not fitted by MR8 model with round-9 flip, either orientation) ===")
res = [m for m in c8 if not (fits(rw_all[m], len(rw_all[m]), tot_of(m), (8,), 0, (9,))
                             or fits(rw_all[m], len(rw_all[m]), tot_of(m), (8,), 1, (9,)))]
print(f"  residual count: {len(res)}")
for m in res[:40]:
    rw = rw_all[m]
    n = len(rw)
    tA, tB = tot_of(m)
    singles = [r for r in range(1, n + 1) if fits(rw, n, tot_of(m), (8,), 0, (r,))]
    singles1 = [r for r in range(1, n + 1) if fits(rw, n, tot_of(m), (8,), 1, (r,))]
    print(f"  {H[m]['map_name']:<11} n={n:>2} header=({tA},{tB}) detected={SIG.get(m)} class={CLS[m]:<15} "
          f"single-flip rounds orient0={singles} orient1={singles1}")
if len(res) > 40:
    print(f"  ... {len(res) - 40} more")

# ------------------------------------------------------------------ T6 build confound for the round-13 defect
print("\n=== T6 round-13 bomb mismatch by build, days 12-25 (planted rounds) ===")
t6 = collections.Counter()
for m in ALLH:
    if m in rw_all and 13 in rw_all[m] and (m, 13) in bside:
        key_ = (H[m]["build_num"], str(12 <= day_of(m)))
        t6[(H[m]["build_num"], "n")] += 1
        if bside[(m, 13)] != rw_all[m][13]:
            t6[(H[m]["build_num"], "mm")] += 1
for b in sorted({k[0] for k in t6}):
    n_ = t6[(b, "n")]
    k_ = t6[(b, "mm")]
    print(f"  build {b}: planted round-13 rows {n_}, mismatches {k_} ({(k_ / n_ if n_ else 0):.3f})")
# ------------------------------------------------------------------ T7 header-independent test of round_end at post-swap rounds
# score-side winner of round r = T if t_score rises by 1 between freeze_end(r) and freeze_end(r+1), CT if ct_score rises.
# Used only for r >= first post-swap round (both freeze_ends post-swap, so the delta is clean). No header totals used.
print("\n=== T7 round_end winner vs score-side winner, days 12-25 (no header totals used) ===")


def score_winner(m, r):
    if (m, r) not in live or (m, r + 1) not in live:
        return None
    t1, c1 = live[(m, r)]
    t2, c2 = live[(m, r + 1)]
    dt, dc = t2 - t1, c2 - c1
    if dt == 1 and dc == 0:
        return T
    if dc == 1 and dt == 0:
        return CT
    return None


agg = collections.defaultdict(lambda: [0, 0])  # key -> [compared, disagree]
for m in ALLH:
    if m not in rw_all:
        continue
    h = ev_h(m)
    b = H[m]["build_num"]
    if h is None:
        continue
    post = h + 1  # first post-swap round
    for r_label, r_name in [(post, "first post-swap round"), (post + 1, "second post-swap round")]:
        if r_label not in rw_all[m]:
            continue
        sw_ = score_winner(m, r_label)
        if sw_ is None:
            continue
        key = (b, f"halftime_after={h}", r_name)
        agg[key][0] += 1
        agg[key][1] += 1 if rw_all[m][r_label] != sw_ else 0
for key in sorted(agg, key=lambda k: (k[0], k[1], k[2])):
    n_, d_ = agg[key]
    print(f"  build={key[0]} {key[1]:<18} {key[2]:<23} compared={n_:>4} round_end disagrees with score side={d_:>4} "
          f"({d_ / n_ if n_ else 0:.3f})")

# control: a non-post-swap round in the same matches (round 5 for MR8, round 5 for MR12)
ctrl = collections.defaultdict(lambda: [0, 0])
for m in ALLH:
    if m not in rw_all or ev_h(m) is None:
        continue
    for r_label in (3, 5, 7):
        if r_label not in rw_all[m]:
            continue
        sw_ = score_winner(m, r_label)
        if sw_ is None:
            continue
        ctrl[(H[m]["build_num"], "pre-swap control rounds 3,5,7")][0] += 1
        ctrl[(H[m]["build_num"], "pre-swap control rounds 3,5,7")][1] += 1 if rw_all[m][r_label] != sw_ else 0
for key in sorted(ctrl):
    n_, d_ = ctrl[key]
    print(f"  build={key[0]} {key[1]:<30} compared={n_:>5} disagree={d_:>4} ({d_ / n_ if n_ else 0:.4f})")

print("\nDONE (part b)")
