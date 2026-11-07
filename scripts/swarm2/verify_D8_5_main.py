"""verify D8-5 (independent recomputation): is the non-round-13 header misfit (D4 other_misfit, 116 matches)
concentrated on four maps, and does it track a per-match halftime schedule (round_state signature)?

Independent of scripts/swarm2/D8_*.py: own loaders, own classifier, own fit/config search, own permutation test.
Only reads 2024 tome (header, round_end, bomb_state, round_state) with column projection and lazy filters.

Conventions (ASSUMPTION, matching the D4/D8 logs so numbers are comparable):
  T = 2, CT = 3 (round_end winner_team_code). Header tA = t_starters_score_final, tB = ct_starters_score_final.
  Schedule (k,) means: the team that started on T is on T for rounds r <= k and on CT for r > k
  (side flips after round k). () means no flip. Orientation 1 means header totals are swapped (tB, tA).
  Complete regulation match: rounds are exactly 1..n, n == tA + tB, n <= 24.
  Round-13 bomb mismatch: bomb_defused -> bside CT, bomb_exploded -> bside T; mismatch if bside != winner at round 13.
  Round-13 flip: flip the round-13 winner between the two teams (as D4).
  Permutation: B = 5000 (ASSUMPTION), seed = 7 (ASSUMPTION, independent of the D8 seed).

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D8_5_main.py
"""
import collections
import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(60)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3
B = 5000     # ASSUMPTION: permutation replicates
SEED = 7     # ASSUMPTION: RNG seed (independent of D8 seed)
rng = np.random.default_rng(SEED)


def fo(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def rd(fd, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fo(fd)], how="vertical_relaxed")


def ch_files(ch):
    return fo(BASE / f"basic_{ch}{SUF}")


def day_of(key):
    return int(key.split("/")[3])


def region_of(server):
    if server is None:
        return "NA"
    for tok in server.split():
        if tok.startswith("us_") or tok.startswith("eu") or tok.startswith("sa") or tok.startswith("as") \
                or tok.startswith("au") or tok.startswith("af") or tok.startswith("cn") or tok.startswith("jp"):
            return tok
    return "other"


# ------------------------------------------------------------------ load
HDR = rd(BASE / "header", ["key", "map_name", "server_name", "providence", "match_type",
                           "t_starters_score_final", "ct_starters_score_final"])
assert HDR["key"].n_unique() == HDR.height
H = {r["key"]: r for r in HDR.iter_rows(named=True)}

RE = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "winner_team_code", "player_count"])
                for f in ch_files("round_end")], how="vertical_relaxed")
print(f"round_end rows (all player_count): {RE.height}; (match,round) groups: "
      f"{RE.group_by(['match_key', 'round']).len().height}")
RE = RE.unique(["match_key", "round"], keep="first")

BSO = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "event_type"])
                 .filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
                 for f in ch_files("bomb_state")], how="vertical_relaxed")
ev = BSO.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
OUT = ev.with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bside")
).select(["match_key", "round", "bside"])

RSF = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
                 .filter(pl.col("event_type") == "round_freeze_end")
                 .select(["match_key", "round", "tick", "t_score", "ct_score"])
                 for f in ch_files("round_state")], how="vertical_relaxed")
nfe = RSF.group_by(["match_key", "round"]).len()
print(f"round_freeze_end rows {RSF.height}; (match,round) groups {nfe.height}; groups with >1 freeze_end: "
      f"{nfe.filter(pl.col('len') > 1).height}")
FE = RSF.sort("tick").unique(["match_key", "round"], keep="first")
live = {(m, r): (int(t), int(c)) for m, r, t, c in FE.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
del RSF, FE

# ------------------------------------------------------------------ signature (same rule as D3/D8: flag swap if
# score deltas between freeze_end r and r+1 are not a normal single-round win)
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

# ------------------------------------------------------------------ per-match round winners
rw_all = collections.defaultdict(dict)
for r in RE.iter_rows(named=True):
    rw_all[r["match_key"]][int(r["round"])] = int(r["winner_team_code"])

bomb_side = {(r["match_key"], int(r["round"])): int(r["bside"]) for r in OUT.iter_rows(named=True)}
mm13 = set()
for m, rw in rw_all.items():
    if 13 in rw and (m, 13) in bomb_side and bomb_side[(m, 13)] != rw[13]:
        mm13.add(m)

# e2 reproduction (all days): rounds with bomb outcome and mismatches
tot_out = sum(1 for (m, r) in bomb_side if r in rw_all.get(m, {}))
tot_mm = sum(1 for (m, r), bs in bomb_side.items() if r in rw_all.get(m, {}) and bs != rw_all[m][r])
print(f"\n=== reproduction ===\nrounds with bomb outcome joined to round_end: {tot_out}; mismatches {tot_mm} "
      f"(e2 log: 15436 / 499)")

# ------------------------------------------------------------------ complete regulation universe
def side_a(r, swaps):
    k = sum(1 for s in swaps if s < r)
    return T if k % 2 == 0 else CT


def wa_of(rw, n, swaps):
    return sum(1 for r in range(1, n + 1) if rw[r] == side_a(r, swaps))


def fit(rw, n, tot, swaps, orient, flip):
    """flip: 'none' | 13 (flip round-13 winner) | 'any' (some single round flip fixes it)."""
    tA = tot[0] if orient == 0 else tot[1]
    wa = wa_of(rw, n, swaps)
    if flip == "none":
        return wa == tA
    if flip == 13:
        if 13 not in rw:
            return False
        wa2 = wa - 1 if rw[13] == side_a(13, swaps) else wa + 1
        return wa2 == tA
    if flip == "any":
        for r in range(1, n + 1):
            wa2 = wa - 1 if rw[r] == side_a(r, swaps) else wa + 1
            if wa2 == tA:
                return True
        return False
    raise ValueError(flip)


COMPLETE = []
for m, rw in rw_all.items():
    if m not in H:
        continue
    tA, tB = int(H[m]["t_starters_score_final"]), int(H[m]["ct_starters_score_final"])
    n = len(rw)
    if not (n == tA + tB and n <= 24 and sorted(rw) == list(range(1, n + 1))):
        continue
    COMPLETE.append(m)
print(f"complete regulation matches (all days): {len(COMPLETE)}; by day group: "
      f"d01-07 {sum(1 for m in COMPLETE if day_of(m) <= 7)}, d12-25 {sum(1 for m in COMPLETE if day_of(m) >= 12)}")

# all matches (any n) in day 12-25 for the population denominators
ALL1225 = [m for m in H if 12 <= day_of(m) <= 25]
print(f"header matches days 12-25: {len(ALL1225)}; with round_end rows: "
      f"{sum(1 for m in ALL1225 if m in rw_all)}")

CLS = {}
for m in COMPLETE:
    rw = rw_all[m]
    tot = (int(H[m]["t_starters_score_final"]), int(H[m]["ct_starters_score_final"]))
    n = len(rw)
    if m in mm13:
        CLS[m] = "affected"
    elif fit(rw, n, tot, (12,), 0, "none"):
        CLS[m] = "fits"
    elif fit(rw, n, tot, (12,), 0, 13):
        CLS[m] = "r13_flip_misfit"
    else:
        CLS[m] = "other_misfit"
cc = collections.Counter(CLS.values())
print(f"D4-style classes (complete): {dict(sorted(cc.items()))}  (D4 log: affected 211, r13_flip_misfit 825, "
      f"other_misfit 116, fits 94)")

OTH = [m for m in COMPLETE if CLS[m] == "other_misfit"]
print(f"other_misfit: {len(OTH)}")
print("other_misfit by map:", sorted(collections.Counter(H[m]["map_name"] for m in OTH).items()))

# ------------------------------------------------------------------ D8-5 claim numbers
print("\n=== claim numbers: detected signature of the 116 ===")
print("other_misfit by detected signature:", sorted(collections.Counter(str(SIG.get(m, ())) for m in OTH).items()))
sig_map_all = collections.Counter()
for m in ALL1225:
    if m in rw_all:
        sig_map_all[(H[m]["map_name"], str(SIG.get(m, ())))] += 1
maps = sorted({k[0] for k in sig_map_all})
sigs = sorted({k[1] for k in sig_map_all})
print("all matches days 12-25 (with round_end): signature x map counts")
print(f"{'map':<13}" + "".join(f"{s:>10}" for s in sigs))
for mp in maps:
    print(f"{mp:<13}" + "".join(f"{sig_map_all.get((mp, s), 0):>10}" for s in sigs))
tot_sig = collections.Counter()
for (mp, s), v in sig_map_all.items():
    tot_sig[s] += v
print("totals by signature (days 12-25):", dict(tot_sig))
four = {"de_inferno", "de_vertigo", "de_overpass", "de_nuke"}
print("(8,) matches on four maps:", sum(v for (mp, s), v in sig_map_all.items() if s == "(8,)" and mp in four),
      "; (8,) on other maps:", sum(v for (mp, s), v in sig_map_all.items() if s == "(8,)" and mp not in four))
print("(12,) matches on four maps:", sum(v for (mp, s), v in sig_map_all.items() if s == "(12,)" and mp in four))

# ------------------------------------------------------------------ fit under each candidate configuration
def configs():
    yield ("()", (), 0, "none")
    yield ("()", (), 1, "none")
    for k in range(0, 24):
        for o in (0, 1):
            for f in ("none", 13, "any"):
                yield (f"({k},)", (k,), o, f)


CFG = list(configs())
print("\n=== config fits for the 116 other_misfit (as-is = orient 0, reversed = orient 1) ===")
res = []
for name, sw, o, f in CFG:
    nfit = sum(1 for m in OTH if fit(rw_all[m], len(rw_all[m]), (int(H[m]["t_starters_score_final"]),
                                                              int(H[m]["ct_starters_score_final"])), sw, o, f))
    res.append((nfit, name, o, f))
res.sort(key=lambda x: -x[0])
print("top 15 configurations (fits / 116, schedule, orientation, flip):")
for nfit, name, o, f in res[:15]:
    print(f"  {nfit:>4}  {name:<8} orient={o} flip={f}")
spec = {(name, o, f): nfit for nfit, name, o, f in res}
for key in [("(8,)", 0, "none"), ("(8,)", 1, "none"), ("(9,)", 0, "none"), ("(9,)", 1, "none"),
            ("(12,)", 0, "none"), ("(12,)", 1, "none"), ("(8,)", 0, 13), ("(8,)", 1, 13), ("(12,)", 0, 13),
            ("(12,)", 1, 13), ("(8,)", 0, "any"), ("(8,)", 1, "any"), ("(9,)", 1, "any"), ("()", 0, "none"),
            ("()", 1, "none")]:
    print(f"  {key[0]:<8} orient={key[1]} flip={key[2]!s:<5} fits={spec.get(key, 'NA')}")

# per-match best reversed schedule for each detected-signature group
print("\n=== other_misfit: fit by detected signature under key configurations ===")
for sg in ["(8,)", "()"]:
    sub = [m for m in OTH if str(SIG.get(m, ())) == sg]
    row = []
    for key in [("(8,)", 0, "none"), ("(8,)", 1, "none"), ("(9,)", 0, "none"), ("(9,)", 1, "none"),
                ("(8,)", 1, 13), ("(12,)", 1, 13), ("()", 1, "none")]:
        sw = (int(key[0].strip("(),")),) if key[0] != "()" else ()
        k_ = sum(1 for m in sub if fit(rw_all[m], len(rw_all[m]),
                                      (int(H[m]["t_starters_score_final"]), int(H[m]["ct_starters_score_final"])),
                                      sw, key[1], key[2]))
        row.append(f"{key[0]} o{key[1]} f{key[2]}={k_}")
    print(f"  sig {sg} (n={len(sub)}): " + "; ".join(row))

# ------------------------------------------------------------------ selection/denominator check
print("\n=== complete-universe coverage by map (days 12-25) ===")
cov = collections.Counter()
cov_c = collections.Counter()
for m in ALL1225:
    cov[H[m]["map_name"]] += 1
for m in COMPLETE:
    if 12 <= day_of(m) <= 25:
        cov_c[H[m]["map_name"]] += 1
for mp in sorted(cov):
    oth = sum(1 for m in OTH if H[m]["map_name"] == mp)
    print(f"  {mp:<12} header matches {cov[mp]:>5}  complete {cov_c[mp]:>5} ({cov_c[mp] / cov[mp]:.3f})  "
          f"other_misfit {oth:>3} ({oth / cov[mp]:.4f} of all header matches)")

# ------------------------------------------------------------------ permutation tests
def het(g, s, y, G, S):
    ns = np.bincount(s, minlength=S).astype(float)
    ys = np.bincount(s, weights=y, minlength=S)
    ps = np.divide(ys, ns, out=np.zeros(S), where=ns > 0)
    pe = ps[s]
    og = np.bincount(g, weights=y, minlength=G)
    eg = np.bincount(g, weights=pe, minlength=G)
    vg = np.bincount(g, weights=pe * (1 - pe), minlength=G)
    ok = vg > 0
    return float(np.sum((og[ok] - eg[ok]) ** 2 / vg[ok]))


def perm_p(g, s, y, G, S, nrep=B):
    obs = het(g, s, y, G, S)
    cnt = 0
    for _ in range(nrep):
        gp = g.copy()
        for val in np.unique(s):
            idx = np.where(s == val)[0]
            gp[idx] = rng.permutation(g[idx])
        if het(gp, s, y, G, S) >= obs - 1e-9:
            cnt += 1
    return obs, (1 + cnt) / (nrep + 1)


def codes(vals):
    u = sorted(set(vals))
    idx = {v: i for i, v in enumerate(u)}
    return np.array([idx[v] for v in vals], dtype=np.int64), len(u)


print("\n=== permutation tests on the 1246 complete matches (y = other_misfit) ===")
mp_g, M = codes([H[m]["map_name"] for m in COMPLETE])
dy_s, D = codes([day_of(m) for m in COMPLETE])
y = np.array([1.0 if CLS[m] == "other_misfit" else 0.0 for m in COMPLETE])
zeros = np.zeros(len(y), dtype=np.int64)
o1, p1 = perm_p(mp_g, zeros, y, M, 1)
print(f"map unadjusted: stat={o1:.2f} perm p={p1:.4f} (B={B}, floor {1 / (B + 1):.4f})")
o2, p2 = perm_p(mp_g, dy_s, y, M, D)
print(f"map | day:       stat={o2:.2f} perm p={p2:.4f}")
sg_g, SG = codes([str(SIG.get(m, ())) for m in COMPLETE])
o3, p3 = perm_p(sg_g, zeros, y, SG, 1)
print(f"signature unadjusted: stat={o3:.2f} perm p={p3:.4f}")
o4, p4 = perm_p(sg_g, dy_s, y, SG, D)
print(f"signature | day:    stat={o4:.2f} perm p={p4:.4f}")

# ------------------------------------------------------------------ within-four-maps: signature vs map
print("\n=== other_misfit rate by detected signature, complete matches, four maps vs other maps ===")
tab = collections.defaultdict(lambda: [0, 0])
for m in COMPLETE:
    grp = "four" if H[m]["map_name"] in four else "other"
    key = (grp, str(SIG.get(m, ())))
    tab[key][0] += 1
    tab[key][1] += 1 if CLS[m] == "other_misfit" else 0
for key in sorted(tab):
    n_, k_ = tab[key]
    print(f"  {key[0]:<6} sig {key[1]:<10} n={n_:>4} other_misfit={k_:>3} rate={k_ / n_ if n_ else 0:.3f}")
ct4 = np.array([[sum(1 for m in COMPLETE if H[m]["map_name"] in four and CLS[m] == "other_misfit"),
                  sum(1 for m in COMPLETE if H[m]["map_name"] in four and CLS[m] != "other_misfit")],
                 [sum(1 for m in COMPLETE if H[m]["map_name"] not in four and CLS[m] == "other_misfit"),
                  sum(1 for m in COMPLETE if H[m]["map_name"] not in four and CLS[m] != "other_misfit")]])
print(f"four-maps vs other-maps other_misfit 2x2 {ct4.tolist()}: Fisher p={stats.fisher_exact(ct4)[1]:.3g}")
sig8_ct = np.array([[sum(1 for m in COMPLETE if str(SIG.get(m, ())) == "(8,)" and CLS[m] == "other_misfit"),
                     sum(1 for m in COMPLETE if str(SIG.get(m, ())) == "(8,)" and CLS[m] != "other_misfit")],
                    [sum(1 for m in COMPLETE if str(SIG.get(m, ())) != "(8,)" and CLS[m] == "other_misfit"),
                     sum(1 for m in COMPLETE if str(SIG.get(m, ())) != "(8,)" and CLS[m] != "other_misfit")]])
print(f"sig (8,) vs other sigs other_misfit 2x2 {sig8_ct.tolist()}: Fisher p={stats.fisher_exact(sig8_ct)[1]:.3g}")

# ------------------------------------------------------------------ metadata: server region, providence, n, totals
print("\n=== metadata of other_misfit vs fits/r13 classes (complete matches) ===")
for label in ["other_misfit", "r13_flip_misfit", "fits", "affected"]:
    sub = [m for m in COMPLETE if CLS[m] == label]
    if not sub:
        continue
    regs = collections.Counter(region_of(H[m]["server_name"]) for m in sub)
    prov = collections.Counter(H[m]["providence"] for m in sub)
    mx = collections.Counter(max(int(H[m]["t_starters_score_final"]), int(H[m]["ct_starters_score_final"])) for m in sub)
    ns = collections.Counter(len(rw_all[m]) for m in sub)
    print(f"  {label:<16} n={len(sub):>4} providence={dict(prov)}")
    print(f"      server regions top: {regs.most_common(6)}")
    print(f"      max final score: {sorted(mx.items())}")
    print(f"      rounds n: {sorted(ns.items())[:12]} ... (min {min(ns)}, max {max(ns)})")

print("\nDONE")


# ================================================================== extension: event-based halftime label
# round_state event round_announce_last_round_half: label L, halftime after round L+1 (D4-5 convention, ASSUMPTION)
HT = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "event_type"])
                .filter(pl.col("event_type") == "round_announce_last_round_half")
                .select(["match_key", "round"]) for f in ch_files("round_state")], how="vertical_relaxed")
HTL = collections.defaultdict(set)
for r in HT.iter_rows(named=True):
    HTL[r["match_key"]].add(int(r["round"]))
print("\n=== event-based halftime label (round_announce_last_round_half) ===")
print(f"matches with event: {len(HTL)}; label sets: {collections.Counter(tuple(sorted(v)) for v in HTL.values()).most_common(6)}")


def ev_h(m):
    labs = HTL.get(m)
    if not labs:
        return None
    return min(labs) + 1


print("complete matches: event halftime_after x class (counts):")
cc2 = collections.Counter((str(ev_h(m)), CLS[m]) for m in COMPLETE)
for (h_, c_), v_ in sorted(cc2.items()):
    print(f"  halftime_after={h_:<6} class={c_:<16} n={v_}")

print("\ncomplete other_misfit by event halftime:", sorted(collections.Counter(str(ev_h(m)) for m in OTH).items()))
print("complete MR8-event (halftime_after=8) by class:",
      sorted(collections.Counter(CLS[m] for m in COMPLETE if ev_h(m) == 8).items()))
print("complete MR8-event by map:", sorted(collections.Counter(H[m]["map_name"] for m in COMPLETE if ev_h(m) == 8).items()))

# ------------------------------------------------------------------ halftime-event coverage over ALL header matches
ALLH = [m for m in H if 12 <= day_of(m) <= 25]
print("\nevent halftime_after by map over ALL header matches days 12-25 (no round_end filter):")
tab_all = collections.defaultdict(collections.Counter)
for m in ALLH:
    tab_all[H[m]["map_name"]][str(ev_h(m))] += 1
for mp in sorted(tab_all):
    print(f"  {mp:<12} {dict(sorted(tab_all[mp].items()))}")

# header-only final-score format (no round_end used): max final score 9 (first to 9) vs 13 (first to 13)
print("\nheader-only max final score x map (days 12-25, all header matches):")
tab_fmt = collections.defaultdict(collections.Counter)
for m in ALLH:
    mx = max(int(H[m]["t_starters_score_final"]), int(H[m]["ct_starters_score_final"]))
    tab_fmt[H[m]["map_name"]][mx] += 1
for mp in sorted(tab_fmt):
    print(f"  {mp:<12} max=9: {tab_fmt[mp].get(9, 0):>4}   max=13: {tab_fmt[mp].get(13, 0):>4}   "
          f"other: {sum(v for k_, v in tab_fmt[mp].items() if k_ not in (9, 13)):>4}")

# ------------------------------------------------------------------ configs using the event-based schedule
print("\n=== other_misfit (116): event-based MR8 schedule, orientation and flip variants ===")
OTH8 = [m for m in OTH if ev_h(m) == 8]
OTH12 = [m for m in OTH if ev_h(m) == 12]
OTHN = [m for m in OTH if ev_h(m) is None]
print(f"other_misfit by event: halftime_after=8: {len(OTH8)}; 12: {len(OTH12)}; none: {len(OTHN)}")


def tot_of(m):
    return (int(H[m]["t_starters_score_final"]), int(H[m]["ct_starters_score_final"]))


for label, sub in [("event MR8 (8)", OTH8), ("all 116", OTH)]:
    print(f"  {label} n={len(sub)}")
    for o in (0, 1):
        for f in ("none", 9, 13, "any"):
            if f == 9:
                # flip at round 9 specifically
                cnt9 = 0
                for m in sub:
                    rw = rw_all[m]
                    n = len(rw)
                    tA = tot_of(m)[0] if o == 0 else tot_of(m)[1]
                    wa = wa_of(rw, n, (8,))
                    wa2 = wa - 1 if rw[9] == side_a(9, (8,)) else wa + 1
                    if wa2 == tA:
                        cnt9 += 1
                print(f"    sched (8,) orient={o} flip@9 (first post-swap round) fits={cnt9}")
            else:
                k_ = sum(1 for m in sub if fit(rw_all[m], len(rw_all[m]), tot_of(m), (8,), o, f))
                print(f"    sched (8,) orient={o} flip={f!s:<5} fits={k_}")

# exact no-flip fits across all schedules for the event MR8 subset, orientation 1 vs 0
print("\n  exact no-flip fit counts for event-MR8 other_misfit by schedule k (orient 0 / orient 1):")
rowk = []
for k in range(0, 17):
    a0 = sum(1 for m in OTH8 if fit(rw_all[m], len(rw_all[m]), tot_of(m), (k,), 0, "none"))
    a1 = sum(1 for m in OTH8 if fit(rw_all[m], len(rw_all[m]), tot_of(m), (k,), 1, "none"))
    rowk.append(f"k={k}:{a0}/{a1}")
print("   " + "  ".join(rowk))

# ------------------------------------------------------------------ build / game version
print("\n=== build_num and demo version of complete matches by class ===")
bld = {}
HB = pl.concat([pl.read_parquet(f, columns=["key", "build_num"]) for f in fo(BASE / "header")], how="vertical_relaxed")
for r in HB.iter_rows(named=True):
    bld[r["key"]] = r["build_num"]
for label in ["affected", "fits", "r13_flip_misfit", "other_misfit"]:
    sub = [m for m in COMPLETE if CLS[m] == label]
    print(f"  {label:<16} n={len(sub):>4} builds={sorted(collections.Counter(bld.get(m) for m in sub).items(), key=lambda kv: -kv[1])[:6]}")
print("  ALL header days 12-25 builds:", sorted(collections.Counter(bld.get(m) for m in ALLH).items(), key=lambda kv: -kv[1])[:8])
print("  other_misfit by build x map:",
      sorted(collections.Counter((bld.get(m), H[m]["map_name"]) for m in OTH).items()))

print("\nDONE (extension)")
