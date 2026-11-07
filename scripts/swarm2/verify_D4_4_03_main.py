r"""Verifier D4-4 main recomputation for the 2024 tome (independent of the claimed scripts).

Sources (per match, per round):
  L(r)  round_end winner_team_code (2 = T, 3 = CT), first row by tick per (match, round)
  B(r)  bomb outcome side from bomb_state: bomb_exploded -> 2, bomb_defused -> 3 (conflicts -> None)
  FE(r) round_state round_freeze_end t_score / ct_score, first row by tick per (match, round)
  H     header t_starters_score_final (tA) and ct_starters_score_final (tB)

Score-step winner S(r), no label used: boundary r -> r+1 between freeze-end fields is either
  noswap: (dt, dc) in {(1,0),(0,1)}   winner = T if dt==1 else CT
  swap:   (t1-c0, c1-t0) in {(1,0),(0,1)}   winner = CT if (t1-c0)==1 else T
  (ambiguous if both fit; bad if neither fits)

Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D4_4_03_main.py
"""
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
T, CT = 2, 3


def files(ch):
    fd = BASE / "header" if ch == "header" else sorted(BASE.glob(f"basic_{ch}.*"))[0]
    return sorted(glob.glob(str(fd / "dataframe_*")))


def day_of(key):
    p = key.split("/")
    return f"2024-{p[2]}-{p[3]}"


def classify(a, b):
    t0, c0 = a
    t1, c1 = b
    ns = (t1 - t0, c1 - c0)
    sw = (t1 - c0, c1 - t0)
    ns_ok = ns in ((1, 0), (0, 1))
    sw_ok = sw in ((1, 0), (0, 1))
    if ns_ok and sw_ok:
        return "ambig", None
    if ns_ok:
        return "noswap", (T if ns == (1, 0) else CT)
    if sw_ok:
        return "swap", (CT if sw == (1, 0) else T)
    return "bad", None


# ---- load (projected columns only, one channel at a time)
H = pl.scan_parquet(files("header")).select(
    ["key", "t_starters_score_final", "ct_starters_score_final"]).collect()
dup_h = H.group_by("key").agg(pl.len().alias("k")).filter(pl.col("k") > 1).height
HD = {}
for r in H.iter_rows(named=True):
    HD.setdefault(r["key"], r)
print(f"header rows={H.height} unique keys={len(HD)} duplicate keys={dup_h}")

RE = pl.scan_parquet(files("round_end")).select(["match_key", "round", "tick", "winner_team_code"]).collect()
dup_re = RE.group_by(["match_key", "round"]).agg(
    pl.len().alias("k"), pl.col("winner_team_code").n_unique().alias("nw")).filter(pl.col("k") > 1)
print(f"round_end rows={RE.height} (match,round) pairs with >1 row={dup_re.height} "
      f"of which conflicting winners={dup_re.filter(pl.col('nw') > 1).height}")
RE1 = RE.sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first", maintain_order=True)

BS = (pl.scan_parquet(files("bomb_state")).select(["match_key", "round", "tick", "event_type"])
      .filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"])).collect())
BO = BS.group_by(["match_key", "round"]).agg(
    (pl.col("event_type") == "bomb_exploded").any().alias("exp"),
    (pl.col("event_type") == "bomb_defused").any().alias("dfs"),
    pl.len().alias("nev"),
)
print(f"bomb outcome rows={BS.height}; (match,round) with outcome={BO.height}; "
      f"with both exploded and defused={BO.filter(pl.col('exp') & pl.col('dfs')).height}")

FE_raw = (pl.scan_parquet(files("round_state")).select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
          .filter(pl.col("event_type") == "round_freeze_end").collect())
dup_fe = FE_raw.group_by(["match_key", "round"]).agg(pl.len().alias("k")).filter(pl.col("k") > 1).height
FE = FE_raw.sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first", maintain_order=True)
print(f"freeze_end rows={FE_raw.height} (match,round) pairs with >1 row={dup_fe}; unique={FE.height}")

# ---- assemble per match / per round
rows = defaultdict(dict)
for r in RE1.iter_rows(named=True):
    rows[r["match_key"]].setdefault(r["round"], {})["L"] = r["winner_team_code"]
for r in BO.iter_rows(named=True):
    d = rows[r["match_key"]].setdefault(r["round"], {})
    d["B"] = T if (r["exp"] and not r["dfs"]) else (CT if (r["dfs"] and not r["exp"]) else None)
    d["hasB"] = True
for r in FE.iter_rows(named=True):
    rows[r["match_key"]].setdefault(r["round"], {})["fe"] = (r["t_score"], r["ct_score"])

keys_round_end = set(RE1["match_key"].unique().to_list())
print(f"matches with round_end={len(keys_round_end)}; with header={len(keys_round_end & set(HD))}; "
      f"round_end matches missing header={len(keys_round_end - set(HD))}; header keys with no round_end={len(set(HD) - keys_round_end)}")

# ---- per-match table
M = {}
for m in keys_round_end:
    if m not in HD:
        continue
    rr = rows[m]
    Lr = sorted(r for r in rr if "L" in rr[r])
    tA = int(HD[m]["t_starters_score_final"])
    tB = int(HD[m]["ct_starters_score_final"])
    n = len(Lr)
    contiguous = Lr == list(range(1, n + 1))
    complete = contiguous and n == tA + tB
    bnd = {}
    for r in range(1, n):
        a = rr.get(r, {}).get("fe")
        b = rr.get(r + 1, {}).get("fe")
        if a is None or b is None:
            bnd[r] = ("missing", None)
        else:
            bnd[r] = classify(a, b)
    S = {r: bnd[r][1] for r in bnd if bnd[r][0] in ("noswap", "swap")}
    swaps = [r for r in bnd if bnd[r][0] == "swap"]
    M[m] = dict(n=n, tA=tA, tB=tB, complete=complete, contiguous=contiguous, rr=rr, bnd=bnd, S=S, swaps=swaps,
                day=day_of(m))


# ---- affected definition (claim): round-13 bomb side != round_end winner (any match, any n)
def b13_mismatch(m):
    d = M[m]["rr"].get(13, {})
    return d.get("B") is not None and "L" in d and d["B"] != d["L"]


r13_bomb_rows = [(m, M[m]["rr"][13]) for m in M if 13 in M[m]["rr"] and "L" in M[m]["rr"][13] and M[m]["rr"][13].get("hasB")]
mm_all = [m for m in M if b13_mismatch(m)]
print(f"\nround-13 rows with bomb outcome and round_end label (all matches): {len(r13_bomb_rows)}")
print(f"  of which bomb side != round_end (mismatch rows): {sum(1 for m, d in r13_bomb_rows if d['B'] is not None and d['B'] != d['L'])}")
print(f"  matches with >=1 round-13 mismatch (all n): {len(mm_all)}")
mm_by_day = Counter(); tot_by_day = Counter()
for m, d in r13_bomb_rows:
    tot_by_day[M[m]["day"]] += 1
    if d["B"] is not None and d["B"] != d["L"]:
        mm_by_day[M[m]["day"]] += 1
print("  round-13 bomb rows and mismatches by day (day, rows, mismatches):")
for dday in sorted(tot_by_day):
    print(f"    {dday} rows={tot_by_day[dday]:>4} mismatch={mm_by_day[dday]:>4}")

CR = [m for m in M if M[m]["complete"] and M[m]["n"] <= 24]
CR_SET = set(CR)
ALLC = [m for m in M if M[m]["complete"]]
OT = [m for m in ALLC if M[m]["n"] > 24]
print(f"\ncomplete matches (round count == header total, rounds 1..n): {len(ALLC)}; n<=24: {len(CR)}; n>24: {len(OT)}")
aff = [m for m in CR if b13_mismatch(m)]
una = [m for m in CR if not b13_mismatch(m)]
AFF_SET = set(aff)
print(f"complete regulation (n<=24): affected={len(aff)} unaffected={len(una)}")
print(f"round-13 bomb row present in unaffected complete regulation matches: "
      f"{sum(1 for m in una if M[m]['rr'].get(13, {}).get('hasB'))}")


# ---- label-only tests (claim's halftime S12) and bomb-at-13 test
def fit(wa, n, tA, tB):
    if (wa, n - wa) == (tA, tB):
        return "ok"
    if (wa, n - wa) == (tB, tA):
        return "swapped"
    return "bad"


def wins_A(m, sched, use_bomb13=False):
    """Team A starts on T; sched = set of rounds after which the sides swap."""
    rr = M[m]["rr"]; n = M[m]["n"]
    wa = 0; swaps = 0
    for r in range(1, n + 1):
        d = rr[r]
        w = d["L"]
        if use_bomb13 and r == 13 and d.get("B") is not None:
            w = d["B"]
        side_A = T if swaps % 2 == 0 else CT
        wa += (w == side_A)
        if r in sched:
            swaps += 1
    return wa


print("\n== header fit, complete regulation (n<=24), affected vs unaffected")
for label, group in [("affected", aff), ("unaffected", una)]:
    for name, sched, bomb in [("S12 end", {12}, False), ("S12 bomb13", {12}, True), ("S13 end", {13}, False)]:
        c = Counter()
        for m in group:
            wa = wins_A(m, sched, bomb)
            c[fit(wa, M[m]["n"], M[m]["tA"], M[m]["tB"])] += 1
        tot = sum(c.values())
        print(f"  {label:<10} {name:<11} n={tot:>5} ok={c['ok']:>5} swapped={c['swapped']:>4} bad={c['bad']:>5}")

eq_mismatch = sum(1 for m in aff if wins_A(m, {12}, True) != wins_A(m, {13}, False))
print(f"  equivalence check (affected): S12-bomb13 wins_A != S13-end wins_A in {eq_mismatch} of {len(aff)} matches")

# ---- single-flip analysis on unaffected misfits (re-implemented)
mis = []
for m in una:
    wa = wins_A(m, {12}, False)
    if fit(wa, M[m]["n"], M[m]["tA"], M[m]["tB"]) != "ok":
        mis.append(m)
fixable = []; fix13 = 0; fix_rounds = Counter(); dsz = Counter()
for m in mis:
    rr = M[m]["rr"]; n = M[m]["n"]; tA = M[m]["tA"]; tB = M[m]["tB"]
    wa = wins_A(m, {12}, False)
    dsz[wa - tA] += 1
    fr = []
    for r in range(1, n + 1):
        side_A = T if r <= 12 else CT
        isA = rr[r]["L"] == side_A
        new_wa = wa - (1 if isA else 0) + (0 if isA else 1)
        if (new_wa, n - new_wa) == (tA, tB):
            fr.append(r)
    if fr:
        fixable.append(m)
        if 13 in fr:
            fix13 += 1
        for r in fr:
            fix_rounds[r] += 1
print(f"\n== unaffected S12 misfits (label-only): {len(mis)} of {len(una)}")
print(f"  misfit size (wa - tA): {dict(sorted(dsz.items()))}")
print(f"  fixable by one flip: {len(fixable)}; of which round 13 is one of the fixing rounds: {fix13}")
print(f"  fixing-round counts: {dict(sorted(fix_rounds.items()))}")

# ---- score-step boundary classes (no labels used)
print("\n== freeze-end boundary classes for complete regulation matches (n<=24)")
bcls = Counter()
for m in CR:
    for r in range(1, M[m]["n"]):
        bcls[(r, M[m]["bnd"][r][0])] += 1
for r in [8, 9, 11, 12, 13, 14, 15]:
    print(f"  boundary {r}->{r+1}: " + ", ".join(f"{k}={bcls[(r, k)]}" for k in ["noswap", "swap", "ambig", "bad", "missing"]))

swap_pos = Counter()
for m in CR:
    swap_pos[tuple(M[m]["swaps"])] += 1
print("  swap boundaries per complete regulation match (matches):")
for k, v in sorted(swap_pos.items(), key=lambda kv: -kv[1])[:12]:
    print(f"    swaps_after={list(k)}: {v}")

# ---- round-13 score step vs label vs bomb
print("\n== round-13 winner: score step (boundary 13->14) vs round_end label vs bomb")
s13_tbl = Counter()
for m in M:
    if 13 not in M[m]["S"] or "L" not in M[m]["rr"].get(13, {}):
        continue
    s = M[m]["S"][13]
    L = M[m]["rr"][13]["L"]
    B = M[m]["rr"][13].get("B")
    if m in AFF_SET:
        grp = "CR-affected"
    elif m in CR_SET:
        grp = "CR-unaffected"
    else:
        grp = "other(n>24/incomplete)"
    s13_tbl[(grp, "S=L" if s == L else "S!=L", "no-bomb" if not M[m]["rr"][13].get("hasB") else ("B=S" if B == s else "B!=S"))] += 1
for k in sorted(s13_tbl):
    print(f"  {k[0]:<22} {k[1]:<6} {k[2]:<8} n={s13_tbl[k]}")

# ---- score-step vs label, all rounds (scope)
dis_by_r = Counter(); tot_by_r = Counter()
for m in CR:
    for r, s in M[m]["S"].items():
        L = M[m]["rr"].get(r, {}).get("L")
        if L is None:
            continue
        tot_by_r[r] += 1
        if s != L:
            dis_by_r[r] += 1
print("\n== score-step winner != round_end label, by round (complete regulation, n<=24)")
print("  " + ", ".join(f"r{r}:{dis_by_r[r]}/{tot_by_r[r]}" for r in sorted(tot_by_r)))


def disagree_rounds(m):
    out = []
    for r, s in M[m]["S"].items():
        L = M[m]["rr"].get(r, {}).get("L")
        if L is not None and s != L:
            out.append(r)
    return out


mat_dis = [m for m in CR if disagree_rounds(m)]
only13 = [m for m in mat_dis if disagree_rounds(m) == [13]]
print(f"  complete regulation matches with any score/label disagreement: {len(mat_dis)}")
print(f"  of which the only disagreeing round is 13: {len(only13)}")

# ---- claim groups x score-step round-13 status
xt = Counter()
for m in CR:
    if 13 not in M[m]["S"] or "L" not in M[m]["rr"].get(13, {}):
        st = "no-score-r13"
    else:
        st = "score!=label" if M[m]["S"][13] != M[m]["rr"][13]["L"] else "score=label"
    g = "affected" if m in AFF_SET else "unaffected"
    b = "bomb13" if M[m]["rr"].get(13, {}).get("hasB") else "nobomb13"
    xt[(g, b, st)] += 1
print("\n== claim groups x score-step round-13 status (complete regulation, n<=24)")
for k in sorted(xt):
    print(f"  {k[0]:<10} {k[1]:<8} {k[2]:<14} n={xt[k]}")

# ---- save compact table for follow-up scripts
out = {m: {"n": M[m]["n"], "tA": M[m]["tA"], "tB": M[m]["tB"], "day": M[m]["day"], "swaps": M[m]["swaps"],
           "L13": M[m]["rr"].get(13, {}).get("L"), "B13": M[m]["rr"].get(13, {}).get("B"),
           "hasB13": bool(M[m]["rr"].get(13, {}).get("hasB")),
           "S13": M[m]["S"].get(13), "CR": m in CR_SET, "aff": m in AFF_SET} for m in CR}
(SCRATCH / "verify_D4_4_cr_table.json").write_text(json.dumps(out), encoding="utf-8")
print(f"\nwrote scratchpad json with {len(out)} complete regulation matches")
