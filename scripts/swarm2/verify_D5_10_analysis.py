"""verify D5_10, step 2: recompute the claimed score-lead numbers and test the mechanism.

Input: scratchpad verify_D5_10 rounds_YYYY.parquet and matches_YYYY.parquet from verify_D5_10_build.py.
Read-only on G:. Prints to stdout; the caller redirects stdout to logs\\swarm2\\verify_D5_10.log.

Sections
  0. Build check against the claim's counts (regime split, 8->9 side swap).
  1. Reproduce the claim: team-A diff (freeze-end team-A score minus cumulative team-A round_end wins
     before the round), matches without an 8->9 swap. Claimed: regime A 2,223 matches, 2,103 differ,
     41,818 states, 15,614 differ; regime B 1,789 matches, 24 differ, 33,903 states, 73 differ.
  2. Boundary classes from freeze-end scores and round_end winners only (no team-A mapping needed).
     For boundary r -> r+1 with winner w(r): N = slot fields, no side swap, winner as recorded;
     S = slot fields after a side swap; Ninv / Sinv = N / S with the winner flipped.
     Ninv at boundary 13 is what the claimed round-13 inversion predicts (no side swap after 13 in MR24).
  3. Header totals (the 2024 header gives final team scores). Team A = team that started on T.
     Complete match = round labels 1..n all have a winner and n == header t + header ct.
  4. What-if: flip only the round-13 winner in regime A. Does the team-A diff and the header test clear?
  5. Per-match cross-tab: round-13 class versus any team-A diff (regime A, no 8->9 swap).
  6. Matches whose first diff is before label 14 (not explained by the round-13 inversion).
  7. Missing / extra round_end rows.
  8. Bomb truth (explode -> T won, defuse -> CT won) by regime, round and day.
  9. 2026 control.
ASSUMPTIONS are marked in the printed text.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_10_analysis.py
"""
import collections
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_10")
T, CT = 2, 3  # winner_team_code: 2 = T side, 3 = CT side (round_end)
MAXR = 60


def swap_expected(r, mr):
    """Same rule as bakeoff.swap_expected, re-typed here (not imported)."""
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


def sched(mr):
    return [r for r in range(1, MAXR) if swap_expected(r, mr)]


SCHEDS = {
    "D5_12_24_27": [12, 24, 27],   # mapping the claim hard-codes (D5_10 / D5_17)
    "BO_MR24": sched(24),          # bakeoff rule with MR24 -> [12, 27, 33, ...]
    "BO_MR16": sched(16),          # bakeoff rule with MR16 -> [8, 19, 25, ...]
}


def a_on_t(r, swaps):
    """Team A starts on T; True if team A is on the T side in round r."""
    return sum(1 for s in swaps if r > s) % 2 == 0


PLACEHOLDER = {"winner": None, "fe_t": None, "fe_ct": None, "n_planted": 0, "n_exploded": 0,
               "n_defused": 0, "planter_side": None}


def load(year):
    R = pl.read_parquet(SCR / f"rounds_{year}.parquet").sort(["match_key", "round"]).to_dicts()
    M = pl.read_parquet(SCR / f"matches_{year}.parquet").to_dicts()
    rounds = collections.defaultdict(dict)
    for x in R:
        rounds[x["match_key"]][int(x["round"])] = x
    mats = {x["match_key"]: x for x in M}
    # 2024 regime B has no round_end and no freeze-end row for round 1. The claim's build keeps a round-1
    # row from round_start; this build does not, so a placeholder (no data) is added here.
    for m in mats:
        if m in rounds and 1 not in rounds[m]:
            rounds[m][1] = {"match_key": m, "round": 1, **PLACEHOLDER}
    return rounds, mats


def regime_of(mat):
    return "A" if (mat["n_round_announce_match_start"] or 0) > 0 else "B"


def swap8_flag(rr):
    """D5_17 rule: side-field swap at 8->9 using the round_end winner of round 8."""
    x8, x9 = rr.get(8), rr.get(9)
    if not x8 or not x9:
        return False
    w8, t8, c8, t9, c9 = x8["winner"], x8["fe_t"], x8["fe_ct"], x9["fe_t"], x9["fe_ct"]
    if None in (w8, t8, c8, t9, c9):
        return False
    a = 1 if w8 == T else 0
    b = 1 if w8 == CT else 0
    ns = (t9 == t8 + a) and (c9 == c8 + b)
    sw = (t9 == c8 + b) and (c9 == t8 + a)
    return sw and not ns


def flipped(w, r, flip_rounds):
    """Winner with T/CT swapped for rounds in flip_rounds (a what-if test, not a fix)."""
    if w is None or r not in flip_rounds:
        return w
    return CT if w == T else T


def states_for(rounds, mats, swaps, m, r1_mode, flip_rounds=frozenset()):
    """D5_10 / D5_17 state logic. Returns [(round, diff)] for rounds >= 2 with a freeze-end team-A score.
    diff = freeze-end team-A score - cumulative team-A round wins before the round."""
    rr = rounds[m]
    reg = regime_of(mats[m])
    fe2 = rr.get(2)
    out = []
    cum = 0
    for r in sorted(rr):
        x = rr[r]
        aT = a_on_t(r, swaps)
        a_fe = None
        if x["fe_t"] is not None and x["fe_ct"] is not None:
            a_fe = x["fe_t"] if aT else x["fe_ct"]
        w = flipped(x["winner"], r, flip_rounds)
        if r == 1 and r1_mode == "fe2":
            a_won = (fe2["fe_t"] if aT else fe2["fe_ct"]) if (fe2 is not None and fe2["fe_t"] is not None) else 0
        else:
            a_won = 0 if w is None else (1 if ((aT and w == T) or ((not aT) and w == CT)) else 0)
        cum_before = cum
        cum += a_won
        if r >= 2 and a_fe is not None and (w is not None or reg == "B"):
            out.append((r, a_fe - cum_before))
    return out


def diff_summary(rounds, mats, swaps, selector, r1_mode, label, flip_rounds=frozenset(), flip_sel=None):
    n_m = n_st = n_nz = m_nz = 0
    by_round = collections.Counter()
    for m in rounds:
        if not selector(m):
            continue
        fr = flip_rounds if (flip_sel is None or flip_sel(m)) else frozenset()
        st = states_for(rounds, mats, swaps, m, r1_mode, fr)
        if not st:
            continue
        n_m += 1
        n_st += len(st)
        nz = [(r, d) for r, d in st if d != 0]
        n_nz += len(nz)
        m_nz += 1 if nz else 0
        for r, _ in nz:
            by_round[r] += 1
    print(f"  {label}: matches {n_m}; states {n_st}; diff!=0 states {n_nz}; matches with any diff!=0 {m_nz}")
    print("    diff!=0 states by round label (first 25):", sorted(by_round.items())[:25])
    return n_m, n_st, n_nz, m_nz


def hits_at(rr, r, flip_rounds=frozenset()):
    """Boundary r -> r+1 classes. Returns a tuple of matching classes, or 'missing' / 'no_pair' / 'other'."""
    x, y = rr.get(r), rr.get(r + 1)
    if not x or not y:
        return "no_pair"
    w = flipped(x["winner"], r, flip_rounds)
    t1, c1, t2, c2 = x["fe_t"], x["fe_ct"], y["fe_t"], y["fe_ct"]
    if w is None or None in (t1, c1, t2, c2):
        return "missing"
    aT = 1 if w == T else 0
    aC = 1 - aT
    ok = {
        "N": (t2 == t1 + aT) and (c2 == c1 + aC),
        "S": (t2 == c1 + aC) and (c2 == t1 + aT),
        "Ninv": (t2 == t1 + aC) and (c2 == c1 + aT),
        "Sinv": (t2 == c1 + aT) and (c2 == t1 + aC),
    }
    hits = tuple(k for k, v in ok.items() if v)
    return hits if hits else "other"


def boundary_classes(rounds, mats, selector, label, max_r=30):
    counts = collections.defaultdict(collections.Counter)
    for m, rr in rounds.items():
        if not selector(m):
            continue
        for r in range(1, max_r + 1):
            h = hits_at(rr, r)
            if isinstance(h, str):
                counts[r][h] += 1
            elif len(h) == 1:
                counts[r][h[0]] += 1
            else:
                counts[r]["amb:" + "/".join(h)] += 1
    print(f"  boundary classes, {label} (r -> r+1; N/S = correct winner; Ninv/Sinv = winner flipped):")
    for r in range(1, max_r + 1):
        c = counts.get(r)
        if not c:
            continue
        tot = sum(v for k, v in c.items() if k not in ("no_pair",))
        print(f"    r={r:>2}: " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())) + f"   (rows {tot})")
    return counts


def header_test(rounds, mats, swaps_for, selector, label, r1_fe2=False, flip_rounds=frozenset(), flip_sel=None):
    res = collections.Counter()
    dist = collections.Counter()
    bad_ex = []
    for m, rr in rounds.items():
        if not selector(m):
            continue
        mat = mats[m]
        t_h, c_h = mat["t_starters_score_final"], mat["ct_starters_score_final"]
        if t_h is None or c_h is None:
            res["no_header_score"] += 1
            continue
        rs = sorted(rr)
        n = len(rs)
        if rs != list(range(1, n + 1)):
            res["labels_not_contiguous"] += 1
            continue
        fr = flip_rounds if (flip_sel is None or flip_sel(m)) else frozenset()
        w = {}
        for r in rs:
            wr = rr[r]["winner"]
            if r == 1 and wr is None and r1_fe2 and rr.get(2) is not None and rr[2]["fe_t"] is not None:
                t2, c2 = rr[2]["fe_t"], rr[2]["fe_ct"]
                wr = T if (t2 == 1 and c2 == 0) else (CT if (c2 == 1 and t2 == 0) else None)
            w[r] = flipped(wr, r, fr)
        if any(w[r] is None for r in rs):
            res["has_round_without_winner"] += 1
            continue
        if n != t_h + c_h:
            res["not_complete_n_ne_header_sum"] += 1
            dist[n - (t_h + c_h)] += 1
            continue
        swaps = swaps_for(mat)
        A = sum(1 for r in rs if (a_on_t(r, swaps) and w[r] == T) or ((not a_on_t(r, swaps)) and w[r] == CT))
        B = n - A
        if (A, B) == (t_h, c_h):
            res["eq"] += 1
        elif (A, B) == (c_h, t_h):
            res["swapped"] += 1
        else:
            res["neither"] += 1
            if len(bad_ex) < 3:
                bad_ex.append((m[-12:], n, t_h, c_h, A, B))
    tot = res["eq"] + res["swapped"] + res["neither"]
    print(f"  header test {label}: complete matches {tot}; {dict(res)}")
    if dist:
        print(f"    not-complete: n minus header sum distribution (top 8): {dist.most_common(8)}")
    if bad_ex:
        print("    examples (key tail, n, header t, header ct, A wins, B wins):", bad_ex)
    return res


def day_of(mat):
    return mat["day"]


def main():
    R24, M24 = load("2024")
    R26, M26 = load("2026")

    print("== 0. Build check (2024) ==")
    print(f"matches {len(M24)}; (match, round) rows {sum(len(v) for v in R24.values())}")
    reg = collections.Counter(regime_of(m) for m in M24.values())
    print(f"regime by round_announce_match_start>0 (A): {dict(reg)}   claim: A 2,478 (=434+1791+253), B 1,957")
    s8 = {m: swap8_flag(R24.get(m, {})) for m in M24}
    print(f"8->9 side swap (D5_17 rule): {sum(s8.values())}  (A {sum(1 for m in M24 if s8[m] and regime_of(M24[m])=='A')}, "
          f"B {sum(1 for m in M24 if s8[m] and regime_of(M24[m])=='B')}); claim 421 (253 A, 168 B)")

    regA_ns = lambda m: regime_of(M24[m]) == "A" and not s8[m]
    regB_ns = lambda m: regime_of(M24[m]) == "B" and not s8[m]
    regA_s8 = lambda m: regime_of(M24[m]) == "A" and s8[m]

    print("\n== 1. Reproduce the claim (team-A diff, 2024, no 8->9 swap) ==")
    print("D5_10 variant: round-1 from round_end winner (regime A) / fe(2) (regime B); map swaps 12,24,27")
    diff_summary(R24, M24, SCHEDS["D5_12_24_27"], regA_ns, "winner", "regime A no-swap8")
    diff_summary(R24, M24, SCHEDS["D5_12_24_27"], regB_ns, "fe2", "regime B no-swap8")
    print("D5_17 variant: round-1 from fe(2) for both regimes")
    diff_summary(R24, M24, SCHEDS["D5_12_24_27"], regA_ns, "fe2", "regime A no-swap8")
    diff_summary(R24, M24, SCHEDS["D5_12_24_27"], regB_ns, "fe2", "regime B no-swap8")
    print("Bakeoff rule (swaps 12, 27): same states for rounds 1-24; differs only for rounds 25-30")
    diff_summary(R24, M24, SCHEDS["BO_MR24"], regA_ns, "winner", "regime A no-swap8 (BO map)")
    diff_summary(R24, M24, SCHEDS["BO_MR24"], regB_ns, "fe2", "regime B no-swap8 (BO map)")

    print("\n== 2. Boundary classes (freeze-end + round_end winners only) ==")
    boundary_classes(R24, M24, regA_ns, "2024 regime A, no 8->9 swap", max_r=30)
    boundary_classes(R24, M24, regB_ns, "2024 regime B, no 8->9 swap", max_r=30)
    boundary_classes(R24, M24, regA_s8, "2024 regime A, WITH 8->9 swap", max_r=16)

    print("\n== 3. Header totals (team A = starters on T) ==")
    header_test(R24, M24, lambda mat: SCHEDS["D5_12_24_27"], regA_ns, "2024 A no-swap8, D5 map")
    header_test(R24, M24, lambda mat: SCHEDS["BO_MR24"], regA_ns, "2024 A no-swap8, BO_MR24 map")
    header_test(R24, M24, lambda mat: SCHEDS["BO_MR24"], regB_ns, "2024 B no-swap8, BO_MR24 map, round 1 from fe(2)",
                r1_fe2=True)
    header_test(R24, M24, lambda mat: SCHEDS["BO_MR16"], regA_s8, "2024 A WITH swap8, BO_MR16 map")
    header_test(R24, M24, lambda mat: SCHEDS["BO_MR24"], regA_s8, "2024 A WITH swap8, BO_MR24 map")

    print("\n== 4. What-if: flip round-13 winner in regime A (no 8->9 swap) ==")
    diff_summary(R24, M24, SCHEDS["D5_12_24_27"], regA_ns, "winner", "regime A no-swap8, round 13 flipped",
                 flip_rounds=frozenset({13}))
    header_test(R24, M24, lambda mat: SCHEDS["BO_MR24"], regA_ns, "2024 A no-swap8, BO_MR24, round 13 flipped",
                flip_rounds=frozenset({13}))
    print("  control: same flip applied to regime B (no 8->9 swap); expected to break regime B")
    diff_summary(R24, M24, SCHEDS["D5_12_24_27"], regB_ns, "fe2", "regime B no-swap8, round 13 flipped",
                 flip_rounds=frozenset({13}))
    header_test(R24, M24, lambda mat: SCHEDS["BO_MR24"], regB_ns, "2024 B no-swap8, round 13 flipped, r1 fe2",
                r1_fe2=True, flip_rounds=frozenset({13}))

    print("\n== 5. Per-match cross-tab: round-13 boundary class vs any team-A diff (regime A, no 8->9 swap) ==")
    xt = collections.Counter()
    for m in M24:
        if not regA_ns(m):
            continue
        st = states_for(R24, M24, SCHEDS["D5_12_24_27"], m, "winner")
        any_diff = any(d != 0 for _, d in st)
        h = hits_at(R24[m], 13)
        if isinstance(h, str):
            cls = h
        elif h == ("Ninv",):
            cls = "Ninv only"
        elif "Ninv" in h:
            cls = "Ninv + ambiguous"
        elif h == ("N",):
            cls = "N only"
        elif "N" in h:
            cls = "N + ambiguous"
        else:
            cls = "other: " + "/".join(h)
        xt[(cls, "diff" if any_diff else "no diff")] += 1
    for k, v in sorted(xt.items()):
        print(f"    {k[0]:<22} {k[1]:<8} {v}")
    print(f"    totals: diff {sum(v for k, v in xt.items() if k[1]=='diff')}, no diff {sum(v for k, v in xt.items() if k[1]=='no diff')}")

    print("\n== 6. Matches with a team-A diff before label 14 (regime A, no 8->9 swap) ==")
    early = []
    for m in M24:
        if not regA_ns(m):
            continue
        st = states_for(R24, M24, SCHEDS["D5_12_24_27"], m, "winner")
        nz = [r for r, d in st if d != 0]
        if nz and min(nz) < 14:
            early.append((m, min(nz), len(nz)))
    print(f"  matches with first diff at label < 14: {len(early)}")
    first_hist = collections.Counter(f for _, f, _ in early)
    print("  first diff label histogram:", sorted(first_hist.items()))
    ecl = collections.Counter()
    for m, f, _ in early:
        h9 = hits_at(R24[m], 9)
        h8 = hits_at(R24[m], 8)
        ecl[("b8", str(h8)), ("b9", str(h9))] += 1
    print("  boundary 8 and 9 classes for these matches (top 6):", ecl.most_common(6))
    day_e = collections.Counter(M24[m]["day"] for m, _, _ in early)
    print("  days of these matches:", sorted(day_e.items()))
    nmax = collections.Counter()
    for m, _, _ in early:
        rr = R24[m]
        nmax[(len(rr), M24[m]["t_starters_score_final"] + M24[m]["ct_starters_score_final"])] += 1
    print("  (rounds present, header sum) for these matches (top 6):", nmax.most_common(6))

    print("\n== 7. Missing / extra round_end and freeze-end rows (2024) ==")
    miss = collections.Counter()
    for m, rr in R24.items():
        for r, x in rr.items():
            has_fe = x["fe_t"] is not None
            if has_fe and x["winner"] is None:
                miss[(regime_of(M24[m]), "fe_without_round_end")] += 1
            if (not has_fe) and x["winner"] is not None:
                miss[(regime_of(M24[m]), "round_end_without_fe")] += 1
    print("  ", dict(miss))
    rnd = collections.Counter()
    for m, rr in R24.items():
        if regime_of(M24[m]) != "A":
            continue
        for r, x in rr.items():
            if (x["fe_t"] is None) and (x["winner"] is not None):
                rnd[r] += 1
    print("  regime A round_end without freeze-end, by round label:", sorted(rnd.items()))

    print("\n== 8. Bomb truth (explode -> T won, defuse -> CT won; one planted bomb, exactly one resolution) ==")
    tab = collections.Counter()
    tab_r13 = collections.Counter()
    by_day13 = collections.Counter()
    mism_rounds_A = collections.Counter()
    for m, rr in R24.items():
        reg_m = regime_of(M24[m])
        for r, x in rr.items():
            if x["n_planted"] >= 1 and (x["n_exploded"] + x["n_defused"]) == 1:
                truth = T if x["n_exploded"] else CT
                w = x["winner"]
                if w is None:
                    tab[(reg_m, "no_round_end_winner")] += 1
                    continue
                res = "match" if truth == w else "mismatch"
                tab[(reg_m, res)] += 1
                if reg_m == "A":
                    if res == "mismatch":
                        mism_rounds_A[r] += 1
                if r == 13:
                    tab_r13[(reg_m, "no 8->9 swap" if not s8[m] else "8->9 swap", res)] += 1
                    if reg_m == "A" and not s8[m]:
                        by_day13[(M24[m]["day"], res)] += 1
    print("  all bomb rounds (regime, result):", dict(tab))
    print("  round-13 bomb rounds (regime, swap8, result):", dict(tab_r13))
    print("  regime A mismatches by round label (top 12):", mism_rounds_A.most_common(12))
    print("  regime A round-13 (no swap8) by day:", sorted(by_day13.items()))

    print("\n== 9. 2026 control ==")
    s8_26 = {m: swap8_flag(R26.get(m, {})) for m in M26}
    print(f"2026 matches {len(M26)}; 8->9 swap {sum(s8_26.values())}; regime-marker A {sum(1 for m in M26 if regime_of(M26[m])=='A')}")
    print(f"2026 header max_rounds distribution: {dict(collections.Counter(M26[m]['max_rounds'] for m in M26))}")
    mr24_26 = lambda m: M26[m]["max_rounds"] == 24
    diff_summary(R26, M26, SCHEDS["BO_MR24"], mr24_26, "winner", "2026 max_rounds 24 (BO map)")
    diff_summary(R26, M26, SCHEDS["D5_12_24_27"], mr24_26, "winner", "2026 max_rounds 24 (D5 map)")
    boundary_classes(R26, M26, mr24_26, "2026 max_rounds 24", max_r=30)
    header_test(R26, M26, lambda mat: sched(mat["max_rounds"]), lambda m: True,
                "2026 all, BO map with header max_rounds")
    b26 = collections.Counter()
    for m in M26:
        if not mr24_26(m):
            continue
        h = hits_at(R26[m], 13)
        b26[h if isinstance(h, str) else "/".join(h)] += 1
    print("  2026 MR24 boundary-13 classes:", dict(b26))
    bm26 = collections.Counter()
    for m, rr in R26.items():
        for r, x in rr.items():
            if x["n_planted"] >= 1 and (x["n_exploded"] + x["n_defused"]) == 1 and x["winner"] is not None:
                truth = T if x["n_exploded"] else CT
                bm26["match" if truth == x["winner"] else "mismatch"] += 1
    print("  2026 bomb rounds (result):", dict(bm26))


if __name__ == "__main__":
    main()
