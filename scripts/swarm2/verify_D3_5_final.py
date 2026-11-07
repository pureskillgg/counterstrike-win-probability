"""Verifier D3-5 (final): independent recomputation of the claimed winner-derived lead disagreement on the 2024
G: tome, decomposed by match format and by independent truth sources, plus the same score-vs-round_end check on
the 2026 tome that the bake-off reads.

Sources (no source is trusted by itself):
  A   round_end.winner_team_code (side code 2 = T, 3 = CT)            [claim: defective in 2024 day 12-25]
  R   round_state score fields at round_freeze_end (side-attached)    [claim: correct]
  C   winner implied by R between freeze r and freeze r+1, using only transitions with a unique winner
      (no schedule assumption)
  D   bomb outcome (exploded -> T, defused -> CT), rounds with an outcome row only
  E   elimination (all five distinct players of one side dead, no bomb planted, 5v5 rounds only)
  P   physical side per player per round from player_info.team_code (used for format and swap timing)

ASSUMPTIONS: 5v5 = modal player_info count (5,5); 2v2 = modal (2,2); MR12 schedule as in the claimed script
(MR=24 for 2024, where the header has no max_rounds; header max_rounds for 2026); score increments are +1.
Read only. Writes nothing (the run wrapper writes logs/swarm2/verify_D3_5.log).
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
T, CT = 2, 3
TOMES = {
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
}


def read(tome, suf, ch, cols):
    files = sorted(glob.glob(str(tome / f"basic_{ch}{suf}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def swap_expected(r, mr):
    # the claimed script's rule (identical to bakeoff.swap_expected)
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


def classify_winners(t, c, t2, c2):
    out = set()
    if (t2, c2) == (t + 1, c):
        out.add(T)
    if (t2, c2) == (t, c + 1):
        out.add(CT)
    if (t2, c2) == (c, t + 1):
        out.add(T)
    if (t2, c2) == (c + 1, t):
        out.add(CT)
    return out


def load(year):
    tome, suf = TOMES[year]
    hdr_files = sorted(glob.glob(str(tome / "header" / "dataframe_*")))
    hcols = set(pl.read_parquet_schema(hdr_files[0]).keys())
    hcols_want = ["key", "t_starters_score_final", "ct_starters_score_final"] + (["max_rounds"] if "max_rounds" in hcols else [])
    HD = pl.concat([pl.read_parquet(f, columns=hcols_want) for f in hdr_files])
    RE = read(tome, suf, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"])
    RS = read(tome, suf, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    BS = read(tome, suf, "bomb_state", ["match_key", "round", "tick", "event_type"])
    DE = read(tome, suf, "player_death", ["match_key", "round", "tick", "player_id", "player_team_code"])
    PI = read(tome, suf, "player_info", ["match_key", "round", "player_id", "team_code"])
    return HD, RE, RS, BS, DE, PI, ("max_rounds" in hcols)


def build(year):
    HD, RE, RS, BS, DE, PI, has_mr = load(year)
    header = {r["key"]: r for r in HD.iter_rows(named=True)}
    A_first, A_all = {}, collections.defaultdict(set)
    for m, r, tk, w, rc in RE.sort("tick").select(["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).iter_rows():
        A_first.setdefault((m, int(r)), int(w))
        A_all[(m, int(r))].add(int(w))
    fe = RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
    FE = {}
    fe_dupe = 0
    for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
        if (m, int(r)) in FE:
            fe_dupe += 1
            continue
        FE[(m, int(r))] = (t, c)
    BOMB = collections.defaultdict(set)
    for m, r, ev in BS.filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"])) \
            .select(["match_key", "round", "event_type"]).iter_rows():
        BOMB[(m, int(r))].add(ev)
    DEATH = collections.defaultdict(lambda: collections.defaultdict(set))
    for m, r, pid, tc in DE.select(["match_key", "round", "player_id", "player_team_code"]).iter_rows():
        DEATH[(m, int(r))][tc].add(pid)
    SIDE = collections.defaultdict(lambda: {T: set(), CT: set()})
    for m, r, pid, tc in PI.select(["match_key", "round", "player_id", "team_code"]).iter_rows():
        if tc in (T, CT):
            SIDE[(m, int(r))][tc].add(pid)
    return dict(header=header, A_first=A_first, A_all=A_all, FE=FE, fe_dupe=fe_dupe, BOMB=BOMB, DEATH=DEATH,
                SIDE=SIDE, has_mr=has_mr)


def day_of(m):
    return int(m.split("/")[3])


def formats(D):
    per_match = collections.defaultdict(collections.Counter)
    for (m, r), v in D["SIDE"].items():
        per_match[m][(len(v[T]), len(v[CT]))] += 1
    out = {}
    for m, cnt in per_match.items():
        (nt, nc), _ = cnt.most_common(1)[0]
        out[m] = "5v5" if (nt, nc) == (5, 5) else ("2v2" if (nt, nc) == (2, 2) else f"other{nt}v{nc}")
    return out


def phys_swaps(D):
    """Rounds r where the T and CT player sets exchanged between r and r+1 (player_info)."""
    out = collections.defaultdict(list)
    rounds_by_m = collections.defaultdict(list)
    for (m, r) in D["SIDE"]:
        rounds_by_m[m].append(r)
    for m, rs in rounds_by_m.items():
        for r in sorted(rs):
            if (m, r + 1) in D["SIDE"]:
                s0, s1 = D["SIDE"][(m, r)], D["SIDE"][(m, r + 1)]
                if s0[T] and s1[T] and s1[T] == s0[CT] and s1[CT] == s0[T]:
                    out[m].append(r)
    return out


def truth_flags(D, m, r, C):
    """Return dict with C (score winner, unique transition), A (round_end), D (bomb), E (elimination)."""
    a = D["A_first"].get((m, r))
    bomb = D["BOMB"].get((m, r), set())
    d = T if "bomb_exploded" in bomb else (CT if "bomb_defused" in bomb else None)
    e = None
    if "bomb_planted" not in bomb and d is None:
        nT, nCT = len(D["DEATH"][(m, r)].get(T, set())), len(D["DEATH"][(m, r)].get(CT, set()))
        if nCT >= 5 and nT < 5:
            e = T
        elif nT >= 5 and nCT < 5:
            e = CT
    return {"A": a, "C": C.get((m, r)), "D": d, "E": e}


def score_winners(D):
    C = {}
    byM = collections.defaultdict(dict)
    for (m, r), v in D["FE"].items():
        byM[m][r] = v
    for m, fes in byM.items():
        for r in fes:
            if (r + 1) not in fes:
                continue
            t, c = fes[r]
            t2, c2 = fes[r + 1]
            ws = classify_winners(t, c, t2, c2)
            if len(ws) == 1:
                C[(m, r)] = next(iter(ws))
    return C


def claimed_derivation(D, MR=24):
    """Re-implementation of scripts/swarm2/D3_lead_derivation_2024.py (same keys, same exclusions)."""
    by_match = collections.defaultdict(dict)
    for (m, r), w in D["A_first"].items():
        by_match[m][r] = w
    live = {}
    for (m, r), (t, c) in D["FE"].items():
        live[(m, r)] = (t, c)
    gap, compared, agree, first_dis = set(), collections.Counter(), collections.Counter(), {}
    for m, rounds in by_match.items():
        rs_sorted = sorted(rounds)
        grp = "day12_25" if day_of(m) >= 12 else "day01_07"
        if rs_sorted != list(range(1, len(rs_sorted) + 1)):
            gap.add(m)
            continue
        side_a = T
        a = b = 0
        for r in rs_sorted:
            derived = (b - a) if side_a == T else (a - b)
            if (m, r) in live:
                lt, lc = live[(m, r)]
                ok = derived == (lc - lt)
                compared[grp] += 1
                agree[grp] += int(ok)
                if not ok and m not in first_dis:
                    first_dis[m] = r
            w = rounds[r]
            if w == side_a:
                a += 1
            elif w == (CT if side_a == T else T):
                b += 1
            if swap_expected(r, MR):
                side_a = CT if side_a == T else T
    return gap, compared, agree, first_dis


def main():
    print("=" * 100)
    print("PART 1. Reproduce the claimed numbers (2024 tome, claimed method, MR=24 for all matches)")
    D24 = build("2024")
    print("2024 header matches:", len(D24["header"]), "| freeze_end duplicate keys dropped:", D24["fe_dupe"])
    gap, compared, agree, first_dis = claimed_derivation(D24, 24)
    print("round-gap matches (labels do not start at 1, excluded):", len(gap),
          "| by group:", dict(collections.Counter("day12_25" if day_of(m) >= 12 else "day01_07" for m in gap)))
    for g in ["day01_07", "day12_25"]:
        n, ag = compared[g], agree[g]
        print(f"  {g}: rounds compared {n}, agree {ag} ({(ag / n if n else 0):.6f})")
    dis_all = collections.Counter(("day12_25" if day_of(m) >= 12 else "day01_07") for m in first_dis)
    print("matches with any disagreement (all non-gap matches):", dict(dis_all))

    # claimed 'clean' set: day 12-25, no round-13 bomb mismatch, no gap
    r13_mismatch = set()
    for (m, r), evs in D24["BOMB"].items():
        if r != 13:
            continue
        if (m, 13) not in D24["A_first"]:
            continue
        bside = T if "bomb_exploded" in evs else (CT if "bomb_defused" in evs else None)
        if bside is not None and bside != D24["A_first"][(m, 13)]:
            r13_mismatch.add(m)
    print("round-13 bomb mismatch matches (no gap):", len(r13_mismatch),
          "| of those, with any lead disagreement:", sum(1 for m in r13_mismatch if m in first_dis),
          "| first disagreeing round:", sorted(collections.Counter(first_dis[m] for m in r13_mismatch if m in first_dis).items()))
    clean = [m for m in D24["header"] if day_of(m) >= 12 and m not in r13_mismatch and m not in gap]
    fd_clean = collections.Counter(first_dis[m] for m in clean if m in first_dis)
    print(f"claimed 'clean' set: {len(clean)} matches; with any disagreement: {sum(1 for m in clean if m in first_dis)}; "
          f"first disagreeing round: {sorted(fd_clean.items())}")

    print("\n" + "=" * 100)
    print("PART 2. Format decomposition of the claimed 'clean' set (player_info modal side sizes)")
    FMT = formats(D24)
    print("formats over all 2024 matches:", collections.Counter(FMT.values()))
    print("formats in claimed clean set:", collections.Counter(FMT.get(m, "none") for m in clean))
    dec = collections.Counter()
    for m in clean:
        if m in first_dis:
            dec[(FMT.get(m, "none"), first_dis[m])] += 1
    print("first disagreeing round by format (format, round): matches:", sorted(dec.items()))
    print("claimed clean matches that are NOT 5v5:", sum(1 for m in clean if FMT.get(m) != "5v5"),
          "| of those with a disagreement:", sum(1 for m in clean if FMT.get(m) != "5v5" and m in first_dis))

    print("\n" + "=" * 100)
    print("PART 3. Truth sources on 2024 day 12-25 and day 01-07 (all formats), per round: A vs C, D vs C, E vs C")
    C24 = score_winners(D24)
    PS24 = phys_swaps(D24)
    rows = []
    for (m, r) in D24["A_first"]:
        f = truth_flags(D24, m, r, C24)
        grp = "day12_25" if day_of(m) >= 12 else "day01_07"
        rows.append((grp, FMT.get(m, "none"), r, f["A"], f["C"], f["D"], f["E"]))
    tot = collections.Counter()
    by_r = collections.Counter()
    for grp, fmt, r, a, c, d, e in rows:
        if c is None:
            tot["no_C"] += 1
            continue
        tot["C_avail"] += 1
        if a is not None:
            tot["A_avail_with_C"] += 1
            tot["A!=C"] += int(a != c)
            if fmt == "5v5":
                tot["5v5 A!=C"] += int(a != c)
                by_r[(grp, r, "A!=C")] += int(a != c)
                by_r[(grp, r, "n")] += 1
        if d is not None:
            tot["D_avail"] += 1
            tot["D!=C"] += int(d != c)
            tot["D!=A (with A)"] += int(a is not None and d != a)
        if e is not None:
            tot["E_avail"] += 1
            tot["E!=C"] += int(e != c)
            tot["E!=A"] += int(a is not None and e != a)
    print("totals:", dict(tot))
    print("5v5 A!=C by group and round (round: n, A!=C):")
    for g in ["day01_07", "day12_25"]:
        print(" ", g, [(r, by_r[(g, r, 'n')], by_r[(g, r, 'A!=C')]) for r in range(1, 27) if by_r[(g, r, 'n')]])

    print("\n5v5 round_end flips (A != C) by whether a physical swap happened at r-1 (player_info):")
    flips_rel = collections.Counter()
    for (m, r) in D24["A_first"]:
        if FMT.get(m) != "5v5":
            continue
        c = C24.get((m, r))
        a = D24["A_first"].get((m, r))
        if c is None or a is None:
            continue
        grp = "day12_25" if day_of(m) >= 12 else "day01_07"
        swapped_before = (r - 1) in PS24.get(m, [])
        flips_rel[(grp, "phys_swap_at_r-1" if swapped_before else "no_swap_at_r-1", "flip" if a != c else "agree")] += 1
    for k, v in sorted(flips_rel.items()):
        print("  ", k, v)

    print("\nE and D truth at flipped 5v5 rounds (day12_25): which source agrees with C vs A")
    ed = collections.Counter()
    for (m, r) in D24["A_first"]:
        if FMT.get(m) != "5v5" or day_of(m) < 12:
            continue
        c = C24.get((m, r))
        a = D24["A_first"].get((m, r))
        if c is None or a is None or a == c:
            continue
        f = truth_flags(D24, m, r, C24)
        if f["E"] is not None:
            ed[("E", "E==C" if f["E"] == c else "E==A" if f["E"] == a else "other")] += 1
        if f["D"] is not None:
            ed[("D", "D==C" if f["D"] == c else "D==A" if f["D"] == a else "other")] += 1
    print("  ", dict(ed))

    print("\n" + "=" * 100)
    print("PART 4. Same checks on the 2026 tome (the bake-off's own data), MR from header")
    D26 = build("2026")
    FMT26 = formats(D26)
    print("2026 header matches:", len(D26["header"]), "| has max_rounds column:", D26["has_mr"],
          "| formats:", collections.Counter(FMT26.values()))
    C26 = score_winners(D26)
    n_rounds = collections.Counter()
    dis26 = collections.Counter()
    mism_match = set()
    byM = collections.defaultdict(dict)
    for (m, r), v in D26["FE"].items():
        byM[m][r] = v
    A26 = collections.defaultdict(dict)
    for (m, r), w in D26["A_first"].items():
        A26[m][r] = w
    for m, rounds in A26.items():
        mr = D26["header"][m].get("max_rounds", 24) if D26["has_mr"] else 24
        rs_sorted = sorted(rounds)
        if rs_sorted != list(range(1, len(rs_sorted) + 1)):
            dis26["gap_matches"] += 1
            continue
        side_a = T
        a = b = 0
        for r in rs_sorted:
            derived = (b - a) if side_a == T else (a - b)
            if r in byM[m]:
                lt, lc = byM[m][r]
                n_rounds["compared"] += 1
                if derived != lc - lt:
                    dis26["lead_disagree"] += 1
                    mism_match.add(m)
            w = rounds[r]
            if w == side_a:
                a += 1
            elif w == (CT if side_a == T else T):
                b += 1
            if swap_expected(r, mr):
                side_a = CT if side_a == T else T
    print("2026 derived-lead rounds compared (MR from header):", n_rounds["compared"],
          "| disagreements:", dis26["lead_disagree"], "| matches with any disagreement:", len(mism_match),
          "| round-gap matches:", dis26["gap_matches"])
    t26 = collections.Counter()
    for (m, r) in D26["A_first"]:
        f = truth_flags(D26, m, r, C26)
        if f["C"] is None:
            continue
        t26["C_avail"] += 1
        if f["A"] is not None:
            t26["A!=C"] += int(f["A"] != f["C"])
        if f["D"] is not None:
            t26["D_avail"] += 1
            t26["D!=C"] += int(f["D"] != f["C"])
            t26["D!=A"] += int(f["A"] is not None and f["D"] != f["A"])
        if f["E"] is not None:
            t26["E_avail"] += 1
            t26["E!=C"] += int(f["E"] != f["C"])
    print("2026 truth-source totals:", dict(t26))


if __name__ == "__main__":
    main()
