"""D3-5 verifier, step K (2024 G: tome): consolidated, independent recomputation of the claim's numbers and the
alternative explanations. Read only. Projected parquet reads. ASSUMPTIONS are printed with each section.

Witnesses (side codes: 2 = T side, 3 = CT side):
  W  round_end.winner_team_code (first row per match-round, file order, as in D3_lead_derivation_2024.py)
  D  round_state freeze_end unit delta (fields are side-attached): T field +1 -> T won, CT field +1 -> CT won
  B  bomb_state: bomb_defused -> CT, bomb_exploded -> T
Halftime h per match = first round_state row with phase == 'second' (ENGINE marker; ASSUMPTION h = 12 if absent).
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(200)
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def rd(ch, cols):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUF}")
    return pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(folder / "dataframe_*")))],
                     how="vertical_relaxed")


print("== 2024 G: tome, consolidated check ==")
H = rd("header", ["key", "t_starters_score_final", "ct_starters_score_final"])
hdr = {k: (t, c) for k, t, c in H.iter_rows()}
RE_raw = rd("round_end", ["match_key", "round", "winner_team_code", "player_count"])
RE = RE_raw.unique(["match_key", "round"], keep="first")  # D3 way (file order)
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"])
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first(),
                                           pl.col("phase").first()))
BS = rd("bomb_state", ["match_key", "round", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
bomb = BS.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
sec = RS.filter(pl.col("phase") == "second").group_by("match_key").agg(pl.col("round").min().alias("h"))
hmap = {m: h for m, h in sec.iter_rows()}

W = collections.defaultdict(dict)
for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows():
    W[m][r] = w
LIVE = collections.defaultdict(dict)
for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    LIVE[m][r] = (t, c)
BOMB = {}
for m, r, evs in bomb.iter_rows():
    BOMB[(m, r)] = CT if "bomb_defused" in evs else T
day = lambda m: int(m.split("/")[3])
grp = lambda m: "12-25" if day(m) >= 12 else "<12"


def cls(m):
    rs = sorted(W[m])
    if rs == list(range(1, len(rs) + 1)):
        return "contig"
    if rs and rs[0] == 2 and rs == list(range(2, len(rs) + 2)):
        return "start2"
    return "other"


# ---------- A. population
pop = collections.Counter((grp(m), cls(m)) for m in W)
print("\n[A] matches by day group and round-1 presence (contig = rounds 1..N present; start2 = round 1 missing):")
for k in sorted(pop):
    print("   ", k, pop[k])

# ---------- B. reproduce D3 claim numbers (MR24 fixed swap rule, D3 definitions)
print("\n[B] reproduce D3 claim (ASSUMPTION MR24 fixed: swap after r == 12 and OT rule (r-27)%6==0):")
r13 = (RE.filter(pl.col("round") == 13)
       .join(bomb.filter(pl.col("round") == 13), on=["match_key", "round"], how="inner"))
aff = set(r13.with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bs")
).filter(pl.col("bs") != pl.col("winner_team_code"))["match_key"].to_list())
print("    round-13 bomb-side mismatch matches (all 2024):", len(aff))
contig_late = [m for m in W if cls(m) == "contig" and day(m) >= 12]
clean = [m for m in contig_late if m not in aff]
print("    contiguous day 12-25 matches:", len(contig_late), "| affected:", sum(m in aff for m in contig_late),
      "| 'clean' (contig, no r13 mismatch):", len(clean))


def swap24(r):
    return r == 12 or (r > 24 and (r - 24 - 3) % 6 == 0)


def first_dis(m, swap):
    side_a, a, b = T, 0, 0
    for r in sorted(W[m]):
        if r not in LIVE[m]:
            continue
        t, c = LIVE[m][r]
        derived = (b - a) if side_a == T else (a - b)
        if derived != c - t:
            return r
        w = W[m][r]
        if w == side_a:
            a += 1
        elif w == (CT if side_a == T else T):
            b += 1
        if swap(r):
            side_a = CT if side_a == T else T
    return None


fd_clean = collections.Counter(first_dis(m, swap24) for m in clean)
n_dis_clean = sum(v for k, v in fd_clean.items() if k is not None)
print(f"    clean matches with any disagreement: {n_dis_clean} of {len(clean)}; first-disagreement round counts: "
      f"{sorted((k, v) for k, v in fd_clean.items() if k is not None)}")
fd_aff = collections.Counter(first_dis(m, swap24) for m in aff if m in W and cls(m) == "contig")
print(f"    affected contiguous matches: first-disagreement rounds {sorted(fd_aff.items(), key=lambda kv: str(kv[0]))}")

# ---------- C. same derivation with the ENGINE halftime (h from phase) instead of MR24
def swap_h(h):
    return lambda r: r == h or (r > 2 * h and (r - 2 * h - 3) % 6 == 0)


print("\n[C] derivation with engine halftime h (ASSUMPTION: OT swap rule (r-2h-3)%6==0, as bakeoff.swap_expected):")
by_h = collections.defaultdict(collections.Counter)
for m in contig_late:
    h = hmap.get(m, 12)
    fd = first_dis(m, swap_h(h))
    key = (f"h{h}" if m in hmap else "hNA", "affected" if m in aff else "clean")
    by_h[key]["n"] += 1
    if fd is not None:
        by_h[key]["dis"] += 1
        by_h[key][f"first_{fd}"] += 1
for k in sorted(by_h):
    print("   ", k, dict(by_h[k]))

# ---------- D. witness agreement by round, contiguous vs start2, halftime type
print("\n[D] witness agreement per round (valid unit delta; transitions at h or OT swap excluded):")
tab = collections.defaultdict(collections.Counter)
for m in W:
    c_ = cls(m)
    if c_ not in ("contig", "start2"):
        continue
    h = hmap.get(m, 12)
    ht = f"h{h}" if m in hmap else "hNA"
    swp = swap_h(h)
    for r in sorted(W[m]):
        if r not in LIVE[m] or (r + 1) not in LIVE[m]:
            continue
        t1, c1 = LIVE[m][r]
        t2, c2 = LIVE[m][r + 1]
        dd = (t2 - t1, c2 - c1)
        k = (c_, ht, r)
        if swp(r):
            tab[k]["transition_swap_excluded"] += 1
            continue
        if dd == (1, 0):
            D = T
        elif dd == (0, 1):
            D = CT
        else:
            tab[k]["unclear"] += 1
            continue
        tab[k]["n"] += 1
        tab[k]["W=D"] += int(W[m][r] == D)
        if (m, r) in BOMB:
            tab[k]["bomb_n"] += 1
            tab[k]["B=D"] += int(BOMB[(m, r)] == D)
            tab[k]["B=W"] += int(BOMB[(m, r)] == W[m][r])
print(f"    {'class':6} {'htype':5} {'r':>3} {'n':>6} {'W=D':>6} {'bomb_n':>6} {'B=D':>6} {'B=W':>6}")
for k in sorted(tab, key=lambda k: (k[0], k[1], k[2])):
    if k[2] > 30:
        continue
    c = tab[k]
    if c["n"] == 0:
        continue
    print(f"    {k[0]:6} {k[1]:5} {k[2]:>3} {c['n']:>6} {c['W=D']:>6} {c['bomb_n']:>6} {c['B=D']:>6} {c['B=W']:>6}")

# ---------- E. header truth test, order-free
print("\n[E] header truth test (order-free). FIELD: {freeze_M fields + winner_M} == {hdrT, hdrC}."
      " WINNER: team totals from winners with engine h (matches with M <= 2h only, no OT assumption).")
res = collections.Counter()
exs = collections.defaultdict(list)
for m in W:
    if m not in hdr or hdr[m][0] is None:
        continue
    c_ = cls(m)
    if c_ not in ("contig", "start2"):
        continue
    ht_, hc_ = hdr[m]
    rounds = sorted(W[m])
    M = rounds[-1]
    if M != ht_ + hc_:
        res[(c_, "not_complete")] += 1
        continue
    h = hmap.get(m, 12)
    htag = f"h{h}" if m in hmap else "hNA"
    res[(c_, "complete", htag)] += 1
    # FIELD
    if M in LIVE[m]:
        t_n, c_n = LIVE[m][M]
        wM = W[m][M]
        tf, cf = t_n + (wM == T), c_n + (wM == CT)
        ok = sorted([tf, cf]) == sorted([ht_, hc_])
        res[(c_, "field_ok", htag)] += int(ok)
        if not ok and len(exs[(c_, "field_bad")]) < 3:
            exs[(c_, "field_bad")].append((m[-45:], M, ht_, hc_, tf, cf))
    # WINNER (no OT only)
    if M <= 2 * h and m in hmap:
        a = 0
        for r in rounds:
            side = T if r <= h else CT
            a += int(W[m][r] == side)
        b = M - a
        ok = (a, b) in [(ht_, hc_), (hc_, ht_)]
        res[(c_, "winner_ok_noOT", htag)] += int(ok)
        res[(c_, "winner_tested_noOT", htag)] += 1
        if not ok and len(exs[(c_, "winner_bad")]) < 3:
            exs[(c_, "winner_bad")].append((m[-45:], M, ht_, hc_, a, b, h))
for k in sorted(res, key=lambda k: str(k)):
    print("   ", k, res[k])
for k, v in exs.items():
    print("   examples", k, v)
print("\n== end of 2024 section ==")
