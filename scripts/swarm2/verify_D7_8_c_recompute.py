"""verify_D7_8 main recompute, independent of every D7_* output. Reads only the raw 2024 tome.

Checks the D7-8/D7-9 claim: swap-pattern counts (3,111 / 253 / 97 / 21 / 734), the round-13 mismatch
replication (605 rows / 434 mismatches), the round-9 'flip' claim, and whether format mix explains the defect.

Sources and definitions:
  freeze_end t_score / ct_score: side-based scores (verified: a halftime shows as value swap, see probe).
  winner_team_code: side code of the round winner (2 = T, 3 = CT).
  bomb side: 3 if bomb_defused, 2 if bomb_exploded (same as e2 / D7_1 convention).
  phase (round_state, freeze_end): 'first' / 'second' per round. A phase change between consecutive rounds
    is treated as a side swap boundary (ASSUMPTION, tested below against score consistency and final scores).
  header t_starters_score_final / ct_starters_score_final: used only as a final-score truth check.
Writes stdout and logs/swarm2/verify_D7_8.log (overwritten each run).
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(240)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
LOG_PATH = ROOT / "logs" / "swarm2" / "verify_D7_8.log"
_LOG = open(LOG_PATH, "w", encoding="utf-8")
INF = 10 ** 9


def P(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    _LOG.write(s + "\n")
    _LOG.flush()


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(ch, cols, pred=None):
    lf = pl.concat([pl.scan_parquet(f) for f in files(BASE / f"basic_{ch}{SUF}")])
    if pred is not None:
        lf = lf.filter(pred)
    return lf.select(cols).collect()


# ------------------------------------------------------------------ load
H = pl.concat([pl.read_parquet(f, columns=["key", "match_date", "map_name", "build_num",
                                             "t_starters_score_final", "ct_starters_score_final"])
               for f in files(BASE / "header")])
H = H.with_columns(
    pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt"),
    pl.col("key").str.extract(r"csds/(\d{4}/\d\d/\d\d)/", 1).alias("key_day"),
)
H = H.with_columns(pl.col("mdt").dt.strftime("%Y-%m-%d").alias("md_date"),
                   pl.col("mdt").dt.strftime("%Y-%m").alias("md_month"))

FE_raw = scan("round_state", ["match_key", "round", "t_score", "ct_score", "phase"],
              pl.col("event_type") == "round_freeze_end")
RE_raw = scan("round_end", ["match_key", "round", "winner_team_code", "player_count", "tick"])
BS = scan("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"],
          pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"]))

# ------------------------------------------------------------------ S0 coverage and duplicates
P("== S0 coverage and duplicate keys")
dupFE = (FE_raw.group_by(["match_key", "round"])
         .agg(pl.len().alias("n"), pl.col("t_score").n_unique().alias("nt"),
              pl.col("ct_score").n_unique().alias("nc"), pl.col("phase").n_unique().alias("np"))
         .filter(pl.col("n") > 1))
FE = FE_raw.unique(subset=["match_key", "round"], keep="first", maintain_order=True)
P("freeze_end rows:", FE_raw.height, "| unique (match,round):", FE.height,
  "| duplicated keys:", dupFE.height, "| with conflicting scores/phase:",
  dupFE.filter((pl.col("nt") > 1) | (pl.col("nc") > 1) | (pl.col("np") > 1)).height)
dupRE = (RE_raw.group_by(["match_key", "round"])
         .agg(pl.len().alias("n"), pl.col("winner_team_code").n_unique().alias("nw"),
              pl.col("player_count").n_unique().alias("np")).filter(pl.col("n") > 1))
RE = RE_raw.sort("tick").unique(subset=["match_key", "round"], keep="first", maintain_order=True)
P("round_end rows:", RE_raw.height, "| unique (match,round):", RE.height,
  "| duplicated keys:", dupRE.height, "| with conflicting winner:", dupRE.filter(pl.col("nw") > 1).height)
if dupRE.height:
    P("duplicated round_end keys:", dupRE.rows())
RE_k = RE.select(["match_key", "round"])
FE_k = FE.select(["match_key", "round"])
P("round_end keys without freeze_end (no score, no ledger):", RE_k.join(FE_k, on=["match_key", "round"], how="anti").height)
P("freeze_end keys without round_end:", FE_k.join(RE_k, on=["match_key", "round"], how="anti").height)
P("header matches:", H.height, "| matches with any round_end:", RE["match_key"].n_unique(),
  "| with any player_count==20 round_end:", RE.filter(pl.col("player_count") == 20)["match_key"].n_unique())
maxr = RE.group_by("match_key").agg(pl.col("round").max().alias("maxr_re"), pl.len().alias("n_re"))
fin = H.select(["key", "t_starters_score_final", "ct_starters_score_final"]).rename({"key": "match_key"})
cov = maxr.join(fin, on="match_key", how="left").with_columns(
    (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final") - pl.col("n_re")).alias("sum_minus_rows"))
P("final-score sum minus round_end row count, per match (0 = complete):",
  cov.group_by("sum_minus_rows").len().sort("sum_minus_rows").rows())

# ------------------------------------------------------------------ per-match freeze_end structures
fe = collections.defaultdict(dict)
for k, r, t, c, ph in FE.select(["match_key", "round", "t_score", "ct_score", "phase"]).iter_rows():
    fe[k][r] = (t, c, ph)


def align(t0, c0, t1, c1, swap):
    """Winner side for round i from freeze_end(i)=(t0,c0) and freeze_end(i+1)=(t1,c1).
    swap=False: side scores keep their meaning. swap=True: side scores are exchanged at this boundary.
    Returns (valid, winner_side) where side 2 = T, 3 = CT."""
    if not swap:
        d = (t1 - t0, c1 - c0)
        if d == (1, 0):
            return True, 2
        if d == (0, 1):
            return True, 3
        return False, None
    d = (t1 - c0, c1 - t0)
    if d == (1, 0):
        return True, 3
    if d == (0, 1):
        return True, 2
    return False, None


# boundary info for every consecutive freeze_end pair
bnd = {}
for k, d in fe.items():
    for r in d:
        if r + 1 in d:
            t0, c0, p0 = d[r]
            t1, c1, p1 = d[r + 1]
            nv, ns = align(t0, c0, t1, c1, False)
            sv, ss = align(t0, c0, t1, c1, True)
            if p0 in ("first", "second") and p1 in ("first", "second"):
                phs = p0 != p1
                pv, ps = align(t0, c0, t1, c1, phs)
            else:
                phs, pv, ps = None, False, None
            bnd[(k, r)] = dict(phs=phs, nv=nv, ns=ns, sv=sv, ss=ss, pv=pv, ps=ps)

# ------------------------------------------------------------------ S1 phase-based format (ledger-free)
P("\n== S1 format from the game's own phase field (freeze_end)")
fs_c = collections.Counter()
ot_c = collections.Counter()
ph_pattern = {}
for k, d in fe.items():
    rs = sorted(d)
    fsec = [r for r in rs if d[r][2] == "second"]
    fs_c[min(fsec) if fsec else None] += 1
    bnds = tuple(r for r in rs if (k, r) in bnd and bnd[(k, r)]["phs"])
    ph_pattern[k] = bnds
P("first 'second'-phase round (13 = halftime after 12; 9 = halftime after 8; None = never reached):",
  sorted(fs_c.items(), key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0)))
P("phase-change boundary pattern (round before change), top 12:",
  collections.Counter(ph_pattern.values()).most_common(12))

# ------------------------------------------------------------------ S2 score consistency of boundaries
P("\n== S2 score consistency of each consecutive freeze_end boundary (1 boundary = 1 round)")
cons = collections.Counter()
for (k, r), b in bnd.items():
    if b["phs"] is None:
        continue
    nos_ok, sw_ok = b["nv"], b["sv"]
    cat = ("both" if nos_ok and sw_ok else "noswap_only" if nos_ok else "swap_only" if sw_ok else "neither")
    key = ("phase_change" if b["phs"] else "no_phase_change", cat)
    cons[key] += 1
for key, n in sorted(cons.items()):
    P(key, n)
byr = collections.Counter()
for (k, r), b in bnd.items():
    if b["phs"]:
        byr[r] += 1
P("phase-change boundaries by round before the change (top 12):", byr.most_common(12))
P("phase-change boundaries whose phase-anchored alignment is valid:",
  sum(1 for b in bnd.values() if b["phs"] and b["pv"]), "of", sum(1 for b in bnd.values() if b["phs"]))

# ------------------------------------------------------------------ S3 claim replication (min-swap DP, claim logic)
P("\n== S3 claim replication: min-swap DP over freeze_end runs (claim logic, sets + pop)")


def solve_segment(bn):
    nb = len(bn)
    F = [[0, 0]]
    for j in range(nb):
        nf = [INF, INF]
        for o0 in (0, 1):
            for o1 in (0, 1):
                s = o0 ^ o1
                v, _ = bn[j][s]
                if v and F[j][o0] < INF:
                    nf[o1] = min(nf[o1], F[j][o0] + s)
        F.append(nf)
    total = min(F[nb])
    if total >= INF:
        return None
    G = [None] * (nb + 1)
    G[nb] = [0, 0]
    for j in range(nb - 1, -1, -1):
        gj = [INF, INF]
        for o0 in (0, 1):
            for o1 in (0, 1):
                s = o0 ^ o1
                v, _ = bn[j][s]
                if v and G[j + 1][o1] < INF:
                    gj[o0] = min(gj[o0], s + G[j + 1][o1])
        G[j] = gj
    out = []
    for j in range(nb):
        sides, swaps = set(), set()
        for o0 in (0, 1):
            for o1 in (0, 1):
                s = o0 ^ o1
                v, side = bn[j][s]
                if not v or F[j][o0] >= INF or G[j + 1][o1] >= INF:
                    continue
                if F[j][o0] + s + G[j + 1][o1] == total:
                    sides.add(side)
                    swaps.add(o0 != o1)
        out.append((sides, swaps))
    return total, out


claim_pair_side, claim_pair_swap_literal, correct_pair_swap = {}, {}, {}
seg_stat = collections.Counter()
pair_stat = collections.Counter()
infeasible_pairs = 0
for k, d in fe.items():
    rs = sorted(d)
    segs, cur = [], [rs[0]]
    for r in rs[1:]:
        if r == cur[-1] + 1:
            cur.append(r)
        else:
            segs.append(cur)
            cur = [r]
    segs.append(cur)
    for seg in segs:
        if len(seg) < 2:
            seg_stat["single_round_segment"] += 1
            continue
        bn = []
        for j in range(len(seg) - 1):
            b = bnd[(k, seg[j])]
            bn.append({0: (b["nv"], b["ns"]), 1: (b["sv"], b["ss"])})
        res = solve_segment(bn)
        if res is None:
            seg_stat["infeasible_segment"] += 1
            infeasible_pairs += len(seg) - 1
            continue
        seg_stat["feasible_segment"] += 1
        total, out = res
        for j, (sides, swaps) in enumerate(out):
            r = seg[j]
            if len(sides) == 1:
                pair_stat["side_unique"] += 1
                claim_pair_side[(k, r)] = next(iter(sides))
                # literal claim behaviour: swaps.pop() on a set that may hold both True and False
                claim_pair_swap_literal[(k, r)] = swaps.copy().pop()
                if len(swaps) == 1:
                    correct_pair_swap[(k, r)] = next(iter(swaps))
                    pair_stat["swap_unique"] += 1
                else:
                    pair_stat["swap_ambiguous_but_side_unique"] += 1
            else:
                pair_stat["side_ambiguous"] += 1
P("segments:", dict(seg_stat), "| infeasible pairs:", infeasible_pairs)
P("pairs:", dict(pair_stat))

# claim pattern per match = rounds whose literal swap flag is True
claim_pat, correct_pat = {}, {}
unresolved_pairs = [kr for kr in bnd if kr not in correct_pair_swap]
unresolved_match = set(k for k, _ in unresolved_pairs)
for k in fe:
    rs = sorted(fe[k])
    claim_pat[k] = tuple(r for r in rs if claim_pair_swap_literal.get((k, r)) is True)
    corr = tuple(r for r in rs if correct_pair_swap.get((k, r)) is True)
    correct_pat[k] = (("AMBIG",) + corr) if k in unresolved_match else corr
P("pairs with unresolved swap flag (ambiguous or infeasible):", len(unresolved_pairs),
  "| matches affected:", len(unresolved_match))

# ------------------------------------------------------------------ S3b claim pattern counts vs claim numbers
P("\n== S3b claim swap-pattern counts (matches with any round_end pc==20, claim-literal flags)")
pc20 = set(RE.filter(pl.col("player_count") == 20)["match_key"].unique().to_list())
pc20_in_fe = [k for k in pc20 if k in fe]
P("matches with pc==20 round_end:", len(pc20), "| of which in freeze_end data:", len(pc20_in_fe))
pat_str = {k: ",".join(str(x) for x in claim_pat[k]) for k in fe}
cnt = collections.Counter(pat_str.get(k, "") for k in pc20)
P("claim-literal pattern counts (pattern: matches):", cnt.most_common(8))
P("claimed (D7_9 log): [('12', 3111), ('', 734), ('8', 253), ('12,27', 97), ('27', 21)] sum 4216")

# corrected: a boundary counts as swap only if the swap flag is unique across optimal paths
cpat_str = {k: ",".join(str(x) for x in correct_pat[k]) for k in fe}
cnt_c = collections.Counter(cpat_str.get(k, "") for k in pc20)
P("corrected pattern counts (swap flag unique on every boundary; AMBIG prefix if not):", cnt_c.most_common(10))
ambig_matches = sum(1 for k in pc20 if correct_pat.get(k) and correct_pat[k][0] == "AMBIG")
P("matches with an ambiguous swap flag on some boundary:", ambig_matches)

P("\nCrosstab: phase-change pattern (rows) x claim-literal pattern (cols), pc==20 matches, top 15:")
xt = collections.Counter((",".join(str(x) for x in ph_pattern.get(k, ())), pat_str.get(k, "")) for k in pc20)
for (a, b), n in xt.most_common(15):
    P(f"  phase=[{a}]  claim=[{b}]  matches={n}")

# ------------------------------------------------------------------ S4 coverage: missing round_end rows vs final scores
P("\n== S4 coverage: rounds missing from round_end / freeze_end, vs header final-score total")
re_rounds = collections.defaultdict(set)
for k, r in RE.select(["match_key", "round"]).iter_rows():
    re_rounds[k].add(r)
fe_rounds = {k: set(d) for k, d in fe.items()}
fin_map = {k: (t, c) for k, t, c in H.select(["key", "t_starters_score_final", "ct_starters_score_final"]).iter_rows()}
miss_re = {k: [r for r in range(1, max(s) + 1) if r not in s] for k, s in re_rounds.items()}
miss_fe = {k: [r for r in range(2, max(s) + 1) if r not in fe_rounds.get(k, set())] for k, s in re_rounds.items()}
dist = collections.Counter()
for k, s in re_rounds.items():
    tf, cf = fin_map[k]
    total_final = tf + cf
    dist[(total_final - len(s)) == len(miss_re[k]), total_final - len(s) >= 0] += 1
P("(sum_final - rounds_in_round_end) == missing_round_end_rounds, and >=0:", dict(dist))
P("matches with >=1 round missing from round_end:", sum(1 for v in miss_re.values() if v), "of", len(miss_re))
P("round 1 missing from round_end:", sum(1 for k in re_rounds if 1 not in re_rounds[k]))
P("round 1 missing from freeze_end (of matches with round_end):", sum(1 for k in re_rounds if 1 not in fe_rounds.get(k, set())))
r1_build = H.filter(pl.col("key").is_in(
    [k for k in re_rounds if 1 not in fe_rounds.get(k, set())])).group_by("build_num").len().rows()
P("round-1-freeze_end-missing matches by build_num:", r1_build)
P("total rounds with round_end but no freeze_end:", sum(len(re_rounds[k] - fe_rounds.get(k, set())) for k in re_rounds))

# ------------------------------------------------------------------ S5 final-score truth check on swap parity
P("\n== S5 final-score check: team totals from round winners (round_end codes) + swap parity")
P("ASSUMPTION: header final scores are team totals; accept either orientation (A=start-T or A=start-CT).")


def totals(winners, swap_bounds):
    A = 0
    for r, w in winners.items():
        S = sum(1 for b in swap_bounds if b < r)
        a_on_t = S % 2 == 0
        if (w == 2) == a_on_t:
            A += 1
    return A, len(winners) - A


win_by_match = collections.defaultdict(dict)
for k, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows():
    win_by_match[k][r] = w

full = [k for k in re_rounds if not miss_re[k] and sum(fin_map[k]) == len(re_rounds[k]) and k in fe]
P("matches with contiguous round_end 1..n, sum(final)==n, and freeze_end present:", len(full))
schemes = {
    "phase_swaps": lambda k: ph_pattern.get(k, ()),
    "claim_swaps": lambda k: claim_pat.get(k, ()),
    "no_swaps": lambda k: (),
}
for name, fn in schemes.items():
    ok = 0
    for k in full:
        A, B = totals(win_by_match[k], fn(k))
        tf, cf = fin_map[k]
        if (A, B) == (tf, cf) or (A, B) == (cf, tf):
            ok += 1
    P(f"  scheme={name}: totals match final score in {ok} of {len(full)} matches")

# S5b: last-frame check, independent of swap parity. Final score in the frame of the last freeze_end round
# = (t + [last winner is T], c + [last winner is CT]). Compare that pair with the header pair.
lastf = collections.Counter()
ex_rows = []
bld = {k: b for k, b in H.select(["key", "build_num"]).iter_rows()}
for k in re_rounds:
    if k not in fe or not fe[k]:
        continue
    m = max(fe[k])
    if m not in win_by_match[k]:
        lastf["no_winner_for_last_freeze_round"] += 1
        continue
    t, c, _ = fe[k][m]
    w = win_by_match[k][m]
    fT, fC = (t + (w == 2)), (c + (w == 3))
    tf, cf = fin_map[k]
    if (fT, fC) == (tf, cf):
        lastf[("frame_exact", bld[k])] += 1
    elif (fT, fC) == (cf, tf):
        lastf[("frame_swapped", bld[k])] += 1
    else:
        lastf[("neither", bld[k])] += 1
P("S5b last-frame check (score after last round vs header finals):", sorted(lastf.items(), key=lambda x: str(x[0])))

# S4b: build split of rounds vs final score
bsplit = collections.Counter()
for k, s in re_rounds.items():
    tf, cf = fin_map[k]
    bsplit[(bld[k], "sum_final-n_re", tf + cf - len(s))] += 1
P("sum(final) - rounds in round_end, by build_num (top 12 per build):")
for b_ in sorted(set(x[0] for x in bsplit)):
    items = sorted([(x[2], n) for x, n in bsplit.items() if x[0] == b_], key=lambda z: -z[1])[:12]
    P("  build", b_, items)

# split: matches where phase and claim swap sets differ
diff_set = [k for k in full if tuple(ph_pattern.get(k, ())) != tuple(claim_pat.get(k, ()))]
same_set = [k for k in full if tuple(ph_pattern.get(k, ())) == tuple(claim_pat.get(k, ()))]
P("full matches where phase swaps != claim swaps:", len(diff_set), "| identical:", len(same_set))
for name, fn in schemes.items():
    ok = 0
    for k in diff_set:
        A, B = totals(win_by_match[k], fn(k))
        tf, cf = fin_map[k]
        if (A, B) == (tf, cf) or (A, B) == (cf, tf):
            ok += 1
    P(f"  on disagreeing matches, scheme={name}: match final score in {ok} of {len(diff_set)}")

# ------------------------------------------------------------------ S6 per-round side comparisons
P("\n== S6 per-round side vs round_end winner_team_code")
fs_first = {}
for k, d in fe.items():
    fsec = [r for r in d if d[r][2] == "second"]
    fs_first[k] = min(fsec) if fsec else None
fmt = {k: ("MR12" if v == 13 else "MR8" if v == 9 else "no_second_half") for k, v in fs_first.items()}
for k in fe:
    fmt.setdefault(k, "no_second_half")
ph_pat_s = {k: ",".join(str(x) for x in ph_pattern.get(k, ())) for k in fe}
cl_pat_s = {k: ",".join(str(x) for x in claim_pat.get(k, ())) for k in fe}
cor_pat_s = {k: ",".join(str(x) for x in correct_pat.get(k, ())) for k in fe}

RR = (RE.join(
        (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
           .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"), pl.len().alias("n_out"))),
        on=["match_key", "round"], how="left")
      .join((BS.filter(pl.col("event_type") == "bomb_planted").sort("tick")
             .group_by(["match_key", "round"]).agg(pl.col("player_team_code").first().alias("planter_side"))),
            on=["match_key", "round"], how="left")
      .join(H.select(["key", "md_date", "key_day", "build_num", "map_name"]).rename({"key": "match_key"}),
            on="match_key", how="left"))
RR = RR.with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(3)
      .when(pl.col("evs").list.contains("bomb_exploded")).then(2).otherwise(None).cast(pl.Int64).alias("bomb_side"))
rows = []
for rec in RR.select(["match_key", "round", "winner_team_code", "player_count", "bomb_side", "planter_side",
                      "md_date", "key_day", "build_num", "n_out"]).iter_rows(named=True):
    k, r = rec["match_key"], rec["round"]
    b = bnd.get((k, r))
    sph = b["ps"] if (b is not None and b["pv"]) else None
    rows.append(dict(match_key=k, round=r, winner=rec["winner_team_code"], pc=rec["player_count"],
                     bomb=rec["bomb_side"], planter=rec["planter_side"], md_date=rec["md_date"],
                     key_day=rec["key_day"], build=rec["build_num"], n_out=rec["n_out"],
                     s_ph=sph, s_cl=claim_pair_side.get((k, r)), s_cor=claim_pair_side.get((k, r))
                     if (k, r) in correct_pair_swap else None,
                     fmt=fmt.get(k), ph_pat=ph_pat_s.get(k, ""), cl_pat=cl_pat_s.get(k, ""),
                     cor_pat=cor_pat_s.get(k, "")))
D = pl.DataFrame(rows, schema_overrides={"s_ph": pl.Int64, "s_cl": pl.Int64, "s_cor": pl.Int64,
                                         "bomb": pl.Int64, "planter": pl.Int64, "n_out": pl.Int64})
D = D.with_columns(
    (pl.col("s_ph").is_not_null() & (pl.col("s_ph") == pl.col("winner"))).alias("ph_eq_w"),
    (pl.col("s_ph").is_not_null() & (pl.col("s_ph") != pl.col("winner"))).alias("ph_ne_w"),
    (pl.col("s_cl").is_not_null() & (pl.col("s_cl") != pl.col("winner"))).alias("cl_ne_w"),
    (pl.col("s_cl").is_not_null() & (pl.col("s_cl") == pl.col("winner"))).alias("cl_eq_w"),
    pl.when(pl.col("key_day").str.slice(-2).cast(pl.Int64) <= 7).then(pl.lit("early_01_07"))
      .when(pl.col("key_day").str.slice(-2).cast(pl.Int64) <= 11).then(pl.lit("mid_08_11"))
      .otherwise(pl.lit("late_12_25")).alias("win_key"),
)
P("round rows (round_end, all):", D.height, "| with bomb outcome:", D.filter(pl.col("bomb").is_not_null()).height)
P("per-round score-side availability: phase-anchored", D.filter(pl.col("s_ph").is_not_null()).height,
  "| claim-literal", D.filter(pl.col("s_cl").is_not_null()).height)
g = D.filter(pl.col("s_ph").is_not_null()).group_by(["win_key", "fmt"]).agg(
    pl.len().alias("n"), pl.col("ph_ne_w").sum().alias("phase_side_ne_winner"),
    pl.col("cl_ne_w").sum().alias("claim_side_ne_winner_(same rows)")).sort(["win_key", "fmt"])
P("phase-anchored side vs winner_team_code, by window and format:")
P(g)
g2 = D.filter(pl.col("s_cl").is_not_null()).group_by(["win_key", "fmt"]).agg(
    pl.len().alias("n"), pl.col("cl_ne_w").sum().alias("claim_side_ne_winner")).sort(["win_key", "fmt"])
P("claim (min-swap DP) side vs winner_team_code, by window and format:")
P(g2)

# halftime-round disagreement: MR12 matches where the claim has no swap at 12 but phase has one
D = D.with_columns(
    pl.when((pl.col("fmt") == "MR12") & (~pl.col("cl_pat").str.contains(r"(^|,)12(,|$)")))
      .then(pl.lit("MR12_claim_no_12")).when(pl.col("fmt") == "MR12").then(pl.lit("MR12_claim_has_12"))
      .when(pl.col("fmt") == "MR8").then(pl.lit("MR8"))
      .otherwise(pl.lit("other")).alias("grp"))
g3 = D.filter(pl.col("s_ph").is_not_null() & (pl.col("round") > 12) & (pl.col("round") < 25)
              & (pl.col("win_key") == "late_12_25")).group_by("grp").agg(
    pl.len().alias("n_rounds_13_24"), pl.col("ph_ne_w").sum().alias("phase_ne_winner"),
    pl.col("cl_ne_w").sum().alias("claim_ne_winner")).sort("grp")
P("MR12 second-half rounds (13-24), late window: phase side vs claim side disagreement with winner:")
P(g3)

# ------------------------------------------------------------------ S7 defect replication and profile
P("\n== S7 bomb vs winner mismatch (player_count==20, bomb outcome present)")
O = D.filter((pl.col("pc") == 20) & pl.col("bomb").is_not_null()).with_columns(
    (pl.col("bomb") != pl.col("winner")).alias("mm_wb"))
P("all such rows:", O.height, "| mismatches:", int(O["mm_wb"].sum()),
  "(claim: 605 round-13 rows / 434 mismatches; other rounds 41 of 14,602)")
r13 = O.filter(pl.col("round") == 13)
P("round 13 rows:", r13.height, "| mismatches:", int(r13["mm_wb"].sum()))
P("round 13 by key-path day (e2 convention):")
P(r13.group_by("key_day").agg(pl.len().alias("n"), pl.col("mm_wb").sum().alias("mm")).sort("key_day"))
oth = O.filter(~pl.col("round").is_in([13, 25]))
P("other rounds (not 13/25): rows", oth.height, "mismatches", int(oth["mm_wb"].sum()))
P("other rounds by window:")
P(oth.group_by("win_key").agg(pl.len().alias("n"), pl.col("mm_wb").sum().alias("mm")).sort("win_key"))
P("mismatch rate by round number, late window (key day 12-25), rounds with n>=20:")
prof = (O.filter(pl.col("win_key") == "late_12_25").group_by("round")
        .agg(pl.len().alias("n"), pl.col("mm_wb").sum().alias("mm"), pl.col("mm_wb").mean().round(3).alias("rate"))
        .filter(pl.col("n") >= 20).sort("round"))
P(prof)
P("mismatch rate by round number, early window (key day 01-07), rounds with n>=20:")
prof_e = (O.filter(pl.col("win_key") == "early_01_07").group_by("round")
          .agg(pl.len().alias("n"), pl.col("mm_wb").sum().alias("mm")).filter(pl.col("n") >= 20).sort("round"))
P(prof_e)
P("round 13 mismatch by build_num:")
P(r13.group_by("build").agg(pl.len().alias("n"), pl.col("mm_wb").sum().alias("mm")).sort("build"))
P("round 13 mismatch by format and claim pattern (late window):")
P(r13.filter(pl.col("win_key") == "late_12_25").group_by(["fmt", "cl_pat"]).agg(
    pl.len().alias("n"), pl.col("mm_wb").sum().alias("mm")).sort("n", descending=True))
P("round 9 by format and claim pattern (all windows, pc==20, bomb):")
r9 = O.filter(pl.col("round") == 9)
P(r9.group_by(["win_key", "fmt", "cl_pat"]).agg(pl.len().alias("n"), pl.col("mm_wb").sum().alias("mm"))
  .sort(["win_key", "fmt", "n"], descending=[False, False, True]).head(20))

# 3-way classification for late-window round-13 rows with a known phase-anchored side
P("\n3-way at round 13, late window, rows with known phase-anchored side and bomb outcome:")
t3 = O.filter((pl.col("round") == 13) & (pl.col("win_key") == "late_12_25") & pl.col("s_ph").is_not_null())
t3 = t3.with_columns(
    pl.when((pl.col("winner") == pl.col("bomb")) & (pl.col("s_ph") == pl.col("bomb"))).then(pl.lit("all_agree"))
      .when((pl.col("winner") != pl.col("bomb")) & (pl.col("s_ph") == pl.col("bomb"))).then(pl.lit("winner_field_odd"))
      .when((pl.col("winner") != pl.col("bomb")) & (pl.col("s_ph") == pl.col("winner"))).then(pl.lit("bomb_field_odd"))
      .otherwise(pl.lit("score_side_odd")).alias("cls"))
P(t3.group_by(["fmt", "cls"]).agg(pl.len().alias("n")).sort(["fmt", "cls"]))
P("planter side on round-13 bomb rows (2 = T expected):",
  r13.group_by("planter").agg(pl.len().alias("n")).sort("planter").rows())
P("bomb rows with n_out>1 (both defused and exploded or duplicate):", O.filter(pl.col("n_out") > 1).height)

# ------------------------------------------------------------------ S8 build vs date vs format
P("\n== S8 build_num vs key-path day (matches) and round-13 mismatch by day x build")
mb = H.with_columns(pl.col("key_day").str.slice(-2).alias("dd")).group_by(["dd", "build_num"]).agg(
    pl.len().alias("matches")).sort(["dd", "build_num"])
P("matches by key day x build_num:", mb.rows())
r13_all = D.filter((pl.col("round") == 13) & pl.col("bomb").is_not_null())
r13_all = r13_all.with_columns(pl.col("key_day").str.slice(-2).alias("dd"),
                               (pl.col("bomb") != pl.col("winner")).alias("mm_wb"))
P("round-13 rows with bomb outcome, ALL player counts (dd, build -> n, mismatches):")
P(r13_all.group_by(["dd", "build"]).agg(pl.len().alias("n"), pl.col("mm_wb").sum().alias("mm")).sort(["dd", "build"]))
P("round-13 rows with bomb outcome, pc==20 only:")
P(r13_all.filter(pl.col("pc") == 20).group_by(["dd", "build"]).agg(pl.len().alias("n"),
  pl.col("mm_wb").sum().alias("mm")).sort(["dd", "build"]))
P("bomb-outcome rounds by player_count (all rounds): n, mismatches (winner vs bomb):")
P(D.filter(pl.col("bomb").is_not_null()).with_columns((pl.col("bomb") != pl.col("winner")).alias("mm"))
  .group_by("pc").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("n", descending=True).head(10))

# S2 consistency by build and window
P("\nboundary consistency by build (phase-change boundaries: swap_only vs both; other: noswap_only vs both):")
bc = collections.Counter()
for (k, r), b in bnd.items():
    if b["phs"] is None:
        continue
    cat = ("both" if b["nv"] and b["sv"] else "noswap_only" if b["nv"] else "swap_only" if b["sv"] else "neither")
    bc[(bld[k], "phase_change" if b["phs"] else "no_change", cat)] += 1
for kk, n in sorted(bc.items()):
    P("  ", kk, n)

# winner_team_code vs phase-anchored score side: disagreement by build and round (late and early window)
# ------------------------------------------------------------------ S9 tie cases, totals by build, map
P("\n== S9 checks: tie-case halftime, team totals by build, map")
# (a) matches where phase has a swap at 12 but min-swap DP has none: which side does round 12 get?
tie = [k for k in fe if 12 in ph_pattern.get(k, ()) and 12 not in claim_pat.get(k, ())]
ta = collections.Counter()
for k in tie:
    w12 = win_by_match[k].get(12)
    ph_s = bnd[(k, 12)]["ps"] if bnd[(k, 12)]["pv"] else None
    dp_s = claim_pair_side.get((k, 12))
    ta[(bld[k], "phase_side==winner" if ph_s == w12 else "phase_side!=winner" if ph_s is not None else "phase_side_none",
        "dp_side==winner" if dp_s == w12 else "dp_side!=winner" if dp_s is not None else "dp_none")] += 1
P("tie matches (phase swap at 12, DP no swap at 12): n =", len(tie))
for kk, n in sorted(ta.items()):
    P("  ", kk, n)
# DP-vs-phase round-12 side for all matches where they differ
dif12 = collections.Counter()
for k in fe:
    if (k, 12) in bnd and bnd[(k, 12)]["pv"] and (k, 12) in claim_pair_side:
        dif12[(bnd[(k, 12)]["ps"] != claim_pair_side[(k, 12)])] += 1
P("round-12 side: phase-anchored vs DP, differ count (True) / agree (False):", dict(dif12))

# (b) MR8 round-9 rows (first round of second half in 8-round halves)
m8 = D.filter((pl.col("fmt") == "MR8") & (pl.col("round") == 9) & pl.col("s_ph").is_not_null())
P("MR8 round-9 rows with phase-anchored side, by build: n, disagree with winner:")
P(m8.group_by("build").agg(pl.len().alias("n"), pl.col("ph_ne_w").sum().alias("dis")).sort("build"))
m8b = D.filter((pl.col("fmt") == "MR8") & (pl.col("round") == 8) & pl.col("s_ph").is_not_null())
P("MR8 round-8 rows (last round of first half), by build: n, disagree:")
P(m8b.group_by("build").agg(pl.len().alias("n"), pl.col("ph_ne_w").sum().alias("dis")).sort("build"))

# (c) team totals: winners from round_end codes vs score-based winners (phase-anchored, RE fallback)
P("\nTeam totals vs header finals, by build (accept either orientation):")
ws_by_match = collections.defaultdict(dict)
for rr in D.select(["match_key", "round", "s_ph", "winner"]).iter_rows(named=True):
    ws_by_match[rr["match_key"]][rr["round"]] = rr["s_ph"] if rr["s_ph"] is not None else rr["winner"]


def ok_totals(wd, bounds, tf, cf):
    A, B = totals(wd, bounds)
    return (A, B) == (tf, cf) or (A, B) == (cf, tf)


res = collections.Counter()
for k in re_rounds:
    if k not in fe:
        continue
    rs = sorted(re_rounds[k])
    if len(rs) == 0:
        continue
    maxr_ = rs[-1]
    gaps_beyond_1 = [r for r in range(2, maxr_ + 1) if r not in re_rounds[k]]
    if gaps_beyond_1:
        res[(bld[k], "skip_gaps_in_round_end")] += 1
        continue
    tf, cf = fin_map[k]
    bounds = ph_pattern.get(k, ())
    if 1 in re_rounds[k]:
        wd_re = {r: w for r, w in win_by_match[k].items()}
        wd_sc = {r: (ws_by_match[k].get(r) if ws_by_match[k].get(r) is not None else win_by_match[k][r])
                 for r in win_by_match[k]}
        if sum((tf, cf)) != len(rs):
            res[(bld[k], "full_but_sum_mismatch")] += 1
            continue
        res[(bld[k], "full", "RE_winners", ok_totals(wd_re, bounds, tf, cf))] += 1
        res[(bld[k], "full", "score_winners", ok_totals(wd_sc, bounds, tf, cf))] += 1
    else:
        # round 1 absent from both sources: its winner is unknown, so allow +1 to either team
        wd_re = {r: w for r, w in win_by_match[k].items()}
        wd_sc = {r: (ws_by_match[k].get(r) if ws_by_match[k].get(r) is not None else win_by_match[k][r])
                 for r in win_by_match[k]}
        ok_pair = []
        for wd in (wd_re, wd_sc):
            A, B = totals(wd, bounds)
            ok = ((tf - A, cf - B) in [(1, 0), (0, 1)]) or ((cf - A, tf - B) in [(1, 0), (0, 1)])
            ok_pair.append(ok)
        res[(bld[k], "round1_absent", "RE_winners", ok_pair[0])] += 1
        res[(bld[k], "round1_absent", "score_winners", ok_pair[1])] += 1
for kk, n in sorted(res.items(), key=lambda x: str(x[0])):
    P("  ", kk, n)

P("\nround-13 disagreement in build 9961 by map (dis / rows with phase side):")
mp = (D.filter((pl.col("round") == 13) & (pl.col("build") == 9961) & pl.col("s_ph").is_not_null())
        .join(H.select(["key", "map_name"]).rename({"key": "match_key"}), on="match_key", how="left")
        .group_by("map_name").agg(pl.len().alias("n"), pl.col("ph_ne_w").sum().alias("dis")).sort("n", descending=True))
P(mp)
P("map distribution of build 9842 vs 9961 (matches):",
  H.group_by(["build_num", "map_name"]).len().sort("len", descending=True).head(14).rows())

# (d) rounds missing from round_end beyond round 1, by build
mg = collections.Counter()
for k, s in re_rounds.items():
    interior = [r for r in range(2, max(s) + 1) if r not in s]
    mg[(bld[k], "interior_missing_rounds_beyond_1", len(interior))] += 1
P("missing round_end rounds beyond round 1 (interior), by build (count of matches):")
P(sorted([(kk[0], kk[2], n) for kk, n in mg.items() if kk[2] <= 3]))

# ------------------------------------------------------------------ S10 winner_team_code vs phase-anchored score side: disagreement by build and round (rounds with >=50 rows)
P("\nwinner_team_code vs phase-anchored score side: disagreements / rows, by build and round (rounds with >=50 rows):")
dis = (D.filter(pl.col("s_ph").is_not_null())
         .with_columns(pl.col("key_day").str.slice(-2).cast(pl.Int64).alias("ddi"))
         .group_by(["build", "round"]).agg(pl.len().alias("n"), pl.col("ph_ne_w").sum().alias("dis"))
         .filter(pl.col("n") >= 50).sort(["build", "round"]))
P(dis.with_columns((pl.col("dis") / pl.col("n")).round(4).alias("rate")))
P("late-window (dd>=12) disagreements by build and format:")
P(D.filter(pl.col("s_ph").is_not_null() & (pl.col("win_key") == "late_12_25")).group_by(["build", "fmt"]).agg(
    pl.len().alias("n"), pl.col("ph_ne_w").sum().alias("dis")).sort(["build", "fmt"]))
P("early-window (dd<=07) disagreements by build and format:")
P(D.filter(pl.col("s_ph").is_not_null() & (pl.col("win_key") == "early_01_07")).group_by(["build", "fmt"]).agg(
    pl.len().alias("n"), pl.col("ph_ne_w").sum().alias("dis")).sort(["build", "fmt"]))

# ------------------------------------------------------------------ S11 swap-parity test with trustworthy winners
P("\n== S11 team-identity test: which swap placement reproduces header finals?")
P("  build 9842 uses round_end codes (they agree with score sides everywhere); build 9961 uses score-side winners.")
trail = collections.Counter()
for k in re_rounds:
    trail[(bld[k], sum(fin_map[k]) - max(re_rounds[k]))] += 1
P("final total minus max round_end round, by build (top 8 per build):")
for b_ in (9842, 9961):
    P("  ", b_, sorted([(x[1], n) for x, n in trail.items() if x[0] == b_], key=lambda z: -z[1])[:8])
schemes = {"phase_swaps": lambda k: ph_pattern.get(k, ()),
           "dp_claim_swaps": lambda k: claim_pat.get(k, ()),
           "no_swaps": lambda k: ()}
r11 = collections.Counter()
for k in re_rounds:
    if k not in fe:
        continue
    rs_ = sorted(re_rounds[k])
    maxr_ = rs_[-1]
    tf, cf = fin_map[k]
    b_ = bld[k]
    sub = "phase_eq_dp" if tuple(ph_pattern.get(k, ())) == tuple(claim_pat.get(k, ())) else "phase_ne_dp"
    if b_ == 9842:
        if 1 in re_rounds[k] or tf + cf != maxr_:
            continue
        wd = {r: win_by_match[k][r] for r in win_by_match[k]}
        for name, fn in schemes.items():
            A, B = totals(wd, fn(k))
            ok = ((tf - A, cf - B) in [(1, 0), (0, 1)]) or ((cf - A, tf - B) in [(1, 0), (0, 1)])
            r11[(b_, "round1_only_missing", sub, name, ok)] += 1
    elif b_ == 9961:
        if 1 not in re_rounds[k] or tf + cf != maxr_:
            continue
        wd = {r: (ws_by_match[k].get(r) if ws_by_match[k].get(r) is not None else win_by_match[k][r])
              for r in win_by_match[k]}
        for name, fn in schemes.items():
            A, B = totals(wd, fn(k))
            r11[(b_, "complete", sub, name, (A, B) == (tf, cf) or (A, B) == (cf, tf))] += 1
P("  counts keyed by (build, coverage, phase-vs-DP subset, parity scheme, matches finals?):")
for kk, n in sorted(r11.items(), key=lambda x: str(x[0])):
    P("   ", kk, n)

# ------------------------------------------------------------------ S12 boundary-level: where does min-swap DP disagree with phase?
P("\n== S12 boundary-level swap flags: phase field vs min-swap DP (all matches)")
bl = collections.Counter()
for (k, r), b in bnd.items():
    if not b["phs"]:
        continue
    dp_flag = claim_pair_swap_literal.get((k, r))
    cat = ("both_valid" if b["nv"] and b["sv"] else "swap_only" if b["sv"] else "noswap_only" if b["nv"] else "neither")
    bl[(r, cat, "DP_swap" if dp_flag else "DP_noswap" if dp_flag is not None else "DP_missing")] += 1
for kk, n in sorted(bl.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
    P("  phase-change boundary after round", kk[0], "| score:", kk[1], "| DP:", kk[2], "->", n)
nb_phase = sum(1 for b in bnd.values() if b["phs"])
nb_dp_miss = sum(n for kk, n in bl.items() if kk[2] == "DP_noswap")
P("phase-change boundaries:", nb_phase, "| DP assigns no swap at these:", nb_dp_miss)
nb_nophase_dp_swap = sum(1 for (k, r), b in bnd.items() if b["phs"] is False and claim_pair_swap_literal.get((k, r)))
P("non-phase-change boundaries where DP assigns a swap:", nb_nophase_dp_swap)

# S12b: round-level side vs round_end winner at ambiguous phase-change boundaries
P("\nAt ambiguous phase-change boundaries: round side vs round_end winner (round = boundary round)")
amb = collections.Counter()
for (k, r), b in bnd.items():
    if not (b["phs"] and b["nv"] and b["sv"]):
        continue
    w = win_by_match[k].get(r)
    if w is None:
        amb[(r, "no_winner")] += 1
        continue
    amb[(r, "phase_side_ok" if b["ps"] == w else "phase_side_BAD",
         "dp_side_ok" if claim_pair_side.get((k, r)) == w else "dp_side_BAD")] += 1
for kk, n in sorted(amb.items(), key=lambda x: str(x[0])):
    P("  ", kk, n)
