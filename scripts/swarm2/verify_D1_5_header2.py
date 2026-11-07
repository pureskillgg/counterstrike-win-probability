"""verify_D1_5 header test v2: unordered team-total pairs (removes the orientation/parity question).

For each complete 2024 match (A6, max_rounds 24, A8), team totals from round_end winners with the
swap rule (A5) are compared as an UNORDERED pair with the header starter totals {tf, cf}.
Candidate corrections:
  raw       round_end winners as recorded
  fix13     round 13 winner := round_state score side (A4), where non-null (the claim's fix)
  fixall    every round with a non-null score side that disagrees with the winner := score side
  fixbomb13 round 13 winner := bomb outcome where a bomb outcome exists, else raw
Stratified by round-13 status: bomb mismatch / non-bomb disagreement / agreement / null.
Also a single-round flip search for matches that fail under fix13.
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
DAYS_HI = {f"{d:02d}" for d in range(12, 26)}
MR = 24  # A8


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def fl(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


def rd(ch, cols=None):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fl(ch)], how="vertical_relaxed")


def swap_expected(r, mr):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


def side_a(r, mr=MR):
    n = sum(1 for s in range(1, r) if swap_expected(s, mr))
    return 2 if n % 2 == 0 else 3


def pair_totals(w, n):
    a = sum(1 for r in range(1, n + 1) if w[r] == side_a(r))
    return a, n - a


def unordered_ok(a, b, tf, cf):
    return sorted((a, b)) == sorted((tf, cf))


# ----------------------------------------------------------------- load
RE = rd("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(["match_key", "round"], keep="first")
BS = rd("bomb_state", ["match_key", "round", "event_type"])
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
HD0 = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(str(BASE / "header" / "dataframe_*")))], how="vertical_relaxed")
HD = HD0.select([pl.col("key").alias("match_key"), "t_starters_score_final", "ct_starters_score_final"]).unique("match_key")

outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
bomb_side = outc.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side"))
bomb_side = {(m, int(r)): int(b) for m, r, b in bomb_side.select(["match_key", "round", "bomb_side"]).iter_rows()}

sc = (RS.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("t_score").sort_by("tick").first().alias("t0"), pl.col("ct_score").sort_by("tick").first().alias("c0"),
    pl.col("t_score").sort_by("tick").last().alias("t1"), pl.col("ct_score").sort_by("tick").last().alias("c1")))
sc = sc.with_columns(pl.when((pl.col("t1") - pl.col("t0")) == 1).then(2)
                     .when((pl.col("c1") - pl.col("c0")) == 1).then(3).otherwise(None).cast(pl.Int64).alias("score_side"))
score_side = {(m, int(r)): int(s) for m, r, s in sc.select(["match_key", "round", "score_side"]).iter_rows() if s is not None}

fin = (RS.sort(["match_key", "tick"]).group_by("match_key").agg(
    pl.col("t_score").last().alias("t_fin"), pl.col("ct_score").last().alias("c_fin")))
fin = {m: (int(t), int(c)) for m, t, c in fin.select(["match_key", "t_fin", "c_fin"]).iter_rows()}

winners = collections.defaultdict(dict)
pcount = {}
for m, r, w, pc in RE.select(["match_key", "round", "winner_team_code", "player_count"]).iter_rows():
    winners[m][int(r)] = int(w)
    pcount[(m, int(r))] = pc
hdrs = {m: (int(tf), int(cf)) for m, tf, cf in HD.select(["match_key", "t_starters_score_final",
                                                          "ct_starters_score_final"]).iter_rows()}

# ----------------------------------------------------------------- complete matches
complete = []
for m, (tf, cf) in hdrs.items():
    w = winners.get(m)
    if not w:
        continue
    n = len(w)
    if sorted(w) != list(range(1, n + 1)) or n != tf + cf:
        continue
    complete.append((m, tf, cf, n))
print("complete matches (A6, contiguous, n == header total):", len(complete), "of", len(hdrs))
print("regulation-only (n<=24):", sum(1 for c in complete if c[3] <= 24), " overtime (n>24):", sum(1 for c in complete if c[3] > 24))


def day_of(m):
    return m.split("/")[2] if False else m[len("csds/2024/02/"):len("csds/2024/02/") + 2]


# ----------------------------------------------------------------- stratified test
rows = []
for m, tf, cf, n in complete:
    w = dict(winners[m])
    r13 = w.get(13)
    ss13 = score_side.get((m, 13))
    bs13 = bomb_side.get((m, 13))
    if bs13 is None:
        st13 = "no_bomb"
    else:
        st13 = "bomb_mismatch" if bs13 != r13 else "bomb_agree"
    if ss13 is None:
        sst13 = "score_null"
    else:
        sst13 = "score_disagree" if ss13 != r13 else "score_agree"
    # raw
    a, b = pair_totals(w, n)
    raw_ok = unordered_ok(a, b, tf, cf)
    # fix13
    w13 = dict(w)
    if ss13 is not None:
        w13[13] = ss13
    a13, b13 = pair_totals(w13, n)
    fix13_ok = unordered_ok(a13, b13, tf, cf)
    # fixall: every round with score side that disagrees
    wall = dict(w)
    for r in range(1, n + 1):
        s = score_side.get((m, r))
        if s is not None:
            wall[r] = s
    aa, ab = pair_totals(wall, n)
    fixall_ok = unordered_ok(aa, ab, tf, cf)
    # fixbomb13
    wb = dict(w)
    if bs13 is not None:
        wb[13] = bs13
    ab_, bb_ = pair_totals(wb, n)
    fixbomb_ok = unordered_ok(ab_, bb_, tf, cf)
    # round_state final totals (parity-aware, team A orientation handled by unordered test)
    t_fin, c_fin = fin[m]
    end_side_a = side_a(n + 1)
    rs_a = t_fin if end_side_a == 2 else c_fin
    rs_b = c_fin if end_side_a == 2 else t_fin
    rs_ok = unordered_ok(rs_a, rs_b, tf, cf)
    # number of score-disagreeing rounds (non-null) other than 13
    other_dis = sum(1 for r in range(1, n + 1) if r != 13 and (m, r) in score_side and score_side[(m, r)] != w[r])
    rows.append({"m": m, "day": m[len("csds/2024/02/"):len("csds/2024/02/") + 2], "n": n,
                 "st13": st13, "sst13": sst13, "raw": raw_ok, "fix13": fix13_ok, "fixall": fixall_ok,
                 "fixbomb13": fixbomb_ok, "rs": rs_ok, "other_dis": other_dis})

def tally(keyfn, label):
    c = collections.Counter()
    t = collections.Counter()
    for r in rows:
        k = keyfn(r)
        t[k] += 1
        for name in ("raw", "fix13", "fixall", "fixbomb13", "rs"):
            if r[name]:
                c[(k, name)] += 1
    print(f"\n{label}")
    print(f"  {'group':55s} {'n':>6s} {'raw':>6s} {'fix13':>6s} {'fixall':>7s} {'fixbomb13':>10s} {'rs_final':>9s}")
    for k in sorted(t, key=lambda x: str(x)):
        print(f"  {str(k):55s} {t[k]:6d} {c[(k,'raw')]:6d} {c[(k,'fix13')]:6d} {c[(k,'fixall')]:7d} "
              f"{c[(k,'fixbomb13')]:10d} {c[(k,'rs')]:9d}")


hdr("H1 unordered header pass rate by round-13 status (count of matches passing each candidate)")
tally(lambda r: r["st13"], "round-13 bomb status (bomb_mismatch / bomb_agree / no_bomb)")
tally(lambda r: r["sst13"], "round-13 score-side status (score_disagree / score_agree / score_null)")
tally(lambda r: (r["st13"], r["sst13"]), "combined round-13 status (bomb status, score status)")
tally(lambda r: ("day", "d12-25" if r["day"] in DAYS_HI else "d01-07"), "day group")

hdr("H2 matches with a non-round-13 score-vs-winner disagreement (other_dis > 0) vs none")
tally(lambda r: ("other_dis>0" if r["other_dis"] > 0 else "other_dis=0"), "other-round score disagreements present")

hdr("H3 single-round flip search for matches still failing under fix13 (which round, if any, repairs the totals)")
flip_hits = collections.Counter()
still = 0
for m, tf, cf, n in complete:
    w = dict(winners[m])
    ss13 = score_side.get((m, 13))
    if ss13 is not None:
        w[13] = ss13
    a, b = pair_totals(w, n)
    if unordered_ok(a, b, tf, cf):
        continue
    still += 1
    fixed_by = []
    for r in range(1, n + 1):
        w2 = dict(w)
        w2[r] = 5 - w2[r]  # flip 2 <-> 3
        a2, b2 = pair_totals(w2, n)
        if unordered_ok(a2, b2, tf, cf):
            fixed_by.append(r)
    if len(fixed_by) == 0:
        flip_hits["no_single_flip"] += 1
    else:
        for r in fixed_by:
            flip_hits[f"round {r}"] += 1
print("matches still failing under fix13:", still)
print("single-round flip repairs (a match counted once per repairing round):",
      dict(sorted(flip_hits.items(), key=lambda kv: (-kv[1]))[:20]))

hdr("H4 summary counts")
print("total complete:", len(rows))
print("raw pass:", sum(r["raw"] for r in rows), " fix13 pass:", sum(r["fix13"] for r in rows),
      " fixall pass:", sum(r["fixall"] for r in rows), " fixbomb13 pass:", sum(r["fixbomb13"] for r in rows),
      " rs_final pass:", sum(r["rs"] for r in rows))
