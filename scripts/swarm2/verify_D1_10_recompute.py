"""verify_D1_10_recompute.py -- independent recomputation of claim D1-10 (swarm2 verifier).

Claim under test (2026 tome): the score side derived from round_state (first vs last tick, claim method)
has no disagreement with round_end.winner_team_code at rounds 9, 13, 25, 28; 100 disagreements in
total for player_count == 10 (81 at round 27, 12 at 33, 5 at 39, 2 at 45). Claimed verdict: "supported".

This script:
  1. Replicates the claim's numbers with its own code (claim method, first vs last round_state tick).
  2. Tests the strongest alternative explanation: the round_state side is wrong (not winner_team_code)
     by comparing against two channels that do not use round_state: score_update (old/new score events)
     and bomb_state (bomb_exploded -> T, bomb_defused -> CT), plus round_end win_reason consistency.
  3. Checks denominators (rounds dropped for a null score side), round gaps, and clustering by match and day.

ASSUMPTIONS (all labelled in output where used):
  A1 team codes: 2 = T, 3 = CT (as in the claim and round_end win_reason table).
  A2 score side from round_state: T if t_score rises by exactly 1 from first to last tick, CT if ct_score does (claim rule).
  A3 score_update: a round's side is the team_code of its single +1 score_update row (exactly one such row required).
  A4 bomb side: round has bomb_exploded and no bomb_defused -> T; bomb_defused and no bomb_exploded -> CT.
  A5 2026 tome is the 2026-10-01_to_08 tome; the match date is taken from the match_key path csds/YYYY/MM/DD/.
Read-only. Only stdout is produced; the runner redirects it to logs\\swarm2\\verify_D1_10.log.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(24)
pl.Config.set_fmt_str_lengths(40)
pl.Config.set_tbl_width_chars(220)

ROOT = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
CLAIM_ROUNDS = [9, 13, 25, 27, 28, 33, 39, 45]


def paths(ch):
    p = sorted(glob.glob(str(ROOT / f"basic_{ch}*" / "dataframe_*")))
    if not p:
        raise SystemExit(f"no parquet files for channel {ch}")
    return p


def load(ch, cols):
    # Column projection; each channel in this tome is a single dataframe file.
    return pl.scan_parquet(paths(ch)).select(cols).collect()


print("python", sys.version.split()[0], "polars", pl.__version__)
print("root", ROOT)

# ---------- round_end (winner label under test) ----------
RE = load("round_end", ["match_key", "round", "winner_team_code", "win_reason_code", "win_reason_message", "player_count"])
n_re = RE.height
n_re_uniq = RE.unique(["match_key", "round"]).height
print(f"\n[1] round_end rows={n_re} unique(match_key,round)={n_re_uniq} duplicate_rows={n_re - n_re_uniq}")
print("    matches:", RE["match_key"].n_unique(), " max round:", RE["round"].max())
print("    player_count dist (top 8):", RE.group_by("player_count").len().sort("len", descending=True).head(8).rows())
print("    winner_team_code dist:", RE.group_by("winner_team_code").len().sort("winner_team_code").rows())

# Internal consistency: does each win_reason_code map to exactly one winner_team_code?
xt = RE.group_by(["win_reason_code", "win_reason_message", "winner_team_code"]).len().sort(["win_reason_code", "winner_team_code"])
multi = RE.group_by("win_reason_code").agg(pl.col("winner_team_code").n_unique().alias("k")).filter(pl.col("k") > 1)
print("\n[1b] win_reason -> winner crosstab:")
for r in xt.iter_rows(named=True):
    print(f"    reason {r['win_reason_code']:>3} {r['win_reason_message'][:34]:<34} winner {r['winner_team_code']} n={r['len']}")
print("    reason codes mapping to >1 winner:", multi.height, "(0 means winner_team_code agrees with the game's win message everywhere)")

RE = RE.unique(["match_key", "round"])
RE = RE.with_columns(pl.col("match_key").str.extract(r"^csds/\d{4}/\d{2}/(\d{2})/", 1).cast(pl.Int64).alias("day"))

# ---------- claim method: round_state first/last tick ----------
RS = load("round_state", ["match_key", "round", "tick", "t_score", "ct_score"])
print(f"\n[2] round_state rows={RS.height}")
g = (RS.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
        pl.col("t_score").sort_by("tick").first().alias("t_b"),
        pl.col("ct_score").sort_by("tick").first().alias("ct_b"),
        pl.col("t_score").sort_by("tick").last().alias("t_a"),
        pl.col("ct_score").sort_by("tick").last().alias("ct_a"),
        (pl.col("tick") == pl.col("tick").min()).sum().alias("n_at_min_tick"),
        (pl.col("tick") == pl.col("tick").max()).sum().alias("n_at_max_tick"),
        pl.len().alias("n_state")))
g = g.with_columns(
    (pl.col("t_a") - pl.col("t_b")).alias("t_d"),
    (pl.col("ct_a") - pl.col("ct_b")).alias("ct_d"))
g = g.with_columns(
    pl.when(pl.col("t_d") == 1).then(pl.lit(2)).when(pl.col("ct_d") == 1).then(pl.lit(3)).otherwise(pl.lit(None, dtype=pl.Int64)).alias("claim_side"))
print("    (round, match) groups:", g.height, " groups with >1 row at min tick:", g.filter(pl.col("n_at_min_tick") > 1).height,
      " at max tick:", g.filter(pl.col("n_at_max_tick") > 1).height)
print("    (t_d, ct_d) combos for all groups (top 10):")
for r in g.group_by(["t_d", "ct_d"]).len().sort("len", descending=True).head(10).iter_rows(named=True):
    print(f"      t_d={r['t_d']:>3} ct_d={r['ct_d']:>3} n={r['len']}")

J = RE.join(g.select(["match_key", "round", "claim_side", "t_d", "ct_d", "n_state"]), on=["match_key", "round"], how="left")
print("    round_end rows with no round_state group:", J.filter(pl.col("n_state").is_null()).height)

# ---------- independent channel 1: score_update ----------
SU = load("score_update", ["match_key", "round", "tick", "team_code", "old_score", "new_score"])
print(f"\n[3] score_update rows={SU.height}")
inc = SU.filter((pl.col("new_score") - pl.col("old_score")) == 1)
su_grp = inc.group_by(["match_key", "round"]).agg(
    pl.len().alias("su_n_inc"),
    pl.col("team_code").min().alias("su_tc_min"),
    pl.col("team_code").max().alias("su_tc_max"))
su_grp = su_grp.with_columns(
    pl.when((pl.col("su_n_inc") == 1)).then(pl.col("su_tc_min")).otherwise(pl.lit(None, dtype=pl.Int64)).alias("su_side"))
print("    rounds with +1 score_update:", su_grp.height,
      " with exactly one:", su_grp.filter(pl.col("su_n_inc") == 1).height,
      " with >1:", su_grp.filter(pl.col("su_n_inc") > 1).height)
J = J.join(su_grp.select(["match_key", "round", "su_side", "su_n_inc"]), on=["match_key", "round"], how="left")

# ---------- independent channel 2: bomb_state ----------
BS = load("bomb_state", ["match_key", "round", "event_type"])
bomb = BS.group_by(["match_key", "round"]).agg(
    (pl.col("event_type") == "bomb_exploded").any().alias("b_exp"),
    (pl.col("event_type") == "bomb_defused").any().alias("b_def"))
bomb = bomb.with_columns(
    pl.when(pl.col("b_exp") & ~pl.col("b_def")).then(pl.lit(2))
      .when(pl.col("b_def") & ~pl.col("b_exp")).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("bomb_side"))
print(f"\n[4] bomb_state rows={BS.height}; rounds with exploded or defused flag: {bomb.height}; "
      f"rounds with both flags: {bomb.filter(pl.col('b_exp') & pl.col('b_def')).height}")
J = J.join(bomb.select(["match_key", "round", "bomb_side"]), on=["match_key", "round"], how="left")

# ---------- rounds with a claim side: reproduce the headline numbers ----------
J = J.with_columns(
    (pl.col("claim_side").is_not_null() & (pl.col("claim_side") != pl.col("winner_team_code"))).alias("claim_dis"),
    (pl.col("su_side").is_not_null() & (pl.col("su_side") != pl.col("winner_team_code"))).alias("su_dis"),
    (pl.col("bomb_side").is_not_null() & (pl.col("bomb_side") != pl.col("winner_team_code"))).alias("bomb_dis"),
)

print("\n[5] HEADLINE (claim method, recomputed)")
for label, sub in [("player_count == 10", J.filter(pl.col("player_count") == 10)), ("all player_count", J)]:
    with_side = sub.filter(pl.col("claim_side").is_not_null())
    n_dis = int(with_side.filter(pl.col("claim_dis"))["claim_dis"].sum()) if with_side.height else 0
    print(f"    {label}: round_end rows={sub.height}, rounds with claim side={with_side.height}, "
          f"claim-side null={sub.height - with_side.height}, disagreements={n_dis}")

# per-round table (player_count 10 and all) for the rounds named in the claim
def per_round(sub):
    return (sub.group_by("round").agg(
        pl.len().alias("n_end"),
        pl.col("claim_side").is_not_null().sum().alias("n_claim"),
        pl.col("claim_dis").sum().alias("claim_dis"),
        pl.col("su_side").is_not_null().sum().alias("n_su"),
        pl.col("su_dis").sum().alias("su_dis"),
        pl.col("bomb_side").is_not_null().sum().alias("n_bomb"),
        pl.col("bomb_dis").sum().alias("bomb_dis"),
    ).sort("round"))

P10 = J.filter(pl.col("player_count") == 10)
t10 = per_round(P10)
tall = per_round(J)
print("\n[6] per-round table, player_count == 10 (n_end=round_end rows, n_claim=rounds with claim side, claim_dis=disagreements)")
hdr = f"    {'round':>5} {'n_end':>6} {'n_claim':>7} {'claim_dis':>9} {'n_su':>6} {'su_dis':>6} {'n_bomb':>6} {'bomb_dis':>8}"
print(hdr)
for r in t10.iter_rows(named=True):
    if r["round"] <= 48 and (r["claim_dis"] or r["su_dis"] or r["bomb_dis"] or r["round"] in CLAIM_ROUNDS or r["round"] <= 2):
        print(f"    {r['round']:>5} {r['n_end']:>6} {r['n_claim']:>7} {r['claim_dis']:>9} {r['n_su']:>6} {r['su_dis']:>6} {r['n_bomb']:>6} {r['bomb_dis']:>8}")

print("\n[6b] per-round table, all player_count (only rows with any disagreement or a claim round)")
print(hdr)
for r in tall.iter_rows(named=True):
    if r["round"] <= 48 and (r["claim_dis"] or r["su_dis"] or r["bomb_dis"] or r["round"] in CLAIM_ROUNDS):
        print(f"    {r['round']:>5} {r['n_end']:>6} {r['n_claim']:>7} {r['claim_dis']:>9} {r['n_su']:>6} {r['su_dis']:>6} {r['n_bomb']:>6} {r['bomb_dis']:>8}")

# ---------- denominators: rounds dropped for null claim side (player_count 10) ----------
print("\n[7] denominators, player_count == 10: round_end rows vs rounds kept by claim method, rounds 22-30")
for r in t10.iter_rows(named=True):
    if 22 <= r["round"] <= 30:
        print(f"    round {r['round']:>2}: round_end={r['n_end']:>5} kept_by_claim={r['n_claim']:>5} dropped={r['n_end'] - r['n_claim']:>5} "
              f"kept_by_score_update={r['n_su']:>5}")
null_total = P10.filter(pl.col("claim_side").is_null()).height
print(f"    total claim-null rounds at pc10: {null_total} of {P10.height}")
top_null = (P10.filter(pl.col("claim_side").is_null()).group_by("round").len().sort("len", descending=True).head(10))
print("    claim-null rounds by round number (top 10):", top_null.rows())

# ---------- round gaps: does round r exist in matches that have round r+1? ----------
pairs = RE.select(["match_key", "round"])
# gap = (match, r) present while (match, r-1) absent, r >= 2
prev = pairs.filter(pl.col("round") >= 2).with_columns((pl.col("round") - 1).alias("round_prev"))
gaps = prev.join(pairs.rename({"round": "round_prev"}), on=["match_key", "round_prev"], how="anti")
print(f"\n[8] round gaps: (match, r) present while (match, r-1) absent: {gaps.height}")
print("    gaps by round r (top 12 by r):", gaps.group_by("round").len().sort("round").head(12).rows())
mx = RE.group_by("match_key").agg(pl.col("round").max().alias("mx"), pl.len().alias("n"))
print(f"    matches where max round != number of rounds (missing rounds somewhere): {mx.filter(pl.col('mx') != pl.col('n')).height} of {mx.height}")

# ---------- clustering: which matches and days carry the claim disagreements (player_count 10) ----------
d10 = P10.filter(pl.col("claim_dis"))
print(f"\n[9] claim disagreements at pc10: {d10.height} rows in {d10['match_key'].n_unique()} matches")
per_match = d10.group_by("match_key").agg(pl.len().alias("n_dis")).sort("n_dis", descending=True)
print("    disagreements per match -> number of matches:", per_match.group_by("n_dis").agg(pl.len().alias("n_matches")).sort("n_dis").rows())
print("    flagged rounds per match (top 5):", per_match.head(5).rows())
print("    disagreements by day (match-date day of month):", d10.group_by("day").len().sort("day").rows())

# ---------- which source is wrong at the flagged rounds? ----------
print("\n[10] for claim-disagreement rows (all pc): who agrees with winner_team_code?")
dd = J.filter(pl.col("claim_dis"))
print(f"    rows: {dd.height}")
print(f"    score_update side present: {dd.filter(pl.col('su_side').is_not_null()).height};"
      f" score_update side == winner_team_code: {dd.filter(pl.col('su_side') == pl.col('winner_team_code')).height};"
      f" score_update side == claim side: {dd.filter(pl.col('su_side') == pl.col('claim_side')).height}")
print(f"    bomb side present: {dd.filter(pl.col('bomb_side').is_not_null()).height};"
      f" bomb side == winner_team_code: {dd.filter(pl.col('bomb_side') == pl.col('winner_team_code')).height};"
      f" bomb side == claim side: {dd.filter(pl.col('bomb_side') == pl.col('claim_side')).height}")
print("    win_reason_code of these rows:", dd.group_by("win_reason_code").len().sort("win_reason_code").rows())
print("    claim side of these rows:", dd.group_by("claim_side").len().rows(), " winner:", dd.group_by("winner_team_code").len().rows())

# ---------- the claim's non-disagreement rounds: are they really checkable? ----------
print("\n[11] rounds named in the claim, player_count 10: rounds where claim side, score_update side and bomb side are all available")
for r in CLAIM_ROUNDS:
    sub = P10.filter(pl.col("round") == r)
    all3 = sub.filter(pl.col("claim_side").is_not_null() & pl.col("su_side").is_not_null()).height
    bomb_av = sub.filter(pl.col("bomb_side").is_not_null()).height
    su_dis = sub.filter(pl.col("su_dis")).height
    cl_dis = sub.filter(pl.col("claim_dis")).height
    print(f"    round {r:>2}: round_end={sub.height:>5} claim+su both={all3:>5} bomb-labelled={bomb_av:>5} "
          f"claim_dis={cl_dis:>3} su_dis={su_dis:>3}")

# ---------- summary numbers for the verdict ----------
print("\n[12] SUMMARY (player_count 10)")
s10 = P10.filter(pl.col("claim_side").is_not_null())
print(f"    claim method: rounds with side={s10.height}, disagreements={s10.filter(pl.col('claim_dis')).height}")
su10 = P10.filter(pl.col("su_side").is_not_null())
print(f"    score_update method: rounds with side={su10.height}, disagreements={su10.filter(pl.col('su_dis')).height}")
bm10 = P10.filter(pl.col("bomb_side").is_not_null())
print(f"    bomb method: rounds with side={bm10.height}, disagreements={bm10.filter(pl.col('bomb_dis')).height}")
print("    claim vs score_update, both present, disagreement on the two methods:",
      P10.filter(pl.col("claim_side").is_not_null() & pl.col("su_side").is_not_null() & (pl.col("claim_side") != pl.col("su_side"))).height)
print("DONE")
