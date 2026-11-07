"""verify_D1_10_mechanism.py -- mechanism test for claim D1-10 (swarm2 verifier).

Usage:  python -I -X utf8 verify_D1_10_mechanism.py 2026
        python -I -X utf8 verify_D1_10_mechanism.py 2024

Question: is the claim's round_state score side (first vs last tick, claim rule) a valid yardstick?
For each (match, round) the round_state first tick (t_b, ct_b) and last tick (t_a, ct_a) are classified
against round_end.winner_team_code (w):
  straight : t_a == t_b + [w==T]  and ct_a == ct_b + [w==CT]   score columns stay with the same side
  swapped  : t_a == ct_b + [w==CT] and ct_a == t_b + [w==T]   score columns follow the teams across a side swap
  both / neither
Independent references (no round_state use): score_update single +1 row (team_code) and bomb_state
(bomb_exploded -> T, bomb_defused -> CT).

ASSUMPTIONS:
  A1 team codes 2 = T, 3 = CT (as in the claim and round_end win_reason table).
  A2 the claim rule is applied unchanged: side T if t_a - t_b == 1, CT if ct_a - ct_b == 1, else null.
  A3 score_update side requires exactly one +1 row in the round.
  A4 bomb side: exploded only -> T, defused only -> CT.
  A5 player_count 10 is the claim's subset; all player_count rows are reported for comparison.
Read-only. Stdout only.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(26)
pl.Config.set_fmt_str_lengths(40)
pl.Config.set_tbl_width_chars(230)

TOMES = {
    "2026": Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"),
    "2024": Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"),
}
label = sys.argv[1] if len(sys.argv) > 1 else "2026"
ROOT = TOMES[label]
print(f"\n==================== TOME {label} : {ROOT} ====================")


KEYS = {"match_key": pl.String, "round": pl.Int64}


def load(ch, cols, optional=False):
    p = sorted(glob.glob(str(ROOT / f"basic_{ch}*" / "dataframe_*")))
    if not p:
        if optional:
            print(f"  channel {ch}: ABSENT in this tome (treated as no information)")
            return None
        raise SystemExit(f"no parquet files for channel {ch}")
    lf = pl.scan_parquet(p)
    have = lf.collect_schema().names()
    missing = [c for c in cols if c not in have]
    if missing:
        raise SystemExit(f"channel {ch} lacks columns {missing}; has {have}")
    print(f"  channel {ch}: {len(p)} file(s)")
    out = lf.select(cols).collect()
    casts = [pl.col(c).cast(t) for c, t in KEYS.items() if c in cols]
    if "tick" in cols:
        casts.append(pl.col("tick").cast(pl.Int64))
    if "t_score" in cols:
        casts += [pl.col("t_score").cast(pl.Int64), pl.col("ct_score").cast(pl.Int64)]
    return out.with_columns(casts)


RE = load("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(["match_key", "round"])
print(f"  round_end unique rounds: {RE.height}; matches: {RE['match_key'].n_unique()}")
print("  player_count (top 6):", RE.group_by("player_count").len().sort("len", descending=True).head(6).rows())

RS = load("round_state", ["match_key", "round", "tick", "t_score", "ct_score"])
g = (RS.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
        pl.col("t_score").sort_by("tick").first().alias("t_b"),
        pl.col("ct_score").sort_by("tick").first().alias("ct_b"),
        pl.col("t_score").sort_by("tick").last().alias("t_a"),
        pl.col("ct_score").sort_by("tick").last().alias("ct_a")))

SU = load("score_update", ["match_key", "round", "team_code", "old_score", "new_score"], optional=True)
if SU is None:
    su = pl.DataFrame(schema={"match_key": pl.String, "round": pl.Int64, "su_n_inc": pl.UInt32,
                              "su_tc": pl.Int64, "su_side": pl.Int64})
else:
    su = (SU.filter((pl.col("new_score") - pl.col("old_score")) == 1)
            .group_by(["match_key", "round"]).agg(
                pl.len().alias("su_n_inc"),
                pl.col("team_code").min().alias("su_tc")))
    su = su.with_columns(pl.when(pl.col("su_n_inc") == 1).then(pl.col("su_tc")).otherwise(pl.lit(None, dtype=pl.Int64)).alias("su_side"))

BS = load("bomb_state", ["match_key", "round", "event_type"], optional=True)
if BS is None:
    BS = pl.DataFrame(schema={"match_key": pl.String, "round": pl.Int64, "event_type": pl.String})
bomb = BS.group_by(["match_key", "round"]).agg(
    (pl.col("event_type") == "bomb_exploded").any().alias("b_exp"),
    (pl.col("event_type") == "bomb_defused").any().alias("b_def"))
bomb = bomb.with_columns(
    pl.when(pl.col("b_exp") & ~pl.col("b_def")).then(pl.lit(2))
      .when(pl.col("b_def") & ~pl.col("b_exp")).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("bomb_side"))

J = RE.join(g, on=["match_key", "round"], how="left")
J = J.join(su.select(["match_key", "round", "su_n_inc", "su_side"]), on=["match_key", "round"], how="left")
J = J.join(bomb.select(["match_key", "round", "bomb_side"]), on=["match_key", "round"], how="left")
J = J.with_columns(
    pl.when((pl.col("t_a") - pl.col("t_b")) == 1).then(pl.lit(2))
      .when((pl.col("ct_a") - pl.col("ct_b")) == 1).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("claim_side"),
    (pl.col("winner_team_code") == 2).cast(pl.Int64).alias("tw"),
    (pl.col("winner_team_code") == 3).cast(pl.Int64).alias("cw"),
    (pl.col("t_b") == pl.col("ct_b")).alias("tied_before"),
)
J = J.with_columns(
    ((pl.col("t_a") == pl.col("t_b") + pl.col("tw")) & (pl.col("ct_a") == pl.col("ct_b") + pl.col("cw"))).alias("straight"),
    ((pl.col("t_a") == pl.col("ct_b") + pl.col("cw")) & (pl.col("ct_a") == pl.col("t_b") + pl.col("tw"))).alias("swapped"),
)
J = J.with_columns(
    pl.when(pl.col("straight") & pl.col("swapped")).then(pl.lit("both"))
      .when(pl.col("straight")).then(pl.lit("straight"))
      .when(pl.col("swapped")).then(pl.lit("swapped"))
      .otherwise(pl.lit("neither")).alias("cat"),
    (pl.col("claim_side").is_not_null() & (pl.col("claim_side") != pl.col("winner_team_code"))).alias("claim_dis"),
    (pl.col("su_side").is_not_null() & (pl.col("su_side") != pl.col("winner_team_code"))).alias("su_dis"),
    (pl.col("bomb_side").is_not_null() & (pl.col("bomb_side") != pl.col("winner_team_code"))).alias("bomb_dis"),
)
print(f"  joined rows: {J.height}; round_state groups missing: {J.filter(pl.col('t_b').is_null()).height}")


def summarise(sub, title):
    print(f"\n--- {title}: round_end rows={sub.height}")
    print("    category counts:", sub.group_by("cat").len().sort("cat").rows())
    print("    claim side present:", sub.filter(pl.col("claim_side").is_not_null()).height,
          " claim disagreements:", sub.filter(pl.col("claim_dis")).height)
    print("    score_update side present:", sub.filter(pl.col("su_side").is_not_null()).height,
          " score_update disagreements:", sub.filter(pl.col("su_dis")).height)
    print("    bomb side present:", sub.filter(pl.col("bomb_side").is_not_null()).height,
          " bomb disagreements:", sub.filter(pl.col("bomb_dis")).height)
    print("    tied_before rounds:", sub.filter(pl.col("tied_before")).height,
          " of which swapped:", sub.filter(pl.col("tied_before") & (pl.col("cat") == "swapped")).height,
          " untied rounds:", sub.filter(~pl.col("tied_before")).height,
          " of which swapped:", sub.filter(~pl.col("tied_before") & (pl.col("cat") == "swapped")).height)


for pc_label, sub in [("player_count == 10", J.filter(pl.col("player_count") == 10)),
                      ("player_count == 20", J.filter(pl.col("player_count") == 20)),
                      ("all player_count", J)]:
    summarise(sub, pc_label)

# per-round table for the claim's subset and the rounds named in the claim
P = J.filter(pl.col("player_count") == 10) if label == "2026" else J
print(f"\n--- per-round table ({'player_count == 10' if label == '2026' else 'all player_count'}), rounds with any flag or a claim round")
print(f"    {'rnd':>3} {'n':>6} {'tied':>5} {'straight':>8} {'swapped':>7} {'both':>5} {'neither':>7} | "
      f"{'claim':>6} {'c_dis':>5} | {'su':>5} {'su_dis':>6} {'su_multi':>8} | {'bomb':>5} {'b_dis':>5}")
per = (P.group_by("round").agg(
    pl.len().alias("n"),
    pl.col("tied_before").sum().alias("tied"),
    (pl.col("cat") == "straight").sum().alias("straight"),
    (pl.col("cat") == "swapped").sum().alias("swapped"),
    (pl.col("cat") == "both").sum().alias("both"),
    (pl.col("cat") == "neither").sum().alias("neither"),
    pl.col("claim_side").is_not_null().sum().alias("claim"),
    pl.col("claim_dis").sum().alias("c_dis"),
    pl.col("su_side").is_not_null().sum().alias("su"),
    pl.col("su_dis").sum().alias("su_dis"),
    (pl.col("su_n_inc") > 1).sum().alias("su_multi"),
    pl.col("bomb_side").is_not_null().sum().alias("bomb"),
    pl.col("bomb_dis").sum().alias("b_dis"),
).sort("round"))
CLAIM_ROUNDS = {9, 12, 13, 25, 27, 28, 33, 39, 45}
for r in per.iter_rows(named=True):
    flagged = any(r[k] for k in ["c_dis", "su_dis", "b_dis", "swapped", "neither", "su_multi"])
    if flagged or r["round"] in CLAIM_ROUNDS:
        print(f"    {r['round']:>3} {r['n']:>6} {r['tied']:>5} {r['straight']:>8} {r['swapped']:>7} {r['both']:>5} {r['neither']:>7} | "
              f"{r['claim']:>6} {r['c_dis']:>5} | {r['su']:>5} {r['su_dis']:>6} {r['su_multi']:>8} | {r['bomb']:>5} {r['b_dis']:>5}")

# claim disagreements: category and tie status
cd = P.filter(pl.col("claim_dis"))
print(f"\n--- claim disagreements in subset: {cd.height}")
print("    by category:", cd.group_by("cat").len().rows())
print("    by tied_before:", cd.group_by("tied_before").len().rows())
print("    by round:", cd.group_by("round").len().sort("round").rows())
print("    claim side == swapped-consistent truth? (cat=='swapped' and claim side != winner) ->",
      cd.filter(pl.col("cat") == "swapped").height, "of", cd.height)

# a few examples of the flagged signature (round 27 or the largest flagged round)
if cd.height:
    rr = cd.group_by("round").len().sort("len", descending=True)["round"][0]
    ex = cd.filter(pl.col("round") == rr).head(6)
    print(f"\n    examples, round {rr} (t_b, ct_b, t_a, ct_a, winner, claim_side, cat):")
    for r in ex.iter_rows(named=True):
        print(f"      t_b={r['t_b']:>2} ct_b={r['ct_b']:>2} t_a={r['t_a']:>2} ct_a={r['ct_a']:>2} "
              f"winner={r['winner_team_code']} claim_side={r['claim_side']} cat={r['cat']} bomb_side={r['bomb_side']}")

# neither-category examples
ne = P.filter(pl.col("cat") == "neither")
print(f"\n--- 'neither' rows (no straight/swapped fit): {ne.height}; by round: {ne.group_by('round').len().sort('round').head(12).rows()}")
for r in ne.head(4).iter_rows(named=True):
    print(f"      round {r['round']} t_b={r['t_b']} ct_b={r['ct_b']} t_a={r['t_a']} ct_a={r['ct_a']} winner={r['winner_team_code']}")

# round 12 coverage: claim blind spot
r12 = P.filter(pl.col("round") == 12)
print(f"\n--- round 12 coverage: rows={r12.height}, claim side present={r12.filter(pl.col('claim_side').is_not_null()).height}, "
      f"score_update side present={r12.filter(pl.col('su_side').is_not_null()).height}, "
      f"bomb side present={r12.filter(pl.col('bomb_side').is_not_null()).height}")

# whole-tome bomb check (all player counts)
ball = J.filter(pl.col("bomb_side").is_not_null())
print(f"\n--- bomb check, all player_count: bomb-labelled rounds={ball.height}, disagreements={ball.filter(pl.col('bomb_dis')).height}")
print("DONE", label)
