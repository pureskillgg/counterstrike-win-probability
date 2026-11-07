"""verify D4-5, step 2: independent recompute of the claimed per-round disagreement counts.

Definitions (same as the claim, re-implemented here, not imported from the D4 scripts):
  - valid round r: swap_expected(r, 24) is False (2024: mr = 24, ASSUMPTION A12), both freeze_end rows
    (r and r+1) exist for the match, and the score delta is unambiguous.
  - score side: CT if dC == 1 and dT == 0; T if dT == 1 and dC == 0; else ambiguous (excluded).
  - disagree: score side != round_end.winner_team_code (first row per match, round).
Also runs the same counts with raw score fields (t_score_raw / ct_score_raw) as a robustness check.
Writes stdout only. Run via verify_D4_5_02_recompute.py from the project interpreter.
"""
import sys

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import polars as pl  # noqa: E402
from verify_D4_5_common import TOMES, scan, header, day_expr, swap_expected_expr, T, CT  # noqa: E402

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(14)


def build(era, score_cols):
    base = TOMES[era]
    rs = scan(base, "round_state", ["match_key", "round", "event_type", "tick"] + score_cols).collect()
    fe = (rs.filter(pl.col("event_type") == "round_freeze_end")
          .sort(["match_key", "round", "tick"])
          .unique(["match_key", "round"], keep="first")
          .select(["match_key", "round", "tick", score_cols[0], score_cols[1]]))
    fe = fe.rename({score_cols[0]: "t0", score_cols[1]: "c0", "tick": "fe_tick"})
    nxt = fe.with_columns((pl.col("round") - 1).alias("round")).rename(
        {"t0": "t1", "c0": "c1", "fe_tick": "fe1_tick"})
    D = fe.join(nxt, on=["match_key", "round"], how="inner").with_columns(
        (pl.col("t1") - pl.col("t0")).alias("dT"), (pl.col("c1") - pl.col("c0")).alias("dC"))
    D = D.with_columns(
        pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
        .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0))
        .alias("score_side"))
    mr = 24
    D = D.with_columns(swap_expected_expr(pl.lit(mr)).alias("sw"))
    return D


def roundend(era):
    base = TOMES[era]
    re_ = scan(base, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                                    "player_count"]).collect()
    return re_.sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first")


print("=== 2024 score side from round_state, unambiguous, valid (no swap after r) ===")
D = build("2024", ["t_score", "ct_score"])
Dv = D.filter((~pl.col("sw")) & (pl.col("score_side") != 0))
print(f"all freeze_end pairs: {D.height}; ambiguous: {int((D['score_side'] == 0).sum())}; "
      f"swap-excluded: {int(D['sw'].sum())}; valid unambiguous: {Dv.height}")
R = roundend("2024")
J = Dv.join(R.select(["match_key", "round", "winner_team_code", "win_reason_code", "tick"]).rename(
    {"tick": "re_tick"}), on=["match_key", "round"], how="inner")
J = J.with_columns(day_expr().alias("day"), (pl.col("score_side") != pl.col("winner_team_code")).alias("dis"))
print(f"joined to round_end: {J.height} (rounds with no round_end row: {Dv.height - J.height})")
J = J.with_columns(pl.when(pl.col("day") <= 7).then(pl.lit("d01-07")).otherwise(
    pl.when(pl.col("day").is_between(12, 25)).then(pl.lit("d12-25")).otherwise(pl.lit("d08-11/26+"))).alias("grp"))
tab = (J.group_by(["grp", "round"]).agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis"))
       .with_columns((pl.col("dis") / pl.col("n")).round(4).alias("rate")).sort(["grp", "round"]))
print("\nper group and round (only rounds with any disagreement, plus round 13 for every group):")
print(tab.filter((pl.col("dis") > 0) | (pl.col("round") == 13)))
print("\ngroup totals:")
print(J.group_by("grp").agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis")).sort("grp"))

print("\nclaimed numbers vs recomputed (days 12-25 2024):")
g = J.filter(pl.col("grp") == "d12-25")
for r in [9, 13, 25, 28]:
    x = g.filter(pl.col("round") == r)
    print(f"  round {r:>2}: recomputed {int(x['dis'].sum())} of {x.height}  "
          f"(claim: {dict(zip([9, 13, 25, 28], ['292 of 3,352', '2,051 of 3,106', '124 of 183', '105 of 152']))[r]})")
other = g.filter(~pl.col("round").is_in([9, 13, 25, 28]))
print(f"  all other rounds in days 12-25: disagreements {int(other['dis'].sum())} of {other.height} (claim: 0)")
e = J.filter(pl.col("grp") == "d01-07")
print(f"  days 01-07, all rounds: disagreements {int(e['dis'].sum())} of {e.height} (claim: 0 of 15,994)")

print("\n=== robustness: same counts with raw score fields (t_score_raw / ct_score_raw) ===")
Dr = build("2024", ["t_score_raw", "ct_score_raw"])
Dr = Dr.filter((~pl.col("sw")) & (pl.col("score_side") != 0))
Jr = Dr.join(R.select(["match_key", "round", "winner_team_code"]), on=["match_key", "round"], how="inner")
Jr = Jr.with_columns(day_expr().alias("day"), (pl.col("score_side") != pl.col("winner_team_code")).alias("dis"))
gr = Jr.filter(pl.col("day").is_between(12, 25))
print(gr.group_by("round").agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis")).filter(
    pl.col("dis") > 0).sort("round"))
print("raw-field total days 12-25:", int(gr["dis"].sum()), "of", gr.height)

print("\n=== 2026 control (same definitions, mr from header max_rounds) ===")
H = header(TOMES["2026"], ["key", "max_rounds"])
mr_map = {k: int(v) for k, v in zip(H["key"].to_list(), H["max_rounds"].to_list())}
D26 = build("2026", ["t_score", "ct_score"])
D26 = D26.with_columns(pl.col("match_key").replace_strict(mr_map, default=24, return_dtype=pl.Int64).alias("mr"))
D26 = D26.with_columns(swap_expected_expr(pl.col("mr")).alias("sw2"))
D26v = D26.filter((~pl.col("sw2")) & (pl.col("score_side") != 0))
R26 = roundend("2026")
J26 = D26v.join(R26.select(["match_key", "round", "winner_team_code"]), on=["match_key", "round"], how="inner")
J26 = J26.with_columns((pl.col("score_side") != pl.col("winner_team_code")).alias("dis"))
print(f"2026 valid joined: {J26.height}; disagreements: {int(J26['dis'].sum())}")
print("2026 max_rounds distribution:", H.group_by("max_rounds").len().sort("max_rounds").rows())
