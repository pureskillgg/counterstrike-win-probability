"""verify D1-6 recompute (swarm2 verifier, independent of D1_followup2.py).

Claim under test: round 9, d12-25, winner_team_code disagrees with the score side in 305 of 3,249 rounds (9.4%);
zero of 805 in d01-07; 10 of 305 have a bomb outcome, all agreeing with the score side.

ASSUMPTIONS (stated; "as cited" marks choices copied from D1_followup2.py):
A1 day = 2-digit day parsed from match_key 'csds/2024/02/DD/...'; d12-25 = days 12..25, d01-07 = days 01..07 (as cited).
A2 player_count == 20 filter on round_end (as cited). Rows without the filter are reported in R3.
A3 winner = round_end.winner_team_code; team code 2 = T, 3 = CT (as cited; bomb side 2 = exploded, 3 = defused).
A4 (ADJ) score side = first vs last round_state row of the round on adjusted t_score/ct_score (as cited):
   T if t_last - t_first == 1, else CT if ct_last - ct_first == 1, else null.
A5 (RAW) same logic on t_score_raw / ct_score_raw (not used by the cited script).
A6 (CONT) continuity side: score change between the first round_state row of round r and the first row of round r+1
   (adjusted columns), same match, requires round r+1 to exist. Independent of the within-round window.
A7 bomb outcome from bomb_state event_type (bomb_defused -> CT 3, bomb_exploded -> T 2), first row per (match, round).
A8 duplicate (match_key, round) groups in round_end are excluded and reported (only one group exists, round 1).
"""
import glob
import math
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(60)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def fl(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def fisher_two_sided(a, b, c, d):
    """Exact two-sided Fisher test on [[a, b], [c, d]] (hypergeometric; no scipy dependency)."""
    n1, n2, k = a + b, c + d, a + c
    n = n1 + n2

    def p_of(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)

    p_obs = p_of(a)
    lo, hi = max(0, k - n2), min(k, n1)
    return sum(p_of(x) for x in range(lo, hi + 1) if p_of(x) <= p_obs * (1 + 1e-9))


# ----------------------------------------------------------------------------- load
RE_all = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "winner_team_code", "player_count"])
                    for f in fl("round_end")], how="vertical_relaxed")
print("round_end rows:", RE_all.height, " matches:", RE_all["match_key"].n_unique())
dups = RE_all.group_by(["match_key", "round"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
print("A8 duplicate (match_key, round) groups in round_end:", dups.height, dups.to_dicts())
RE_all = RE_all.join(dups.select(["match_key", "round"]), on=["match_key", "round"], how="anti")
RE_all = RE_all.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
print("A1 day parse failures:", RE_all.filter(pl.col("day").is_null()).height)
RE_all = RE_all.with_columns(
    pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
print("dg counts (all round_end rows):", RE_all.group_by("dg").len().sort("dg").to_dicts())

RS = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "tick", "event_type",
                                            "t_score", "ct_score", "t_score_raw", "ct_score_raw"])
                for f in fl("round_state")], how="vertical_relaxed")
print("round_state rows:", RS.height)
RS = RS.sort(["match_key", "round", "tick"])
S = RS.group_by(["match_key", "round"], maintain_order=True).agg(
    pl.col("t_score").first().alias("t0"), pl.col("ct_score").first().alias("c0"),
    pl.col("t_score").last().alias("t1"), pl.col("ct_score").last().alias("c1"),
    pl.col("t_score_raw").first().alias("rt0"), pl.col("ct_score_raw").first().alias("rc0"),
    pl.col("t_score_raw").last().alias("rt1"), pl.col("ct_score_raw").last().alias("rc1"),
    pl.len().alias("n_states"),
    pl.col("event_type").eq("round_end").any().alias("has_round_end_evt"),
)
print("round_state (match, round) groups:", S.height)

# A6 continuity: first row of round r+1 minus first row of round r, same match
S = S.sort(["match_key", "round"]).with_columns(
    pl.col("round").shift(-1).over("match_key").alias("round_next"),
    pl.col("t0").shift(-1).over("match_key").alias("t0_next"),
    pl.col("c0").shift(-1).over("match_key").alias("c0_next"),
).with_columns(
    pl.when((pl.col("round_next") == pl.col("round") + 1) & ((pl.col("t0_next") - pl.col("t0")) == 1)).then(pl.lit(2))
      .when((pl.col("round_next") == pl.col("round") + 1) & ((pl.col("c0_next") - pl.col("c0")) == 1)).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("cont_side"))

# A4 / A5 within-round sides
S = S.with_columns(
    pl.when((pl.col("t1") - pl.col("t0")) == 1).then(pl.lit(2))
      .when((pl.col("c1") - pl.col("c0")) == 1).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("adj_side"),
    pl.when((pl.col("rt1") - pl.col("rt0")) == 1).then(pl.lit(2))
      .when((pl.col("rc1") - pl.col("rc0")) == 1).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("raw_side"),
)

# A7 bomb outcome
BS = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "event_type"]) for f in fl("bomb_state")],
               how="vertical_relaxed")
BO = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
      .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side"))
      .select(["match_key", "round", "bomb_side"]))
print("bomb outcome (match, round) rows:", BO.height)

# ----------------------------------------------------------------------------- join
A = (RE_all.join(S.select(["match_key", "round", "adj_side", "raw_side", "cont_side", "n_states",
                           "has_round_end_evt", "t0", "c0", "t1", "c1"]),
                 on=["match_key", "round"], how="left")
     .join(BO, on=["match_key", "round"], how="left"))
print("round_end rows with no round_state group:", A.filter(pl.col("n_states").is_null()).height)

A = A.with_columns(
    (pl.col("adj_side").is_not_null() & (pl.col("adj_side") != pl.col("winner_team_code"))).alias("adj_mm"),
    (pl.col("raw_side").is_not_null() & (pl.col("raw_side") != pl.col("winner_team_code"))).alias("raw_mm"),
    (pl.col("cont_side").is_not_null() & (pl.col("cont_side") != pl.col("winner_team_code"))).alias("cont_mm"),
    (pl.col("bomb_side").is_not_null() & (pl.col("bomb_side") != pl.col("winner_team_code"))).alias("bomb_mm"),
)
A20 = A.filter(pl.col("player_count") == 20)

# ----------------------------------------------------------------------------- R1: round 9 headline recompute
hdr("R1 round 9 headline recompute (player_count == 20; A4 adjusted score side as cited)")
for dg in ["d12-25", "d01-07"]:
    x = A20.filter((pl.col("round") == 9) & (pl.col("dg") == dg))
    xs = x.filter(pl.col("adj_side").is_not_null())
    mm = xs.filter(pl.col("adj_mm"))
    ok = xs.filter(~pl.col("adj_mm"))
    print(f"{dg}: round-9 rows={x.height}, rows with ADJ side={xs.height}, "
          f"ADJ disagreements={mm.height} ({mm.height / max(xs.height, 1):.4f})")
    print(f"       distinct matches with ADJ disagreement at round 9: {mm['match_key'].n_unique()}")
    print(f"       bomb outcome among ADJ-disagreeing rows: {mm.filter(pl.col('bomb_side').is_not_null()).height} of {mm.height}")
    print(f"       bomb outcome among ADJ-agreeing rows:    {ok.filter(pl.col('bomb_side').is_not_null()).height} of {ok.height}")
    if mm.height and ok.height:
        a = mm.filter(pl.col("bomb_side").is_not_null()).height
        b = mm.height - a
        c = ok.filter(pl.col("bomb_side").is_not_null()).height
        d = ok.height - c
        print(f"       Fisher exact two-sided p (bomb presence, disagree vs agree): {fisher_two_sided(a, b, c, d):.3g}")

hdr("R1b round 9 d12-25 cross-tab: ADJ vs RAW vs CONT side vs winner (player_count == 20)")
x = A20.filter((pl.col("round") == 9) & (pl.col("dg") == "d12-25"))
print(x.group_by(["adj_side", "raw_side", "cont_side", "winner_team_code"]).len().sort("len", descending=True))

hdr("R2 disagreement counts by round, player_count == 20: ADJ / RAW / CONT")
for dg in ["d12-25", "d01-07"]:
    x = A20.filter(pl.col("dg") == dg)
    t = (x.group_by("round").agg(
            pl.col("adj_side").is_not_null().sum().alias("n_adj"),
            pl.col("adj_mm").sum().alias("adj_mm"),
            pl.col("raw_side").is_not_null().sum().alias("n_raw"),
            pl.col("raw_mm").sum().alias("raw_mm"),
            pl.col("cont_side").is_not_null().sum().alias("n_cont"),
            pl.col("cont_mm").sum().alias("cont_mm"))
         .sort("round"))
    print(f"\n-- {dg}")
    print(t)
