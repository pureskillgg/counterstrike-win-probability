"""verify D1-6 swap check (swarm2 verifier).

S1 swap signature by round: a within-round state change is labelled SWAP_T if (t1, c1) == (c0, t0 + 1)
   (the two counters exchanged, T-labelled counter gained one) and SWAP_CT if (t1, c1) == (c0 + 1, t0).
   Counted per round for the 305 set and for the d12-25 control (player_count == 20).
S2 round-8 detail for the 305 set: winner of round 8 vs swap type.
S3 fixed-label test: if round 9 is relabelled from the score side (the claim's proposed fix) and rounds 1..8 keep
   winner_team_code, does the cumulative identity at round-10 start hold? Compared with the unfixed identity.
S4 halftime control: at the round 12 -> 13 boundary, does (t0_13, c0_13) == (c1_12, t1_12) (side swap) hold in
   the control? Confirms the swap signature exists naturally at halftime.

ASSUMPTIONS: A1 days; A2 player_count == 20; A8 duplicate exclusion; A3 team codes 2 = T, 3 = CT.
A11 ASSUMPTION: a state change that exactly exchanges the two counters is a label error, not a game event,
    except at a halftime boundary (round 12 -> 13) where a real side swap happens.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(50)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def fl(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


RE = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "winner_team_code", "player_count"])
                for f in fl("round_end")], how="vertical_relaxed")
dups = RE.group_by(["match_key", "round"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
RE = RE.join(dups.select(["match_key", "round"]), on=["match_key", "round"], how="anti")
RE = RE.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
RE = RE.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))

RS = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "tick", "t_score", "ct_score"])
                for f in fl("round_state")], how="vertical_relaxed").sort(["match_key", "round", "tick"])
S = RS.group_by(["match_key", "round"], maintain_order=True).agg(
    pl.col("t_score").first().alias("t0"), pl.col("ct_score").first().alias("c0"),
    pl.col("t_score").last().alias("t1"), pl.col("ct_score").last().alias("c1"))

M = RE.join(S, on=["match_key", "round"], how="left")
M = M.with_columns(
    pl.when((pl.col("t1") - pl.col("t0")) == 1).then(pl.lit(2))
      .when((pl.col("c1") - pl.col("c0")) == 1).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("adj_side"))
M = M.with_columns(
    pl.when((pl.col("t1") == pl.col("c0")) & ((pl.col("c1") - pl.col("t0")) == 1)).then(pl.lit("SWAP_T"))
      .when((pl.col("c1") == pl.col("t0")) & ((pl.col("t1") - pl.col("c0")) == 1)).then(pl.lit("SWAP_CT"))
      .otherwise(pl.lit("no_swap")).alias("swap_type"))
M = M.with_columns(pl.when(pl.col("swap_type") != "no_swap").then(pl.col("swap_type")).otherwise(pl.lit("no_swap"))
                   .alias("swap_type"))

set305 = set(M.filter((pl.col("round") == 9) & (pl.col("dg") == "d12-25") & (pl.col("player_count") == 20)
                      & pl.col("adj_side").is_not_null() & (pl.col("adj_side") != pl.col("winner_team_code")))
             ["match_key"].to_list())
M = M.with_columns(pl.col("match_key").is_in(list(set305)).alias("in305"))

# ----------------------------------------------------------------------------- S1
hdr("S1 swap signature by round (d12-25, player_count == 20): 305 set vs control")
x = M.filter((pl.col("dg") == "d12-25") & (pl.col("player_count") == 20) & (pl.col("round") <= 13))
t = (x.group_by(["in305", "round"]).agg(
        pl.len().alias("n"),
        (pl.col("swap_type") == "SWAP_T").sum().alias("swap_T"),
        (pl.col("swap_type") == "SWAP_CT").sum().alias("swap_CT"),
        pl.col("adj_side").is_null().sum().alias("null_side"))
     .sort(["in305", "round"]))
print(t)

# ----------------------------------------------------------------------------- S2
hdr("S2 round 8 in the 305 set: winner of round 8 by swap type")
x8 = M.filter((pl.col("round") == 8) & pl.col("in305"))
print(x8.group_by(["swap_type", "winner_team_code"]).len().sort(["swap_type", "winner_team_code"]))

# ----------------------------------------------------------------------------- S3
hdr("S3 cumulative identity at round-10 start (305 set): unfixed vs claim's fix (round 9 from score side)")
W = M.filter(pl.col("round") <= 9).sort(["match_key", "round"])
fixed = W.with_columns(
    pl.when((pl.col("round") == 9) & pl.col("in305") & pl.col("adj_side").is_not_null())
      .then(pl.col("adj_side")).otherwise(pl.col("winner_team_code")).alias("w_fixed"))
agg = fixed.group_by("match_key").agg(
    (pl.col("winner_team_code") == 2).sum().alias("T_w"),
    (pl.col("winner_team_code") == 3).sum().alias("CT_w"),
    (pl.col("w_fixed") == 2).sum().alias("T_f"),
    (pl.col("w_fixed") == 3).sum().alias("CT_f"),
    pl.col("round").n_unique().alias("nr"))
r10 = S.filter(pl.col("round") == 10).select(["match_key", pl.col("t0").alias("T10"), pl.col("c0").alias("CT10")])
J = agg.join(r10, on="match_key", how="inner").filter(pl.col("nr") == 9)
J = J.with_columns(pl.col("match_key").is_in(list(set305)).alias("in305"))
J = J.with_columns(
    ((pl.col("T10") == pl.col("T_w")) & (pl.col("CT10") == pl.col("CT_w"))).alias("id_unfixed"),
    ((pl.col("T10") == pl.col("T_f")) & (pl.col("CT10") == pl.col("CT_f"))).alias("id_fixed"),
    ((pl.col("T10") == pl.col("CT_w")) & (pl.col("CT10") == pl.col("T_w"))).alias("id_swapped_unfixed"))
print(J.group_by("in305").agg(pl.len().alias("n"), pl.col("id_unfixed").sum().alias("unfixed_holds"),
                              pl.col("id_fixed").sum().alias("fixed_holds"),
                              pl.col("id_swapped_unfixed").sum().alias("unfixed_swapped_holds")))

# ----------------------------------------------------------------------------- S4
hdr("S4 halftime boundary 12 -> 13 (player_count == 20, d12-25 and d01-07): side swap at halftime, control")
H = (S.filter(pl.col("round").is_in([12, 13])).select(["match_key", "round", "t0", "c0", "t1", "c1"]))
h12 = H.filter(pl.col("round") == 12).select(["match_key", pl.col("t1").alias("t12e"), pl.col("c1").alias("c12e")])
h13 = H.filter(pl.col("round") == 13).select(["match_key", pl.col("t0").alias("t13s"), pl.col("c0").alias("c13s")])
hh = h12.join(h13, on="match_key", how="inner").with_columns(
    ((pl.col("t13s") == pl.col("c12e")) & (pl.col("c13s") == pl.col("t12e"))).alias("halftime_swapped"),
    ((pl.col("t13s") == pl.col("t12e")) & (pl.col("c13s") == pl.col("c12e"))).alias("no_swap"))
dg_pc = M.filter(pl.col("round") == 13).select(["match_key", "dg", "player_count"]).unique()
hh = hh.join(dg_pc, on="match_key", how="left").filter(pl.col("player_count") == 20)
hh = hh.with_columns(pl.col("match_key").is_in(list(set305)).alias("in305"))
print(hh.group_by(["dg", "in305"]).agg(pl.len().alias("n"), pl.col("halftime_swapped").sum().alias("swapped"),
                                       pl.col("no_swap").sum().alias("no_swap")).sort(["dg", "in305"]))
