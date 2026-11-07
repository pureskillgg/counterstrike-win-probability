"""verify D1-6 2026 check (swarm2 verifier): does the round-9 ADJ-vs-winner disagreement exist in the 2026 tome?

Same ADJ rule as verify_D1_6_recompute.py (A4). Reports round 9 (and rounds 8, 10, 13 for context), with
player_count == 20 and with no player_count filter, plus the engine win-reason agreement with winner_team_code.
ASSUMPTIONS: A3 team codes 2 = T, 3 = CT. A8 duplicate (match, round) exclusion. The 2026 tome has no day split
(one date range 2026-09-18..2026-10-08), so all matches are reported together.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(50)

T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")


def fl(ch):
    return sorted(glob.glob(str(T26 / f"basic_{ch}*" / "dataframe_*")))


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


re_files = fl("round_end")
rs_files = fl("round_state")
print("round_end files:", len(re_files), " round_state files:", len(rs_files))
RE = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "winner_team_code", "player_count",
                                            "win_reason_message"]) for f in re_files], how="vertical_relaxed")
dups = RE.group_by(["match_key", "round"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
print("round_end rows:", RE.height, " matches:", RE["match_key"].n_unique(), " duplicate (match, round) groups:", dups.height)
RE = RE.join(dups.select(["match_key", "round"]), on=["match_key", "round"], how="anti")

lf = pl.scan_parquet(rs_files).select(["match_key", "round", "tick", "t_score", "ct_score"]).sort(["match_key", "round", "tick"])
S = (lf.group_by(["match_key", "round"], maintain_order=True).agg(
        pl.col("t_score").first().alias("t0"), pl.col("ct_score").first().alias("c0"),
        pl.col("t_score").last().alias("t1"), pl.col("ct_score").last().alias("c1"))
     .collect(engine="streaming"))
print("round_state (match, round) groups:", S.height)
S = S.with_columns(
    pl.when((pl.col("t1") - pl.col("t0")) == 1).then(pl.lit(2))
      .when((pl.col("c1") - pl.col("c0")) == 1).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("adj_side"))

A = RE.join(S.select(["match_key", "round", "adj_side", "t0", "c0"]), on=["match_key", "round"], how="left")
A = A.with_columns((pl.col("adj_side").is_not_null() & (pl.col("adj_side") != pl.col("winner_team_code"))).alias("adj_mm"))

hdr("2026 round-level ADJ disagreement (any player_count), rounds 1..13")
t = (A.filter(pl.col("round") <= 13).group_by("round").agg(
        pl.col("adj_side").is_not_null().sum().alias("n_side"),
        pl.col("adj_mm").sum().alias("disagree"),
        pl.col("adj_side").is_null().sum().alias("null_side"))
     .sort("round"))
print(t)

hdr("2026 round 9 with player_count == 20 (the claim's filter)")
x9 = A.filter((pl.col("round") == 9) & (pl.col("player_count") == 20))
x9s = x9.filter(pl.col("adj_side").is_not_null())
print("round-9 rows pc20:", x9.height, " with ADJ side:", x9s.height, " disagree:", int(x9s["adj_mm"].sum()))
print("round-9 rows any pc with ADJ side:", A.filter((pl.col("round") == 9) & pl.col("adj_side").is_not_null()).height,
      " disagree:", int(A.filter((pl.col("round") == 9))["adj_mm"].sum()))

hdr("2026 win-reason agreement with winner_team_code (all rounds), for context")
print(A.group_by(["win_reason_message", "winner_team_code"]).len().sort("len", descending=True))

hdr("2026 round 9: rows with ADJ disagreement (listing, if any)")
print(A.filter((pl.col("round") == 9) & pl.col("adj_mm")).select(["match_key", "winner_team_code", "adj_side", "win_reason_message", "t0", "c0"]).head(20))
