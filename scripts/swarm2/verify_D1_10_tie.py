"""D1-10 verifier: when does the claim's round_state rule fail? Tie check.

Hypothesis: the claim rule (t_after - t_before == 1 -> T, else ct_after - ct_before == 1 -> CT) gives the
wrong side only when a side swap is recorded after the round's final score AND the scores were tied before
the round (so the swapped delta is exactly +1 on the wrong team).
Tabulates, for 2026 pc=10 rounds and for 2024 all-pc rounds: swap_after_re x tied_before x claim_disagrees.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(12)

T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")


def fl(root, ch):
    return sorted(glob.glob(str(root / f"basic_{ch}*" / "dataframe_*")))


def run(root, label, pc_filter):
    RE = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "winner_team_code", "player_count"]).collect() for f in fl(root, "round_end")])
    RE = RE.unique(["match_key", "round"])
    if pc_filter is not None:
        RE = RE.filter(pl.col("player_count") == pc_filter)
    RS = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).collect() for f in fl(root, "round_state")])
    s = RS.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
        pl.col("t_score").sort_by("tick").first().alias("t_b"), pl.col("ct_score").sort_by("tick").first().alias("ct_b"),
        pl.col("t_score").sort_by("tick").last().alias("t_a"), pl.col("ct_score").sort_by("tick").last().alias("ct_a"))
    rer = (RS.filter(pl.col("event_type") == "round_end").group_by(["match_key", "round"])
           .agg(pl.col("t_score").first().alias("t_re"), pl.col("ct_score").first().alias("ct_re")))
    s = s.join(rer, on=["match_key", "round"], how="left")
    s = s.with_columns(
        pl.when((pl.col("t_a") - pl.col("t_b")) == 1).then(pl.lit(2))
          .when((pl.col("ct_a") - pl.col("ct_b")) == 1).then(pl.lit(3))
          .otherwise(pl.lit(None, dtype=pl.Int64)).alias("claim_side"),
        ((pl.col("t_a") == pl.col("ct_re")) & (pl.col("ct_a") == pl.col("t_re")) & (pl.col("t_re") != pl.col("ct_re"))).alias("swap_after"),
        (pl.col("t_b") == pl.col("ct_b")).alias("tied_before"))
    J = RE.join(s.select(["match_key", "round", "claim_side", "swap_after", "tied_before"]), on=["match_key", "round"], how="left")
    J = J.filter(pl.col("claim_side").is_not_null()).with_columns(
        (pl.col("claim_side") != pl.col("winner_team_code")).alias("dis"),
        pl.col("swap_after").fill_null(False), pl.col("tied_before").fill_null(False))
    print(f"\n== {label}: rounds with claim side {J.height}, claim disagreements {int(J['dis'].sum())}")
    print("   swap_after x tied_before x disagree (count):")
    for r in J.group_by(["swap_after", "tied_before", "dis"]).len().sort(["swap_after", "tied_before", "dis"]).iter_rows(named=True):
        print(f"     swap_after={r['swap_after']!s:<5} tied_before={r['tied_before']!s:<5} dis={r['dis']!s:<5} n={r['len']}")
    d = J.filter(pl.col("dis"))
    print("   disagreements by round (top 12):", sorted(d.group_by("round").len().rows(), key=lambda x: -x[1])[:12])
    print("   disagreements with swap_after=True:", int(d["swap_after"].sum()), " with tied_before=True:", int(d["tied_before"].sum()))
    return J


run(T26, "2026 pc=10", 10)
run(T26, "2026 all pc", None)
run(T24, "2024 all pc", None)
print("\nDONE")
