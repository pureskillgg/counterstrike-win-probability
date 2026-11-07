"""verify D1-7 (swarm2 verifier): 2026 score_update as an independent scorer signal vs winner_team_code and round_state counters."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(16)
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")


def scan(name):
    return pl.scan_parquet(sorted(glob.glob(str(T26 / f"basic_{name}*" / "dataframe_*"))))


RE = scan("round_end").select(["match_key", "round", "winner_team_code", "player_count", "tick"]).unique(["match_key", "round"]).collect()
print("2026 round_end rows (dedup):", RE.height)
su = scan("score_update").select(["match_key", "round", "tick", "team_code", "old_score", "new_score"]).collect()
print("score_update rows:", su.height, " rounds with any score_update:", su.select(["match_key", "round"]).unique().height)
su = su.with_columns((pl.col("new_score") - pl.col("old_score")).alias("delta"))
# per round: count of +1 events and the team codes of those events
one = su.filter(pl.col("delta") == 1)
agg = one.group_by(["match_key", "round"]).agg(pl.len().alias("n_plus1"), pl.col("team_code").unique().alias("plus1_teams"))
print("\n+1 events per (match, round):", agg.group_by("n_plus1").len().sort("n_plus1").to_dicts())
agg = agg.with_columns(pl.when((pl.col("n_plus1") == 1)).then(pl.col("plus1_teams").list.first()).otherwise(None).alias("su_side"))
J = RE.join(agg.select(["match_key", "round", "su_side", "n_plus1"]), on=["match_key", "round"], how="left")
J = J.with_columns(
    pl.when(pl.col("su_side").is_null()).then(pl.lit("no_single_plus1")).when(pl.col("su_side") == pl.col("winner_team_code")).then(pl.lit("agree")).otherwise(pl.lit("disagree")).alias("su_vs_winner"))
print("\n2026 all rounds, score_update(+1) side vs winner_team_code:")
print(J.group_by("su_vs_winner").len().sort("su_vs_winner"))
print("\n2026 per-round table, rounds 1..45 (pc any): n rounds, n agree, n disagree, n no_single")
tab = (J.group_by("round").agg(
    pl.len().alias("n"),
    (pl.col("su_vs_winner") == "agree").sum().alias("agree"),
    (pl.col("su_vs_winner") == "disagree").sum().alias("disagree"),
    (pl.col("su_vs_winner") == "no_single_plus1").sum().alias("no_single")).sort("round"))
print(tab.filter(pl.col("round") <= 45))
print("\n2026 pc==10 per-round disagreements (score_update vs winner):")
tab10 = (J.filter(pl.col("player_count") == 10).group_by("round").agg(
    pl.len().alias("n"), (pl.col("su_vs_winner") == "disagree").sum().alias("disagree"),
    (pl.col("su_vs_winner") == "agree").sum().alias("agree")).sort("round"))
print(tab10.filter(pl.col("disagree") > 0))
print("pc==10 rounds 25..45 n/agree/disagree:")
print(tab10.filter(pl.col("round").is_between(25, 45)))
pass  # no file output in this check
