"""Verifier D2-4 diagnostic: why is sx null on some valid 5v5 rounds (2024)?

Uses only the player_info channel. Valid = 10 distinct players, 5 on code 2, 5 on code 3 (same rule as the claim).
For each valid round, counts how many team-X players (membership = round-F code 2) are flagged and how the code
2 / code 3 split falls among them, then tallies the patterns where sx is null.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_4_sxcov.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import polars as pl  # noqa: E402

from verify_D2_4_build import channel  # noqa: E402

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(12)

label = "2024"
pi = channel(label, "player_info", ["match_key", "round", "player_id", "team_code"]).unique(
    ["match_key", "round", "player_id"], keep="first", maintain_order=True)
cnt = pi.group_by(["match_key", "round"]).agg(
    pl.len().alias("np"), (pl.col("team_code") == 2).sum().alias("n2"), (pl.col("team_code") == 3).sum().alias("n3"))
valid = cnt.filter((pl.col("np") == 10) & (pl.col("n2") == 5) & (pl.col("n3") == 5)).select(["match_key", "round"])
print("(match, round) keys in player_info:", cnt.height, "  valid 5v5:", valid.height)
print("invalid keys by (np, n2, n3) pattern, top 8:")
print(cnt.join(valid, on=["match_key", "round"], how="anti").group_by(["np", "n2", "n3"]).len().sort("len", descending=True).head(8))

first = valid.group_by("match_key").agg(pl.col("round").min().alias("F"))
roster = pi.join(first, on="match_key").filter(pl.col("round") == pl.col("F"))
x_ids = roster.filter(pl.col("team_code") == 2).select(["match_key", "player_id"]).unique().with_columns(pl.lit(True).alias("isX"))
pv = pi.join(valid, on=["match_key", "round"]).join(x_ids, on=["match_key", "player_id"], how="left").with_columns(
    pl.col("isX").fill_null(False))
agg = pv.group_by(["match_key", "round"]).agg(
    pl.col("isX").sum().alias("nx"),
    ((pl.col("isX")) & (pl.col("team_code") == 2)).sum().alias("xon2"),
    ((pl.col("isX")) & (pl.col("team_code") == 3)).sum().alias("xon3"),
    ((~pl.col("isX")) & (pl.col("team_code") == 2)).sum().alias("yon2"),
    ((~pl.col("isX")) & (pl.col("team_code") == 3)).sum().alias("yon3"))
print("valid rounds:", agg.height)
print("sx by my rule (nx==5 & all X on one side):",
      agg.filter(((pl.col("nx") == 5) & ((pl.col("xon2") == 5) | (pl.col("xon3") == 5)))).height)
nulls = agg.filter(~((pl.col("nx") == 5) & ((pl.col("xon2") == 5) | (pl.col("xon3") == 5))))
print("valid rounds with sx null:", nulls.height)
print("pattern (nx, xon2, xon3, yon2, yon3) for null-sx rounds, top 12:")
print(nulls.group_by(["nx", "xon2", "xon3", "yon2", "yon3"]).len().sort("len", descending=True).head(12))

# claim-style rule: x_t == n_x and y_t == 0 -> 2; x_t == 0 and y_t == n_y -> 3
claim_rule = agg.with_columns(
    pl.when((pl.col("xon2") == pl.col("nx")) & (pl.col("yon2") == 0)).then(2)
    .when((pl.col("xon2") == 0) & (pl.col("yon3") == 0)).then(3)
    .otherwise(None).alias("sx_claim_like"))
print("sx by claim-style rule (non-null):", claim_rule.filter(pl.col("sx_claim_like").is_not_null()).height)
