"""Debug helper for verify_D7_8: re-runs verify_D7_8_main silently and inspects intermediate tables (stdout only)."""
import io
import contextlib
import sys
from pathlib import Path

import polars as pl

src = Path(r"G:\pureskill\csds-cs2\scripts\swarm2\verify_D7_8_main.py").read_text(encoding="utf-8")
src = src.replace('sys.stdout.reconfigure(encoding="utf-8")', "pass")
g = {"__name__": "verify_main_debug"}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(src, "verify_D7_8_main.py", "exec"), g)
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
FE, SA, P, X, PM, L, RE, ST, starts, dp_map = (g[k] for k in ["FE", "SA", "P", "X", "PM", "L", "RE", "ST", "starts", "dp_map"])

def show_match(k, rounds):
    fe = FE.filter(pl.col("match_key") == k).sort("round").select(["round", "t_score", "ct_score"])
    sa = SA.filter(pl.col("match_key") == k).select(["round", "sideA"])
    x = X.filter(pl.col("match_key") == k).select(["round", "winner_team_code", "wside_p", "ledger_side", "player_count", "bomb_side"])
    p = P.filter(pl.col("match_key") == k).select(["round", "dA", "dB", "won"])
    t = fe.join(sa, on="round", how="left").join(x, on="round", how="left").join(p, on="round", how="left").sort("round")
    t = t.filter(pl.col("round").is_in(rounds))
    print("match", k[-50:], "| pl starts", starts.get(k, []), "| dp swaps", dp_map.get(k, []))
    print(t.rows())

print("== example: pl pattern '12', DP pattern ''")
ex = PM.filter((pl.col("pl_pat") == "12") & (pl.col("dp_pat") == "")).head(2)["match_key"].to_list()
for k in ex:
    show_match(k, list(range(1, 27)))
print("\n== example: pl pattern '8' in day<12 (f8 before onset)")
ex8 = PM.filter((pl.col("pl_pat") == "8") & (pl.col("day") < "12")).head(3)
print("count pl '8' with day<12:", PM.filter((pl.col("pl_pat") == "8") & (pl.col("day") < "12")).height,
      "| pl '8' with day>=12:", PM.filter((pl.col("pl_pat") == "8") & (pl.col("day") >= "12")).height)
for k in ex8["match_key"].to_list():
    show_match(k, list(range(6, 12)))

print("\n== round 9 rows for pl '8' matches: presence in X and reasons")
k8 = PM.filter(pl.col("pl_pat") == "8")["match_key"].to_list()
r9 = X.filter(pl.col("match_key").is_in(k8) & (pl.col("round") == 9))
print("X rows for round 9 of pl '8' matches:", r9.height, "of", len(k8))
print("  winner null:", r9.filter(pl.col("winner_team_code").is_null()).height,
      "| wside_p null:", r9.filter(pl.col("wside_p").is_null()).height,
      "| player_count != 20:", r9.filter(pl.col("player_count") != 20).height)
print("  round-9 player_count (top):", r9.group_by("player_count").len().sort("len", descending=True).rows())
print("  by day:", r9.group_by("day").len().sort("day").rows())
print("pl '8' matches by day:", PM.filter(pl.col("pl_pat") == "8").group_by("day").len().sort("day").rows())

print("\n== does round 9 exist in freeze_end for pl '8' matches?")
print("FE round 9 rows for pl '8' matches:", FE.filter(pl.col("match_key").is_in(k8) & (pl.col("round") == 9)).height)
print("RE round 9 rows for pl '8' matches:", RE.filter(pl.col("match_key").is_in(k8) & (pl.col("round") == 9)).height)
print("RE round 9 player_count for pl '8' matches:", RE.filter(pl.col("match_key").is_in(k8) & (pl.col("round") == 9)).group_by("player_count").len().rows())

print("\n== disagreement rows between D7 ledger side and player winner side (127 expected)")
dis = X.filter(pl.col("ledger_side").is_not_null() & pl.col("wside_p").is_not_null() & (pl.col("ledger_side") != pl.col("wside_p")))
print("count:", dis.height, "| by round (top):", dis.group_by("round").len().sort("len", descending=True).head(8).rows())
print("by build:", dis.group_by("build_num").len().rows())
print("matches involved:", dis["match_key"].n_unique())
