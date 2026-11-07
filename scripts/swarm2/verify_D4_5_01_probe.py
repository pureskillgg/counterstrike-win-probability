"""verify D4-5, step 1: probe keys, duplicate rows, reason-code mapping, round_state score fields, header finals.

Output goes to stdout (captured to logs\\swarm2\\verify_D4_5_01_probe.log). Read-only.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D4_5_01_probe.py
"""
import sys

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import polars as pl  # noqa: E402
from verify_D4_5_common import TOMES, scan, header, day_expr, T, CT  # noqa: E402

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(14)
pl.Config.set_fmt_str_lengths(60)

base = TOMES["2024"]
print("== example match_key values")
rs = scan(base, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score",
                                "t_score_raw", "ct_score_raw", "phase", "is_warmup"]).collect()
print(rs.head(3))
print("distinct match_key:", rs["match_key"].n_unique())
print("rows per match (describe):")
print(rs.group_by("match_key").len()["len"].describe())

print("\n== freeze_end duplicates per (match, round)")
fe = rs.filter(pl.col("event_type") == "round_freeze_end")
dup = fe.group_by(["match_key", "round"]).agg(pl.len().alias("n"),
                                              pl.col("t_score").n_unique().alias("nu_t"),
                                              pl.col("ct_score").n_unique().alias("nu_c"))
print("groups:", dup.height, " groups with n>1:", int((dup["n"] > 1).sum()),
      " of which score values differ:", int(((dup["nu_t"] > 1) | (dup["nu_c"] > 1)).sum()))
print("max rows per group:", int(dup["n"].max()))

print("\n== round_end duplicates per (match, round)")
re_ = scan(base, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                               "player_count"]).collect()
rd = re_.group_by(["match_key", "round"]).agg(pl.len().alias("n"),
                                              pl.col("winner_team_code").n_unique().alias("nu_w"))
print("round_end groups:", rd.height, " n>1:", int((rd["n"] > 1).sum()),
      " winner differs within group:", int((rd["nu_w"] > 1).sum()))
print("round_end rows:", re_.height)

print("\n== reason code vs winner (all 2024 round_end rows, first per group)")
first = re_.unique(["match_key", "round"], keep="first")
print(first.group_by(["win_reason_code", "winner_team_code"]).len().sort(["win_reason_code", "winner_team_code"]))

print("\n== round_state score fields: t_score vs t_score_raw equality")
print("rows:", rs.height, " t_score==t_score_raw:", int((rs["t_score"] == rs["t_score_raw"]).sum()),
      " ct_score==ct_score_raw:", int((rs["ct_score"] == rs["ct_score_raw"]).sum()))

print("\n== header finals (2024)")
H = header(base, ["key", "match_date", "t_starters_score_final", "ct_starters_score_final", "tick_rate"])
print(H.head(3))
print(H.select(["t_starters_score_final", "ct_starters_score_final"]).describe())
print("header keys distinct:", H["key"].n_unique(), " rows:", H.height)
print("round_state match_key distinct vs header key:",
      rs["match_key"].n_unique(), H["key"].n_unique(),
      " overlap:", len(set(rs["match_key"].unique().to_list()) & set(H["key"].to_list())))
