"""D5 step 4: per-label dump of raw round ticks for a handful of matches (2024 tome).

For each chosen match prints, by round label:
  round_start channel tick, round_state round_start tick, round_freeze_end tick (round_state),
  round_end tick, winner_team_code, reason, legacy_code, player_count, freeze_end scores,
  and bomb outcome rows (tick, event) with their labels.
Match picks are deterministic: first N keys (sorted) in each group, plus the first
aff_r13 match whose round-13 winner disagrees with the round-13 score chain.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D5_04_inspect.py
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from D5_common import read  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)

SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5")
R = pl.read_parquet(SCR / "rounds.parquet")
M = pl.read_parquet(SCR / "matches.parquet")

LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]
M = M.with_columns(grp=pl.when(pl.col("aff_r13")).then(pl.lit("aff_r13"))
                   .when(pl.col("day").is_in(LATE)).then(pl.lit("ctrl_late"))
                   .when(pl.col("day").is_in(EARLY)).then(pl.lit("early")).otherwise(pl.lit("other")))

# aff match where the score chain (13->14) disagrees with round_end winner 13
r13 = R.filter(pl.col("round") == 13).select(["match_key", "fe_t_score", "fe_ct_score", "winner"])
nx = R.filter(pl.col("round") == 14).select(["match_key", "fe_t_score", "fe_ct_score"]).rename(
    {"fe_t_score": "t14", "fe_ct_score": "ct14"})
chk = r13.join(nx, on="match_key", how="inner").with_columns(
    dt=pl.col("t14") - pl.col("fe_t_score"), dct=pl.col("ct14") - pl.col("fe_ct_score")).with_columns(
    chain=pl.when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(pl.lit(2))
    .when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(pl.lit(3)).otherwise(pl.lit(0)))
ctrl_bad = chk.filter((pl.col("chain") != 0) & (pl.col("chain") != pl.col("winner"))).join(
    M.select(["match_key", "grp"]), on="match_key").filter(pl.col("grp") == "ctrl_late")
aff_bad = chk.filter((pl.col("chain") != 0) & (pl.col("chain") != pl.col("winner"))).join(
    M.select(["match_key", "grp"]), on="match_key").filter(pl.col("grp") == "aff_r13")

picks = []
for g, n in [("early", 2), ("ctrl_late", 2), ("aff_r13", 2)]:
    picks += M.filter(pl.col("grp") == g).sort("match_key").select("match_key").head(n)["match_key"].to_list()
picks += aff_bad.sort("match_key").head(1)["match_key"].to_list()
picks += ctrl_bad.sort("match_key").head(2)["match_key"].to_list()
print("picked matches:")
for k in picks:
    print("  ", k, M.filter(pl.col("match_key") == k).select(["grp", "map_name", "match_date", "final_sum"]).to_dicts())

SEL = set(picks)
RST = read("round_start", ["match_key", "round", "tick"]).filter(pl.col("match_key").is_in(list(SEL)))
RE = read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                        "legacy_code", "player_count"]).filter(pl.col("match_key").is_in(list(SEL)))
RS = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).filter(
    pl.col("match_key").is_in(list(SEL)))
BS = read("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"]).filter(
    pl.col("match_key").is_in(list(SEL)) & pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"]))

for k in picks:
    print("\n" + "=" * 120)
    print(k)
    rs_start = RS.filter((pl.col("match_key") == k) & (pl.col("event_type") == "round_start")).select(
        ["round", pl.col("tick").alias("rs_round_start_tick")]).unique("round")
    fe = RS.filter((pl.col("match_key") == k) & (pl.col("event_type") == "round_freeze_end")).sort("tick").group_by("round").agg(
        pl.col("tick").first().alias("fe_tick"), pl.col("t_score").first().alias("fe_t"), pl.col("ct_score").first().alias("fe_ct"))
    ch = RST.filter(pl.col("match_key") == k).select(["round", pl.col("tick").alias("ch_start_tick")]).unique("round")
    en = RE.filter(pl.col("match_key") == k).sort("tick").group_by("round").agg(
        pl.col("tick").first().alias("end_tick"), pl.col("winner_team_code").first().alias("winner"),
        pl.col("win_reason_code").first().alias("reason"), pl.col("legacy_code").first().alias("legacy"),
        pl.col("player_count").first().alias("pc"), pl.len().alias("n_end"))
    tab = (fe.join(ch, on="round", how="full", coalesce=True).join(en, on="round", how="full", coalesce=True)
           .join(rs_start, on="round", how="full", coalesce=True).sort("round"))
    print(tab)
    bs = BS.filter(pl.col("match_key") == k).sort("tick")
    print("bomb rows (round label, tick, event, team):")
    print(bs.select(["round", "tick", "event_type", "player_team_code"]))
