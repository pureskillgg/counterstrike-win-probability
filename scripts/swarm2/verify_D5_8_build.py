"""Verifier for D5-8 (inverted round-13 label). Step 1: independent per-round table from the 2024 tome.

Reads only the 2024 tome channels needed (round_end, round_state, bomb_state, header), one channel at a time.
Writes intermediate parquet to the session scratchpad (not to the repo, not under E:).
Every derived field is an ASSUMPTION unless stated otherwise; see the analysis script for the checks.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_8_build.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

B = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
OUT = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_8")
OUT.mkdir(parents=True, exist_ok=True)
SUF = "2023-12-28,2024-02-25"


def rd(ch, cols):
    d = B / "header" if ch == "header" else B / f"basic_{ch}.{SUF}"
    fs = sorted(glob.glob(str(d / "dataframe_*")))
    lfs = [pl.scan_parquet(f).select(cols) for f in fs]
    return pl.concat(lfs, how="vertical_relaxed").collect()


# header: final team scores (header key == match_key, verified)
H = rd("header", ["key", "match_date", "map_name", "t_starters_score_final", "ct_starters_score_final"]).rename(
    {"key": "match_key"})
print("header rows:", H.height, "unique keys:", H["match_key"].n_unique())

# round_end: winner_team_code (side code 2=T, 3=CT), reason code
RE_raw = rd("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
dup = RE_raw.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
print("round_end rows:", RE_raw.height, "dup (match,round) keys:", dup.height)
RE = (RE_raw.sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first", maintain_order=True))

# round_state: freeze-end scores; round_announce_match_start marks regime A
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw"])
regA = (RS.filter(pl.col("event_type") == "round_announce_match_start").select("match_key").unique()
        .with_columns(regA=pl.lit(True)))
FE_raw = RS.filter(pl.col("event_type") == "round_freeze_end")
print("freeze_end rows:", FE_raw.height)
FE = (FE_raw.sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first", maintain_order=True)
      .select(["match_key", "round", pl.col("t_score").alias("t_fe"), pl.col("ct_score").alias("ct_fe"),
               pl.col("t_score_raw").alias("t_fe_raw"), pl.col("ct_score_raw").alias("ct_fe_raw")]))

# bomb_state: outcome (defused -> CT side 3, exploded -> T side 2) and planter side
BS = rd("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
print("bomb_state event types:", BS.group_by("event_type").len().sort("len", descending=True).to_dicts())
outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"])
        .agg(pl.col("event_type").unique().alias("bomb_evs"), pl.col("tick").min().alias("bomb_out_tick")))
plant = (BS.filter(pl.col("event_type") == "bomb_planted").sort("tick")
         .group_by(["match_key", "round"]).agg(pl.col("player_team_code").first().alias("planter_side")))

# assemble per-round table
R = (RE.join(H.select(["match_key", "match_date", "map_name", "t_starters_score_final", "ct_starters_score_final"]),
             on="match_key", how="left")
     .join(regA, on="match_key", how="left")
     .with_columns(regA=pl.col("regA").fill_null(False))
     .join(FE, on=["match_key", "round"], how="left")
     .join(FE.with_columns(round=pl.col("round") - 1).rename({"t_fe": "t_fe_next", "ct_fe": "ct_fe_next",
                                                             "t_fe_raw": "t_fe_next_raw", "ct_fe_raw": "ct_fe_next_raw"}),
           on=["match_key", "round"], how="left")
     .join(outc, on=["match_key", "round"], how="left")
     .join(plant, on=["match_key", "round"], how="left")
     .with_columns(day=pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1)))

R = R.with_columns(
    bomb_side=pl.when(pl.col("bomb_evs").list.contains("bomb_defused")).then(pl.lit(3))
    .when(pl.col("bomb_evs").list.contains("bomb_exploded")).then(pl.lit(2)).otherwise(None).cast(pl.Int64),
    # reason code -> side (CS2 round_end codes; ASSUMPTION: standard CS:GO/CS2 mapping)
    reason_side=pl.when(pl.col("win_reason_code").is_in([1, 9, 13, 17])).then(pl.lit(2))
    .when(pl.col("win_reason_code").is_in([7, 8, 11, 12, 18])).then(pl.lit(3)).otherwise(None).cast(pl.Int64),
    # naive freeze-end step (same convention as scripts\swarm\D5 scratch analysis): +1 on t field -> side 2
    dt=pl.col("t_fe_next") - pl.col("t_fe"), dct=pl.col("ct_fe_next") - pl.col("ct_fe"),
)
R = R.with_columns(
    step_side=pl.when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(pl.lit(2))
    .when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(pl.lit(3)).otherwise(None).cast(pl.Int64),
)

R.write_parquet(OUT / "rounds2024.parquet")
H.write_parquet(OUT / "header2024.parquet")
print("per-round table rows:", R.height, "matches:", R["match_key"].n_unique())
print("regime A matches:", R.filter(pl.col("regA")).select("match_key").n_unique(),
      "regime B matches:", R.filter(~pl.col("regA")).select("match_key").n_unique())
print("bomb_side not null:", R.filter(pl.col("bomb_side").is_not_null()).height,
      "reason_side null:", R.filter(pl.col("reason_side").is_null()).height,
      "step_side null:", R.filter(pl.col("step_side").is_null()).height)
print("reason codes:", R.group_by("win_reason_code").len().sort("win_reason_code").to_dicts())
print("day counts (matches):", R.select(["match_key", "day"]).unique().group_by("day").len().sort("day").to_dicts())
