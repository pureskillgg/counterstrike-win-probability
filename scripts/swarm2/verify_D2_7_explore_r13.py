"""D2-7 verifier, exploration: is the 2024 round-13 flip a winner-label error or a side-swap artifact?
Independent sources per (match_key, round): round_end winner and win_reason, bomb outcome (bomb_state and
round_state), freeze-end scores (round_state) for round k and k+1, header match_date and final scores."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(24)
pl.Config.set_fmt_str_lengths(40)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def read(folder, cols):
    fs = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def read_ch(ch, cols):
    return read(BASE / f"basic_{ch}{SUF}", cols)


RE = read_ch("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
RS = read_ch("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw"])
FZ = (RS.filter(pl.col("event_type") == "round_freeze_end")
      .select(["match_key", "round", "tick", "t_score", "ct_score", "t_score_raw", "ct_score_raw"])
      .unique(["match_key", "round"], keep="first"))
BS = read_ch("bomb_state", ["match_key", "round", "tick", "event_type"])
hdr_schema = pl.read_parquet_schema(sorted(glob.glob(str(BASE / "header" / "dataframe_*")))[0])
print("header schema:", dict(hdr_schema))
HD = read(BASE / "header", ["key", "match_date", "t_starters_score_final", "ct_starters_score_final"])
HD = HD.rename({"key": "match_key"})
HD = HD.with_columns(pl.col("match_date").cast(pl.String).str.slice(0, 10).alias("date"))
DAY = HD.select(["match_key", "date"]).with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))

RE1 = RE.unique(["match_key", "round"], keep="first")
bomb_bs = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
           .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("bs_ev")))
bomb_rs = (RS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
           .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("rs_ev")))

base = (RE1.join(FZ.rename({"tick": "freeze_tick"}), on=["match_key", "round"], how="left")
        .join(bomb_bs, on=["match_key", "round"], how="left")
        .join(bomb_rs, on=["match_key", "round"], how="left")
        .join(DAY, on="match_key", how="left"))
nxt = FZ.select(["match_key", "round", pl.col("t_score").alias("t_next"), pl.col("ct_score").alias("ct_next"),
                 pl.col("round").alias("r_prev")]).with_columns((pl.col("round") - 1).alias("round"))
nxt = nxt.select(["match_key", "round", "t_next", "ct_next"])
base = base.join(nxt, on=["match_key", "round"], how="left")
base = base.with_columns(
    pl.when(pl.col("bs_ev").list.contains("bomb_defused")).then(pl.lit("defused"))
    .when(pl.col("bs_ev").list.contains("bomb_exploded")).then(pl.lit("exploded")).otherwise(pl.lit("none")).alias("bomb_bs"),
    pl.when(pl.col("rs_ev").list.contains("bomb_defused")).then(pl.lit("defused"))
    .when(pl.col("rs_ev").list.contains("bomb_exploded")).then(pl.lit("exploded")).otherwise(pl.lit("none")).alias("bomb_rs"),
    pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
    .when(pl.col("round") == 28).then(pl.lit("r28")).otherwise(pl.lit("other")).alias("rbin"),
    pl.col("winner_team_code").alias("winner"),
)
base = base.with_columns(
    # score-derived winner for round k: increment of the score that is on the same side in k and k+1
    # (no-swap reading: next T score = T score + T win)
    pl.when((pl.col("t_next") - pl.col("t_score") == 1) & (pl.col("ct_next") - pl.col("ct_score") == 0)).then(pl.lit(T))
    .when((pl.col("ct_next") - pl.col("ct_score") == 1) & (pl.col("t_next") - pl.col("t_score") == 0)).then(pl.lit(CT))
    .otherwise(pl.lit(None)).alias("score_winner_noswap"),
    # swap reading: next T score belongs to the team that was CT in round k
    pl.when((pl.col("t_next") - pl.col("ct_score") == 1) & (pl.col("ct_next") - pl.col("t_score") == 0)).then(pl.lit(T))
    .when((pl.col("ct_next") - pl.col("ct_score") == 1) & (pl.col("t_next") - pl.col("t_score") == 0)).then(pl.lit(CT))
    .otherwise(pl.lit(None)).alias("score_winner_swap"),
)
base.write_parquet(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\d27_rounds_2024.parquet")
print("rounds table rows:", base.height)

print("\n== bomb outcome from bomb_state vs round_state (all rounds) ==")
print(base.group_by(["bomb_bs", "bomb_rs"]).len().sort(["bomb_bs", "bomb_rs"]))

print("\n== round 13 : winner x bomb_bs x win_reason (all days) ==")
r13 = base.filter(pl.col("round") == 13)
print(r13.group_by(["winner", "win_reason_code", "bomb_bs"]).len().sort(["winner", "win_reason_code", "bomb_bs"]))

print("\n== round 13 : winner vs score-derived winner (noswap / swap readings) ==")
print(r13.group_by(["winner", "score_winner_noswap", "score_winner_swap"]).len().sort(["winner", "score_winner_noswap", "score_winner_swap"]))

print("\n== round 13 by day: n, winner==bomb_bs mismatch, winner==score_noswap, winner==score_swap ==")
r13 = r13.with_columns(
    (pl.col("bomb_bs") == "exploded").cast(pl.Int64).alias("x"),
    pl.when(pl.col("bomb_bs") == "exploded").then(pl.lit(T)).when(pl.col("bomb_bs") == "defused").then(pl.lit(CT)).otherwise(pl.lit(None)).alias("bomb_side"),
)
g = r13.group_by("day").agg(
    pl.len().alias("n"),
    (pl.col("bomb_side").is_not_null() & (pl.col("bomb_side") != pl.col("winner"))).sum().alias("bomb_vs_winner_mm"),
    pl.col("bomb_side").is_not_null().sum().alias("n_bomb"),
    (pl.col("score_winner_noswap") == pl.col("winner")).sum().alias("noswap_agree"),
    pl.col("score_winner_noswap").is_not_null().sum().alias("noswap_n"),
    (pl.col("score_winner_swap") == pl.col("winner")).sum().alias("swap_agree"),
    pl.col("score_winner_swap").is_not_null().sum().alias("swap_n"),
).sort("day")
print(g)
