"""verify_D1_5_c: why do the D4 last-victim rule and the alive-count rule disagree on round 13 (d12-25)?

For non-plant 5v5 round-13 rounds (d12-25) decided by alive count (player_info sides):
  - t_zero: tick at which the side with 0 alive lost its last member (max victim tick of that side)
  - last_any: tick of the last death in the round (<= round_end); last_side = its victim side
  - gap_zero = round_end - t_zero (ticks), gap_last = round_end - last_any (ticks), live = round_freeze_end tick
Splits rounds by whether winner_team_code agrees with the alive-count winner, then prints example timelines.

Appends to logs\\swarm2\\verify_D1_5.log. Read-only on the tome.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
LOG = ROOT / "logs" / "swarm2" / "verify_D1_5.log"
_fh = open(LOG, "a", encoding="utf-8")
HI = [f"{d:02d}" for d in range(12, 26)]
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)


def emit(*parts):
    s = " ".join(str(p) for p in parts)
    print(s, flush=True)
    _fh.write(s + "\n")
    _fh.flush()


def hdr(t):
    emit("\n" + "=" * 96 + "\n" + t + "\n" + "=" * 96)


def fdir(ch):
    d = sorted(BASE.glob(f"basic_{ch}.*"))
    assert len(d) == 1, (ch, d)
    return d[0]


def files(ch):
    return sorted(glob.glob(str(fdir(ch) / "dataframe_*")))


def rd(ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files(ch)], how="vertical_relaxed")


hdr("F0. load")
RE = (rd("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
      .unique(["match_key", "round"], keep="first").rename({"tick": "end_tick"}))
RE20 = RE.filter(pl.col("player_count") == 20).with_columns(
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
RE20 = RE20.with_columns(pl.when(pl.col("day").is_in(HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
BS = rd("bomb_state", ["match_key", "round", "event_type"])
PLANTED = BS.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round"]).unique().with_columns(
    pl.lit(True).alias("planted"))
RSX = rd("round_state", ["match_key", "round", "tick", "event_type"])
LIVE = (RSX.filter(pl.col("event_type") == "round_freeze_end").group_by(["match_key", "round"])
        .agg(pl.col("tick").min().alias("live")))
PI = rd("player_info", ["match_key", "round", "player_id_fixed", "team_code"]).with_columns(
    pl.col("player_id_fixed").cast(pl.Int64).alias("pid"), pl.col("team_code").cast(pl.Int64))
dupk = PI.group_by(["match_key", "round", "pid"]).len().filter(pl.col("len") > 1).select("match_key").unique()
PI = PI.join(dupk, on="match_key", how="anti")
ROS = PI.group_by(["match_key", "round"]).agg((pl.col("team_code") == 2).sum().alias("nT"),
                                              (pl.col("team_code") == 3).sum().alias("nCT"))
DE = rd("player_death", ["match_key", "round", "tick", "player_id_fixed"]).with_columns(
    pl.col("player_id_fixed").cast(pl.Int64).alias("pid"))
# victims with side and tick, deduplicated per round, restricted to ticks <= round_end
V = (DE.join(RE20.select(["match_key", "round", "end_tick"]), on=["match_key", "round"], how="inner")
       .filter(pl.col("tick") <= pl.col("end_tick"))
       .join(PI.select(["match_key", "round", "pid", "team_code"]), on=["match_key", "round", "pid"], how="inner")
       .unique(["match_key", "round", "pid"], keep="first"))
emit("victim rows (<= round_end, roster-matched, deduplicated):", V.height)

VT = V.filter(pl.col("team_code") == 2).group_by(["match_key", "round"]).agg(
    pl.col("tick").max().alias("t_last_T"), pl.len().alias("vT"))
VCT = V.filter(pl.col("team_code") == 3).group_by(["match_key", "round"]).agg(
    pl.col("tick").max().alias("t_last_CT"), pl.len().alias("vCT"))
LAST = V.group_by(["match_key", "round"]).agg(pl.col("tick").max().alias("last_any"))
LASTSIDE = (V.join(LAST, on=["match_key", "round"]).filter(pl.col("tick") == pl.col("last_any"))
              .group_by(["match_key", "round"]).agg(pl.col("team_code").first().alias("last_side")))

F = (RE20.join(ROS, on=["match_key", "round"], how="inner")
     .filter((pl.col("nT") == 5) & (pl.col("nCT") == 5))
     .join(PLANTED, on=["match_key", "round"], how="left").with_columns(pl.col("planted").fill_null(False))
     .filter(~pl.col("planted"))
     .join(VT, on=["match_key", "round"], how="left").join(VCT, on=["match_key", "round"], how="left")
     .join(LAST, on=["match_key", "round"], how="left").join(LASTSIDE, on=["match_key", "round"], how="left")
     .join(LIVE, on=["match_key", "round"], how="left")
     .with_columns(pl.col("vT").fill_null(0), pl.col("vCT").fill_null(0))
     .with_columns(
         (pl.col("nT") - pl.col("vT")).alias("aT"), (pl.col("nCT") - pl.col("vCT")).alias("aCT"))
     .with_columns(
         pl.when((pl.col("aT") == 0) & (pl.col("aCT") > 0)).then(pl.lit(3))
           .when((pl.col("aCT") == 0) & (pl.col("aT") > 0)).then(pl.lit(2))
           .otherwise(pl.lit(0)).cast(pl.Int64).alias("alive_win"))
     .filter(pl.col("alive_win").is_in([2, 3]))
     .with_columns(
         # the tick at which the side with zero alive lost its last member
         pl.when(pl.col("alive_win") == 3).then(pl.col("t_last_T")).otherwise(pl.col("t_last_CT")).alias("t_zero"),
         (pl.col("winner_team_code") != pl.col("alive_win")).alias("disagree"))
     .with_columns(
         (pl.col("end_tick") - pl.col("t_zero")).alias("gap_zero"),
         (pl.col("end_tick") - pl.col("last_any")).alias("gap_last"),
         (pl.col("last_side") == pl.col("alive_win")).alias("last_is_losing_side"),
         (pl.col("t_zero") - pl.col("live")).alias("zero_after_live"))
     .with_columns(pl.col("alive_win").alias("alive_win_code")))
emit("F decided d12-25 and d01-07 round 13 and 25 and 28 (non-plant 5v5):", F.height)

for rnd in [13]:
    G = F.filter((pl.col("round") == rnd) & (pl.col("dg") == "d12-25"))
    hdr(f"F1. round {rnd}, d12-25, non-plant 5v5 decided: split by disagreement")
    emit("rows:", G.height, " disagree (winner != alive):", G.filter(pl.col("disagree")).height)
    emit(G.group_by("disagree").agg(
        pl.len().alias("n"),
        pl.col("gap_zero").quantile(0.5, "nearest").alias("gap_zero_p50"),
        pl.col("gap_zero").max().alias("gap_zero_max"),
        pl.col("gap_last").quantile(0.5, "nearest").alias("gap_last_p50"),
        pl.col("gap_last").max().alias("gap_last_max"),
        pl.col("last_is_losing_side").mean().round(4).alias("share_last_victim_is_losing_side"),
        pl.col("zero_after_live").quantile(0.5, "nearest").alias("zero_after_live_p50")))
    emit("winner_team_code x alive_win x last-victim side (is last victim on the alive_win loser side?):")
    emit(G.group_by(["winner_team_code", "alive_win", "last_side"]).len().sort(["winner_team_code", "alive_win", "last_side"]))

hdr("F2. example timelines (round 13, d12-25, disagreeing rows): victims sorted by tick")
ex = (F.filter((pl.col("round") == 13) & (pl.col("dg") == "d12-25") & pl.col("disagree"))
        .select(["match_key", "end_tick", "live", "winner_team_code", "alive_win", "nT", "nCT", "t_zero", "last_any"])
        .head(3))
emit(ex)
for row in ex.iter_rows(named=True):
    d = (V.filter((pl.col("match_key") == row["match_key"]) & (pl.col("round") == 13))
          .select(["tick", "team_code", "pid"]).sort("tick"))
    emit("match", row["match_key"], " winner", row["winner_team_code"], " alive_win", row["alive_win"],
         " live", row["live"], " end", row["end_tick"], " victims", d.height)
    emit(d.rows())

hdr("F3. same metrics for round 13 d01-07 (clean era) for comparison")
G0 = F.filter((pl.col("round") == 13) & (pl.col("dg") == "d01-07"))
emit("rows:", G0.height, " disagree:", G0.filter(pl.col("disagree")).height)
emit(G0.group_by("disagree").agg(pl.col("gap_zero").quantile(0.5, "nearest").alias("gap_zero_p50"),
                                  pl.col("gap_last").quantile(0.5, "nearest").alias("gap_last_p50"),
                                  pl.col("last_is_losing_side").mean().round(4).alias("share_last_is_losing")))

hdr("F4. round_end timing relative to the losing side's last death: round 13 vs other rounds, d12-25 disagreeing rows")
allD = F.filter(pl.col("dg") == "d12-25").with_columns(
    pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
      .when(pl.col("round") == 28).then(pl.lit("r28")).otherwise(pl.lit("other")).alias("rb"))
emit(allD.group_by(["rb", "disagree"]).agg(pl.len().alias("n"),
                                          pl.col("gap_zero").quantile(0.5, "nearest").alias("gap_zero_p50"),
                                          pl.col("gap_zero").quantile(0.95, "nearest").alias("gap_zero_p95"),
                                          pl.col("last_is_losing_side").mean().round(4).alias("share_last_is_losing"))
     .sort(["rb", "disagree"]))
_fh.close()
print("appended", LOG)
