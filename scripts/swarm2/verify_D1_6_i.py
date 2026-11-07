"""verify D1-6 (swarm2 verifier), part I: referee check on the other ledger-mismatch loci (25, 27, 28), d01-07 and d12-25.
Elimination (all CT dead -> T; all T dead and no bomb_planted -> CT) and bomb (defused -> CT, exploded -> T) versus the
ledger and versus winner_team_code. Round 27 is the overtime side-swap round (bake-off rule: swap after round 27), so the
freeze-to-freeze ledger for round 27 crosses a swap and is expected to be inverted by construction; this script tests that.
ASSUMPTIONS: as in parts B-H. Ledger = freeze increment r -> r+1 without swap adjustment (as in part B).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(24)
pl.Config.set_fmt_str_lengths(40)

T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF24 = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]
ROUNDS = [25, 27, 28]


def f24(ch):
    return sorted(glob.glob(str(T24 / f"basic_{ch}{SUF24}" / "dataframe_*")))


def rd(files, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


RE = (rd(f24("round_end"), ["match_key", "round", "tick", "winner_team_code", "player_count"])
      .unique(["match_key", "round"], keep="first", maintain_order=True).rename({"tick": "end_tick"}))
RS = rd(f24("round_state"), ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
FZ = (RS.filter(pl.col("event_type") == "round_freeze_end")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("t0"), pl.col("ct_score").first().alias("c0")))
FZn = FZ.with_columns((pl.col("round") - 1).alias("round")).select(
    ["match_key", "round", pl.col("t0").alias("t1"), pl.col("c0").alias("c1")])
LED = FZ.join(FZn, on=["match_key", "round"], how="left").with_columns(
    pl.when(((pl.col("t1") - pl.col("t0")) == 1) & ((pl.col("c1") - pl.col("c0")) == 0)).then(pl.lit(2))
      .when(((pl.col("c1") - pl.col("c0")) == 1) & ((pl.col("t1") - pl.col("t0")) == 0)).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("ledger_side")).select(["match_key", "round", "ledger_side"])
A = (RE.join(LED, on=["match_key", "round"], how="left")
     .with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")))
A = A.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
A = A.filter(pl.col("round").is_in(ROUNDS))

PI = rd(f24("player_info"), ["match_key", "round", "player_id_fixed", "team_code"]).unique(
    ["match_key", "round", "player_id_fixed"], keep="first")
NTC = PI.group_by(["match_key", "round"]).agg((pl.col("team_code") == 2).sum().alias("nT"),
                                              (pl.col("team_code") == 3).sum().alias("nCT"))
PD = (rd(f24("player_death"), ["match_key", "round", "tick", "player_id_fixed", "player_team_code"])
      .drop_nulls(["player_id_fixed", "player_team_code"]).with_columns(pl.col("player_id_fixed").cast(pl.Int64))
      .join(RE.select(["match_key", "round", "end_tick"]), on=["match_key", "round"], how="inner")
      .filter(pl.col("tick") <= pl.col("end_tick"))
      .unique(["match_key", "round", "player_id_fixed"], keep="first"))
DTH = PD.group_by(["match_key", "round"]).agg((pl.col("player_team_code") == 2).sum().alias("dT"),
                                              (pl.col("player_team_code") == 3).sum().alias("dCT"))
BS = rd(f24("bomb_state"), ["match_key", "round", "event_type"])
BO = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
      .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3))
                    .otherwise(pl.lit(2)).alias("bomb_side")).select(["match_key", "round", "bomb_side"]))
PL = (BS.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round"]).unique()
      .with_columns(pl.lit(1).alias("planted")))
G = (A.join(NTC, on=["match_key", "round"], how="left").join(DTH, on=["match_key", "round"], how="left")
     .join(BO, on=["match_key", "round"], how="left").join(PL, on=["match_key", "round"], how="left"))
G = G.with_columns(pl.col("dT", "dCT", "planted").fill_null(0))
G = G.with_columns(
    pl.when((pl.col("dCT") == pl.col("nCT")) & (pl.col("nCT") > 0)).then(pl.lit(2))
      .when((pl.col("dT") == pl.col("nT")) & (pl.col("nT") > 0) & (pl.col("planted") == 0)).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("elim_side"))
G = G.with_columns(((pl.col("ledger_side").is_not_null()) & (pl.col("ledger_side") != pl.col("winner_team_code")))
                   .alias("mm"))

hdr("I1. ledger vs winner, elimination and bomb referees by round and day group")
for dgv in ["d12-25", "d01-07"]:
    for rr in ROUNDS:
        g = G.filter((pl.col("dg") == dgv) & (pl.col("round") == rr) & pl.col("ledger_side").is_not_null())
        if g.height == 0:
            print(f"{dgv} round {rr}: no rows")
            continue
        mm = g.filter(pl.col("mm"))
        ok = g.filter(~pl.col("mm"))
        e_mm = mm.filter(pl.col("elim_side").is_not_null())
        b_mm = mm.filter(pl.col("bomb_side").is_not_null())
        e_ok = ok.filter(pl.col("elim_side").is_not_null())
        b_ok = ok.filter(pl.col("bomb_side").is_not_null())
        print(f"{dgv} round {rr}: rows={g.height}  mismatch={mm.height}")
        print(f"   MISMATCH elim known={e_mm.height} elim==ledger {int((e_mm['elim_side'] == e_mm['ledger_side']).sum())}"
              f" elim==winner {int((e_mm['elim_side'] == e_mm['winner_team_code']).sum())}"
              f" | bomb known={b_mm.height} bomb==ledger {int((b_mm['bomb_side'] == b_mm['ledger_side']).sum())}"
              f" bomb==winner {int((b_mm['bomb_side'] == b_mm['winner_team_code']).sum())}")
        print(f"   AGREE    elim known={e_ok.height} elim==winner {int((e_ok['elim_side'] == e_ok['winner_team_code']).sum())}"
              f" | bomb known={b_ok.height} bomb==winner {int((b_ok['bomb_side'] == b_ok['winner_team_code']).sum())}")
