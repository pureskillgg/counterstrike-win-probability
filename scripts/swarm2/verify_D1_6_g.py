"""verify D1-6 (swarm2 verifier), part G: round-record coverage around the defect loci, and independent elimination and
bomb checks on the round-13 and round-9 mismatch rows (d12-25, all player_counts unless stated).
  G1 round_end presence: for each d12-25 match, is round 8/9/10 and 11/12/13/14 present in round_end? Mismatch rate of
     round 9 by presence of round 8 and 10; of round 13 by presence of round 12 and 14.
  G2 elimination (all CT dead -> T; all T dead and no bomb_planted -> CT) and bomb (defused -> CT, exploded -> T) agreement
     with the ledger and with the winner, for round-13 and round-9 rows (mismatch and agreeing).
ASSUMPTIONS: as in parts B-F. Ledger = freeze-to-freeze increment (no swap between r and r+1 for r = 9, 13).
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
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("t0"),
                                            pl.col("ct_score").first().alias("c0")))
FZn = FZ.with_columns((pl.col("round") - 1).alias("round")).select(
    ["match_key", "round", pl.col("t0").alias("t1"), pl.col("c0").alias("c1")])
LED = FZ.join(FZn, on=["match_key", "round"], how="left").with_columns(
    pl.when(((pl.col("t1") - pl.col("t0")) == 1) & ((pl.col("c1") - pl.col("c0")) == 0)).then(pl.lit(2))
      .when(((pl.col("c1") - pl.col("c0")) == 1) & ((pl.col("t1") - pl.col("t0")) == 0)).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("ledger_side")).select(["match_key", "round", "ledger_side"])
A = (RE.join(LED, on=["match_key", "round"], how="left")
     .with_columns(((pl.col("ledger_side").is_not_null()) & (pl.col("ledger_side") != pl.col("winner_team_code")))
                   .alias("mm"),
                   pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")))
A = A.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))

# ----------------------------------------------------------------------------- G1 presence
hdr("G1a. round_end presence of rounds 8..14 among d12-25 matches that have round 11 (any player_count)")
P = (A.filter(pl.col("dg") == "d12-25").select(["match_key", "round"]).with_columns(pl.lit(1).alias("p")))
mk11 = P.filter(pl.col("round") == 11)["match_key"].unique()
pres = (P.filter(pl.col("match_key").is_in(mk11)).pivot(on="round", index="match_key", values="p", aggregate_function="first")
        .fill_null(0))
cols = [str(r) for r in range(8, 15) if str(r) in pres.columns]
print("matches with round 11 in round_end:", pres.height)
print("presence rates:", {c: round(float(pres[c].mean()), 4) for c in cols})

hdr("G1b. round-13 ledger mismatch rate by presence of round 12 and round 14 (d12-25)")
r13 = A.filter((pl.col("round") == 13) & (pl.col("dg") == "d12-25") & pl.col("ledger_side").is_not_null())
pres_small = pres.select(["match_key"] + [c for c in ["12", "14"] if c in pres.columns])
x = r13.join(pres_small, on="match_key", how="left").with_columns(pl.col("12").fill_null(0), pl.col("14").fill_null(0))
print(x.group_by(["12", "14"]).agg(pl.len().alias("rows"), pl.col("mm").sum().alias("mismatch"),
                                   pl.col("mm").mean().round(4).alias("rate")).sort(["12", "14"]))

hdr("G1c. round-9 ledger mismatch rate by presence of round 8 and round 10 (d12-25)")
r9 = A.filter((pl.col("round") == 9) & (pl.col("dg") == "d12-25") & pl.col("ledger_side").is_not_null())
pres_s9 = pres.select(["match_key"] + [c for c in ["8", "10"] if c in pres.columns])
x9 = r9.join(pres_s9, on="match_key", how="left").with_columns(pl.col("8").fill_null(0), pl.col("10").fill_null(0))
print(x9.group_by(["8", "10"]).agg(pl.len().alias("rows"), pl.col("mm").sum().alias("mismatch"),
                                   pl.col("mm").mean().round(4).alias("rate")).sort(["8", "10"]))

# ----------------------------------------------------------------------------- G2 independent elimination and bomb
hdr("G2. elimination and bomb referees on round-13 and round-9 rows (d12-25, all player_counts)")
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
G = (A.filter(pl.col("dg") == "d12-25").join(NTC, on=["match_key", "round"], how="left")
     .join(DTH, on=["match_key", "round"], how="left").join(BO, on=["match_key", "round"], how="left")
     .join(PL, on=["match_key", "round"], how="left"))
G = G.with_columns(pl.col("dT", "dCT", "planted").fill_null(0))
G = G.with_columns(
    pl.when((pl.col("dCT") == pl.col("nCT")) & (pl.col("nCT") > 0)).then(pl.lit(2))
      .when((pl.col("dT") == pl.col("nT")) & (pl.col("nT") > 0) & (pl.col("planted") == 0)).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("elim_side"))
for rr in [13, 9]:
    g = G.filter(pl.col("round") == rr)
    for lab, cond in [("MISMATCH (ledger != winner)", pl.col("mm")), ("AGREE (ledger == winner)", ~pl.col("mm"))]:
        s = g.filter(cond & pl.col("ledger_side").is_not_null())
        n_e = s.filter(pl.col("elim_side").is_not_null())
        n_b = s.filter(pl.col("bomb_side").is_not_null())
        print(f"round {rr} {lab}: rows={s.height}")
        print(f"   elimination known={n_e.height}  elim==ledger {int((n_e['elim_side'] == n_e['ledger_side']).sum())}"
              f"  elim==winner {int((n_e['elim_side'] == n_e['winner_team_code']).sum())}")
        print(f"   bomb known={n_b.height}  bomb==ledger {int((n_b['bomb_side'] == n_b['ledger_side']).sum())}"
              f"  bomb==winner {int((n_b['bomb_side'] == n_b['winner_team_code']).sum())}")
