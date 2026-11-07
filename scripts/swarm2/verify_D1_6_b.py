"""verify D1-6 (swarm2 verifier), part B: independent winner signals for round 9.
Signals per round (no bomb channel needed for the main test):
  LEDGER  : side whose round_freeze_end score rises by 1 between round r and round r+1 (no side swap after round 9).
  REASON  : side implied by win_reason_code (mapping learned from the 2026 tome, part A section 4).
  ELIM    : side implied by eliminations: all 5 CT dead -> T wins; all 5 T dead and no bomb_planted -> CT wins.
  BOMB    : bomb_defused -> CT, bomb_exploded -> T (the claim's signal).
  CLAIM   : claim's score side (first/last round_state row, all event types).
ASSUMPTIONS:
  B1 round 9 and round 8 have no side swap after them (swap only after round 12 in regulation; bake-off swap rule).
  B2 deaths counted once per (match, round, player_id_fixed) with tick <= round_end tick.
  B3 2024 universe = player_count 20 (verified in part A to be 10 players per round: 5 T, 5 CT).
  B4 2026 universe = player_count 10.
  B5 REASON mapping: 1->T, 7->CT, 8->CT, 9->T, 11->CT, 12->CT, 13->T, 17->CT, 18->T (from the 2026 tome, zero exceptions).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(24)
pl.Config.set_fmt_str_lengths(50)

T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
SUF24 = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]
REASON_SIDE = {1: 2, 7: 3, 8: 3, 9: 2, 11: 3, 12: 3, 13: 2, 17: 3, 18: 2}


def f24(ch):
    return sorted(glob.glob(str(T24 / f"basic_{ch}{SUF24}" / "dataframe_*")))


def f26(ch):
    return sorted(glob.glob(str(T26 / f"basic_{ch}*" / "dataframe_*")))


def rd(files, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def side_all(RS):
    sc = (RS.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
        pl.col("t_score").sort_by("tick").first().alias("t_before"),
        pl.col("ct_score").sort_by("tick").first().alias("ct_before"),
        pl.col("t_score").sort_by("tick").last().alias("t_after"),
        pl.col("ct_score").sort_by("tick").last().alias("ct_after")))
    return sc.with_columns(
        pl.when((pl.col("t_after") - pl.col("t_before")) == 1).then(pl.lit(2))
          .when((pl.col("ct_after") - pl.col("ct_before")) == 1).then(pl.lit(3))
          .otherwise(pl.lit(None, dtype=pl.Int64)).alias("score_side")).select(["match_key", "round", "score_side"])


def load(tome):
    fx = f24 if tome == "2024" else f26
    pcv = 20 if tome == "2024" else 10
    RE = (rd(fx("round_end"), ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
          .unique(["match_key", "round"], keep="first", maintain_order=True)
          .rename({"tick": "end_tick"}))
    RS = rd(fx("round_state"), ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    FZ = (RS.filter(pl.col("event_type") == "round_freeze_end")
          .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("t0"),
                                                pl.col("ct_score").first().alias("c0")))
    FZn = FZ.with_columns((pl.col("round") - 1).alias("round")).select(
        ["match_key", "round", pl.col("t0").alias("t1"), pl.col("c0").alias("c1")])
    LED = FZ.join(FZn, on=["match_key", "round"], how="inner").with_columns(
        pl.when(((pl.col("t1") - pl.col("t0")) == 1) & ((pl.col("c1") - pl.col("c0")) == 0)).then(pl.lit(2))
          .when(((pl.col("c1") - pl.col("c0")) == 1) & ((pl.col("t1") - pl.col("t0")) == 0)).then(pl.lit(3))
          .otherwise(pl.lit(None, dtype=pl.Int64)).alias("ledger_side")).select(["match_key", "round", "ledger_side"])
    SC = side_all(RS)
    PI = rd(fx("player_info"), ["match_key", "round", "player_id_fixed", "team_code"]).unique(
        ["match_key", "round", "player_id_fixed"], keep="first")
    NTC = PI.group_by(["match_key", "round"]).agg((pl.col("team_code") == 2).sum().alias("nT"),
                                                  (pl.col("team_code") == 3).sum().alias("nCT"))
    PD = (rd(fx("player_death"), ["match_key", "round", "tick", "player_id_fixed", "player_team_code"])
          .drop_nulls(["player_id_fixed", "player_team_code"])
          .with_columns(pl.col("player_id_fixed").cast(pl.Int64))
          .join(RE.select(["match_key", "round", "end_tick"]), on=["match_key", "round"], how="inner")
          .filter(pl.col("tick") <= pl.col("end_tick"))
          .unique(["match_key", "round", "player_id_fixed"], keep="first"))
    DTH = PD.group_by(["match_key", "round"]).agg((pl.col("player_team_code") == 2).sum().alias("dT"),
                                                  (pl.col("player_team_code") == 3).sum().alias("dCT"))
    BS = rd(fx("bomb_state"), ["match_key", "round", "event_type"])
    BO = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
          .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
          .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3))
                        .otherwise(pl.lit(2)).alias("bomb_side")).select(["match_key", "round", "bomb_side"]))
    PL = (BS.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round"]).unique()
          .with_columns(pl.lit(1).alias("planted")))
    reason = None
    for code, side in REASON_SIDE.items():
        cond = pl.col("win_reason_code") == code
        reason = pl.when(cond).then(pl.lit(side)) if reason is None else reason.when(cond).then(pl.lit(side))
    reason = reason.otherwise(pl.lit(None, dtype=pl.Int64)).alias("reason_side")
    R = (RE.join(SC, on=["match_key", "round"], how="left")
         .join(LED, on=["match_key", "round"], how="left")
         .join(NTC, on=["match_key", "round"], how="left")
         .join(DTH, on=["match_key", "round"], how="left")
         .join(BO, on=["match_key", "round"], how="left")
         .join(PL, on=["match_key", "round"], how="left"))
    R = R.with_columns(pl.col("dT", "dCT", "planted").fill_null(0), reason)
    R = R.with_columns(
        pl.when((pl.col("dCT") == pl.col("nCT")) & (pl.col("nCT") > 0)).then(pl.lit(2))
          .when((pl.col("dT") == pl.col("nT")) & (pl.col("nT") > 0) & (pl.col("planted") == 0)).then(pl.lit(3))
          .otherwise(pl.lit(None, dtype=pl.Int64)).alias("elim_side"))
    if tome == "2024":
        R = R.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
        R = R.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25"))
                           .otherwise(pl.lit("d01-07")).alias("dg"))
    else:
        R = R.with_columns(pl.lit("2026").alias("dg"))
    R = R.with_columns(
        ((pl.col("score_side").is_not_null()) & (pl.col("score_side") != pl.col("winner_team_code"))).alias("mm"))
    return R.filter(pl.col("player_count") == pcv), RE


def agree(df, a, b):
    d = df.filter(pl.col(a).is_not_null() & pl.col(b).is_not_null())
    return d.height, int((d[a] == d[b]).sum())


def signals(df, label):
    print(f"\n-- {label}: rows={df.height}")
    for name in ["ledger_side", "reason_side", "elim_side", "bomb_side", "score_side"]:
        n_w, eq_w = agree(df, name, "winner_team_code")
        print(f"   {name:<12} vs winner_team_code: non-null={n_w:>5}  equal={eq_w:>5}  "
              f"({(eq_w / n_w * 100 if n_w else float('nan')):.2f}%)")
    n_l, eq_ls = agree(df, "ledger_side", "score_side")
    print(f"   ledger vs claim score_side:     non-null={n_l:>5}  equal={eq_ls:>5}")


# ----------------------------------------------------------------------------- 2024
R24, RE24 = load("2024")
hdr("B1. 2024 (pc==20) round 9: signals in mismatch rows vs agreeing rows, d12-25")
r9 = R24.filter((pl.col("round") == 9) & (pl.col("dg") == "d12-25") & pl.col("score_side").is_not_null())
mm9 = r9.filter(pl.col("mm"))
ok9 = r9.filter(~pl.col("mm"))
print("round 9 d12-25 rows with claim score side:", r9.height, " mismatches:", mm9.height, " agreeing:", ok9.height)
signals(mm9, "MISMATCH rows (winner != claim score side)")
signals(ok9, "AGREEING rows (control)")
hdr("B1b. 2024 round 9 mismatch rows: crosstab of signals")
print(mm9.group_by(["winner_team_code", "score_side", "ledger_side", "reason_side", "elim_side", "bomb_side"])
      .len().sort("len", descending=True))
hdr("B1c. 2024 round 9 d01-07 (pc==20) control, signals")
signals(R24.filter((pl.col("round") == 9) & (pl.col("dg") == "d01-07") & pl.col("score_side").is_not_null()),
        "d01-07 round 9 rows with claim score side")

hdr("B2. 2024 ledger vs winner vs claim score side by round (pc==20), d12-25 and d01-07")
tab = (R24.filter(pl.col("ledger_side").is_not_null() & pl.col("score_side").is_not_null())
       .with_columns((pl.col("ledger_side") == pl.col("winner_team_code")).alias("L_eq_W"),
                     (pl.col("ledger_side") == pl.col("score_side")).alias("L_eq_S"))
       .group_by(["dg", "round"]).agg(pl.len().alias("n"), pl.col("L_eq_W").sum().alias("ledger=winner"),
                                      pl.col("L_eq_S").sum().alias("ledger=claimscore"))
       .sort(["dg", "round"]))
print(tab.filter(pl.col("round") <= 30))

hdr("B3. 2024 round 9 d12-25: mismatch rows, ledger direction and reason vs winner (per match)")
print("mismatch matches:", mm9["match_key"].n_unique(), " rows:", mm9.height)
print("ledger == winner_team_code:", int((mm9["ledger_side"] == mm9["winner_team_code"]).sum()),
      " ledger == claim score_side:", int((mm9["ledger_side"] == mm9["score_side"]).sum()),
      " ledger null:", int(mm9["ledger_side"].is_null().sum()))
print("elim side non-null:", int(mm9["elim_side"].is_not_null().sum()),
      " elim == winner:", int((mm9["elim_side"] == mm9["winner_team_code"]).sum()),
      " elim == score:", int((mm9["elim_side"] == mm9["score_side"]).sum()))

# ----------------------------------------------------------------------------- 2026 control
R26, RE26 = load("2026")
hdr("B4. 2026 (pc==10) round 9: signals, all rows with claim score side (control)")
r26 = R26.filter((pl.col("round") == 9) & pl.col("score_side").is_not_null())
signals(r26, "2026 round 9 pc10 rows with claim score side")
hdr("B4b. 2026 (pc==10) round 27 and round 13: claim score side vs winner (control for derivation failure modes)")
for rr in [13, 27]:
    signals(R26.filter((pl.col("round") == rr) & pl.col("score_side").is_not_null()), f"2026 round {rr} pc10")
