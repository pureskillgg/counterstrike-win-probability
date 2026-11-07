"""D2-7 verifier, exploration: fourth source. Round-end reason vs death events in round 13 (and others).
Reason 9 = Terrorists win (all CTs eliminated); reason 8 = CTs win (all Ts eliminated or time out).
If reason 9 is right, the CT side must have 5 distinct deaths in the round (all_CT_dead)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
TMP = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
base = pl.read_parquet(TMP / "d27_rounds_2024.parquet")


def read(folder, cols):
    fs = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


DE = read(BASE / f"basic_player_death{SUF}", ["match_key", "round", "tick", "player_id", "player_team_code"])
PI = read(BASE / f"basic_player_info{SUF}", ["match_key", "round", "player_id", "team_code"])
# distinct deaths per side per round; team from death row, fallback from player_info (same round)
pi1 = PI.unique(["match_key", "round", "player_id"], keep="first")
d = (DE.join(pi1, on=["match_key", "round", "player_id"], how="left")
     .with_columns(pl.coalesce(["player_team_code", "team_code"]).alias("side"))
     .filter(pl.col("side").is_in([2, 3]))
     .unique(["match_key", "round", "player_id"], keep="first"))
cnt = d.group_by(["match_key", "round", "side"]).len().pivot(on="side", index=["match_key", "round"], values="len").fill_null(0)
cnt = cnt.rename({"2": "t_deaths", "3": "ct_deaths"}) if "2" in cnt.columns else cnt
print("columns after pivot:", cnt.columns)
cnt = cnt.rename({c: ("t_deaths" if c in ("2", 2) else "ct_deaths") for c in cnt.columns if c in ("2", "3", 2, 3)})
j = base.join(cnt, on=["match_key", "round"], how="left").with_columns(
    pl.col("t_deaths").fill_null(0), pl.col("ct_deaths").fill_null(0))
j = j.with_columns(
    pl.when(pl.col("bomb_bs") == "defused").then(pl.lit(3)).when(pl.col("bomb_bs") == "exploded").then(pl.lit(2)).otherwise(pl.lit(None)).alias("bomb_side"),
    pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
    .when(pl.col("round") == 28).then(pl.lit("r28")).otherwise(pl.lit("other")).alias("rbin"))
j = j.with_columns(
    (pl.col("winner") != pl.col("bomb_side")).alias("w_vs_bomb_mm"),
    ((pl.col("day").cast(pl.Int64) >= 12)).alias("late"))

print("\n== reason 9 (T win, all CT dead expected) : all_CT_dead rate, by bucket and day group ==")
r9 = j.filter(pl.col("win_reason_code") == 9)
print(r9.with_columns((pl.col("ct_deaths") == 5).alias("all_ct_dead"))
      .group_by(["rbin", "late"]).agg(pl.len().alias("n"), pl.col("all_ct_dead").mean().alias("rate_all_ct_dead"))
      .sort(["rbin", "late"]))
print("\n== reason 8 (CT win, all T dead expected if elimination) : all_T_dead rate ==")
r8 = j.filter(pl.col("win_reason_code") == 8)
print(r8.with_columns((pl.col("t_deaths") == 5).alias("all_t_dead"))
      .group_by(["rbin", "late"]).agg(pl.len().alias("n"), pl.col("all_t_dead").mean().alias("rate_all_t_dead"))
      .sort(["rbin", "late"]))

print("\n== round 13 flagged rows (winner vs bomb mismatch): reason x CT/T death counts ==")
fl = j.filter((pl.col("round") == 13) & pl.col("w_vs_bomb_mm"))
print(fl.group_by(["winner", "win_reason_code", "bomb_bs"]).agg(
    pl.len().alias("n"), pl.col("ct_deaths").mean().alias("mean_ct_deaths"),
    pl.col("t_deaths").mean().alias("mean_t_deaths"), (pl.col("ct_deaths") == 5).sum().alias("n_all_ct_dead"),
    (pl.col("t_deaths") == 5).sum().alias("n_all_t_dead")).sort(["winner", "bomb_bs"]))
print("\n== round 13 agreeing rows (winner == bomb), days >= 12, for comparison ==")
ok = j.filter((pl.col("round") == 13) & pl.col("late") & pl.col("bomb_side").is_not_null() & ~pl.col("w_vs_bomb_mm"))
print(ok.group_by(["winner", "win_reason_code", "bomb_bs"]).agg(
    pl.len().alias("n"), pl.col("ct_deaths").mean().alias("mean_ct_deaths"),
    pl.col("t_deaths").mean().alias("mean_t_deaths"), (pl.col("ct_deaths") == 5).sum().alias("n_all_ct_dead"),
    (pl.col("t_deaths") == 5).sum().alias("n_all_t_dead")).sort(["winner", "bomb_bs"]))
