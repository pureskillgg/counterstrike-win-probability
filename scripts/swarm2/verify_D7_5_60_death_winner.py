"""verify_D7_5_60_death_winner: third, non-bomb check for the round winner side.

Death-based label (W_death): in a round that ended by elimination (win_reason_code 8 = CTs_Win or 9 = Terrorists_Win)
with no bomb outcome, the last player killed belongs to the losing side, so the winner is the other side.
Uses basic_player_death.player_team_code (victim side at death) and tick ordering inside (match, round).
Compared against W_code, W_within and W_dp by period and round number.
ASSUMPTION: player_team_code is a side code (2 = T, 3 = CT) at the time of death (as used for bomb planter side).
ASSUMPTION: a round with a unique latest death tick is decisive; ties are excluded and counted.
"""
import glob
import sys
from pathlib import Path
from collections import Counter
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
R = pl.read_parquet(str(SCRATCH / "verify_D7_5_rounds.parquet"))

fs = sorted(glob.glob(str(BASE / f"basic_player_death{SUF}" / "dataframe_*")))
D = pl.concat([pl.scan_parquet(f) for f in fs]).select(
    ["match_key", "round", "tick", "player_team_code"]).filter(pl.col("player_team_code").is_in([2, 3])).collect()
last = D.group_by(["match_key", "round"]).agg(
    pl.col("tick").max().alias("last_tick"),
    pl.len().alias("n_deaths"))
lastd = D.join(last, on=["match_key", "round"]).filter(pl.col("tick") == pl.col("last_tick"))
ties = lastd.group_by(["match_key", "round"]).agg(pl.col("player_team_code").n_unique().alias("nuniq"), pl.len().alias("nrows"))
amb = ties.filter(pl.col("nuniq") > 1).height
print("rounds with deaths:", last.height, "| rounds with same-tick last death on both sides (ambiguous, excluded):", amb)
lastside = (lastd.group_by(["match_key", "round"]).agg(pl.col("player_team_code").first().alias("victim_side_last"))
            .join(ties.filter(pl.col("nuniq") == 1).select(["match_key", "round"]), on=["match_key", "round"]))
lastside = lastside.with_columns(
    pl.when(pl.col("victim_side_last") == 2).then(3).otherwise(2).alias("W_death"))  # the other side wins

X = R.join(lastside.select(["match_key", "round", "W_death", "victim_side_last"]), on=["match_key", "round"], how="left")
X = X.filter(pl.col("W_bomb").is_null() & pl.col("reason").is_in([8, 9]))
print("elimination-type rounds (reason 8/9, no bomb outcome):", X.height)
print("with W_death:", X.filter(pl.col("W_death").is_not_null()).height)
for lab in ["W_code", "W_within", "W_dp"]:
    s = X.filter(pl.col("W_death").is_not_null() & pl.col(lab).is_not_null())
    eq = s.filter(pl.col(lab) == pl.col("W_death")).height
    print(f"W_death == {lab}: {eq} of {s.height} ({eq / s.height * 100:.2f}%)")

print("\n== by period and round (W_death agreement with each label; n = rounds with W_death and label) ==")
for p in ["pre", "post"]:
    s = X.filter((pl.col("period") == p) & pl.col("W_death").is_not_null())
    g = s.group_by("round").agg(
        pl.len().alias("n"),
        (pl.col("W_code") == pl.col("W_death")).sum().alias("code_eq"),
        pl.col("W_code").is_not_null().sum().alias("n_code"),
        (pl.col("W_within") == pl.col("W_death")).sum().alias("within_eq"),
        pl.col("W_within").is_not_null().sum().alias("n_within"),
        (pl.col("W_dp") == pl.col("W_death")).sum().alias("dp_eq"),
        pl.col("W_dp").is_not_null().sum().alias("n_dp"),
    ).sort("round")
    print(p)
    for row in g.filter(pl.col("round").is_in([1, 2, 8, 9, 12, 13, 14, 24, 25, 26, 27, 28, 29, 30]) | (pl.col("n") > 0)).rows():
        rnd, n, ce, nc, we, nw, de, nd = row
        if rnd in (1, 2, 8, 9, 12, 13, 14, 24, 25, 26, 27, 28, 29, 30):
            print(f"  r{rnd:<3d} n={n:5d} | code {ce:5d}/{nc:5d} | within {we:5d}/{nw:5d} | dp {de:5d}/{nd:5d}")
    tot = s.select(pl.len().alias("n"), (pl.col("W_code") == pl.col("W_death")).sum().alias("code_eq"),
                   (pl.col("W_within") == pl.col("W_death")).sum().alias("within_eq"),
                   pl.col("W_within").is_not_null().sum().alias("n_within")).row(0)
    print(f"  total {p}: n={tot[0]} | code agrees {tot[1]} | within agrees {tot[2]} of {tot[3]}")

print("\n== post, round 13 and 25/28 elimination rounds: death-based vs code, split by map ==")
s = X.filter((pl.col("period") == "post") & pl.col("round").is_in([13, 25, 28, 9]) & pl.col("W_death").is_not_null())
print(s.group_by(["round", "W_death", "W_code"]).len().sort(["round", "W_death", "W_code"]).rows())
