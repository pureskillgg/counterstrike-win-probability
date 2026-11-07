"""D7-11: onset of the winner-code flip using the ledger on ALL first-after-swap rounds (not only bomb rounds).
fas = round r such that the min-swap ledger path has a swap between r-1 and r (first round after a side change).
Flip = round_end winner_team_code != ledger side. Time = header match_date (UTC).
Also: flip rate for fas rounds vs all other rounds, by period; best single-step onset over matches with a fas round.
ASSUMPTIONS: ledger path from D7_8 (minimum swaps); player_count == 20.
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
X = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_rounds_with_ledger_dp.parquet"))
L = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_ledger_dp.parquet"))
# swap at row r means swap between r and r+1 -> round r+1 is first-after-swap
fas = L.filter(pl.col("swap") == True).with_columns((pl.col("round") + 1).alias("round")).select(["match_key", "round"]).with_columns(pl.lit(True).alias("fas"))
Y = X.join(fas, on=["match_key", "round"], how="left").with_columns(pl.col("fas").fill_null(False))
Y = Y.filter(pl.col("w_ok").is_not_null())
print("ledger-known winner rows:", Y.height, "| first-after-swap rows:", int(Y["fas"].sum()))
print("\nflip rate (winner != ledger) by period x fas:")
print(Y.group_by(["period", "fas"]).agg(pl.len().alias("n"), (~pl.col("w_ok")).sum().alias("flips"),
      (~pl.col("w_ok")).mean().round(4).alias("rate")).sort(["period", "fas"]))
print("\nfirst-after-swap rounds by round number and period (n, flips):")
print(Y.filter(pl.col("fas")).group_by(["period", "round"]).agg(pl.len().alias("n"), (~pl.col("w_ok")).sum().alias("flips"))
      .sort(["period", "round"]).rows())

# onset over first-after-swap rounds in regulation-half changes (round 9 and 13), one row per match-round
F = Y.filter(pl.col("fas") & pl.col("round").is_in([9, 13])).sort("mdt")
y = F["w_ok"].cast(pl.Int8).to_list()   # 1 = agrees (pre), 0 = flipped
t = F["mdt"].to_list()
n = len(y)
pre_flip = [0]*(n+1); pre_ok = [0]*(n+1)
for i, v in enumerate(y):
    pre_flip[i+1] = pre_flip[i] + (1 - v); pre_ok[i+1] = pre_ok[i] + v
best = min(range(n+1), key=lambda c: pre_flip[c] + (pre_ok[n] - pre_ok[c]))
err = pre_flip[best] + (pre_ok[n] - pre_ok[best])
print(f"\nonset over first-after-swap rounds 9/13 (n={n}): best single step errors={err}")
print("  last agreeing before cut:", t[best-1] if best > 0 else None, "| first flip at/after cut:", t[best] if best < n else None)
print("  flips before cut:", pre_flip[best], "| agreeing after cut:", pre_ok[n]-pre_ok[best])
print("  all agreeing rows, latest:", max(t[i] for i in range(n) if y[i] == 1))
print("  all flipped rows, earliest:", min(t[i] for i in range(n) if y[i] == 0))
for lo, hi in [((2024,2,6,0), (2024,2,7,0)), ((2024,2,7,0), (2024,2,7,6)), ((2024,2,7,6), (2024,2,8,0))]:
    import datetime as dt
    a = dt.datetime(*lo); b = dt.datetime(*hi)
    seg = [(t[i], y[i]) for i in range(n) if a <= t[i] < b]
    print(f"  window {a} .. {b}: rows={len(seg)} flipped={sum(1 for _, v in seg if v == 0)} agree={sum(1 for _, v in seg if v == 1)}")
print("\nfirst 12 post-onset fas rows (time, agrees?, round, map):")
Fs = F.filter(pl.col("mdt") >= pl.datetime(2024, 2, 6, 20)).sort("mdt").head(12)
print(Fs.select(["mdt", "round", "w_ok", "map_name"]))
