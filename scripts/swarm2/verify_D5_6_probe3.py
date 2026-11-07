"""verify D5-6 probe 3: which round channel is aligned with player_death and round_end labels.

Compares round_start, round_freeze_end (round_state) and round_end (round_end channel) ticks per label,
and checks how many victim deaths fall inside each candidate window. Reads the 2024 tome on G: with
column projection and event filters. Prints one example match and global summaries.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_probe3.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(40)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
K = ["match_key", "round"]


def scan(ch, cols):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.scan_parquet(fs).select(cols)


RST = scan("round_start", ["match_key", "round", "tick"]).collect().group_by(K).agg(
    pl.col("tick").min().alias("st"))
RE = scan("round_end", ["match_key", "round", "tick", "winner_team_code"]).collect().group_by(K).agg(
    pl.col("tick").min().alias("et"), pl.col("winner_team_code").first().alias("rw"))
RS = scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).filter(
    pl.col("event_type").is_in(["round_freeze_end", "round_start", "round_end", "bomb_planted", "bomb_defused",
                                "bomb_exploded"])).collect()
FE = (RS.filter(pl.col("event_type") == "round_freeze_end").group_by(K)
      .agg(pl.col("tick").min().alias("fe"), pl.col("t_score").first().alias("fe_t"),
           pl.col("ct_score").first().alias("fe_ct")))
RSST = RS.filter(pl.col("event_type") == "round_start").group_by(K).agg(pl.col("tick").min().alias("rs_st"))
RSEND = RS.filter(pl.col("event_type") == "round_end").group_by(K).agg(pl.col("tick").min().alias("rs_et"))
DE = scan("player_death", ["match_key", "round", "tick", "player_team_code"]).filter(
    pl.col("player_team_code").is_in([2, 3])).collect()

T = (RST.join(RE, on=K, how="full", coalesce=True).join(FE, on=K, how="full", coalesce=True)
     .join(RSST, on=K, how="full", coalesce=True).join(RSEND, on=K, how="full", coalesce=True))

mk = "csds/2024/02/01/m_9805aa614020949e127f8783/csds"
print("example match rounds 1-4, by label:")
print(T.filter(pl.col("match_key") == mk).sort("round").head(8))
print("example match victim-death tick range by label (min,max,n):")
print(DE.filter(pl.col("match_key") == mk).group_by("round").agg(
    pl.col("tick").min().alias("dmin"), pl.col("tick").max().alias("dmax"), pl.len().alias("n")).sort("round").head(8))

# global: ordering of the three channels per label
Tn = T.filter(pl.col("st").is_not_null() & pl.col("fe").is_not_null() & pl.col("et").is_not_null())
print("\npairs with st, fe, et all present:", Tn.height)
print("share fe - st  (quantiles, ticks):")
print(Tn.select((pl.col("fe") - pl.col("st")).alias("d")).select(
    pl.col("d").quantile(0.01).alias("q01"), pl.col("d").median().alias("med"), pl.col("d").quantile(0.99).alias("q99")))
print("share et - fe (quantiles, ticks):")
print(Tn.select((pl.col("et") - pl.col("fe")).alias("d")).select(
    pl.col("d").quantile(0.01).alias("q01"), pl.col("d").median().alias("med"), pl.col("d").quantile(0.99).alias("q99")))
print("share et - st (quantiles, ticks):")
print(Tn.select((pl.col("et") - pl.col("st")).alias("d")).select(
    pl.col("d").quantile(0.01).alias("q01"), pl.col("d").median().alias("med"), pl.col("d").quantile(0.99).alias("q99")))
print("fraction et < st:", Tn.select((pl.col("et") < pl.col("st")).mean()).item(),
      " fraction et < fe:", Tn.select((pl.col("et") < pl.col("fe")).mean()).item())

# candidate windows for the victim death of label r
D = DE.join(T.select(K + ["st", "fe", "et"]), on=K, how="left")
print("\nvictim deaths:", D.height)
for name, cond in [
    ("[fe_r, et_r]", (pl.col("tick") >= pl.col("fe")) & (pl.col("tick") <= pl.col("et"))),
    ("[st_r, et_r]", (pl.col("tick") >= pl.col("st")) & (pl.col("tick") <= pl.col("et"))),
    ("[fe_r, fe_r+1)", None),
]:
    if cond is None:
        continue
    print(f"deaths in window {name}: {D.select(cond.cast(pl.Int32).sum()).item()} of {D.height}  "
          f"({D.select(cond.mean()).item():.4f})")

# death in window of label r-1 round_end: i.e. death tick <= et_r and > et_{r-1}
Tp = T.select(K + ["et"]).with_columns(round=pl.col("round") + 1).rename({"et": "et_prev"})
D2 = D.join(Tp, on=K, how="left")
cond_prev = (pl.col("tick") > pl.col("et_prev")) & (pl.col("tick") <= pl.col("et"))
print("deaths in (et_{r-1}, et_r]:", D2.filter(cond_prev.is_not_null()).select(cond_prev.cast(pl.Int32).sum()).item()
      if False else D2.filter(pl.col("et_prev").is_not_null()).select(
          ((pl.col("tick") > pl.col("et_prev")) & (pl.col("tick") <= pl.col("et"))).cast(pl.Int32).sum()).item(),
      "of", D2.filter(pl.col("et_prev").is_not_null()).height)
