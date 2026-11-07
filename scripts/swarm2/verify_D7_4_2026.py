"""verify_D7_4 2026 check: any negative bomb-outcome-to-round_end lag in the 2026 tome (builds 10896, 10924), all player counts,
and outcome-round mismatch count (claim: 0 of 7,427). Reads round_end, bomb_state and header only.
"""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
T26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def channel(name):
    d = sorted(glob.glob(str(T26 / f"basic_{name}.*")))[0]
    return pl.scan_parquet(files(Path(d)))


H = pl.read_parquet(files(T26 / "header")[0], columns=["key", "build_num"]).rename({"key": "match_key"})
RE = channel("round_end").select(["match_key", "round", "tick", "winner_team_code", "player_count"]).collect()
RE = RE.unique(["match_key", "round"], keep="first", maintain_order=True)
BS = (channel("bomb_state").filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
        .select(["match_key", "round", "tick", "event_type"]).collect())
OUT = BS.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("out_tick"), (pl.col("event_type") == "bomb_defused").any().alias("d"))
O = (RE.join(OUT, on=["match_key", "round"], how="inner").join(H, on="match_key", how="left")
       .with_columns(pl.when(pl.col("d")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"),
                     (pl.col("out_tick") - pl.col("tick")).alias("lag"))
       .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm")))
print("2026 outcome rounds (all player counts):", O.height, "| mismatches:", int(O["mm"].sum()))
print("2026 by build: n, lag==0, lag<0, lag==-429, lag>0, mm:",
      O.group_by("build_num").agg(pl.len().alias("n"), (pl.col("lag") == 0).sum().alias("n0"), (pl.col("lag") < 0).sum().alias("neg"),
                                  (pl.col("lag") == -429).sum().alias("n429"), (pl.col("lag") > 0).sum().alias("pos"),
                                  pl.col("mm").sum().alias("mm")).sort("build_num").rows())
