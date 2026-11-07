"""verify_D9_10 step 8: per-player team_code strings over rounds 1-30 for three build-9961 overtime matches (stdout only)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D9_10")
R = pl.read_parquet(SCR / "round_table.parquet")
PI = pl.scan_parquet(sorted(glob.glob(str(BASE / f"basic_player_info{SUF}" / "dataframe_*")))).select(
    ["match_key", "round", "player_id", "team_code"]).collect()
PIu = PI.unique(subset=["match_key", "round", "player_id"], keep="last")
ot = R.filter((pl.col("build_num") == 9961) & (pl.col("round") == 25)).select("match_key").unique()
for mk in ot.head(3)["match_key"].to_list():
    sub = PIu.filter(pl.col("match_key") == mk)
    print("MATCH", mk[-12:])
    for pid in sub["player_id"].unique().sort().to_list()[:10]:
        s = sub.filter(pl.col("player_id") == pid).sort("round")
        d = {r: c for r, c in zip(s["round"].to_list(), s["team_code"].to_list())}
        print(f"  p{pid % 100000:05d} " + "".join(str(d.get(r, "-")) for r in range(1, 31)))
    w = R.filter(pl.col("match_key") == mk).sort("round")
    print("  winner " + "".join(str(x) for x in w["winner_team_code"].to_list()))
    print("  bomb   " + "".join(("d" if d else ("e" if e else ".")) for d, e in zip(w["defused"].to_list(), w["exploded"].to_list())))
    print("  plant  " + "".join(str(x if x is not None else "-") for x in w["planter_side"].to_list()))
