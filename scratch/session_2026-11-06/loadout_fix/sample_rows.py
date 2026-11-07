import json, sys
import numpy as np, polars as pl
S = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/loadout_fix/"
keys = json.load(open(S + "match_keys.json"))
rf = np.load(S + "round_freeze.npy").astype(np.int64); rm = np.load(S + "round_match.npy"); rn = np.load(S + "round_num.npy").astype(np.int32)
exp = pl.DataFrame({"match_key": [keys[i] for i in rm.tolist()], "round": rn, "freeze": rf}).unique(subset=["match_key","round"], keep="first")
for name, f in [("status_1hz","2026-09-01.parquet"),("status_16hz","2026-09-01.parquet"),("status_8hz","2026-09-01.parquet"),("status_random","2026-09-01.parquet"),("status_8new","2026-09-01.parquet")]:
    df = pl.read_parquet("G:/pureskill/csds-cs2/data/" + name + "/" + f, columns=["match_key","round","tick","second","player_id","side"])
    mk = df.get_column("match_key")[0]; rnd = df.filter(pl.col("match_key")==mk).get_column("round")[0]
    sub = df.filter((pl.col("match_key")==mk) & (pl.col("round")==rnd) & (pl.col("player_id")==df.filter((pl.col("match_key")==mk)&(pl.col("round")==rnd)).get_column("player_id")[0])).sort("tick")
    fr = exp.filter((pl.col("match_key")==mk)&(pl.col("round")==rnd)).get_column("freeze")
    fr = int(fr[0]) if len(fr) else None
    print("==", name, "match", mk[:40], "round", rnd, "export freeze", fr, "rows in group", sub.height)
    sub = sub.with_columns((pl.col("tick") - (fr if fr is not None else 0)).alias("tick_minus_freeze"))
    print(sub.head(12).write_csv(None))
    print("second min/max in group:", sub.get_column("second").min(), sub.get_column("second").max(), "tick span:", sub.get_column("tick").min(), sub.get_column("tick").max())
