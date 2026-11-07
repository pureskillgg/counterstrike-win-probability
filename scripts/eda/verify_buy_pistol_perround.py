"""Pistol-lens follow-up: per-round pistol-start signature (money + equipment ~ 1000 per player). Prints to stdout only."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"G:\pureskill\csds-cs2")
SS = ROOT / "data" / "state_sample"
KEY = ["match_key", "round"]


def main():
    pl = pd.read_parquet(SS / "players_1hz.parquet")
    fe_min = pl.groupby(KEY, as_index=False)["tick"].min().rename(columns={"tick": "fe_tick"})
    fe = pl.merge(fe_min, on=KEY, how="left")
    fe = fe[fe["tick"] == fe["fe_tick"]].copy()
    fe["s"] = fe["money"].astype(float) + fe["equipment_value_calc"].astype(float)
    fe["start_like"] = fe["s"] <= 1100

    side = fe.groupby(["match_key", "round", "team_code"]).agg(
        n=("player_id", "size"), all_start=("start_like", "all"), max_money=("money", "max"),
        mean_s=("s", "mean")).reset_index()
    rnd = fe.groupby("round").agg(playerrows=("player_id", "size"),
                                   share_start_like=("start_like", "mean"),
                                   share_money_gt800=("money", lambda x: (x > 800).mean()),
                                   median_s=("s", "median"))
    sides = side.groupby("round").agg(sides=("n", "size"), sides_all_start=("all_start", "sum"))
    rounds_n = fe.groupby("round")["match_key"].nunique().rename("matches")
    t = rnd.join(sides).join(rounds_n)
    print("round  matches  playerrows  share(s<=1100)  share(money>800)  median(s)  sides  sides_all_start_like")
    for r, row in t.iterrows():
        print(f"{int(r):5d}  {int(row['matches']):7d}  {int(row['playerrows']):10d}  {row['share_start_like']:14.4f}  "
              f"{row['share_money_gt800']:16.4f}  {row['median_s']:9.1f}  {int(row['sides']):5d}  {int(row['sides_all_start']):20d}")
    print("")
    ss = side[side["all_start"]]
    print("sides where every player has s<=1100 (pistol-start-like), by round:")
    print(ss.groupby("round").size().to_dict())
    print("")
    print(f"total pistol-start-like sides: {len(ss)}; of which round 1 or 13: {int(ss['round'].isin([1, 13]).sum())}")
    print(f"sides in rounds 1 and 13 that are NOT pistol-start-like: "
          f"{int(((side['round'].isin([1, 13])) & (~side['all_start'])).sum())} of {int(side['round'].isin([1, 13]).sum())}")
    print(f"player-rows with money>800 in rounds 1 or 13: {int(((fe['round'].isin([1, 13])) & (fe['money'] > 800)).sum())}")
    print(f"max money in rounds 1 or 13: {int(fe.loc[fe['round'].isin([1, 13]), 'money'].max())}")


if __name__ == "__main__":
    main()
