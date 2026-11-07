"""Pistol-lens follow-up: money + equipment identity and armor carryover at freeze-end. Prints to stdout only."""
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

    groups = [
        ("R1", fe["round"] == 1),
        ("R13", fe["round"] == 13),
        ("R12", fe["round"] == 12),
        ("R2-11", fe["round"].between(2, 11)),
        ("R14-24", fe["round"].between(14, 24)),
    ]
    print("money + equipment_value_calc at freeze-end (pistol start money 800 + free default pistol ~200 -> ~1000 expected in R1)")
    for name, m in groups:
        sub = fe[m]
        s = sub["s"].to_numpy()
        print(f"{name:7s} n={len(sub):5d} s q05/q50/q95={np.quantile(s,0.05):.0f}/{np.quantile(s,0.5):.0f}/{np.quantile(s,0.95):.0f} "
              f"share s<=1000: {(s <= 1000).mean():.4f}  share s<=1100: {(s <= 1100).mean():.4f} max={int(s.max())}")
    print("")

    for name, m in [("R1", fe["round"] == 1), ("R13", fe["round"] == 13)]:
        sub = fe[m]
        o = sub[sub["equipment_value_calc"] > 1000]
        print(f"{name}: player-rows with equipment>1000: {len(o)} of {len(sub)}; their money: "
              f"{sorted(o['money'].astype(int).tolist())}")
        print(f"{name}: their equipment: {sorted(o['equipment_value_calc'].astype(int).tolist())}")
        print(f"{name}: their armor values: {sorted(o['armor'].astype(int).tolist())}; "
              f"has_helmet: {int(o['has_helmet'].sum())} of {len(o)}")
        print(f"{name}: distinct rounds containing those rows: {o[KEY].drop_duplicates().shape[0]}")
    print("")

    for name, m in [("R1", fe["round"] == 1), ("R12", fe["round"] == 12), ("R13", fe["round"] == 13),
                    ("R2-11", fe["round"].between(2, 11))]:
        sub = fe[m]
        print(f"{name:6s} armor>0 share: {(sub['armor'] > 0).mean():.4f}  helmet share: {sub['has_helmet'].mean():.4f}  "
              f"defuser share: {sub['has_defuser'].mean():.4f}  n={len(sub)}")
    print("")

    # side-level R13 outlier check: side totals above 5000 in pistol rounds
    side = fe[fe["round"].isin([1, 13])].groupby(["match_key", "round", "team_code"]).agg(
        eq=("equipment_value_calc", "sum"), money=("money", "sum"), n=("player_id", "size")).reset_index()
    big = side[side["eq"] > 4000].sort_values("eq", ascending=False)
    print(f"pistol side-rounds with equipment total > 4000: {len(big)}")
    print(big.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
