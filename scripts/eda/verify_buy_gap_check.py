"""Pistol-lens follow-up: round coverage per match (gaps, missing round 1 / 13). Prints to stdout only."""
from pathlib import Path

import pandas as pd

ROOT = Path(r"G:\pureskill\csds-cs2")
SS = ROOT / "data" / "state_sample"


def main():
    rd = pd.read_parquet(SS / "rounds.parquet")
    g = rd.groupby("match_key").agg(n=("round", "nunique"), mn=("round", "min"), mx=("round", "max"))
    g["has1"] = g.index.isin(rd.loc[rd["round"] == 1, "match_key"])
    g["has13"] = g.index.isin(rd.loc[rd["round"] == 13, "match_key"])
    g["gap"] = g["mx"] - g["mn"] + 1 - g["n"]
    print(f"matches: {len(g)}")
    print(f"matches whose first round is not 1: {int((g['mn'] != 1).sum())}")
    print(f"matches with missing round numbers inside [min,max] (gap>0): {int((g['gap'] > 0).sum())}; "
          f"total missing inside range: {int(g['gap'].sum())}")
    print(f"matches with fewer than 13 rounds: {int((g['n'] < 13).sum())}")
    short = g[g["n"] < 13].sort_values("n")
    print("short matches (n, min, max, has1, has13, gap):")
    print(short[["n", "mn", "mx", "has1", "has13", "gap"]].to_string())
    print("")
    print("matches with >=13 rounds but no round 13:")
    print(g[(g["n"] >= 13) & (~g["has13"])][["n", "mn", "mx", "has1", "has13", "gap"]].to_string())
    print("")
    print("matches with no round 1:")
    print(g[~g["has1"]][["n", "mn", "mx", "has1", "has13", "gap"]].to_string())
    print("")
    full = g[(g["n"] == 24) | ((g["mx"] == 24) & (g["gap"] == 0))]
    print(f"matches with exactly rounds 1..24 contiguous (n==24, gap 0): {int(((g['mn'] == 1) & (g['mx'] == 24) & (g['gap'] == 0)).sum())}")
    print(f"matches with round numbers 1..13 all present: {int(((g['mn'] == 1) & (g['gap'] == 0) & (g['mx'] >= 13)).sum())}")
    print(f"distribution of n_rounds across all matches: {g['n'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
