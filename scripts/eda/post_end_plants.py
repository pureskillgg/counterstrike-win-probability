"""What do bomb_planted rows logged after round_end look like? (read-only)

Run: pixi run python -I scripts\\eda\\post_end_plants.py
"""
import collections
import glob
from pathlib import Path

import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUFFIX = ".2026-09-18,2026-10-08"
TICK_RATE = 64


def read(ch, cols=None):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUFFIX}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def main():
    RE = read("round_end", ["match_key", "round", "tick", "win_reason_code"])
    BS = read("bomb_state", ["match_key", "round", "tick", "event_type"])
    end = {}
    wr = {}
    for m, r, t, w in zip(RE["match_key"].to_list(), RE["round"].to_list(), RE["tick"].to_list(),
                          RE["win_reason_code"].to_list()):
        end[(m, r)] = t
        wr[(m, r)] = w
    plants = BS.filter(pl.col("event_type") == "bomb_planted")
    gaps = []
    by_reason = collections.Counter()
    all_reason = collections.Counter(wr.values())
    for m, r, t in zip(plants["match_key"].to_list(), plants["round"].to_list(), plants["tick"].to_list()):
        k = (m, r)
        if k in end and t > end[k]:
            gaps.append((t - end[k]) / TICK_RATE)
            by_reason[wr[k]] += 1
    gaps.sort()
    print(f"post-end plant rows: {len(gaps)}")
    print(f"- seconds after round_end: min {gaps[0]:.2f}, median {gaps[len(gaps)//2]:.2f}, "
          f"max {gaps[-1]:.2f}; <=1 s: {sum(1 for g in gaps if g <= 1)}; <=5 s: {sum(1 for g in gaps if g <= 5)}")
    print("- win_reason of the round for post-end plants (count, share of all rounds with that reason):")
    for k, v in sorted(by_reason.items()):
        print(f"    {k}: {v} of {all_reason[k]} rounds ({100.0 * v / all_reason[k]:.1f}%)")
    print(f"- rounds with a post-end plant: {len(by_reason) and sum(by_reason.values())} (one plant per round)")


if __name__ == "__main__":
    main()
