"""Second targeted check pass for Task 1 goofy items (read-only).

Run: pixi run python -I scripts\\eda\\flow_checks2.py
"""
import collections
import glob
from pathlib import Path

import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUFFIX = ".2026-09-18,2026-10-08"
TICK_RATE = 64


def read(ch):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUFFIX}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")


def main():
    hdr = read("header")
    RE = read("round_end")
    RS = read("round_state")
    DE = read("player_death")
    BS = read("bomb_state")
    RST = read("round_start")

    # (a) raw player_id == 0 deaths inside the live window: duplicates per round (bakeoff victim dedupe)
    live = {}
    for m, r, t in zip(*[RS.filter(pl.col("event_type") == "round_freeze_end")[c].to_list()
                         for c in ("match_key", "round", "tick")]):
        live.setdefault((m, r), t)
    end = {}
    for m, r, t in zip(RE["match_key"].to_list(), RE["round"].to_list(), RE["tick"].to_list()):
        end.setdefault((m, r), t)
    z_in = collections.Counter()
    z_total_in = 0
    all_in = 0
    for m, r, t, pid in zip(DE["match_key"].to_list(), DE["round"].to_list(), DE["tick"].to_list(),
                            DE["player_id"].to_list()):
        k = (m, r)
        if k not in live or k not in end:
            continue
        if live[k] <= t <= end[k]:
            all_in += 1
            if pid == 0:
                z_in[k] += 1
                z_total_in += 1
    print(f"## (a) raw player_id==0 deaths in live window: {z_total_in} of {all_in} in-window deaths")
    print(f"- rounds with >=2 in-window raw-pid-0 deaths (bakeoff victim dedupe would drop extras): "
          f"{sum(1 for v in z_in.values() if v >= 2)}")
    print(f"- rounds with >=1 in-window raw-pid-0 death: {len(z_in)}")

    # (b) player_count != 10 vs wingman/max_rounds
    print("\n## (b) round_end player_count vs header is_wingman and max_rounds")
    hw = dict(zip(hdr["key"].to_list(), zip(hdr["is_wingman"].to_list(), hdr["max_rounds"].to_list(),
                                            hdr["match_type"].to_list())))
    per = collections.defaultdict(list)
    for m, pc in zip(RE["match_key"].to_list(), RE["player_count"].to_list()):
        per[m].append(pc)
    ct = collections.Counter()
    for m, pcs in per.items():
        w, mr, mt = hw[m]
        all10 = all(p == 10 for p in pcs)
        none10 = all(p != 10 for p in pcs)
        ct[(bool(w), mr, "all10" if all10 else ("none10" if none10 else "mixed"))] += 1
    for k, v in sorted(ct.items(), key=lambda kv: str(kv[0])):
        print(f"- is_wingman={k[0]}, max_rounds={k[1]}, {k[2]}: {v} matches")
    print(f"- header match_type counts: {dict(collections.Counter(hdr['match_type'].to_list()))}")
    print(f"- header is_wingman counts: {dict(collections.Counter(hdr['is_wingman'].to_list()))}")
    nonten_rounds_by_pc = collections.Counter(pc for pc in RE["player_count"].to_list() if pc != 10)
    print(f"- rounds with player_count != 10 by value: {dict(sorted(nonten_rounds_by_pc.items()))}")

    # (c) planted events outside the round window
    print("\n## (c) bomb_planted outside [round_start, round_end]")
    rst_start = {}
    for m, r, t in zip(RST["match_key"].to_list(), RST["round"].to_list(), RST["tick"].to_list()):
        rst_start.setdefault((m, r), t)
    plants = BS.filter(pl.col("event_type") == "bomb_planted")
    before = after = inside = 0
    for m, r, t in zip(plants["match_key"].to_list(), plants["round"].to_list(), plants["tick"].to_list()):
        k = (m, r)
        if k not in rst_start or k not in end:
            continue
        if t < rst_start[k]:
            before += 1
        elif t > end[k]:
            after += 1
        else:
            inside += 1
    print(f"- bomb_state bomb_planted rows: {plants.height}; inside window {inside}; "
          f"before round_start {before}; after round_end {after}")
    # plant time relative to live start for inside-window vs after-end
    # (d) round 1 round_start tick 66 durations
    print("\n## (d) round_start tick 66 (round 1) and long durations")
    s66 = RST.filter(pl.col("tick") == 66)
    keys66 = set(zip(s66["match_key"].to_list(), s66["round"].to_list()))
    long_dur = []
    for k, s in rst_start.items():
        if k in end and (end[k] - s) / TICK_RATE > 200:
            long_dur.append(k)
    print(f"- rounds with round_start->round_end > 200 s: {len(long_dur)}; of those round_start tick==66: "
          f"{sum(1 for k in long_dur if k in keys66)}; round numbers: "
          f"{dict(collections.Counter(k[1] for k in long_dur).most_common(5))}")
    print(f"- tick-66 round_start rows: {len(keys66)}; all round number 1: "
          f"{all(k[1] == 1 for k in keys66)}")
    print(f"- round 1 rows with round_start tick != 66: "
          f"{sum(1 for k, s in rst_start.items() if k[1] == 1 and s != 66)}")


if __name__ == "__main__":
    main()
