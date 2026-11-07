"""Verify two Task 1 claims (read-only).

(a) player_count == 4 rounds all belong to is_wingman matches; wingman rounds per match.
(b) raw player_id == 0 rounds with >=2 in-window deaths: are any of them in rounds that survive the
    bakeoff round filters (player_count == 10, 5 T and 5 CT at start, no disconnect/takeover,
    round_start present, live start present, non-null scores)? Only those would trigger the
    first-death-per-victim skip in build_states.

Run: pixi run python -I scripts\\eda\\wingman_and_sentinels.py
"""
import collections
import glob
from pathlib import Path

import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUFFIX = ".2026-09-18,2026-10-08"
T, CT, START = 2, 3, 5


def read(ch, cols=None):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUFFIX}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def main():
    hdr = read("header", ["key", "is_wingman", "max_rounds"])
    wing = {k for k, w in zip(hdr["key"].to_list(), hdr["is_wingman"].to_list()) if w}
    RE = read("round_end", ["match_key", "round", "tick", "player_count"])
    pc4 = RE.filter(pl.col("player_count") == 4)
    print(f"(a) rounds with player_count 4: {pc4.height}; of them in is_wingman matches: "
          f"{pc4.filter(pl.col('match_key').is_in(list(wing))).height}")
    wr = collections.Counter(RE.filter(pl.col("match_key").is_in(list(wing)))["match_key"].to_list())
    print(f"    wingman matches: {len(wing)}; rounds in wingman matches: {sum(wr.values())}; "
          f"rounds per wingman match min {min(wr.values())} max {max(wr.values())}")
    pcs = collections.Counter(RE.filter(pl.col("match_key").is_in(list(wing)))["player_count"].to_list())
    print(f"    player_count values inside wingman matches: {dict(sorted(pcs.items()))}")

    # (b) survivors-with-duplicate-raw-pid check
    RS = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    RST = read("round_start", ["match_key", "round", "tick"])
    DE = read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code"])
    PI = read("player_info", ["match_key", "round", "player_id", "team_code"])
    DC = read("player_disconnect", ["match_key", "round", "tick"])
    BT = read("bot_takeover", ["match_key", "round", "tick"])
    live = {}
    for m, r, t, et, ts, cs in zip(RS["match_key"].to_list(), RS["round"].to_list(), RS["tick"].to_list(),
                                   RS["event_type"].to_list(), RS["t_score"].to_list(), RS["ct_score"].to_list()):
        if et == "round_freeze_end" and (m, r) not in live:
            live[(m, r)] = (t, ts, cs)
    end = {}
    rd = {}
    for m, r, t, pc in zip(RE["match_key"].to_list(), RE["round"].to_list(), RE["tick"].to_list(),
                           RE["player_count"].to_list()):
        end.setdefault((m, r), t)
        rd[(m, r)] = pc
    rst_keys = set(zip(RST["match_key"].to_list(), RST["round"].to_list()))
    players = collections.defaultdict(dict)
    for m, r, pid, tc in zip(PI["match_key"].to_list(), PI["round"].to_list(), PI["player_id"].to_list(),
                             PI["team_code"].to_list()):
        players[(m, r)][pid] = tc
    disc = set(zip(DC["match_key"].to_list(), DC["round"].to_list())) | \
        set(zip(BT["match_key"].to_list(), BT["round"].to_list()))
    surv = set()
    for k, pc in rd.items():
        pl_ = players.get(k, {})
        n_t = sum(1 for v in pl_.values() if v == T)
        n_ct = sum(1 for v in pl_.values() if v == CT)
        if pc != 10 or not (len(pl_) == 10 and n_t == START and n_ct == START):
            continue
        if k in disc or k not in rst_keys or k not in live:
            continue
        _, ts, cs = live[k]
        if ts is None or cs is None:
            continue
        surv.add(k)
    print(f"\n(b) rounds surviving the round filters: {len(surv)} of {len(rd)}")
    zero_in = collections.Counter()
    for m, r, t, pid in zip(DE["match_key"].to_list(), DE["round"].to_list(), DE["tick"].to_list(),
                            DE["player_id"].to_list()):
        k = (m, r)
        if pid != 0 or k not in live or k not in end:
            continue
        if live[k][0] <= t <= end[k]:
            zero_in[k] += 1
    dup_rounds = [k for k, v in zero_in.items() if v >= 2]
    print(f"    rounds with >=2 in-window raw-pid-0 deaths: {len(dup_rounds)}; of those surviving the filters: "
          f"{sum(1 for k in dup_rounds if k in surv)}")
    print(f"    rounds with any in-window raw-pid-0 death that survive: "
          f"{sum(1 for k in zero_in if k in surv)}")


if __name__ == "__main__":
    main()
