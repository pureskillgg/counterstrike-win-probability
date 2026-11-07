# verify_A2_3_disc_timing.py
# Sixth-and-a-half verifier script for finding A2-3. Appends to logs\swarm2\verify_A2_3.log.
# Stage (b) of cross_period.py drops a round when ANY player_disconnect row carries the same (match, round)
# label, with no tick window. For kept-match rounds that pass the 5 T / 5 CT check (stage a), this script
# splits the stage-(b) drops by whether every disconnect tick is after the round_end tick (post-round leave)
# or some disconnect tick is at or before the round_end tick. Final rounds and non-final rounds are split.
# Projected columns only. Tick arithmetic is raw ticks; no tick rate assumed.

import collections
import glob
import os

import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
TOME24 = os.path.join(ROOT, "tomes", "2024-02-g", "tome", "csds")
SUFFIX24 = ".2023-12-28,2024-02-25"
LOG_PATH = os.path.join(ROOT, "logs", "swarm2", "verify_A2_3.log")
T, CT, START = 2, 3, 5

LOG = open(LOG_PATH, "a", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def read(ch, cols):
    d = os.path.join(TOME24, f"basic_{ch}{SUFFIX24}")
    files = sorted(glob.glob(os.path.join(d, "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


say("")
say("== verify_A2_3_disc_timing.py ==")
RE = read("round_end", ["match_key", "round", "tick"])
PI = read("player_info", ["match_key", "round", "player_id", "team_code"])
DC = read("player_disconnect", ["match_key", "round", "tick"])
RST = read("round_start", ["match_key", "round"])
BS = read("bomb_state", ["match_key"])
DE = read("player_death", ["match_key"])
RS = read("round_state", ["match_key"])

maxE = RE.group_by("match_key").agg(pl.col("round").max().alias("mx"))
mx = {m: int(v) for m, v in maxE.iter_rows()}
end_tick = {}
for m, r, tk in RE.iter_rows():
    end_tick.setdefault((m, int(r)), int(tk))
dense = [set(RE["match_key"].to_list()), set(RST["match_key"].to_list()), set(RS["match_key"].to_list()),
         set(DE["match_key"].to_list()), set(PI["match_key"].to_list()), set(BS["match_key"].to_list())]
kept = set(mx)
for h in dense:
    kept &= h
say(f"kept matches: {len(kept)}")

players = collections.defaultdict(dict)
for m, r, pid, tc in PI.iter_rows():
    if m in kept:
        players[(m, int(r))][pid] = tc
disc = collections.defaultdict(list)
for m, r, tk in DC.iter_rows():
    if m in kept:
        disc[(m, int(r))].append(int(tk))

res = collections.Counter()
for m in kept:
    for k in range(1, mx[m] + 1):
        if (m, k) not in end_tick:
            continue
        pl_ = players.get((m, k), {})
        n_t = sum(1 for v in pl_.values() if v == T)
        n_ct = sum(1 for v in pl_.values() if v == CT)
        if not (n_t == START and n_ct == START):
            continue
        if (m, k) not in disc:
            continue
        kind = "final" if k == mx[m] else "nonfinal"
        et = end_tick[(m, k)]
        ticks = disc[(m, k)]
        if all(t > et for t in ticks):
            res[(kind, "all disconnects after round_end tick")] += 1
        else:
            res[(kind, "some disconnect at or before round_end tick")] += 1
say("stage (b) drops (round passes 5 T / 5 CT, has a disconnect row with same key) by disconnect timing:")
for key, n in sorted(res.items()):
    say(f"   {key[0]:<8} {key[1]:<45} {n}")
say("DONE")
LOG.close()
