# verify_A2_3_final.py
# Sixth verifier script for finding A2-3. Appends to logs\swarm2\verify_A2_3.log.
# (a) Round 1 check: any 2024 round_start row before the round_end of the lowest label?
#     Where does round_start label 1 fall relative to round_end 1 and 2 (P1 matches)?
# (b) The 141 stage-(c) final rounds: is round_start label N-1 inside the final round window, and does the
#     final round have a freeze_end row? (Explains why the window/freeze count is 140 not 141.)
# (c) Label-corrected join (round k paired with round_start label k-1, the convention 2026 uses for label k):
#     count kept rounds that pass stages (a) and (b) but would have no round_start row under that join.
# ASSUMPTION: tick rate 64; kept set = channel rule of verify_A2_3_recount.py (214 dropped, 4,221 kept).
# Projected columns only.

import collections
import glob
import os

import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
TOME24 = os.path.join(ROOT, "tomes", "2024-02-g", "tome", "csds")
SUFFIX24 = ".2023-12-28,2024-02-25"
LOG_PATH = os.path.join(ROOT, "logs", "swarm2", "verify_A2_3.log")
T, CT, START = 2, 3, 5
TICK_RATE = 64  # ASSUMPTION

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
say("== verify_A2_3_final.py ==")
RE = read("round_end", ["match_key", "round", "tick"])
RST = read("round_start", ["match_key", "round", "tick"])
PI = read("player_info", ["match_key", "round", "player_id", "team_code"])
DC = read("player_disconnect", ["match_key", "round", "tick"])
RS = read("round_state", ["match_key", "round", "tick", "event_type"]).filter(
    pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "tick"])

E, S, F = (collections.defaultdict(dict) for _ in range(3))
for m, r, tk in RE.iter_rows():
    E[m].setdefault(int(r), int(tk))
for m, r, tk in RST.iter_rows():
    S[m].setdefault(int(r), int(tk))
for m, r, tk in RS.iter_rows():
    F[m].setdefault(int(r), int(tk))

# kept set (channel rule, as in verify_A2_3_recount.py)
BS = read("bomb_state", ["match_key"])
DE = read("player_death", ["match_key"])
dense = [set(RE["match_key"].to_list()), set(RST["match_key"].to_list()), set(RS["match_key"].to_list()),
         set(DE["match_key"].to_list()), set(PI["match_key"].to_list()), set(BS["match_key"].to_list())]
all_m = set(E)
kept = set(all_m)
for h in dense:
    kept &= h
say(f"kept matches: {len(kept)}")

# (a) round 1 check
p1 = [m for m in E if set(E[m]) == set(range(1, max(E[m]) + 1)) and m in S and min(S[m]) == 1]
before_e1 = sum(1 for m in p1 if min(S[m].values()) < E[m][1])
s1_between = sum(1 for m in p1 if E[m][1] < S[m][1] < E[m][2])
say(f"(a) P1-type matches (E labels 1..N, S present): {len(p1)}")
say(f"    matches with any round_start row before round_end 1 tick: {before_e1}")
say(f"    matches with round_start label 1 between round_end 1 and round_end 2: {s1_between}")
say(f"    matches with round_start label 0 present: {sum(1 for m in S if 0 in S[m])}")

# (b) the stage-(c) finals
players = collections.defaultdict(dict)
for m, r, pid, tc in PI.iter_rows():
    if m in kept:
        players[(m, int(r))][pid] = tc
disc = {(m, int(r)) for m, r, _ in DC.iter_rows() if m in kept}
rst_keys = {(m, int(r)) for m in kept for r in S.get(m, {})}
stage = {}
for m in kept:
    for k in E[m]:
        pl_ = players.get((m, k), {})
        n_t = sum(1 for v in pl_.values() if v == T)
        n_ct = sum(1 for v in pl_.values() if v == CT)
        if not (n_t == START and n_ct == START):
            stage[(m, k)] = "a"
        elif (m, k) in disc:
            stage[(m, k)] = "b"
        elif (m, k) not in rst_keys:
            stage[(m, k)] = "c"
        else:
            stage[(m, k)] = "pass"
say(f"    stage counts over kept rounds: {dict(collections.Counter(stage.values()))}")

fin_c = [(m, max(E[m])) for m in kept if stage.get((m, max(E[m]))) == "c"]
window_ok = freeze_ok = both_ok = 0
for m, N in fin_c:
    s_ok = (N - 1) in S[m] and E[m][N - 1] < S[m][N - 1] < E[m][N]
    f_ok = N in F[m]
    window_ok += int(s_ok)
    freeze_ok += int(f_ok)
    both_ok += int(s_ok and f_ok)
say(f"(b) stage-(c) final rounds: {len(fin_c)}; round_start label N-1 inside final window: {window_ok}; "
    f"final freeze_end present: {freeze_ok}; both: {both_ok}")

# (c) label-corrected join: round k uses round_start label k-1
corr_fail = collections.Counter()
for m in kept:
    for k in E[m]:
        if stage[(m, k)] in ("a", "b"):
            continue
        if (k - 1) not in S.get(m, {}):
            corr_fail["round k lacks round_start label k-1"] += 1
            corr_fail["of which round 1"] += int(k == 1)
            corr_fail["of which final round"] += int(k == max(E[m]))
say(f"(c) under label-corrected join, rounds passing stages a and b with no start row: {dict(corr_fail)}")
say("    (current join: stage c = 141, all final rounds)")
say("DONE")
LOG.close()
