# verify_A2_3_shift.py
# Third verifier script for finding A2-3. Appends to logs\swarm2\verify_A2_3.log.
# Splits the 141 stage-(c) "no round_start row" rounds by match label pattern:
#   pattern P1 = round_end labels 1..N and round_start labels 1..N-1 (final round_start row absent)
#   pattern P2 = round_end labels 2..N and round_start labels 1..N-1 (whole-match label offset of 1)
# and checks which label convention round_state (freeze_end scores) follows in P2 matches.
# Uses projected columns only.

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
say("== verify_A2_3_shift.py ==")
RE = read("round_end", ["match_key", "round", "tick"])
RST = read("round_start", ["match_key", "round"])
PI = read("player_info", ["match_key", "round", "player_id", "team_code"])
DC = read("player_disconnect", ["match_key", "round"])
RS = read("round_state", ["match_key", "round", "event_type", "t_score", "ct_score"])

E = collections.defaultdict(set)
for m, r in RE.select(["match_key", "round"]).unique().iter_rows():
    E[m].add(int(r))
S = collections.defaultdict(set)
for m, r in RST.select(["match_key", "round"]).unique().iter_rows():
    S[m].add(int(r))

# pattern of each match
pattern = {}
for m in E:
    e, s = E[m], S.get(m, set())
    if not s:
        pattern[m] = "P0_no_round_start_rows"
    elif e == set(range(1, max(e) + 1)) and s == set(range(1, max(s) + 1)) and max(e) == max(s) + 1:
        pattern[m] = "P1_final_rst_absent"
    elif e == set(range(2, max(e) + 1)) and s == set(range(1, max(s) + 1)) and max(e) == max(s) + 1:
        pattern[m] = "P2_label_offset"
    else:
        pattern[m] = "other"
say("match pattern counts (round_end labels vs round_start labels):",
    dict(collections.Counter(pattern.values())))
say("rounds in each pattern (round_end pairs):",
    dict(collections.Counter({p: sum(len(E[m]) for m in E if pattern[m] == p) for p in set(pattern.values())})))

# channel-kept set (same rule as verify_A2_3_recount.py): zero-row dense channel drops
BS = read("bomb_state", ["match_key"])
DE = read("player_death", ["match_key"])
RS_keys = set(RS["match_key"].unique().to_list())
dense_have = [set(RE["match_key"].to_list()), set(RST["match_key"].to_list()), RS_keys,
              set(DE["match_key"].to_list()), set(PI["match_key"].to_list()),
              set(BS["match_key"].to_list())]
kept = set(E)
for have in dense_have:
    kept &= have
say(f"kept matches (dense-channel rule): {len(kept)}")

# replicate stage order for kept matches, classify each stage-(c) round by pattern
players = collections.defaultdict(dict)
for m, r, pid, tc in PI.iter_rows():
    if m in kept:
        players[(m, int(r))][pid] = tc
disc = set((m, int(r)) for m, r in DC.iter_rows() if m in kept)
rst_keys = {(m, int(r)) for m in kept for r in S.get(m, set())}
rd_keys = [(m, r) for m in kept for r in sorted(E[m])]
stage_c = []
stage_counts = collections.Counter()
for k in rd_keys:
    pl_ = players.get(k, {})
    n_t = sum(1 for v in pl_.values() if v == T)
    n_ct = sum(1 for v in pl_.values() if v == CT)
    if not (n_t == START and n_ct == START):
        stage_counts["a_not5v5"] += 1
        continue
    if k in disc:
        stage_counts["b_disc"] += 1
        continue
    if k not in rst_keys:
        stage_counts["c_no_rst"] += 1
        stage_c.append(k)
        continue
    stage_counts["pass"] += 1
say(f"stage counts over kept-match rounds ({len(rd_keys)} round_end labels): {dict(stage_counts)}")

by_pat = collections.Counter()
final_c = 0
for m, r in stage_c:
    by_pat[pattern[m]] += 1
    if r == max(E[m]):
        final_c += 1
say(f"stage-(c) rounds by pattern: {dict(by_pat)}; of which final round: {final_c}")

# round_state convention test for P2 matches: freeze_end labels and the score at the lowest label
fz = RS.filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "t_score", "ct_score"])
fz_by = collections.defaultdict(dict)
for m, r, ts, cs in fz.iter_rows():
    if m in pattern and pattern[m] == "P2_label_offset":
        fz_by[m].setdefault(int(r), (ts, cs))
min_label = collections.Counter()
score_at_min = collections.Counter()
for m, d in fz_by.items():
    lo = min(d)
    min_label[lo] += 1
    score_at_min[(d[lo][0], d[lo][1])] += 1
say(f"P2 matches with freeze_end rows: {len(fz_by)}; lowest freeze_end label counts: {dict(min_label)}")
say(f"P2 score at lowest freeze_end label (t, ct): {score_at_min.most_common(5)}")
s1 = collections.Counter(d[1] for d in fz_by.values() if 1 in d)
s2 = collections.Counter(d[2] for d in fz_by.values() if 2 in d)
say(f"P2 freeze_end score at label 1 (t, ct): {s1.most_common(3)}  (n={sum(s1.values())})")
say(f"P2 freeze_end score at label 2 (t, ct): {s2.most_common(3)}  (n={sum(s2.values())})")
say("DONE")
LOG.close()
