# verify_A2_3_ticks.py
# Fourth verifier script for finding A2-3. Appends to logs\swarm2\verify_A2_3.log.
# Question: in P2 matches (round_end labels 2..N, round_start labels 1..N-1), is the round_start row
# labelled k aligned with round k (tick between round_end k-1 and round_end k), or shifted (tick between
# round_end k and round_end k+1)? Same test for round_state freeze_end rows, and for P1 matches as control.
# Only interior labels are tested (k-1 and k+1 both present in round_end).
# Projected columns only.

import collections
import glob
import os

import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
TOME24 = os.path.join(ROOT, "tomes", "2024-02-g", "tome", "csds")
SUFFIX24 = ".2023-12-28,2024-02-25"
LOG_PATH = os.path.join(ROOT, "logs", "swarm2", "verify_A2_3.log")

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
say("== verify_A2_3_ticks.py ==")
RE = read("round_end", ["match_key", "round", "tick"])
RST = read("round_start", ["match_key", "round", "tick"])
RS = read("round_state", ["match_key", "round", "tick", "event_type"]).filter(
    pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "tick"])

E_t = collections.defaultdict(dict)
for m, r, tk in RE.iter_rows():
    E_t[m].setdefault(int(r), int(tk))
S_t = collections.defaultdict(dict)
for m, r, tk in RST.iter_rows():
    S_t[m].setdefault(int(r), int(tk))
F_t = collections.defaultdict(dict)
for m, r, tk in RS.iter_rows():
    F_t[m].setdefault(int(r), int(tk))

pattern = {}
for m in E_t:
    e = set(E_t[m])
    s = set(S_t.get(m, {}))
    if not s:
        continue
    if e == set(range(1, max(e) + 1)) and s == set(range(1, max(s) + 1)) and max(e) == max(s) + 1:
        pattern[m] = "P1"
    elif e == set(range(2, max(e) + 1)) and s == set(range(1, max(s) + 1)) and max(e) == max(s) + 1:
        pattern[m] = "P2"

# monotonicity of round_end ticks in label order (sanity)
mono_bad = collections.Counter()
for m in pattern:
    labs = sorted(E_t[m])
    ok = all(E_t[m][labs[i]] < E_t[m][labs[i + 1]] for i in range(len(labs) - 1))
    mono_bad[(pattern[m], ok)] += 1
say("round_end tick strictly increasing in label (pattern, ok): ", dict(mono_bad))


def test(pat, table, name):
    aligned = shifted = neither = 0
    for m, p in pattern.items():
        if p != pat:
            continue
        for k, tk in table.get(m, {}).items():
            if (k - 1) not in E_t[m] or (k + 1) not in E_t[m]:
                continue
            lo_al, hi_al = E_t[m][k - 1], E_t[m][k]
            lo_sh, hi_sh = E_t[m][k], E_t[m][k + 1]
            if lo_al < tk < hi_al:
                aligned += 1
            elif lo_sh < tk < hi_sh:
                shifted += 1
            else:
                neither += 1
    tot = aligned + shifted + neither
    say(f"  {pat} {name}: n={tot} aligned(k in (end k-1, end k))={aligned} "
        f"shifted(k in (end k, end k+1))={shifted} neither={neither}")


say("tick-order test on interior labels (k-1 and k+1 present in round_end):")
test("P1", S_t, "round_start tick")
test("P2", S_t, "round_start tick")
test("P1", F_t, "round_state freeze_end tick")
test("P2", F_t, "round_state freeze_end tick")

# final-round start of P2 matches: is the round_start labelled N-1 the final round (tick after end of N-1)?
fin = collections.Counter()
for m, p in pattern.items():
    if p != "P2":
        continue
    N = max(E_t[m])
    if (N - 1) in S_t[m] and (N - 1) in E_t[m]:
        fin["S_(N-1) after E_(N-1) end"] += int(S_t[m][N - 1] > E_t[m][N - 1])
        fin["S_(N-1) before E_(N-1) end"] += int(S_t[m][N - 1] < E_t[m][N - 1])
    if 1 in S_t[m] and 2 in E_t[m]:
        fin["S_1 before E_2 end"] += int(S_t[m][1] < E_t[m][2])
        fin["S_1 after E_2 end"] += int(S_t[m][1] > E_t[m][2])
say(f"P2 end-of-match checks: {dict(fin)}")
say("DONE")
LOG.close()
