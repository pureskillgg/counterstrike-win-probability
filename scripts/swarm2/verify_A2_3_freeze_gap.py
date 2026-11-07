# verify_A2_3_freeze_gap.py
# Fifth verifier script for finding A2-3. Appends to logs\swarm2\verify_A2_3.log.
# Purpose: decide whether a 2024 round_start row labelled k is the start of round k (same label as
# round_end k) or of round k+1, using the time from round_start to round_state freeze_end of the
# round it belongs to. A true round start should precede its own freeze_end by about the freeze time.
# Compared with the 2026 tome (control, labels aligned by construction of the anti-join check).
# ASSUMPTION: tick rate 64 ticks per second (feature_validation.TICK_RATE).
# Projected columns only.

import collections
import glob
import os
import statistics

import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
TICK_RATE = 64  # ASSUMPTION
TOME24 = os.path.join(ROOT, "tomes", "2024-02-g", "tome", "csds")
SUFFIX24 = ".2023-12-28,2024-02-25"
TOME26 = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
SUFFIX26 = ".2026-09-18,2026-10-08"
LOG_PATH = os.path.join(ROOT, "logs", "swarm2", "verify_A2_3.log")

LOG = open(LOG_PATH, "a", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def read(tome, suffix, ch, cols):
    d = os.path.join(tome, f"basic_{ch}{suffix}")
    files = sorted(glob.glob(os.path.join(d, "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def tables(tome, suffix):
    RE = read(tome, suffix, "round_end", ["match_key", "round", "tick"])
    RST = read(tome, suffix, "round_start", ["match_key", "round", "tick"])
    RS = read(tome, suffix, "round_state", ["match_key", "round", "tick", "event_type"]).filter(
        pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "tick"])
    E, S, F = (collections.defaultdict(dict) for _ in range(3))
    for m, r, tk in RE.iter_rows():
        E[m].setdefault(int(r), int(tk))
    for m, r, tk in RST.iter_rows():
        S[m].setdefault(int(r), int(tk))
    for m, r, tk in RS.iter_rows():
        F[m].setdefault(int(r), int(tk))
    return E, S, F


def summarize(tag, E, S, F):
    res = collections.Counter()
    gap_shift, gap_align = [], []
    for m in E:
        e, s, f = E[m], S.get(m, {}), F.get(m, {})
        for k in s:
            if (k - 1) in e and k in e and (k + 1) in e and (k + 1) in f and k in f:
                if e[k - 1] < s[k] < e[k]:
                    res["S_k inside round k (aligned)"] += 1
                    gap_align.append((f[k] - s[k]) / TICK_RATE)
                elif e[k] < s[k] < e[k + 1]:
                    res["S_k inside round k+1 (shifted +1)"] += 1
                    gap_shift.append((f[k + 1] - s[k]) / TICK_RATE)
                else:
                    res["neither"] += 1
    say(f"{tag}: label relation of round_start rows (interior labels with freeze_end present): {dict(res)}")
    for name, g in (("aligned: freeze_end(k) - start(k)", gap_align),
                    ("shifted: freeze_end(k+1) - start(k)", gap_shift)):
        if g:
            g = sorted(g)
            say(f"   {name} seconds: n={len(g)} p10={g[len(g)//10]:.2f} median={statistics.median(g):.2f} "
                f"p90={g[(9*len(g))//10]:.2f} min={g[0]:.2f} max={g[-1]:.2f}")
    # final round of each match: is there a round_start row labelled N-1 inside the final round?
    fin = collections.Counter()
    fin_gap = []
    for m in E:
        e, s, f = E[m], S.get(m, {}), F.get(m, {})
        if not e:
            continue
        N = max(e)
        if (N - 1) in s and (N - 1) in e and N in f and e[N - 1] < s[N - 1] < e[N]:
            fin["S_(N-1) inside final round N"] += 1
            fin_gap.append((f[N] - s[N - 1]) / TICK_RATE)
        elif N in s:
            fin["S_N present"] += 1
        else:
            fin["no S row inside final round (no S_N, no S_(N-1) inside N)"] += 1
    say(f"   final-round check: {dict(fin)}")
    if fin_gap:
        g = sorted(fin_gap)
        say(f"   final round: freeze_end(N) - start(N-1) seconds: n={len(g)} median={statistics.median(g):.2f} "
            f"p10={g[len(g)//10]:.2f} p90={g[(9*len(g))//10]:.2f}")


say("")
say("== verify_A2_3_freeze_gap.py ==")
E24, S24, F24 = tables(TOME24, SUFFIX24)
summarize("2024-02-g", E24, S24, F24)
E26, S26, F26 = tables(TOME26, SUFFIX26)
summarize("2026 control", E26, S26, F26)
say("DONE")
LOG.close()
