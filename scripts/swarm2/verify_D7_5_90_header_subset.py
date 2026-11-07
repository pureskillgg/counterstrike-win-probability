"""verify_D7_5_90_header_subset: header-total test restricted to matches where the header totals equal the round count.

Only matches with tF + cF == N (N = last round number, rounds 1..N all present in round_end, i.e. contiguous)
are used. For each such match the team totals are computed under three label sources, with the DP orientation:
  A = team on T at round 1. tF is compared with A's total (direct) or with B's total (swapped).
Reports how often each source reproduces the header total, by period. Header semantics are not assumed beyond this.
"""
import sys
import pickle
from pathlib import Path
from collections import Counter
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
R = pl.read_parquet(str(SCRATCH / "verify_D7_5_rounds.parquet")).sort(["match_key", "round"])
with open(str(SCRATCH / "verify_D7_5_state.pkl"), "rb") as fh:
    ST = pickle.load(fh)
hdr = ST["hdr"]

res = Counter()
examples = []
for (mk,), g in R.group_by(["match_key"], maintain_order=True):
    rounds = g["round"].to_list()
    N = max(rounds)
    if rounds != list(range(1, N + 1)):
        res[(g["period"][0], "noncontig")] += 1
        continue
    h = hdr[mk]
    tF, cF = h["t_starters_score_final"], h["ct_starters_score_final"]
    if tF + cF != N:
        continue
    o = g["orient"].to_list()
    if any(x is None for x in o):
        res[(g["period"][0], "no_orient")] += 1
        continue
    per = g["period"][0]
    for src in ["W_code", "W_within", "W_dp"]:
        s = g[src].to_list()
        if src == "W_dp":
            # rounds 1..N-1 from the ledger; the last round is not in the ledger and is inferred from the header
            s_part = s[:-1]
            if any(x is None for x in s_part):
                res[(per, src, "missing")] += 1
                continue
            A1 = sum(1 for x, oo in zip(s_part, o[:-1]) if (x == 2 and oo == 0) or (x == 3 and oo == 1))
            B1 = (N - 1) - A1
            dA, dB = tF - A1, cF - B1
            if dA in (0, 1) and dB == 1 - dA:
                v = "direct_lenient"
            elif (tF - B1) in (0, 1) and (cF - A1) == 1 - (tF - B1):
                v = "swapped_lenient"
            else:
                v = "fail"
            res[(per, src, v)] += 1
            continue
        # lenient for every source: rounds 1..N-1 only, last round inferred from the header (same rule as W_dp)
        s_part = s[:-1]
        if any(x is None for x in s_part):
            res[(per, src, "missing_1_to_N-1")] += 1
            continue
        A1 = sum(1 for x, oo in zip(s_part, o[:-1]) if (x == 2 and oo == 0) or (x == 3 and oo == 1))
        B1 = (N - 1) - A1
        if tF - A1 in (0, 1):
            v = "direct_lenient"
        elif tF - B1 in (0, 1):
            v = "swapped_lenient"
        else:
            v = "fail_lenient"
        res[(per, src, v)] += 1
        if src == "W_code" and v == "fail_lenient" and len(examples) < 5:
            examples.append((mk, N, tF, cF, A1, B1))
        # exact (full) test with the last-round label, for the code only
        if src == "W_code":
            A = A1 + (1 if (s[-1] == 2 and o[-1] == 0) or (s[-1] == 3 and o[-1] == 1) else 0)
            B = N - A
            vf = "direct_full" if (A == tF and B == cF) else ("swapped_full" if (A == cF and B == tF) else "fail_full")
            res[(per, src, vf)] += 1
for k in sorted(res, key=lambda x: str(x)):
    print(k, res[k])
print("examples code fail (key, N, tF, cF, A, B):", examples)
