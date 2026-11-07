"""A7: does cross_period's max_rounds rule (2 x first half-announcement round + 2, else 24) agree with
the score evidence?

For 2024: the rule value is CONTRADICTED by the scores when it has any unambiguous boundary
inconsistent with its schedule (bad > 0). Matches where the score evidence is tied between candidates
are reported separately (the rule cannot be confirmed or refuted by scores there).
For 2026: rule vs header max_rounds.

Feeds is_pistol_round and is_overtime (model inputs) in the cross-period 2024 test.
"""
import collections
import importlib.util
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
spec = importlib.util.spec_from_file_location("A7", ROOT / "scripts" / "swarm2" / "A7_score_coverage.py")
A7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A7)

HALF_EVENT = "round_announce_last_round_half"


def half_rounds(base, suf):
    rs = pl.scan_parquet(str(base / f"basic_round_state{suf}" / "dataframe_*")).filter(
        pl.col("event_type") == HALF_EVENT).group_by("match_key").agg(pl.col("round").min().alias("r")).collect()
    return {m: int(r) for m, r in rs.iter_rows()}


for label in ("2024", "2026"):
    base, suf = A7.TOMES[label]
    H, W, F, info = A7.load_tome(label, base, suf)
    half = half_rounds(base, suf)
    rule = {m: (half[m] * 2 + 2 if m in half else 24) for m in H}
    print(f"== {label}: matches {len(H)}; with a half announcement {len(half)}")
    print(f"   rule MR distribution: {dict(sorted(collections.Counter(rule.values()).items()))}")
    if label == "2026":
        hdr = {m: int(H[m]["max_rounds"]) for m in H}
        print(f"   rule vs header max_rounds: agree {sum(1 for m in H if rule[m] == hdr[m])} of {len(H)}")
        print(f"   disagreements (rule, header): {dict(collections.Counter((rule[m], hdr[m]) for m in H if rule[m] != hdr[m]))}")
        continue
    contra, consistent_best, tied_only, rule_not_best = [], 0, 0, []
    for m in H:
        full = A7.mr_fit_full(F.get(m, {}), (rule[m],))
        bad_rule, ok_rule = full[rule[m]]
        best_bad = min(A7.mr_fit_full(F.get(m, {}), A7.MR24_CANDIDATES).values(), key=lambda x: x[0])[0]
        if bad_rule > 0:
            contra.append((m, rule[m], bad_rule))
        elif bad_rule == best_bad:
            consistent_best += 1
        else:
            rule_not_best.append(m)
    print(f"   rule value contradicted by unambiguous score boundaries: {len(contra)}")
    print(f"   rule value consistent with the scores (bad 0): {len(H) - len(contra)}")
    print(f"   contradicted by scores, by rule value: {dict(collections.Counter(c[1] for c in contra))}")
    m16 = [m for m in H if rule[m] == 16]
    print(f"   rule MR16 matches: {len(m16)}; of these contradicted: {sum(1 for c in contra if c[1] == 16)}")
    m24 = [m for m in H if rule[m] == 24]
    print(f"   rule MR24 matches: {len(m24)}; of these contradicted: {sum(1 for c in contra if c[1] == 24)}")
    print(f"   rule MR16 matches with the score fit tied (ok count) vs a 24 candidate: "
          f"{sum(1 for m in m16 if A7.mr_best(F.get(m, {}), A7.MR24_CANDIDATES, prefer=16)[2] > 1)}")
