"""A7: the 30 2024 matches where the cross-period halftime rule gives 16 but the score-only fit gives 24.
Shows the score-only evidence for each candidate (bad, ok) and the halftime announcement round."""
import collections, sys
from pathlib import Path
import importlib.util
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
spec = importlib.util.spec_from_file_location("A7", ROOT / "scripts" / "swarm2" / "A7_score_coverage.py")
A7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(A7)
RULE = A7.half_rule(*A7.TOMES["2024"])
base, suf = A7.TOMES["2024"]
H, W, F, info = A7.load_tome("2024", base, suf)
hl = pl.scan_parquet(str(base / f"basic_round_state{suf}" / "dataframe_*")).filter(
    pl.col("event_type") == "round_announce_last_round_half").group_by("match_key").agg(pl.col("round").min().alias("r")).collect()
half = {m: int(r) for m, r in hl.iter_rows()}
bad_rule = []
for m in H:
    rule = half[m] * 2 + 2 if m in half else 24
    best, bbad, ties, full = A7.mr_best(F.get(m, {}), A7.MR24_CANDIDATES, prefer=RULE.get(m, 24))
    if rule != best:
        bad_rule.append((m, rule, best, half.get(m), A7.mr_fit_full(F.get(m, {}), (16, 24))))
print("disagreements:", len(bad_rule))
print("rule -> fit:", collections.Counter((x[1], x[2]) for x in bad_rule))
print("half-announcement round:", collections.Counter(x[3] for x in bad_rule).most_common(6))
print("score-only (bad, ok) at 16 vs 24 for these matches:", collections.Counter((x[4][16], x[4][24]) for x in bad_rule).most_common(6))
print("days:", collections.Counter(A7.day_of(x[0]) for x in bad_rule))
