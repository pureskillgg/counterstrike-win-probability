"""A7: which source is right on the flipped rounds? Bomb outcomes as an independent check.

For every round with an unambiguous, schedule-consistent score boundary (score-inferred winner w_s)
and a bomb outcome in the same round (bomb_exploded -> T won, code 2; bomb_defused -> CT won, code 3),
counts agreement of the recorded winner (round_end) and of the score-inferred winner with the bomb,
split by whether the round is a WINMM round (recorded winner != score-inferred winner).
Rounds with no bomb outcome cannot be checked this way and are reported as a count only.
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
RULE = A7.half_rule(*A7.TOMES["2024"])
T, CT = A7.T, A7.CT


def bomb_outcomes(base, suf):
    bs = pl.scan_parquet(str(base / f"basic_bomb_state{suf}" / "dataframe_*")).filter(
        pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"])).select(
        ["match_key", "round", "event_type"]).collect()
    out = {}
    for m, r, ev in bs.iter_rows():
        out.setdefault((m, int(r)), set()).add(ev)
    res = {}
    for k, evs in out.items():
        if len(evs) == 1:
            res[k] = T if "bomb_exploded" in evs else CT
        else:
            res[k] = None  # both events in one round: unresolved
    return res


for label in ("2026", "2024"):
    base, suf = A7.TOMES[label]
    H, W, F, info = A7.load_tome(label, base, suf)
    bo = bomb_outcomes(base, suf)
    cnt = collections.Counter()
    for m in H:
        Fm, Wm = F.get(m, {}), W.get(m, {})
        if label == "2026":
            mr = int(H[m]["max_rounds"])
        else:
            mr = A7.mr_best(Fm, A7.MR24_CANDIDATES, prefer=RULE.get(m, 24))[0]
        for r in sorted(Fm):
            if r + 1 not in Fm:
                continue
            t, c = Fm[r]
            t2, c2 = Fm[r + 1]
            cls = A7.boundary_class(t, c, t2, c2)
            if cls in ("ambig", "neither"):
                cnt["boundary_ambig_or_neither"] += 1
                continue
            sched = A7.swap_expected(r, mr)
            if cls != ("swap" if sched else "noswap"):
                cnt["boundary_inconsistent"] += 1
                continue
            if cls == "noswap":
                w_s = T if (t2 - t) == 1 else CT
            else:
                w_s = CT if (t2 - c) == 1 else T
            w_rec = Wm.get(r)
            if w_rec not in (T, CT):
                cnt["no_recorded_winner"] += 1
                continue
            outcome = bo.get((m, r), "absent")
            if outcome == "absent":
                cnt["no_bomb_outcome"] += 1
                continue
            if outcome is None:
                cnt["bomb_both_events"] += 1
                continue
            flip = w_rec != w_s
            cnt["rounds_checked"] += 1
            cnt["flip_rounds" if flip else "agree_rounds"] += 1
            if flip:
                cnt["flip_bomb_matches_recorded"] += int(outcome == w_rec)
                cnt["flip_bomb_matches_score"] += int(outcome == w_s)
                cnt["flip_bomb_matches_neither"] += int(outcome not in (w_rec, w_s))
            else:
                cnt["agree_bomb_matches_recorded"] += int(outcome == w_rec)
                cnt["agree_bomb_matches_score"] += int(outcome == w_s)
    print(f"== {label}")
    for k in ("rounds_checked", "agree_rounds", "flip_rounds", "agree_bomb_matches_recorded",
              "agree_bomb_matches_score", "flip_bomb_matches_recorded", "flip_bomb_matches_score",
              "flip_bomb_matches_neither", "no_bomb_outcome", "bomb_both_events", "no_recorded_winner",
              "boundary_ambig_or_neither", "boundary_inconsistent"):
        print(f"   {k:<32} {cnt.get(k, 0)}")
