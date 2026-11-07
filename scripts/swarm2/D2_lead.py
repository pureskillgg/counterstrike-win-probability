"""D2 lead check (corrected): team-tracked score lead from recorded winners vs round_state lead.

Score fields are side-attached (ct_score = score of the team on CT at that round). A team's score
accumulates across the halftime swap, so the cumulative lead must track teams, not sides.

For each match with contiguous rounds 1..rmax, team X = the five players on T in the first valid round.
Team scores scoreX, scoreY start at 0. Before round r:
  tracked_lead(r) = scoreX - scoreY if X is on CT in round r (sx == 3), else scoreY - scoreX.
  round_state lead(r) = ct_score - t_score at round_freeze_end (side-attached fields).
After round r, the winning team is X if the winning side code == sx(r), else Y. The winning side code is:
  R (recorded winner_team_code)            -> "winner-derived" lead (what a pipeline would use)
  T (side-attached truth from D2_truth.py) -> "truth-derived" lead (sanity check)

Counts rounds where the tracked lead differs from the round_state lead, by year.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\D2_lead.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
STATES_PER_ROUND = 234265 / 28683  # ASSUMPTION: bake-off ratio of kept states to kept rounds


def check(label):
    t = pl.read_parquet(ROOT / "logs" / "swarm2" / f"D2_truth_{label}.parquet")
    t = t.select(["match_key", "round", "sx", "R", "T", "t0", "ct0", "flip"])
    by = defaultdict(dict)
    for mk, rd, sx, R, T, t0, ct0, fl in t.iter_rows():
        by[mk][rd] = (sx, R, T, t0, ct0, fl)
    out = {"matches_contig": 0, "rounds_checked_R": 0, "R_lead_bad_rounds": 0, "R_lead_bad_matches": set(),
           "rounds_checked_T": 0, "T_lead_bad_rounds": 0, "first_bad_round_hist": defaultdict(int),
           "rounds_with_rs_lead": 0}
    for mk, byr in by.items():
        rmax = max(byr)
        if min(byr) != 1 or rmax != len(byr):
            continue
        out["matches_contig"] += 1
        # --- winner-derived (R) tracking
        scoreX = scoreY = 0
        first_bad = None
        ok_chain = True
        for rd in range(1, rmax + 1):
            sx, R, T, t0, ct0, fl = byr[rd]
            if sx is None or R is None:
                ok_chain = False
                break
            rs_lead = None if (t0 is None or ct0 is None) else ct0 - t0
            tracked = (scoreX - scoreY) if sx == 3 else (scoreY - scoreX)
            if rs_lead is not None:
                out["rounds_with_rs_lead"] += 1
                out["rounds_checked_R"] += 1
                if tracked != rs_lead:
                    out["R_lead_bad_rounds"] += 1
                    out["R_lead_bad_matches"].add(mk)
                    if first_bad is None:
                        first_bad = rd
            if R == sx:
                scoreX += 1
            else:
                scoreY += 1
        if first_bad is not None:
            out["first_bad_round_hist"][first_bad] += 1
        # --- truth (T) tracking, sanity check only
        scoreX = scoreY = 0
        for rd in range(1, rmax + 1):
            sx, R, T, t0, ct0, fl = byr[rd]
            if sx is None or T is None:
                break
            rs_lead = None if (t0 is None or ct0 is None) else ct0 - t0
            tracked = (scoreX - scoreY) if sx == 3 else (scoreY - scoreX)
            if rs_lead is not None:
                out["rounds_checked_T"] += 1
                if tracked != rs_lead:
                    out["T_lead_bad_rounds"] += 1
            if T == sx:
                scoreX += 1
            else:
                scoreY += 1
    n_bad = out["R_lead_bad_rounds"]
    print(f"-- {label}: matches with contiguous rounds 1..max: {out['matches_contig']}")
    print(f"   rounds with round_state lead and a full team-tracked history: {out['rounds_checked_R']}")
    print(f"   winner-derived (R) lead != round_state lead: {n_bad} rounds in {len(out['R_lead_bad_matches'])} matches")
    print(f"   truth-derived (T) lead != round_state lead (sanity, circular by construction): {out['T_lead_bad_rounds']} rounds")
    print(f"   first bad round per affected match (round: matches): {sorted(out['first_bad_round_hist'].items())[:12]}")
    print(f"   ASSUMPTION states affected approx = rounds x {STATES_PER_ROUND:.2f}: {n_bad * STATES_PER_ROUND:,.0f}")
    return out


def main():
    for label in ["2024", "2026"]:
        check(label)


if __name__ == "__main__":
    main()
