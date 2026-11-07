"""A8 check 9: which source is wrong for 2024 round 13, the winner code or the bomb events?

Idea: round_state freeze-time scores are a separate engine field. For a round r whose next round
has a freeze row and no side swap is scheduled after r, the team whose score rises from round r to
round r+1 is the team that won round r (fields are team-attached in the bake-off check). So the
field that rises gives an observed winner side, independent of round_end.winner_team_code and of
the bomb events. Compare that observed side with (a) winner_team_code and (b) the bomb-event rule
(exploded -> T, defused -> CT). Swap rule (bakeoff.swap_expected) uses max_rounds, which the 2024
header lacks, so only rounds where no swap can fall after r are used: round 13 (MR24 and MR16 both
have no swap after round 13), and a sanity sample of other rounds with no swap after r for MR24.
Rounds with max_rounds unknown are excluded. ASSUMPTION: max_rounds 24 for matches whose header
has no halftime-based value is not needed here, because the rule is applied only to rounds where
the swap is impossible for either 16 or 24 round formats (round 13 transitions only; see below).
Run from the project root:
  .pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A8_round13_score_check.py
"""

import glob
import sys
from collections import Counter
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
TOMES = {
    "2026-10-01_to_08": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"),
    "2024-02-g": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds"),
}
T, CT = 2, 3


def read_channel(root, ch, cols):
    folders = sorted(glob.glob(str(root / f"basic_{ch}.*")))
    if not folders:
        folders = sorted(glob.glob(str(root / f"basic_{ch}")))
    files = sorted(glob.glob(str(Path(folders[0]) / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def main():
    for name, root in TOMES.items():
        print(f"== {name} ==")
        RE = read_channel(root, "round_end", ["match_key", "round", "tick", "winner_team_code"])
        BS = read_channel(root, "bomb_state", ["match_key", "round", "tick", "event_type"])
        RS = read_channel(root, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
        fr = (RS.filter(pl.col("event_type") == "round_freeze_end")
              .with_columns(pl.col("round").cast(pl.Int64))
              .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
        fmap = {(m, int(r)): (t, c) for m, r, t, c in fr.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
        ev = (BS.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
              .with_columns(pl.col("round").cast(pl.Int64))
              .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
        evm = {(m, int(r)): set(e) for m, r, e in ev.select(["match_key", "round", "evs"]).iter_rows()}
        win = {(m, int(r)): int(w) for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows()}

        # round 13 transitions (no swap after round 13 in MR16 or MR24)
        c13 = Counter()
        for (m, r), w in win.items():
            if r != 13:
                continue
            f0 = fmap.get((m, 13))
            f1 = fmap.get((m, 14))
            if f0 is None or f1 is None or f0[0] is None or f1[0] is None:
                c13["missing_freeze_or_null"] += 1
                continue
            dt = f1[0] - f0[0]
            dc = f1[1] - f0[1]
            if (dt, dc) == (1, 0):
                obs = T
            elif (dt, dc) == (0, 1):
                obs = CT
            else:
                c13[f"score_change_not_one_point {(dt, dc)}"] += 1
                continue
            e = evm.get((m, 13), set())
            bomb = None
            if e and not ("bomb_exploded" in e and "bomb_defused" in e):
                bomb = T if "bomb_exploded" in e else CT
            c13["transitions"] += 1
            c13["observed_eq_winner_code" if obs == w else "observed_ne_winner_code"] += 1
            if bomb is not None:
                c13["bomb_testable"] += 1
                c13["observed_eq_bomb" if obs == bomb else "observed_ne_bomb"] += 1
                if obs != w and obs == bomb:
                    c13["winner_code_wrong_bomb_and_score_agree"] += 1
                if obs == w and obs != bomb:
                    c13["bomb_disagrees_score_and_winner_agree"] += 1
                if obs != w and obs != bomb:
                    c13["all_three_disagree"] += 1
        print("round 13 -> 14 (no swap possible after round 13):")
        for k, v in sorted(c13.items()):
            print(f"  {k}: {v}")

        # sanity: rounds 1..11 and 14..20 where no swap can fall after r in MR24 (r not 12, not 3rd of OT)
        # and the bomb rule is testable; observed-vs-winner agreement by round number
        agree = Counter()
        for (m, r), w in win.items():
            if r in (12,) or r > 23 or r == 0:
                continue
            f0 = fmap.get((m, r))
            f1 = fmap.get((m, r + 1))
            if f0 is None or f1 is None or f0[0] is None or f1[0] is None:
                continue
            dt, dc = f1[0] - f0[0], f1[1] - f0[1]
            if (dt, dc) == (1, 0):
                obs = T
            elif (dt, dc) == (0, 1):
                obs = CT
            else:
                continue
            if (1 <= r <= 7) or (9 <= r <= 11) or (14 <= r <= 18):
                agree["checked"] += 1
                agree["agree" if obs == w else "disagree"] += 1
        print(f"sanity (rounds 1-11 and 14-20, MR24-type rounds, no swap assumed): {dict(agree)}")
        print()
    print("Reading: if observed == bomb and observed != winner code for the disagreeing round-13 rows,")
    print("the recorded winner code is the wrong field. ASSUMPTION: the score fields and the bomb events are")
    print("each correct unless shown otherwise. The 2026 tome is the control.")


if __name__ == "__main__":
    main()
