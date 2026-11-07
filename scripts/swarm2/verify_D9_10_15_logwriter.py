"""verify_D9_10 step 15: write logs\\swarm2\\verify_D9_10.log by running the reproduction scripts and storing their stdout.

The only file this writes is logs\\swarm2\\verify_D9_10.log. Scripts are run with the project interpreter, -I -X utf8.
"""
import subprocess
from pathlib import Path

ROOT = Path(r"G:\pureskill\csds-cs2")
PY = str(ROOT / ".pixi" / "envs" / "default" / "python.exe")
LOG = ROOT / "logs" / "swarm2" / "verify_D9_10.log"
SCRIPTS = [
    "verify_D9_10_12_summary.py",
    "verify_D9_10_13_format.py",
    "verify_D9_10_14_mr8_control.py",
]

HEADER = """verify_D9_10 - independent verification of the D9-10 claim (2024-02-g tome, build 9961 and 9842 controls)

CLAIM UNDER TEST (as relayed)
  Build 9961, pc20, non-bomb rounds: round 13 flipped in 1,580 of 1,778 scored matches (88.9%); round 25 in 88 of 88;
  round 28 in 84 of 84; round 9 in 294 of 1,873 (293 without a round-13 flip). Claim: inverted rounds are the first
  round after a side switch. Round-9 subset documented as unexplained (about 16% of build-9961 matches).
  Proposed fix: round index plus per-match evidence on rounds 13, 25, 28, with the round-9 subset checked separately.

VERDICT: PARTIAL
  - The counts reproduce exactly (see [3]). The 88.9% uses N rounds excluded from the denominator (claim convention).
  - Round 9 is NOT unexplained: 290 of the 294 flipped matches are 8-round-half matches (regulation about 16 rounds,
    no overtime; final score sums 13-17, mostly 15-16; maps de_inferno, de_vertigo, de_overpass, de_nuke) whose halftime
    is after round 8. The game marks the second half at round 8 (phase), the team side changes at 9, and the
    bomb-verified winner at round 9 is inverted in 10 of 10 bomb rounds of those matches (0 of 448 elsewhere).
    The other 4 flips are noise (3 at round 9 in 13-halftime matches, 1 with no detected halftime).
    The same format in build 9842 (206 matches) shows no inversion (0 of 8 bomb rounds).
  - Round 25 is inverted WITHOUT a side switch: the planter stays on T from round 24 to 25 (65 of 65) and no
    player_info side change occurs at round 25. The 'first round after a side switch' description does not hold
    for 25; the bomb-verified inversion is 24 of 24 and defused rounds at 25 and 28 have CT defusers (37 of 37
    recorded as T). Round 28 IS a true side switch (planter T 3->2 at 28) and is inverted 31 of 31.
  - Round 13 is the first round after a true side switch in 13-halftime matches: bomb-verified inversion 434 of 434;
    the 1,580 F of 1,586 decided (99.6%) in those matches is the real rate. The 88.9% headline is diluted by the
    8-round-half matches where round 13 is not first after halftime (191 R of 198 R at round 13).
  - Denominator: 294 of 2,478 build-9961 matches = 11.9% (the claim's 16% uses 1,873 scored round-9 matches).
  - Defect bins: the e2 'other' bin (41 of 14,602) excludes round 25 only; it includes round 9 (10) and round 28 (31).

WHAT WAS TESTED AND REJECTED AS THE EXPLANATION
  - Death-label inversion: every non-null death player_team_code agrees with player_info (316,914 of 316,914).
    The 467 apparent disagreements in the first pass were null death sides, not inverted labels.
  - Round-index shift (winner of round r really belongs to r-1 or r+1): the bomb side agrees with the neighbouring
    rounds' recorded winners in only 42-65% of bomb rounds at labels 9, 13, 25 and 28, while in 100% of the flipped
    bomb rounds the bomb side is opposite to the round's own recorded winner.
  - Date window: build 9842 has 528 matches on day 12 and 3,459 bomb rounds on days 12-25 with 0 inversions.
  - round_end win_reason_message: not independent. It is inverted together with winner_team_code (for example
    'Terrorists_Win' on defused rounds), so it cannot validate the winner field.

FIX EVALUATION on build 9961 bomb rounds (n = 9,031; ground truth = bomb outcome)
  - no fix: 499 errors
  - round-index only (flip 13, 25, 28 in every match): 17 errors (10 at round 9 left unflipped, 7 at round 13 wrongly flipped)
  - per-match: flip the round after each detected halftime (9 or 13), plus 25 and 28: 0 errors
  - per-match halftime only (no 25/28 flip): 55 errors
  The claimed fix is right in direction only if it uses per-match halftime; a round-index rule is wrong for the
  8-round-half matches. The round-9 subset should be documented as the 8-round-half format, not as unexplained.

ASSUMPTIONS
  - ASSUMPTION: bomb_defused -> side 3 (CT), bomb_exploded -> side 2 (T) is the ground truth for the round winner.
  - ASSUMPTION: a defuse is a CT action (defuser side 3) and a plant is a T action (planter side 2); both are
    supported by the data (defused rounds: 5,047 of 5,047 with defuser side 3).
  - ASSUMPTION: halftime = first round r in {9, 13} where more than half of the players' player_info team_code
    changes from round r-1 (team side change). Confirmed by the planter's own side code at the same round.
  - ASSUMPTION: death test as in the claim (R / F / N); bomb rounds are excluded from it.
  - ASSUMPTION: the 8-round-half format is inferred from the game's phase marker (second half from round 8), the
    halftime side change at round 9, and final score sums 13-17. It is not read from a format field.
  - NOT CHECKED: the 2026 tome; the claim's 1,631 single-round-inversion count (regex-based, not used in the verdict).
  - Small samples: round 9 bomb check n = 10 (flipped) and 8 (build 9842 control); round 25 n = 24; round 28 n = 31.

NOTE ON 'None' IN LOG LINES
  A tuple or group marked None means a null value (for example a death row with no player_team_code), not a label.

SCRIPTS (all in scripts\\swarm2, prefix verify_D9_10_)
  00 schema, 01 field probe, 02 per-round table, 03 claim recompute, 04 alternatives, 05 side signatures,
  06 round-9 vs round-13, 07 overtime, 08 debug, 09 defuser and planter, 11 reconcile, 12 summary, 13 format, 14 control.
  Output below is from 12, 13 and 14 (the reproduction set).
"""


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "w", encoding="utf-8") as f:
        f.write(HEADER)
        for s in SCRIPTS:
            f.write(f"\n##### {s} #####\n")
            p = subprocess.run([PY, "-I", "-X", "utf8", str(ROOT / "scripts" / "swarm2" / s)],
                               capture_output=True, text=True, encoding="utf-8")
            f.write(p.stdout)
            if p.returncode:
                f.write(f"\n[exit {p.returncode}] {p.stderr}\n")
    print("wrote", LOG)


if __name__ == "__main__":
    main()
