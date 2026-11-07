"""verify_D8_5_final: final verifier summary appended to the log (supersedes the earlier SUMMARY block in this file).

Every number below was printed by a verify_D8_5_* script in this session (see the section named in brackets).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D8_5_final.py
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
LOG = Path(r"G:\pureskill\csds-cs2\logs\swarm2\verify_D8_5.log")
TEXT = """
==================================================================================================
FINAL VERIFIER SUMMARY (verify_D8_5_final.py). This block supersedes the earlier SUMMARY block above
(its 61-62% anomaly rates pooled both builds; its "no build confound" wording is withdrawn).
==================================================================================================
CLAIM D8-5: non-round-13 header misfit concentrated on four maps; tracks a halftime regime; claimed verdict "supported";
fix = key on per-match detected swap schedule, not map_name; resolve header orientation first.

VERDICT: PARTIAL

1. Reproduced exactly [check A-D]: D4 classes affected 211 / r13_flip 825 / other_misfit 116 / fits 94 (1,246 complete);
   other_misfit inferno 43/202, vertigo 49/234, overpass 10/83, nuke 14/185, zero on the other five maps;
   permutation (B=20000) map stat 128.85 and map|day stat 119.23, zero exceedances (p < 5e-5; claim 0.0002 consistent);
   (8,) signature 333 matches in days 12-25, all on the four maps; (12,) on the four maps 1,155.
2. Halftime regime [events, format, maps]: the game's own event round_announce_last_round_half gives MR8 (after round 8)
   for 161 complete matches and MR12 (after round 12) for 1,007; 78 complete have no announcement and fit MR12 exactly.
   other_misfit by announced format: MR8 115, MR12 1. The scoreboard (8,) detector finds 109 of the 161 MR8 matches; the
   34 other_misfit no-swap cases are MR8 halftimes it missed. The claim's 81/35 split is a detector artifact.
   Map is not sufficient: MR8 is 161 of 704 complete matches on the four maps, and the four maps also host 1,155 standard
   MR12 matches (days 12-25). Map is a proxy for the MR8 format, not the key.
3. BUILD CONFOUND (not in the claim; strongest alternative) [build_check, build_labels, build_steps]:
   - header build_num: 9842 (1,957 matches), 9961 (2,478).
   - all 1,246 complete matches and all 116 other_misfit are build 9961.
   - build 9842 (973 matches in days 12-25): round_end labels start at 2 in 973 of 973 matches and rows equal the header
     total in 0 of 973, so no build 9842 match can reach the complete set. Halftime step alignment is fine in both builds
     (the halftime step is a swap or both in each build), so the labels are aligned with the scoreboard at the halftime.
   - first post-halftime round, scoreboard vs round_end: build 9961 MR8 round 9 anomaly 241 of 292 (82.5%), MR12 round 13
     1,653 of 1,927 (85.8%); build 9842 MR8 0 of 94 and MR12 0 of 774 (round_end agrees with the scoreboard everywhere).
   - round-13 bomb mismatch, days 12-25: build 9961 434 of 441 (98.4%); build 9842 0 of 177.
   So the defect is a build 9961 effect. Day and map controls cannot separate it; build must be stratified.
4. Not a different defect [check E, profile, union, xtab]: within build 9961 the first post-halftime round is the defective
   round in both formats. MR12: flipping round 13 fits 1,005 of 1,007 complete matches (normal orientation); placebo rounds
   fit at most 555. MR8: the round-9 flip with either orientation fits 161 of 161 complete matches; the best placebo round
   reaches 104. Round 9 is the defective round for MR8, round 13 for MR12. This is the same mechanism as D8-1, not a
   different one.
5. Header orientation is not one convention [orient]: from the scoreboard's final side scores (independent of round 9),
   MR8 normal 37 / swapped 85 / tie 38 (no final 1); MR12 normal 948 / swapped 3 / tie 49. A global orientation rule for
   MR8 would mishandle about 37 matches; orientation has to be decided per match.
6. The proposed fix does not reconcile as written [check E-2, xtab, union]: detected (8,) schedule with no flip fits 0 of
   109 detected (8,) complete matches; the claimed F-3 test flipped only round 13, which is why it reports 0 fits. With the
   announced halftime and the round-9 flip, orientation chosen per match, 161 of 161 MR8 complete matches fit. Under the
   claimed fix alone (swap schedule, no flip) 1 of 161 MR8 complete matches fits.
7. Selection [select]: none of the 774 MR12 matches with a clean round-13 step is complete; those are the build 9842
   matches (round labels start at 2, rows short of the header total). Per-map rates are conditional on the complete set,
   and that set is build 9961 only. Denominators for per-map rates, all days 12-25 header matches (any build): inferno
   566, vertigo 580, overpass 302, nuke 437; other_misfit counts 43, 49, 10, 14 give 7.6%, 8.5%, 3.3%, 3.2%.
8. Days 01-07 contribute no complete matches (build 9842 labels and shortfalls), so the analysis covers days 12-25 only.

CONFIRMED: the reported counts, the four-map concentration, that map is not the key, and that the misfit tracks an
MR8 halftime regime (announced).
NOT SUPPORTED: "a different defect from round 13" (same first-post-halftime defect, build 9961, round 9 in MR8);
"tracks the detected (8,) signature" (detector misses 34 of 115); "key on the detected schedule" as the fix (0 of 109
detected (8,) fit as-is; needs the round-9 correction and per-match orientation); "supported" as a verdict.

REQUIRED FIXES
  (a) Take halftime position from round_announce_last_round_half (label + 1), not the scoreboard (8,) detector.
  (b) Re-label the non-round-13 misfit as the first-post-halftime defect (round 9 in MR8), the same as D8-1; drop "different defect".
  (c) Stratify every D8 analysis by build_num (or restrict to build 9961) before any map or day claim; day and map controls do not control build.
  (d) Replacement for the first post-halftime winner: the scoreboard side where it disagrees with round_end, in build 9961 only.
  (e) Decide MR8 header orientation per match from the scoreboard final side scores; do not use a global convention.
  (f) Redo F-3 with the announced halftime and the round h+1 flip; the 0-fit result is a test-design artifact.
  (g) State the denominator for per-map rates (complete set, build 9961, vs all days 12-25 header matches).

ASSUMPTIONS (marked in the scripts): permutation B=20000 and seed 20261014; announcement label v => halftime after v+1
(verified against the scoreboard halftime step in both builds); build 9842 round-label offset taken at face value;
freeze_end = first by tick, as in D8; complete = labels 1..n, n = header total, n <= 24.
"""
with open(LOG, "a", encoding="utf-8", newline="\n") as f:
    f.write(TEXT)
print("FINAL DONE")
