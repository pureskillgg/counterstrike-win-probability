"""verify_D8_5_maps: map x announced-format table and the final verifier summary (D8-5 verifier).

Run last in the chain: it appends the map table and the summary block (numbers as printed by the earlier sections of the
same log run) to logs\\swarm2\\verify_D8_5.log.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D8_5_maps.py
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
LOG = ROOT / "logs" / "swarm2" / "verify_D8_5.log"
OUT = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D8_5")
LOGF = open(LOG, "a", encoding="utf-8", newline="\n")
FOUR = ["de_inferno", "de_vertigo", "de_overpass", "de_nuke"]


def say(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    LOGF.write(s + "\n")
    LOGF.flush()


RS = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "event_type"])
                for f in sorted(glob.glob(str(BASE / f"basic_round_state{SUF}" / "dataframe_*")))],
               how="vertical_relaxed")
ann = (RS.filter(pl.col("event_type") == "round_announce_last_round_half")
       .select(["match_key", "round"]).unique().group_by("match_key").agg(pl.col("round").sort()))
ANN = {r["match_key"]: r["round"] for r in ann.iter_rows(named=True)}
R = pl.read_parquet(OUT / "rounds.parquet").sort(["match_key", "round"])
MAT = {}
for x in R.iter_rows(named=True):
    m = MAT.setdefault(x["match_key"], {"map": x["map_name"], "tA": int(x["tA"]), "tB": int(x["tB"]),
                                        "day": int(x["day"]), "rd": []})
    m["rd"].append(x)
for m in MAT.values():
    n = len(m["rd"])
    m["complete"] = n == m["tA"] + m["tB"] and n <= 24 and [x["round"] for x in m["rd"]] == list(range(1, n + 1))

say("\n=== verify_D8_5 MAP x FORMAT TABLE and SUMMARY (appended, last section of the chain) ===")
comp_tab = collections.defaultdict(collections.Counter)
for k, m in MAT.items():
    if m["complete"]:
        a = ANN.get(k, [])
        key = "MR8" if a == [7] else "MR12" if a == [11] else "none"
        comp_tab[m["map"]][key] += 1
        comp_tab[m["map"]]["n"] += 1
say("complete matches by map (days 12-25): n, MR8-announced, MR12-announced, no announcement:")
four_n = four_mr8 = 0
for mp in sorted(comp_tab):
    c = comp_tab[mp]
    say(f"  {mp:<12} n={c['n']:>4}  MR8={c['MR8']:>4}  MR12={c['MR12']:>4}  none={c['none']:>3}")
    if mp in FOUR:
        four_n += c["n"]
        four_mr8 += c["MR8"]
say(f"four maps combined: complete={four_n}, MR8-announced={four_mr8} ({four_mr8 / four_n:.3f})")

say("""
SUMMARY (verifier D8-5; numbers are from the sections above in this same log)
  1. Reproduced: D4 classes affected 211 / r13_flip_misfit 825 / other_misfit 116 / fits 94 (1246 complete, all days 12-25).
     other_misfit by map: inferno 43/202, vertigo 49/234, overpass 10/83, nuke 14/185; zero on the other five maps.
     Permutation (B=20000): map unadjusted stat 128.85 and map|day stat 119.23, both p <= 5e-5 (claim: 0.0002, consistent).
     Scoreboard signature split of other_misfit: (8,) 81, () 35; (8,) occurs only on the four maps (333 in days 12-25).
  2. Concentration on four maps: reproduced. Map is not sufficient: the four maps also host 1,155 standard (12,) matches
     (days 12-25), and 501 complete MR12 matches. The MR8 halftime is confined to the four maps (161 of 704 complete).
  3. Halftime regime: the game's own event round_announce_last_round_half marks MR8 (round 8) and MR12 (round 12).
     other_misfit by announced format: MR8 115, MR12 1. The scoreboard (8,) detector finds 109 of the 161 complete MR8
     matches; the 34 other_misfit no-swap cases are MR8 halftimes it missed (ambiguous score gaps of 0 or 1).
     The claim's 81/35 split is therefore a detector artifact, not two groups.
  4. Not a different defect: the first post-halftime round is the defective round in both formats. Scoreboard anomaly
     at h+1 in 62% of MR8 matches (241 of 386) and 61% of MR12 matches (1653 of 2701), days 12-25. Complete matches show
     no clean step at h+1 in either format. Flipping round 13 fits 1005 of 1007 MR12 complete matches (normal orientation); flipping round 9
     with either orientation fixes 161 of 161 MR8 complete matches, while the best placebo round reaches 104 of 161.
     (Round 13 in MR12 fits 1005 of 1007 with normal orientation; the other 2 are a 13-round and a no-data case.)
  5. Header orientation is not one convention. Orientation from the scoreboard's final side scores (independent of
     round 9): MR8 normal 37 / swapped 85 / tie 38; MR12 normal 948 / swapped 3 / tie 49. A global orientation flip for
     MR8 would mis-handle about 37 matches; per-match orientation is required.
  6. The claim's F-3 test flipped only round 13 under the detected schedule; that is why it reports 0 fits for (8,).
     With the announced halftime and the round-9 flip, 161 of 161 complete MR8 matches fit under one of the two
     orientations (per-match choice). Under the claimed fix alone (swap schedule, no flip), 1 of 161 fits.
  7. Selection caveat: none of the 774 MR12 matches with a clean round-13 step is complete; they carry round-count
     shortfalls against the header total (most common -2, -3, -8, i.e. fewer rounds than the header total), which is
     why they are outside the complete set. Complete-set rates are conditional on completeness
     (complete share by map: 26% to 51% across the nine maps; four-map range 27% to 42%).
  8. Days 01-07 contribute no complete matches (round counts fall short of the header totals), so the analysis covers
     days 12-25 only.

VERDICT: PARTIAL.
  Confirmed: the reported counts, the four-map concentration, the claim that map is not the key, and that the misfit
  tracks a halftime regime (the MR8 announced halftime).
  Not supported: "a different defect from round 13" (same first-post-halftime defect, different round and format);
  "tracks the detected (8,) signature" (detector misses 34 of 115 MR8 cases); "key on the detected schedule" as a fix
  (schedule alone reconciles 0 of 161 MR8 matches; the round-9 correction and per-match header orientation are also
  needed).
  Required fixes: (a) take the halftime from round_announce_last_round_half, not the scoreboard; (b) correct the first
  post-halftime round winner in both formats (round 13 MR12, round 9 MR8); (c) set the header orientation per match from
  the scoreboard final side scores, not a global convention; (d) redo F-3 with the announced halftime and round h+1;
  (e) report rates with completeness by map and format, since the complete set excludes clean MR12 matches.
""")
LOGF.close()
print("MAPS DONE")
