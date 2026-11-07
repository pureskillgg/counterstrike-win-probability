"""A8 check 4: round label versus bomb outcome, 2026 tome and 2024-02 tome (read-only).

Rule (CS2 engine behaviour): a round with a bomb_exploded event must have winner_team_code 2 (T);
a round with a bomb_defused event must have winner_team_code 3 (CT). Rounds with neither event
are not testable by this rule. Counts are per round, from round_end winner_team_code and the
bomb_state events in the same round. Round 13 is reported separately because it is the first
round after the halftime swap in MR12 matches.
Run from the project root:
  .pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A8_label_bomb_check.py
"""

import glob
import sys
from collections import Counter
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
TOMES = {
    "2026-10-01_to_08": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds",
    "2024-02-g": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
}
T, CT = 2, 3


def read_channel(root, ch, cols):
    folders = sorted(glob.glob(str(root / f"basic_{ch}.*")))
    if not folders:
        raise SystemExit(f"no folder for {ch} in {root}")
    files = sorted(glob.glob(str(Path(folders[0]) / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def main():
    for name, root in TOMES.items():
        print(f"== {name} ==")
        RE = read_channel(root, "round_end", ["match_key", "round", "tick", "winner_team_code"])
        BS = read_channel(root, "bomb_state", ["match_key", "round", "tick", "event_type"])
        n_rounds = RE.height
        ev = (BS.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
              .group_by(["match_key", "round"])
              .agg(pl.col("event_type").unique().alias("evs")))
        ev_map = {(m, int(r)): set(e) for m, r, e in ev.select(["match_key", "round", "evs"]).iter_rows()}
        c = Counter()
        r13 = Counter()
        by_round_n = Counter()
        for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows():
            r = int(r)
            evs = ev_map.get((m, r), set())
            if not evs:
                c["no_bomb_outcome"] += 1
                continue
            if "bomb_exploded" in evs and "bomb_defused" in evs:
                c["both_exploded_and_defused"] += 1
            expect = T if "bomb_exploded" in evs else CT
            ok = int(w) == expect
            c["match" if ok else "mismatch"] += 1
            by_round_n[(r, ok)] += 1
            if r == 13:
                r13["match" if ok else "mismatch"] += 1
        print(f"round_end rows: {n_rounds}; rounds with a bomb outcome: "
              f"{c['match'] + c['mismatch'] + c['both_exploded_and_defused']}")
        print(f"  label agrees with bomb outcome: {c['match']}; disagrees: {c['mismatch']}; "
              f"no bomb outcome in round: {c['no_bomb_outcome']}; both events in one round: "
              f"{c['both_exploded_and_defused']}")
        print(f"  round 13: agrees {r13['match']}, disagrees {r13['mismatch']}")
        other_mis = c["mismatch"] - r13["mismatch"]
        other_tot = (c["match"] + c["mismatch"]) - (r13["match"] + r13["mismatch"])
        print(f"  all other rounds with a bomb outcome: disagrees {other_mis} of {other_tot}")
        mis_by_r = sorted(((r, n) for (r, ok), n in by_round_n.items() if not ok))[:20]
        print(f"  disagreements by round number (first 20 round numbers): {mis_by_r}")
    print("\nReading: 0 disagreements means the winner codes match the bomb outcome in every testable round.")


if __name__ == "__main__":
    main()
