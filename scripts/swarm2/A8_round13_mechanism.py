"""A8 check 11: mechanism of the 2024 round-13 winner-code mismatch (read-only diagnostic).

For each 2024 match with the needed freeze rows, the observed winner of round r (from the one-point score
rise between the freeze rows of round r and r+1, valid when no swap falls after r) is compared with
winner_team_code of round r, for r = 11, 12, 13, 14. Also tabulates whether the recorded round-13 code equals
the observed round-12 winner (a shifted-row pattern), and the bomb rule for round 13.
Run from the project root:
  .pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A8_round13_mechanism.py
"""

import glob
import sys
from collections import Counter
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
TOME = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
T, CT = 2, 3


def read_channel(ch, cols):
    folders = sorted(glob.glob(str(TOME / f"basic_{ch}.*")))
    files = sorted(glob.glob(str(Path(folders[0]) / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def main():
    RE = read_channel("round_end", ["match_key", "round", "tick", "winner_team_code"])
    RS = read_channel("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    BS = read_channel("bomb_state", ["match_key", "round", "tick", "event_type"])
    fr = (RS.filter(pl.col("event_type") == "round_freeze_end")
          .with_columns(pl.col("round").cast(pl.Int64))
          .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
    fmap = {(m, int(r)): (t, c) for m, r, t, c in fr.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
    win = {(m, int(r)): int(w) for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows()}
    ev = (BS.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
          .with_columns(pl.col("round").cast(pl.Int64))
          .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
    evm = {(m, int(r)): set(e) for m, r, e in ev.select(["match_key", "round", "evs"]).iter_rows()}

    def obs(m, r):
        f0, f1 = fmap.get((m, r)), fmap.get((m, r + 1))
        if f0 is None or f1 is None or f0[0] is None or f1[0] is None:
            return None
        d = (f1[0] - f0[0], f1[1] - f0[1])
        return T if d == (1, 0) else (CT if d == (0, 1) else None)

    matches = sorted({m for m, _ in win})
    cnt = Counter()
    pattern13 = Counter()
    for m in matches:
        o12, o13, o14 = obs(m, 12), obs(m, 13), obs(m, 14)
        w12, w13, w14 = win.get((m, 12)), win.get((m, 13)), win.get((m, 14))
        if o13 is None or w13 is None:
            cnt["r13 not observable"] += 1
            continue
        cnt["r13 observable"] += 1
        cnt["r13 code == observed"] += int(w13 == o13)
        cnt["r13 code != observed"] += int(w13 != o13)
        if w12 is not None and o12 is not None:
            cnt["r12 code == observed"] += int(w12 == o12)
        e13 = evm.get((m, 13), set())
        bomb13 = None
        if e13 and not ("bomb_exploded" in e13 and "bomb_defused" in e13):
            bomb13 = T if "bomb_exploded" in e13 else CT
        key = ("code==obs" if w13 == o13 else "code!=obs",
               "bomb=obs" if bomb13 == o13 else ("bomb=code" if bomb13 == w13 else ("no bomb" if bomb13 is None else "bomb other")))
        pattern13[key] += 1
        # shifted-row test: does the recorded round-13 code equal the observed round-12 winner?
        if o12 is not None and w13 is not None and w13 != o13:
            cnt["r13 mismatch rows: code == observed r12 winner"] += int(w13 == o12)
            cnt["r13 mismatch rows total"] += 1
        # is the mismatch tied to the round-12 code disagreeing with its observation?
        if w12 is not None and o12 is not None and w13 is not None and o13 is not None:
            cnt[f"r12 code {'==' if w12 == o12 else '!='} obs; r13 code {'==' if w13 == o13 else '!='} obs"] += 1
    print("== 2024-02 round 13 mechanism ==")
    for k, v in sorted(cnt.items()):
        print(f"  {k}: {v}")
    print("\npattern for round 13 (code vs observed score winner; bomb vs observed):")
    for k, v in sorted(pattern13.items(), key=lambda x: -x[1]):
        print(f"  {k[0]:<10} {k[1]:<12} {v}")
    # the 2026 control
    print("\n(2026 control: run A8_round13_score_check.py; all 1,649 observable round-13 transitions agree)")


if __name__ == "__main__":
    main()
