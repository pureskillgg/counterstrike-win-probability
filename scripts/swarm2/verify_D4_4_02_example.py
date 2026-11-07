r"""Verifier D4-4: example-match inspection around the halftime boundary (rounds 10-16).
Shows freeze-end score fields, round_end labels and bomb outcomes for two example matches.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D4_4_02_example.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
EXAMPLES = ["csds/2024/02/12/05d51edb-04bc-420c-91b9-4bb95317ca19", "csds/2024/02/12/01648186"]


def scan(ch, cols):
    fd = sorted(BASE.glob(f"basic_{ch}.*"))[0]
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.scan_parquet(files).select(cols)


for ex in EXAMPLES:
    print(f"===== match prefix {ex}")
    rs = (scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"])
          .filter(pl.col("match_key").str.starts_with(ex))
          .filter(pl.col("event_type") == "round_freeze_end")
          .collect().sort(["round", "tick"]))
    re_ = (scan("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"])
           .filter(pl.col("match_key").str.starts_with(ex)).collect().sort(["round", "tick"]))
    be = (scan("bomb_state", ["match_key", "round", "tick", "event_type"])
          .filter(pl.col("match_key").str.starts_with(ex))
          .filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
          .collect().sort(["round", "tick"]))
    print("freeze_end rows (round, tick, t_score, ct_score, phase):")
    for r in rs.iter_rows(named=True):
        if 9 <= r["round"] <= 16:
            print(f"   r{r['round']:>2} tick={r['tick']:>7} t={r['t_score']:>2} ct={r['ct_score']:>2} phase={r['phase']}")
    print("round_end rows (round, tick, winner, reason):")
    for r in re_.iter_rows(named=True):
        if 9 <= r["round"] <= 16:
            print(f"   r{r['round']:>2} tick={r['tick']:>7} winner={r['winner_team_code']} reason={r['win_reason_code']}")
    print("bomb outcome rows (round, tick, event):")
    for r in be.iter_rows(named=True):
        if 9 <= r["round"] <= 16:
            print(f"   r{r['round']:>2} tick={r['tick']:>7} {r['event_type']}")
