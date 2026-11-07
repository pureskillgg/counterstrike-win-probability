"""verify_D1_7 probe (swarm2 verifier, independent of the claimant's scripts).

Purpose: learn the semantics of round_state score fields, round_end win reasons,
and bomb events around overtime round 27 before any recomputation.
Memory: lazy scans with column projection, one channel at a time, no E: access.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(50)

T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")


def files(root, ch):
    pat = str(root / f"basic_{ch}*" / "dataframe_*")
    return sorted(glob.glob(pat))


def scan(root, ch, cols):
    fs = files(root, ch)
    return pl.concat([pl.scan_parquet(f).select(cols) for f in fs], how="vertical_relaxed")


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100, flush=True)


for name, root in [("2024", T24), ("2026", T26)]:
    hdr(f"{name}: file counts")
    for ch in ["round_end", "round_state", "bomb_state", "player_info", "player_death"]:
        print(ch, len(files(root, ch)), flush=True)

    hdr(f"{name}: round_state event_type and phase counts")
    rs = scan(root, "round_state", ["match_key", "round", "tick", "event_type", "phase", "is_warmup", "t_score", "ct_score", "t_score_raw", "ct_score_raw"]).collect()
    print("rows", rs.height)
    print(rs.group_by("event_type").len().sort("len", descending=True).head(15))
    print(rs.group_by("phase").len().sort("len", descending=True).head(15))
    print("is_warmup counts:", rs.group_by("is_warmup").len().to_dicts())
    print("raw vs display equal rate:",
          round(float((rs["t_score"] == rs["t_score_raw"]).mean()), 4),
          round(float((rs["ct_score"] == rs["ct_score_raw"]).mean()), 4))
    del rs

    hdr(f"{name}: round_end win_reason_message counts (all rounds) and round 27 only")
    re_ = scan(root, "round_end", ["match_key", "round", "winner_team_code", "win_reason_code", "win_reason_message", "player_count"]).collect()
    print("round_end rows", re_.height, "unique (match,round)", re_.unique(["match_key", "round"]).height)
    print(re_.group_by("win_reason_message").len().sort("len", descending=True).head(12))
    print("round 27 win reasons:")
    print(re_.filter(pl.col("round") == 27).group_by(["win_reason_message", "winner_team_code"]).len().sort("len", descending=True))
    print("player_count x round 27:", re_.filter(pl.col("round") == 27).group_by("player_count").len().to_dicts())
    del re_
