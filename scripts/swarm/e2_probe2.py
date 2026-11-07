"""E2 probe 2: 2024 round structure, duplicates, player id encodings. Reads G: only."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
}


def read(base, suffix, ch):
    fd = base / ("header" if ch == "header" else f"basic_{ch}{suffix}")
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")


for era, (base, suffix) in TOMES.items():
    print(f"===== {era}")
    hdr = read(base, suffix, "header")
    print("header rows", hdr.height, "unique key", hdr["key"].n_unique(), "unique demo_id", hdr["demo_id"].n_unique() if "demo_id" in hdr.columns else None)
    re_ = read(base, suffix, "round_end")
    rs = read(base, suffix, "round_state")
    rst = read(base, suffix, "round_start")
    print("round_end rows", re_.height, "distinct (match,round)", re_.select(["match_key", "round"]).unique().height)
    print("round_end player_count:", re_.group_by("player_count").len().sort("player_count").rows())
    print("round_end winner:", re_.group_by("winner_team_code").len().rows())
    print("round_end win_reason:", re_.group_by("win_reason_code").len().sort("win_reason_code").rows())
    print("round_start channel rows", rst.height, "distinct", rst.select(["match_key", "round"]).unique().height)
    print("round_state freeze_end rows", rs.filter(pl.col("event_type") == "round_freeze_end").height)
    print("round_state distinct match", rs["match_key"].n_unique(), "round_end-channel distinct match", re_["match_key"].n_unique())
    # rounds per match distribution
    rpm = re_.group_by("match_key").len()
    print("rounds per match (round_end channel) describe:", rpm["len"].describe().rows())
    # per-match duplicate round numbers in round_end
    dup = re_.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
    print("duplicated (match,round) in round_end:", dup.height)
    if dup.height:
        ex = dup.head(1).row(0)
        print("example dup key", ex)
        print(re_.filter((pl.col("match_key") == ex[0]) & (pl.col("round") == ex[1])))
    # round_state round_end vs channel
    rs_re = rs.filter(pl.col("event_type") == "round_end").group_by("match_key").len()
    print("round_state round_end per match describe:", rs_re["len"].describe().rows())
    # round numbers range
    print("round range", re_["round"].min(), re_["round"].max())
    # player id encodings in deaths
    pdth = read(base, suffix, "player_death")
    for c in ["player_id", "attacker_id", "assister_id", "player_id_fixed", "attacker_id_fixed"]:
        vc = pdth.group_by(c).len().sort("len", descending=True).head(6).rows()
        nulls = pdth[c].null_count()
        print(f"deaths {c}: nulls={nulls} top={vc}")
    print("death rows", pdth.height, "death matches", pdth["match_key"].n_unique())
    print("header map counts", hdr.group_by("map_name").len().sort("len", descending=True).rows())
