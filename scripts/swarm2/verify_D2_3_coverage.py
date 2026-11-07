"""verify_D2_3 coverage: rows per round number (rounds 9-31) in the 2024 and 2026 tomes.

Checks whether rounds 26 and 27 (absent from the claim's truth table) exist in the raw tables and
whether they are dropped by the player_count filter, the 5v5 player_info validity check, or the
freeze-end join. Read-only; writes nothing.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_3_coverage.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(20)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
PC = {"2024": 20, "2026": 10}


def tbl(label, ch, cols):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))[0]
    return pl.scan_parquet(sorted(glob.glob(fd + "/dataframe_*"))).select(cols).collect()


def main():
    for label in ["2024", "2026"]:
        re_all = tbl(label, "round_end", ["match_key", "round", "tick", "player_count"])
        print(f"\n######## {label} round_end rows by round (all player_count), rounds 9-31 ########")
        pc_by_round = (re_all.filter(pl.col("round").is_between(9, 31))
                       .group_by(["round"]).agg(pl.len().alias("rows_all"),
                                                (pl.col("player_count") == PC[label]).sum().alias("rows_pc"))
                       .sort("round"))
        print(pc_by_round)
        print("player_count value counts (rounds 24-30):")
        print(re_all.filter(pl.col("round").is_between(24, 30)).group_by(["round", "player_count"]).len()
              .sort(["round", "player_count"]))

        pi = (tbl(label, "player_info", ["match_key", "round", "player_id", "team_code"])
              .unique(["match_key", "round", "player_id"], keep="first", maintain_order=True))
        cnt = pi.group_by(["match_key", "round"]).agg(
            pl.len().alias("np"), (pl.col("team_code") == 2).sum().alias("nt"), (pl.col("team_code") == 3).sum().alias("nct"))
        valid = cnt.filter((pl.col("np") == 10) & (pl.col("nt") == 5) & (pl.col("nct") == 5))
        print(f"\n######## {label} player_info (match, round) keys by round, and valid 5v5 keys ########")
        pr = (cnt.filter(pl.col("round").is_between(9, 31)).group_by("round").agg(pl.len().alias("keys"))
              .join(valid.filter(pl.col("round").is_between(9, 31)).group_by("round").agg(pl.len().alias("valid_5v5")),
                    on="round", how="left").sort("round"))
        print(pr)

        rs = tbl(label, "round_state", ["match_key", "round", "event_type"])
        fe = rs.filter(pl.col("event_type") == "round_freeze_end").group_by("round").agg(
            pl.col("match_key").n_unique().alias("matches_with_freeze_end"))
        print(f"\n######## {label} round_freeze_end matches by round (rounds 9-31) ########")
        print(fe.filter(pl.col("round").is_between(9, 31)).sort("round"))

        # matches that contain round 25: which rounds does each such match have in round_end (all pc)?
        m25 = re_all.filter(pl.col("round") == 25)["match_key"].unique()
        sub = re_all.filter(pl.col("match_key").is_in(m25.implode()) & pl.col("round").is_between(20, 31))
        per = sub.group_by("match_key").agg(pl.col("round").sort().alias("rounds"))
        print(f"\n######## {label}: matches with a round-25 row: {m25.len()}; round sets (rounds 20-31, all pc) ########")
        print(per.group_by("rounds").len().sort("len", descending=True).head(15))


if __name__ == "__main__":
    main()
