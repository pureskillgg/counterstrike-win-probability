"""A6 probe: round 13 / 25 inversion per match, round-1 coverage, and 2026 multi-side score rounds.

Q1. 2024: is the round_end-vs-score inversion at round 13 (and 25) a per-match property?
    Uses A6_threeway_2024.parquet written by A6_threeway_probe.py (W, S_ns, S_sw, B per round).
Q2. 2024: for matches with no round_end row for round 1, is there a round_start / round_freeze_end row for round 1?
Q3. 2026: for the 106 rounds with two +1 score_update rows, where do the two rows sit relative to round_end?
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A6_r13_match_probe.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(50)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
BASE26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
T, CT = 2, 3


def files(base, ch):
    return sorted(glob.glob(str(sorted(base.glob(f"basic_{ch}.*"))[0] / "dataframe_*")))


def load(base, ch, cols):
    return pl.scan_parquet(files(base, ch)).select(cols).collect()


def main():
    P2 = pl.read_parquet(Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad") / "A6_threeway_2024.parquet")
    print("== Q1 2024 inversion per match (rows with S_ns known and W, B present)")
    for r in (13, 25):
        s = P2.filter((pl.col("round") == r) & pl.col("S_ns").is_not_null()).with_columns(
            (pl.col("W") != pl.col("S_ns")).alias("inv"))
        print(f"round {r}: rows with S_ns={s.height}; inverted (W != S_ns)={s.filter(pl.col('inv')).height}")
        print(s.group_by("day").agg(pl.len().alias("n"), pl.col("inv").sum().alias("inv")).sort("day"))
    r13 = P2.filter((pl.col("round") == 13) & pl.col("S_ns").is_not_null()).with_columns(
        (pl.col("W") != pl.col("S_ns")).alias("inv13"))
    r25 = P2.filter((pl.col("round") == 25) & pl.col("S_ns").is_not_null()).with_columns(
        (pl.col("W") != pl.col("S_ns")).alias("inv25"))
    pm = r13.select(["match_key", "day", "inv13"]).join(r25.select(["match_key", "inv25"]), on="match_key", how="left")
    print("per-match pattern (r13 inverted, r25 inverted, rows) for matches with r13 and r25 present:")
    print(pm.filter(pl.col("inv25").is_not_null()).group_by(["inv13", "inv25"]).len().sort(["inv13", "inv25"]))
    print("per-match r13 inversion, matches (any day >=12):",
          pm.filter(pl.col("day").cast(pl.Int32) >= 12).group_by("inv13").len().sort("inv13").rows())
    # is the inversion related to the bomb outcome presence?
    bb = P2.filter((pl.col("round") == 13) & pl.col("S_ns").is_not_null())
    print("r13 inverted by bomb-side availability:",
          bb.with_columns((pl.col("W") != pl.col("S_ns")).alias("inv"), pl.col("B").is_not_null().alias("hasB"))
          .group_by(["hasB", "inv"]).len().sort(["hasB", "inv"]).rows())

    print("\n== Q2 2024 round 1 coverage")
    RE = load(BASE24, "round_end", ["match_key", "round"]).unique(["match_key", "round"])
    have1 = set(RE.filter(pl.col("round") == 1)["match_key"].to_list())
    all_m = set(RE["match_key"].to_list())
    no1 = all_m - have1
    print("matches in round_end:", len(all_m), " with round 1 row:", len(have1), " without:", len(no1))
    RST = load(BASE24, "round_start", ["match_key", "round"]).unique(["match_key", "round"])
    rs1 = set(RST.filter(pl.col("round") == 1)["match_key"].to_list())
    print("of those without round_end round 1: with round_start round 1:", len(no1 & rs1))
    RS = load(BASE24, "round_state", ["match_key", "round", "event_type"]).filter(
        pl.col("round") == 1).filter(pl.col("event_type").is_in(["round_freeze_end", "round_end"]))
    fe1 = set(RS.filter(pl.col("event_type") == "round_freeze_end")["match_key"].to_list())
    re1s = set(RS.filter(pl.col("event_type") == "round_end")["match_key"].to_list())
    print("of those without round_end round 1: with round_state freeze_end r1:", len(no1 & fe1),
          " with round_state round_end r1:", len(no1 & re1s))
    print("of matches with round_end round 1: with round_state round_end r1:", len(have1 & re1s))
    # how do round_end rows of matches without round 1 start (min round) vs header max rounds
    print("in matches without round 1, round_end rounds are 2..N (checked in A6_label_audit P1).")

    print("\n== Q3 2026 rounds with two +1 score_update rows")
    su = load(BASE26, "score_update", ["match_key", "round", "tick", "team_code", "old_score", "new_score"])
    up = su.filter(pl.col("new_score") - pl.col("old_score") == 1)
    multi = up.group_by(["match_key", "round"]).agg(pl.col("team_code").n_unique().alias("nt"),
                                                    pl.len().alias("n")).filter(pl.col("nt") > 1)
    RE26 = load(BASE26, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"])
    mj = multi.join(RE26, on=["match_key", "round"], how="left")
    pair = up.join(multi.select(["match_key", "round"]), on=["match_key", "round"], how="inner").group_by(
        ["match_key", "round"]).agg(pl.col("tick").min().alias("t_first"), pl.col("tick").max().alias("t_second"),
                                    pl.col("team_code").sort().alias("teams"))
    mj = mj.join(pair, on=["match_key", "round"], how="left")
    print("multi rounds:", mj.height)
    print("round_end tick - t_first quantiles:", mj.select((pl.col("tick") - pl.col("t_first")).quantile(0.05).alias("p05"),
                                                             (pl.col("tick") - pl.col("t_first")).median().alias("p50"),
                                                             (pl.col("tick") - pl.col("t_first")).quantile(0.95).alias("p95")).rows())
    print("round_end tick - t_second quantiles:", mj.select((pl.col("tick") - pl.col("t_second")).quantile(0.05).alias("p05"),
                                                              (pl.col("tick") - pl.col("t_second")).median().alias("p50"),
                                                              (pl.col("tick") - pl.col("t_second")).quantile(0.95).alias("p95")).rows())
    print("multi rounds by round number:", mj.group_by("round").len().sort("round").rows())
    print("multi rounds by winner and code:", mj.group_by(["winner_team_code", "win_reason_code"]).len().sort(
        ["winner_team_code", "win_reason_code"]).rows())
    print("multi rounds: teams sorted sample:", mj.group_by("teams").len().rows())


if __name__ == "__main__":
    main()
