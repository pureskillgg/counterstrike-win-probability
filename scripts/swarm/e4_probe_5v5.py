"""E4 probe: why round_end.player_count == 10 is absent in the 2024 tome; compare with a player_info 5v5 rule.

Read-only, G: only. Run: cd G:\\pureskill\\csds-cs2 ; pixi run python -I -X utf8 scripts\\swarm\\e4_probe_5v5.py
"""
import collections
import glob
from pathlib import Path

import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
}


def read(tk, ch, cols=None):
    base, suf = TOMES[tk]
    folder = base / ("header" if ch == "header" else f"basic_{ch}{suf}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


for tk in ["2026", "2024"]:
    re_ = read(tk, "round_end", ["match_key", "round", "player_count", "winner_team_code"])
    pc = collections.Counter(re_["player_count"].to_list())
    print(f"\n== {tk} round_end.player_count distribution (rows {re_.height}): "
          + ", ".join(f"{k}={v}" for k, v in sorted(pc.items(), key=lambda kv: str(kv[0]))))
    pi = read(tk, "player_info", ["match_key", "round", "player_id", "team_code"])
    print(f"   player_info rows {pi.height}; team_code values "
          f"{dict(sorted(collections.Counter(pi['team_code'].to_list()).items(), key=lambda kv: str(kv[0])))}")
    uniq = pi.unique(subset=["match_key", "round", "player_id"], keep="first")
    print(f"   player_info unique (match, round, player) rows {uniq.height} (dupes {pi.height - uniq.height})")
    cnt = (uniq.group_by(["match_key", "round"]).agg([
        (pl.col("team_code") == 2).sum().alias("nT"),
        (pl.col("team_code") == 3).sum().alias("nCT"),
        pl.col("player_id").count().alias("n"),
    ]))
    rounds = re_.unique(subset=["match_key", "round"], keep="first")
    j = rounds.join(cnt, on=["match_key", "round"], how="left")
    j = j.with_columns(
        pl.col("nT").fill_null(0), pl.col("nCT").fill_null(0), pl.col("n").fill_null(0))
    five = j.filter((pl.col("nT") == 5) & (pl.col("nCT") == 5))
    print(f"   rounds {j.height}; rounds with player_info exactly 5 T and 5 CT: {five.height} "
          f"({five.height / j.height:.4f}); rounds with no player_info rows: "
          f"{j.filter(pl.col('n') == 0).height}")
    cross = collections.Counter(zip(j["player_count"].to_list(),
                                    [(a == 5 and b == 5) for a, b in zip(j["nT"].to_list(), j["nCT"].to_list())]))
    print(f"   cross-tab (player_count, is_5v5_by_player_info) -> rounds: "
          + ", ".join(f"({k[0]},{k[1]})={v}" for k, v in sorted(cross.items(), key=lambda kv: str(kv[0]))))
    dist_nT = collections.Counter(zip(j["nT"].to_list(), j["nCT"].to_list()))
    top = dist_nT.most_common(8)
    print("   most common (nT, nCT) pairs: " + ", ".join(f"({a},{b})={c}" for (a, b), c in top))
