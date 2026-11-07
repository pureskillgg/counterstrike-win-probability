"""Discriminating checks for claim D1-9 (verifier run). Read-only.

Question: is the 2024 round_end.player_count = 20 a doubled count (2 x 10 players), or a
different population? Tests:
  A. Match-level: build_num and (2026) is_wingman vs modal player_count.
  B. Wingman analogue: 2026 wingman matches vs 2024 matches with 4 distinct players.
  C. Independent player count per round from basic_player_spawn (not player_info): if spawn
     shows 10 distinct players where player_info shows 4 or 9, player_info is incomplete and the
     round_end value 20 is 2 x 10, which supports the doubling reading.
  D. Clustering of the rounds where player_count is 20 but player_info shows fewer than 10 players.
"""
import glob
import sys
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_cols(12)
pl.Config.set_fmt_str_lengths(60)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
}
T, CT = 2, 3


def folder(tag, ch):
    base, suf = TOMES[tag]
    return base / ("header" if ch == "header" else f"basic_{ch}{suf}")


def files(tag, ch):
    fs = sorted(glob.glob(str(folder(tag, ch) / "dataframe_*")))
    if not fs:
        raise SystemExit(f"no dataframe files in {folder(tag, ch)}")
    return fs


def cols_of(tag, ch):
    return pq.read_schema(files(tag, ch)[0]).names


def read(tag, ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files(tag, ch)],
                     how="vertical_relaxed")


print("== A. Match-level: build_num, wingman flag, modal player_count ==")
MPC = {}
for tag in TOMES:
    re_ = read(tag, "round_end", ["match_key", "player_count"])
    MPC[tag] = re_.group_by("match_key").agg(
        pl.col("player_count").mode().first().alias("mode_pc"),
        pl.col("player_count").n_unique().alias("n_distinct_pc"))
    hcols = ["key", "build_num"] + (["is_wingman", "max_rounds"] if tag == "2026" else [])
    h = read(tag, "header", hcols)
    j = h.join(MPC[tag], left_on="key", right_on="match_key", how="left")
    print(f"\n[{tag}] header rows {h.height:,}; header keys with round_end: "
          f"{j.filter(pl.col('mode_pc').is_not_null()).height:,}")
    print(f"[{tag}] build_num -> matches, modal pc (top 10):")
    print(j.group_by(["build_num", "mode_pc"]).len().sort("len", descending=True).head(10))
    if tag == "2026":
        print("[2026] is_wingman x modal player_count:")
        print(j.group_by(["is_wingman", "mode_pc"]).len().sort("len", descending=True).head(10))
        print("[2026] max_rounds x modal player_count (top 8):")
        print(j.group_by(["max_rounds", "mode_pc"]).len().sort("len", descending=True).head(8))

print("\n== B/C/D. Round-level player counts from three sources ==")
for tag in TOMES:
    re_ = read(tag, "round_end", ["match_key", "round", "player_count"]).unique(
        ["match_key", "round"], keep="first")
    cols = cols_of(tag, "player_info")
    pid = "player_id_fixed" if "player_id_fixed" in cols else "player_id"
    pi = read(tag, "player_info", ["match_key", "round", "team_code", pid])
    pr = pi.group_by(["match_key", "round"]).agg(
        pl.col(pid).n_unique().alias("nd_pinfo"),
        pl.col(pid).filter(pl.col("team_code") == T).n_unique().alias("nT_pinfo"),
        pl.col(pid).filter(pl.col("team_code") == CT).n_unique().alias("nCT_pinfo"))
    sp_cols = cols_of(tag, "player_spawn")
    spid = "player_id_fixed" if "player_id_fixed" in sp_cols else "player_id"
    sp_team = "team_code" if "team_code" in sp_cols else ("player_team_code" if "player_team_code" in sp_cols else None)
    print(f"\n[{tag}] basic_player_spawn cols: {sp_cols}")
    spawn_read = [ "match_key", "round", spid] + ([sp_team] if sp_team else [])
    sp = read(tag, "player_spawn", spawn_read)
    if sp_team:
        sr = sp.group_by(["match_key", "round"]).agg(
            pl.col(spid).n_unique().alias("ns"),
            pl.col(spid).filter(pl.col(sp_team) == T).n_unique().alias("sT"),
            pl.col(spid).filter(pl.col(sp_team) == CT).n_unique().alias("sCT"))
    else:
        sr = sp.group_by(["match_key", "round"]).agg(pl.col(spid).n_unique().alias("ns"))
    print(f"[{tag}] spawn rows {sp.height:,}; rounds with spawn rows {sr.height:,}; "
          f"spawn distinct per round (top 8):")
    print(sr.group_by("ns").len().sort("len", descending=True).head(8))
    j = re_.join(pr, on=["match_key", "round"], how="left").join(sr, on=["match_key", "round"], how="left")
    print(f"[{tag}] rounds: {j.height:,}; with player_info: {j.filter(pl.col('nd_pinfo').is_not_null()).height:,}; "
          f"with spawn: {j.filter(pl.col('ns').is_not_null()).height:,}")
    print(f"[{tag}] crosstab player_count x spawn distinct (top 12):")
    print(j.group_by(["player_count", "ns"]).len().sort("len", descending=True).head(12))
    if tag == "2024":
        odd = j.filter((pl.col("player_count") == 20) & (pl.col("nd_pinfo") != 10))
        print(f"[2024] pc==20 rounds where player_info distinct != 10: {odd.height:,}")
        print(f"[2024] of those, spawn distinct == 10: "
              f"{odd.filter(pl.col('ns') == 10).height:,}; spawn distinct == 4: "
              f"{odd.filter(pl.col('ns') == 4).height:,}; spawn null: "
              f"{odd.filter(pl.col('ns').is_null()).height:,}")
        print("[2024] spawn distinct for pc==20 rounds (all), top 8:")
        print(j.filter(pl.col("player_count") == 20).group_by("ns").len().sort("len", descending=True).head(8))
        print("[2024] spawn distinct for pc==8 rounds, top 6:")
        print(j.filter(pl.col("player_count") == 8).group_by("ns").len().sort("len", descending=True).head(6))
        print("[2024] spawn distinct == 2 * player_count/2 check, pc==20 -> ns==10:",
              f"{j.filter((pl.col('player_count') == 20) & (pl.col('ns') == 10)).height:,} of "
              f"{j.filter((pl.col('player_count') == 20) & pl.col('ns').is_not_null()).height:,} with spawn")
        per_m = odd.group_by("match_key").len().sort("len", descending=True)
        print(f"[2024] pc==20 & pinfo!=10 rounds: matches {per_m.height}; "
              f"top matches by count (top 8):")
        print(per_m.head(8))
        mm = odd.group_by("match_key").len()
        big = mm.filter(pl.col("len") >= 10).height
        print(f"[2024] matches with >=10 such rounds: {big}")
        allround = j.filter(pl.col("nd_pinfo").is_not_null()).group_by("match_key").agg(
            pl.len().alias("n"), (pl.col("nd_pinfo") != 10).sum().alias("n_not10"))
        print(f"[2024] matches with >=1 pinfo!=10 round: "
              f"{allround.filter(pl.col('n_not10') > 0).height}; all rounds pinfo!=10: "
              f"{allround.filter((pl.col('n_not10') == pl.col('n')) & (pl.col('n') > 0)).height}")
    if tag == "2026":
        print("[2026] spawn distinct for pc==10 rounds, top 6:")
        print(j.filter(pl.col("player_count") == 10).group_by("ns").len().sort("len", descending=True).head(6))
        print("[2026] spawn distinct for pc==4 rounds, top 6:")
        print(j.filter(pl.col("player_count") == 4).group_by("ns").len().sort("len", descending=True).head(6))
