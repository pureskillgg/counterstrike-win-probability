"""Independent check of claim D1-9 (verifier run).

Recomputes, from the parquet tomes only (read-only, column-projected, one channel at a time):
  - round_end.player_count distribution in the 2024-02 and 2026-10 tomes
  - player_info duplicate (match, round, player) keys, in both tomes
  - round_end duplicate (match, round) rows
  - whether round_end.player_count equals the number of distinct players in player_info
Claimed values are printed next to recomputed values at the end.
"""
import glob
import sys
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(70)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
}
# scripts/model/feature_validation.py lines 38-39
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


def pct(n, d):
    return f"{n:,} of {d:,} ({100.0 * n / d:.2f}%)"


CLAIMED = {}

print("== 0. Folders and columns ==")
for tag in TOMES:
    print(tag, "folder:", folder(tag, "round_end"))
    print(tag, "header cols:", cols_of(tag, "header"))
    print(tag, "round_end cols:", cols_of(tag, "round_end"))
    print(tag, "player_info cols:", cols_of(tag, "player_info"))

print("\n== 1. Header match counts ==")
for tag in TOMES:
    hc = cols_of(tag, "header")
    h = read(tag, "header", ["key"])
    print(f"{tag}: header rows {h.height:,}; distinct key {h['key'].n_unique():,}")

RE = {}
print("\n== 2. round_end player_count ==")
for tag in TOMES:
    re_ = read(tag, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
    RE[tag] = re_
    n = re_.height
    nu = re_.unique(["match_key", "round"]).height
    print(f"\n[{tag}] rows {n:,}; distinct (match,round) {nu:,}; duplicate rows {n - nu}; "
          f"matches {re_['match_key'].n_unique():,}")
    print(re_.group_by("player_count").len().sort("len", descending=True))
    n20 = re_.filter(pl.col("player_count") == 20).height
    n10 = re_.filter(pl.col("player_count") == 10).height
    neven = re_.filter(pl.col("player_count") % 2 == 0).height
    print(f"[{tag}] player_count==20: {pct(n20, n)}")
    print(f"[{tag}] player_count==10: {pct(n10, n)}")
    print(f"[{tag}] even player_count (all values): {pct(neven, n)}")
    per_m = re_.group_by("match_key").agg(
        pl.col("player_count").n_unique().alias("nu"),
        pl.col("player_count").mode().first().alias("mode_pc"),
        pl.len().alias("n_rounds"))
    print(f"[{tag}] matches by number of distinct player_count values: "
          f"{per_m.group_by('nu').len().sort('nu').rows()}")
    print(f"[{tag}] matches by modal player_count (top 8): "
          f"{per_m.group_by('mode_pc').len().sort('len', descending=True).head(8).rows()}")
    bucket = re_.with_columns(
        pl.when(pl.col("round") <= 12).then(pl.lit("r01-12"))
        .when(pl.col("round") <= 24).then(pl.lit("r13-24"))
        .otherwise(pl.lit("r25+")).alias("bucket"))
    print(f"[{tag}] by round bucket:")
    print(bucket.group_by("bucket").agg(
        pl.len().alias("rounds"),
        (pl.col("player_count") == 20).mean().alias("share_pc20"),
        (pl.col("player_count") == 10).mean().alias("share_pc10")).sort("bucket"))
    day = re_.with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day"))
    dsum = day.group_by("day").agg(
        pl.len().alias("rounds"),
        (pl.col("player_count") == 20).mean().alias("share_pc20"),
        (pl.col("player_count") == 10).mean().alias("share_pc10")).sort("day")
    print(f"[{tag}] by match day (DD):")
    print(dsum)

CLAIMED["2024 round_end rows"] = 83527
CLAIMED["2024 player_count==20 rounds"] = 78596
CLAIMED["2026 round_end rows"] = 35992
CLAIMED["2026 player_count==10 rounds"] = 32041

print("\n== 3. player_info: distinct players per round vs player_count ==")
PI = {}
PID = {}
for tag in TOMES:
    cols = cols_of(tag, "player_info")
    pid = "player_id_fixed" if "player_id_fixed" in cols else "player_id"
    PID[tag] = pid
    pi = read(tag, "player_info", ["match_key", "round", "team_code", pid]
              + (["player_id"] if pid != "player_id" and "player_id" in cols else []))
    PI[tag] = pi
    print(f"\n[{tag}] player_info rows {pi.height:,}; identity column used: {pid}")
    pr = pi.group_by(["match_key", "round"]).agg(
        pl.len().alias("rows"),
        pl.col(pid).n_unique().alias("nd"),
        pl.col(pid).filter(pl.col("team_code") == T).n_unique().alias("nT"),
        pl.col(pid).filter(pl.col("team_code") == CT).n_unique().alias("nCT"),
    )
    print(f"[{tag}] player_info (match,round) groups: {pr.height:,}; rows per group (top 10):")
    print(pr.group_by("rows").len().sort("len", descending=True).head(10))
    rej = RE[tag].unique(["match_key", "round"], keep="first").select(["match_key", "round", "player_count"])
    j = rej.join(pr, on=["match_key", "round"], how="left")
    nomatch = j.filter(pl.col("nd").is_null()).height
    print(f"[{tag}] round_end rounds with no player_info rows: {nomatch:,}")
    jj = j.filter(pl.col("nd").is_not_null())
    print(f"[{tag}] crosstab player_count x distinct players (top 15):")
    print(jj.group_by(["player_count", "nd"]).len().sort("len", descending=True).head(15))
    n = jj.height
    print(f"[{tag}] player_count == 2 * distinct players: {pct(jj.filter(pl.col('player_count') == 2 * pl.col('nd')).height, n)}")
    print(f"[{tag}] player_count == distinct players:     {pct(jj.filter(pl.col('player_count') == pl.col('nd')).height, n)}")
    print(f"[{tag}] distinct players == 10 (5 T and 5 CT):  "
          f"{pct(jj.filter((pl.col('nd') == 10) & (pl.col('nT') == 5) & (pl.col('nCT') == 5)).height, n)}")
    pc20 = jj.filter(pl.col("player_count") == 20)
    if pc20.height:
        print(f"[{tag}] among player_count==20 rounds, distinct players == 10 and 5/5: "
              f"{pct(pc20.filter((pl.col('nd') == 10) & (pl.col('nT') == 5) & (pl.col('nCT') == 5)).height, pc20.height)}")
    pc10 = jj.filter(pl.col("player_count") == 10)
    if pc10.height:
        print(f"[{tag}] among player_count==10 rounds, distinct players == 10 and 5/5: "
              f"{pct(pc10.filter((pl.col('nd') == 10) & (pl.col('nT') == 5) & (pl.col('nCT') == 5)).height, pc10.height)}")

print("\n== 4. Duplicate keys ==")
DUP_MATCHES = {}
for tag in TOMES:
    pi = PI[tag]
    pid = PID[tag]
    pid_names = [pid] + (["player_id"] if pid != "player_id" and "player_id" in pi.columns else [])
    for pn in pid_names:
        k = ["match_key", "round", pn]
        d = pi.filter(pl.len().over(k) > 1)
        keys = d.select(k).unique()
        conf = (d.group_by(k).agg(pl.col("team_code").n_unique().alias("nt"))
                .filter(pl.col("nt") > 1))
        print(f"\n[{tag}] player_info by ({pn}): dup rows {d.height:,}; dup keys {keys.height:,}; "
              f"matches with dup keys {d['match_key'].n_unique()}; keys with conflicting team_code "
              f"{conf.height}")
        print(f"[{tag}] dup rows per match: {d.group_by('match_key').len().sort('len', descending=True).rows()}")
        print(f"[{tag}] conflicting keys per round: {conf.group_by('round').len().sort('round').rows()}")
        if tag not in DUP_MATCHES:
            DUP_MATCHES[tag] = set(d["match_key"].unique().to_list())
    full_dup = pi.height - pi.unique().height
    print(f"[{tag}] exact full-row duplicate rows in player_info (all columns): {full_dup:,}")
    if tag == "2024":
        CLAIMED["2024 player_info dup keys"] = 104
        CLAIMED["2024 player_info dup matches"] = 3
        CLAIMED["2024 dup keys with conflicting team"] = 34

    rd = RE[tag].filter(pl.len().over(["match_key", "round"]) > 1)
    print(f"[{tag}] round_end duplicate rows: {rd.height}; matches: {rd['match_key'].unique().to_list()}")
    DUP_MATCHES[tag + "_re"] = set(rd["match_key"].unique().to_list())
    if tag == "2024":
        CLAIMED["2024 round_end dup rows"] = 1

union_2024 = DUP_MATCHES["2024"] | DUP_MATCHES["2024_re"]
print(f"\n[2024] matches with ANY duplicate (player_info or round_end): {len(union_2024)}")
for m in sorted(union_2024):
    src = []
    if m in DUP_MATCHES["2024"]:
        src.append("player_info")
    if m in DUP_MATCHES["2024_re"]:
        src.append("round_end")
    pc = RE["2024"].filter(pl.col("match_key") == m)["player_count"].unique().to_list()
    print(f"  {m} sources={src} distinct player_count in match={pc}")
print(f"[2024] claim's drop list has 3 matches; round_end-dup match not in list: "
      f"{sorted(DUP_MATCHES['2024_re'] - DUP_MATCHES['2024'])}")

print("\n== 5. Claimed vs recomputed ==")
rec = {
    "2024 round_end rows": RE["2024"].height,
    "2024 player_count==20 rounds": RE["2024"].filter(pl.col("player_count") == 20).height,
    "2026 round_end rows": RE["2026"].height,
    "2026 player_count==10 rounds": RE["2026"].filter(pl.col("player_count") == 10).height,
    "2024 player_info dup keys": None,
    "2024 player_info dup matches": None,
    "2024 dup keys with conflicting team": None,
    "2024 round_end dup rows": RE["2024"].height - RE["2024"].unique(["match_key", "round"]).height,
}
pi24 = PI["2024"]
pid24 = PID["2024"]
d24 = pi24.filter(pl.len().over(["match_key", "round", pid24]) > 1)
rec["2024 player_info dup keys"] = d24.select(["match_key", "round", pid24]).unique().height
rec["2024 player_info dup matches"] = d24["match_key"].n_unique()
rec["2024 dup keys with conflicting team"] = (
    d24.group_by(["match_key", "round", pid24]).agg(pl.col("team_code").n_unique().alias("nt"))
    .filter(pl.col("nt") > 1).height)
for k, v in CLAIMED.items():
    print(f"{k:<40} claimed {v:>8}   recomputed {rec[k]:>8}   {'MATCH' if v == rec[k] else 'DIFF'}")
