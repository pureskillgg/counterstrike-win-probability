"""Probe for claim D1-9 (verifier run). Read-only.

Are the 2024 matches with round_end.player_count 20 but only 1-4 spawned players (or 4 distinct
player_info players) wingman-like 2v2 matches? Compares rounds per match, team counts, and round
numbers with 2026 wingman matches. Also counts 2024 rounds that fit neither pc == 2*ns nor
pc == 2*nd.
"""
import glob
import sys
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_cols(14)
pl.Config.set_fmt_str_lengths(64)

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


def pct(n, d):
    return f"{n:,} of {d:,} ({100.0 * n / d:.2f}%)"


def rounds(tag):
    re_ = read(tag, "round_end", ["match_key", "round", "player_count"]).unique(
        ["match_key", "round"], keep="first")
    pcols = cols_of(tag, "player_info")
    pid = "player_id_fixed" if "player_id_fixed" in pcols else "player_id"
    pi = read(tag, "player_info", ["match_key", "round", "team_code", pid])
    pr = pi.group_by(["match_key", "round"]).agg(
        pl.col(pid).n_unique().alias("nd"),
        pl.col(pid).filter(pl.col("team_code") == T).n_unique().alias("nT"),
        pl.col(pid).filter(pl.col("team_code") == CT).n_unique().alias("nCT"))
    spid = "player_id_fixed" if "player_id_fixed" in cols_of(tag, "player_spawn") else "player_id"
    sp = read(tag, "player_spawn", ["match_key", "round", spid])
    sr = sp.group_by(["match_key", "round"]).agg(pl.col(spid).n_unique().alias("ns"))
    return (re_.join(pr, on=["match_key", "round"], how="left")
               .join(sr, on=["match_key", "round"], how="left"))


R24 = rounds("2024")
R26 = rounds("2026")

print("== 2024: rounds that fit neither pc == 2*ns nor pc == 2*nd (both known) ==")
k = R24.filter(pl.col("ns").is_not_null() & pl.col("nd").is_not_null())
neither = k.filter((pl.col("player_count") != 2 * pl.col("ns")) & (pl.col("player_count") != 2 * pl.col("nd")))
print(f"rounds with both ns and nd: {k.height:,}; fit neither: {pct(neither.height, k.height)}")
print("fit-neither by (pc, nd, ns) top 15:")
print(neither.group_by(["player_count", "nd", "ns"]).len().sort("len", descending=True).head(15))

print("\n== 2024: pc==20 rounds with spawn distinct in {1,2,3,4} (doubling contradicted by spawn) ==")
c = R24.filter((pl.col("player_count") == 20) & pl.col("ns").is_between(1, 4))
print(f"rounds: {c.height:,}")
print("nT x nCT (player_info) top 10:")
print(c.group_by(["nT", "nCT"]).len().sort("len", descending=True).head(10))
print("round numbers (top 10):")
print(c.group_by("round").len().sort("len", descending=True).head(10))
mc = c.group_by("match_key").agg(pl.len().alias("n_bad"))
full = R24.group_by("match_key").agg(pl.len().alias("rounds"), pl.col("round").max().alias("max_round"))
mj = mc.join(full, on="match_key", how="left")
print(f"matches containing these rounds: {mj.height}")
print("max_round per match, these matches (top 12 counts):")
print(mj.group_by("max_round").len().sort("len", descending=True).head(12))
print("rounds per match, these matches (top 12 counts):")
print(mj.group_by("rounds").len().sort("len", descending=True).head(12))

print("\n== 2024 all matches: rounds per match and max round (for comparison) ==")
print("rounds per match (top 12 counts):")
print(full.group_by("rounds").len().sort("len", descending=True).head(12))

print("\n== 2026: wingman matches vs rounds per match (header is_wingman) ==")
h26 = read("2026", "header", ["key", "is_wingman", "max_rounds"])
per26 = R26.group_by("match_key").agg(pl.len().alias("rounds"), pl.col("round").max().alias("max_round"),
                                      pl.col("nd").mode().first().alias("nd_mode"))
j26 = per26.join(h26, left_on="match_key", right_on="key", how="left")
print(j26.group_by(["is_wingman", "max_rounds"]).agg(pl.len().alias("matches"),
                                                     pl.col("rounds").max().alias("max_rounds_seen"),
                                                     pl.col("max_round").max().alias("max_round_seen")))
print("2026 rounds per match for wingman matches (top 8 counts):")
print(j26.filter(pl.col("is_wingman")).group_by("rounds").len().sort("len", descending=True).head(8))

print("\n== 2024 matches with pc_mode 20 and nd_mode 4 or ns_mode 4: rounds per match (top 12) ==")
mm = R24.group_by("match_key").agg(pl.len().alias("rounds"),
                                   pl.col("player_count").mode().first().alias("pc_mode"),
                                   pl.col("nd").mode().first().alias("nd_mode"),
                                   pl.col("ns").mode().first().alias("ns_mode"))
sel = mm.filter((pl.col("pc_mode") == 20) & ((pl.col("nd_mode") == 4) | (pl.col("ns_mode") == 4)))
print(f"matches: {sel.height}")
print(sel.group_by("rounds").len().sort("len", descending=True).head(12))
