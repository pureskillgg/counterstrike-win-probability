"""Match-level follow-up for claim D1-9 (verifier run). Read-only.

Classifies each match by its round-level player counts from player_info (nd) and
player_spawn (ns), and compares with round_end.player_count (pc). Tests whether the 2024
pc = 20 rounds that do not show 10 players are a separate population (a match subset), and
whether the 2026 wingman matches behave as pc = nd.
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


def build_rounds(tag):
    re_ = read(tag, "round_end", ["match_key", "round", "player_count"]).unique(
        ["match_key", "round"], keep="first")
    pcols = cols_of(tag, "player_info")
    pid = "player_id_fixed" if "player_id_fixed" in pcols else "player_id"
    pi = read(tag, "player_info", ["match_key", "round", "team_code", pid])
    pr = pi.group_by(["match_key", "round"]).agg(
        pl.col(pid).n_unique().alias("nd"),
        pl.col(pid).filter(pl.col("team_code") == T).n_unique().alias("nT"),
        pl.col(pid).filter(pl.col("team_code") == CT).n_unique().alias("nCT"))
    scols = cols_of(tag, "player_spawn")
    spid = "player_id_fixed" if "player_id_fixed" in scols else "player_id"
    sp = read(tag, "player_spawn", ["match_key", "round", spid])
    sr = sp.group_by(["match_key", "round"]).agg(pl.col(spid).n_unique().alias("ns"))
    return (re_.join(pr, on=["match_key", "round"], how="left")
               .join(sr, on=["match_key", "round"], how="left"))


def match_table(j):
    return j.group_by("match_key").agg(
        pl.len().alias("rounds"),
        pl.col("player_count").mode().first().alias("pc_mode"),
        pl.col("player_count").n_unique().alias("pc_nuniq"),
        pl.col("nd").mode().first().alias("nd_mode"),
        (pl.col("nd") == 10).mean().alias("share_nd10"),
        (pl.col("nd") == 4).mean().alias("share_nd4"),
        pl.col("ns").mode().first().alias("ns_mode"),
        (pl.col("ns") == 10).mean().alias("share_ns10"),
    )


def pct(n, d):
    return f"{n:,} of {d:,} ({100.0 * n / d:.2f}%)"


MT = {}
ROUNDS = {}
for tag in TOMES:
    j = build_rounds(tag)
    ROUNDS[tag] = j
    mt = match_table(j)
    MT[tag] = mt
    print(f"\n== [{tag}] matches: {mt.height:,}; rounds: {j.height:,} ==")
    print(f"[{tag}] matches by share of rounds with nd==10 (bins):")
    print(mt.with_columns(pl.when(pl.col("share_nd10") == 0).then(pl.lit("0"))
                          .when(pl.col("share_nd10") < 0.5).then(pl.lit("(0,0.5)"))
                          .when(pl.col("share_nd10") < 1).then(pl.lit("[0.5,1)"))
                          .otherwise(pl.lit("1")).alias("bin")).group_by("bin").len().sort("bin"))
    print(f"[{tag}] matches with nd_mode x pc_mode (top 12):")
    print(mt.group_by(["nd_mode", "pc_mode"]).len().sort("len", descending=True).head(12))
    print(f"[{tag}] matches with pc_mode x ns_mode (top 12):")
    print(mt.group_by(["pc_mode", "ns_mode"]).len().sort("len", descending=True).head(12))
    print(f"[{tag}] matches where pc is constant across all rounds: "
          f"{mt.filter(pl.col('pc_nuniq') == 1).height:,} of {mt.height:,}")

# 2024: matches with no round showing 10 players in player_info
mt24 = MT["2024"]
j24 = ROUNDS["2024"]
zero10 = mt24.filter(pl.col("share_nd10") == 0)
print(f"\n== 2024: matches with NO round showing 10 distinct players in player_info: "
      f"{zero10.height:,} ==")
print("pc_mode distribution:", zero10.group_by("pc_mode").len().sort("len", descending=True).rows())
print("nd_mode distribution:", zero10.group_by("nd_mode").len().sort("len", descending=True).rows())
print("ns_mode distribution:", zero10.group_by("ns_mode").len().sort("len", descending=True).rows())
print("rounds in these matches:", int(zero10["rounds"].sum()))
print("pc==20 rounds in these matches:",
      j24.join(zero10.select("match_key"), on="match_key").filter(pl.col("player_count") == 20).height)
print("pc_nuniq==1 in these matches:", pct(zero10.filter(pl.col("pc_nuniq") == 1).height, zero10.height))

ex = zero10.sort("rounds", descending=True).head(4)["match_key"].to_list()
for m in ex:
    seq = (j24.filter(pl.col("match_key") == m).sort("round")
           .select(["round", "player_count", "nd", "nT", "nCT", "ns"]).head(8))
    print(f"\n[2024] example {m}")
    print(seq)

print("\n== 2024: round-level pc==20 and nd!=10, by whether the match has any nd==10 round ==")
flag = mt24.select(["match_key", pl.col("share_nd10")])
x = j24.join(flag, on="match_key").with_columns(
    pl.when(pl.col("share_nd10") == 0).then(pl.lit("match has no nd==10 round"))
    .otherwise(pl.lit("match has nd==10 rounds")).alias("grp"))
print(x.filter(pl.col("player_count") == 20).group_by(["grp", pl.col("nd") == 10]).len().sort("grp"))

# 2026 analogue: wingman matches
h26 = read("2026", "header", ["key", "is_wingman", "build_num"])
m26 = MT["2026"].join(h26, left_on="match_key", right_on="key", how="left")
print("\n== 2026: wingman flag vs nd_mode and pc_mode ==")
print(m26.group_by(["is_wingman", "nd_mode", "pc_mode"]).len().sort("len", descending=True).head(10))

h24 = read("2024", "header", ["key", "build_num"])
m24 = MT["2024"].join(h24, left_on="match_key", right_on="key", how="left")
print("\n== 2024: build_num x (share_nd10==0) and pc_mode ==")
print(m24.group_by(["build_num", "pc_mode"]).len().sort("len", descending=True).head(8))
print(m24.with_columns((pl.col("share_nd10") == 0).alias("no_nd10")).group_by(["build_num", "no_nd10"]).len().sort(["build_num", "no_nd10"]))

# 2024 ratio check on spawn: pc == 2 * ns
for tag in TOMES:
    j = ROUNDS[tag]
    w = j.filter(pl.col("ns").is_not_null())
    n = w.height
    print(f"\n[{tag}] rounds with spawn: {n:,}; pc == 2*ns: {pct(w.filter(pl.col('player_count') == 2 * pl.col('ns')).height, n)}; "
          f"pc == ns: {pct(w.filter(pl.col('player_count') == pl.col('ns')).height, n)}")
    w2 = j.filter(pl.col("ns").is_not_null() & pl.col("nd").is_not_null())
    n2 = w2.height
    print(f"[{tag}] rounds with spawn and player_info: {n2:,}; "
          f"pc == 2*nd: {pct(w2.filter(pl.col('player_count') == 2 * pl.col('nd')).height, n2)}; "
          f"pc == nd: {pct(w2.filter(pl.col('player_count') == pl.col('nd')).height, n2)}; "
          f"ns == nd: {pct(w2.filter(pl.col('ns') == pl.col('nd')).height, n2)}")
