r"""Verifier for D4-5, step G: direct swap-schedule test from player side codes (2024 and 2026 control).

A player's side code (player_team_code in player_death, the victim's side at death) is fixed within a round.
If a side swap happens after round r, a player who dies in both round r and round r+1 shows the opposite side code.
For each round boundary (r -> r+1) in each match we count:
  pairs   = players with a death in both round r and round r+1 (side code known in both)
  flips   = pairs whose side code differs
A boundary with flips/pairs near 1 is a swap boundary; near 0 is no swap.
Compared with the assumed schedule (2024: swap after 12, then OT rule with mr=24; 2026: header max_rounds).

Scope: boundaries 1..30. Reports per era, day group and boundary, and the set of boundaries with flip rate > 0.5.
Reads projected columns only.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D4_5_g_swap_flips.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(16)
pl.Config.set_tbl_width_chars(220)
ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
         "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
T, CT = 2, 3


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def swap_expected(r, mr):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


for era, base in TOMES.items():
    print(f"\n######################## ERA {era} ########################")
    if era == "2026":
        H = read_dir(base / "header", ["key", "max_rounds"])
        mr_map = {k: int(v) for k, v in zip(H["key"].to_list(), H["max_rounds"].to_list())}
    else:
        mr_map = None
    fs_de = sorted(glob.glob(str(sorted(base.glob("basic_player_death.*"))[0] / "dataframe_*")))
    de = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "player_id", "player_team_code"]) for f in fs_de],
                   how="vertical_relaxed").filter(pl.col("player_team_code").is_in([T, CT]))
    # one side code per (match, round, player): mode (usually unique)
    pr = (de.group_by(["match_key", "round", "player_id"]).agg(pl.col("player_team_code").n_unique().alias("nside"),
                                                              pl.col("player_team_code").first().alias("side")))
    print(f"player-round side pairs: {pr.height}; with conflicting side in a round: {pr.filter(pl.col('nside') > 1).height}")
    a = pr.select(["match_key", "round", "player_id", "side"])
    b = a.with_columns((pl.col("round") - 1).alias("round")).rename({"side": "side_next"})
    # pairs (r, r+1): join on match, player, round=r (a has round r+1 after shifting b)
    pairs = a.join(b, on=["match_key", "round", "player_id"], how="inner")
    # pairs now: side = side in round r+1, side_next = side in round r+1 shifted? correct it below
    # b has round = (r_original - 1); joining a.round == b.round links a(round=x) with b(original round=x+1)
    # so 'side' is side in round x and 'side_next' is side in round x+1 (original round of b).
    pairs = pairs.with_columns(pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day")).with_columns(
        pl.when(pl.col("day") == "12").then(pl.lit("d12"))
        .when(pl.col("day").is_in(["01", "02", "03", "04", "05", "06", "07"])).then(pl.lit("d01-07"))
        .otherwise(pl.lit("d13-25")).alias("grp") if era == "2024" else pl.lit("all").alias("grp"))
    pairs = pairs.with_columns((pl.col("side") != pl.col("side_next")).alias("flip"))
    # match-level: fraction flipped per boundary, and the number of players per match-boundary
    mb = pairs.group_by(["grp", "round"]).agg(
        pl.len().alias("pairs"), pl.col("flip").sum().alias("flips"), (pl.col("flip").mean()).alias("flip_rate"),
        pl.col("match_key").n_unique().alias("matches")).sort(["grp", "round"])
    print("boundary r -> r+1 (keyed by round r); flip_rate = share of players whose side code changed:")
    print(mb.filter(pl.col("round") <= 30))
    # per-match swap detection: a match-boundary is a 'swap' if majority of its pairs flipped (>=2 pairs)
    mm = pairs.group_by(["match_key", "round"]).agg(pl.len().alias("n"), pl.col("flip").mean().alias("fr"))
    mm = mm.filter(pl.col("n") >= 2).with_columns((pl.col("fr") > 0.5).alias("swapped"))
    print("\nmatch-boundaries with >=2 players observed: count of swapped boundaries by round:")
    sm = mm.group_by("round").agg(pl.len().alias("boundaries"), pl.col("swapped").sum().alias("swapped")).sort("round")
    print(sm.filter(pl.col("round") <= 30))
    # per match: which boundaries are swaps (compare to assumed schedule)
    if mr_map is None:
        exp = {r: swap_expected(r, 24) for r in range(1, 31)}
    else:
        mrs = pl.DataFrame({"match_key": list(mr_map.keys()), "mr": list(mr_map.values())})
        mm2 = mm.join(mrs, on="match_key", how="left")
        mm2 = mm2.with_columns(pl.struct(["round", "mr"]).map_elements(
            lambda s: swap_expected(s["round"], s["mr"]), return_dtype=pl.Boolean).alias("exp_swap"))
        print("\n2026 agreement: observed swap vs expected (mr from header) on boundaries with >=2 pairs:")
        print(mm2.group_by(["swapped", "exp_swap"]).len().sort(["swapped", "exp_swap"]))
        continue
    mm = mm.with_columns(pl.col("round").map_elements(lambda r: exp.get(r, False), return_dtype=pl.Boolean).alias("exp_swap"))
    print("\n2024 agreement: observed swap vs expected (mr=24 assumption), boundaries with >=2 pairs:")
    print(mm.group_by(["swapped", "exp_swap"]).len().sort(["swapped", "exp_swap"]))
    print("\n2024 observed swap boundaries NOT in expected schedule (round, count):")
    print(mm.filter(pl.col("swapped") & ~pl.col("exp_swap")).group_by("round").len().sort("round"))
    print("\n2024 expected swap boundaries that are NOT observed (round, count):")
    print(mm.filter(~pl.col("swapped") & pl.col("exp_swap")).group_by("round").len().sort("round"))
    print("\n2024 matches with an observed swap at boundary 9, 13, 25 or 28 (count by day group):")
    spec = mm.filter(pl.col("round").is_in([9, 13, 25, 28]) & pl.col("swapped"))
    print(spec.with_columns(pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day")).group_by(["day", "round"]).len().sort(["day", "round"]))
