"""verify_D1_5 build: 2024 build_num x day x round disagreement (round_end vs freeze-end-derived side).
Question: is the disagreement a build (game version) property, and does round 9 follow the same build?
ASSUMPTIONS: A1 side codes, A5 swap rule (MR24, A8), derived winner from freeze(r), freeze(r+1)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)

T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF24 = ".2023-12-28,2024-02-25"
MR = 24


def swap_expected(r, mr=MR):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


def fl(ch):
    return sorted(glob.glob(str(T24 / f"basic_{ch}{SUF24}" / "dataframe_*")))


re24 = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "winner_team_code", "player_count"]) for f in fl("round_end")],
                 how="vertical_relaxed").unique(["match_key", "round"], keep="first")
rs24 = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "tick", "event_type", "t_score", "ct_score"]) for f in fl("round_state")],
                 how="vertical_relaxed")
fe_df = (rs24.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
         .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("t"), pl.col("ct_score").first().alias("c")))
fe = {(m, int(r)): (int(t), int(c)) for m, r, t, c in fe_df.select(["match_key", "round", "t", "c"]).iter_rows()}


def fe_winner(m, r):
    a = fe.get((m, r))
    b = fe.get((m, r + 1))
    if a is None or b is None:
        return None
    t_r, c_r = a
    t_n, c_n = b
    if swap_expected(r):
        dT, dC = c_n - t_r, t_n - c_r
    else:
        dT, dC = t_n - t_r, c_n - c_r
    if dT == 1 and dC == 0:
        return 2
    if dC == 1 and dT == 0:
        return 3
    return None


rows = []
for m, r, w, pc in re24.select(["match_key", "round", "winner_team_code", "player_count"]).iter_rows():
    fw = fe_winner(m, int(r))
    if fw is None:
        continue
    day = m[len("csds/2024/02/"):len("csds/2024/02/") + 2]
    rows.append((m, day, int(r), int(w), fw))
D = pl.DataFrame(rows, schema=["match_key", "day", "round", "w", "fw"], orient="row").with_columns(
    (pl.col("w") != pl.col("fw")).alias("dis"))
HD0 = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(str(T24 / "header" / "dataframe_*")))], how="vertical_relaxed")
HDm = HD0.select([pl.col("key").alias("match_key"), "build_num", "map_name", "match_date"]).unique("match_key")
D = D.join(HDm, on="match_key", how="left")

print("build_num by day (matches in header, count):")
print(HDm.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
      .group_by(["day", "build_num"]).len().sort(["day", "build_num"]).to_dicts())
print("\ndisagreement rate by build_num, round, for rounds 9, 13, 25, 28 and all other rounds (d01-25 pooled):")
D = D.with_columns(pl.when(pl.col("round").is_in([9, 13, 25, 28])).then(pl.col("round").cast(pl.Utf8)).otherwise(pl.lit("other")).alias("rg"))
print(D.group_by(["build_num", "rg"]).agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis"),
                                          pl.col("dis").mean().round(4).alias("rate")).sort(["rg", "build_num"]))
print("\nround 9 disagreements by build and day:")
print(D.filter(pl.col("round") == 9).group_by(["build_num", "day"]).agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis")).sort(["build_num", "day"]))
print("\nall disagreements (any round) by build, and by day group:")
print(D.group_by("build_num").agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis")))
print(D.with_columns(pl.col("day").is_in([f"{d:02d}" for d in range(12, 26)]).alias("d1225"))
      .group_by(["d1225", "build_num"]).agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis")).sort(["d1225", "build_num"]))
