"""verify D7-2 (07): independent brute-force recomputation of the key realizable counts (no prefix sums), plus
the round-27 pre-switch flips in build 9842 and a full raw check of the winner codes used in the D7 parquet.
Single threshold rule, flip-after: predict 'flip' when value >= thr. Candidate thr = each distinct value (and +inf).
ASSUMPTION: flip = not w_ok (D7 ledger-vs-winner label).
"""
import sys, glob, datetime as dt
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(200)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
LOG = ROOT / "logs" / "swarm2"
S = pl.read_parquet(str(Path("C:/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/e0a18ad2-aa9e-4a3c-9946-29641b2cde82/scratchpad") / "verify_D7_2_S.parquet"))

def brute(vals, flips):
    best = None
    for thr in sorted(set(vals)) + [None]:
        if thr is None:
            pred = [False] * len(vals)
        else:
            pred = [v >= thr for v in vals]
        err = sum(1 for p, f in zip(pred, flips) if p != f)
        if best is None or err < best[0]:
            best = (err, thr)
    return best

def brute_by_day(vals, flips):
    best = None
    for thr in sorted(set(vals)) + [None]:
        pred = [False] * len(vals) if thr is None else [v >= thr for v in vals]
        err = sum(1 for p, f in zip(pred, flips) if p != f)
        if best is None or err < best[0]:
            best = (err, thr)
    return best

flips = S["flip"].to_list()
e_m, t_m = brute(S["t_match"].to_list(), flips)
e_p, t_p = brute_by_day(S["t_path"].to_list(), flips)
e_b, t_b = brute(S["build_num"].to_list(), flips)
print(f"brute-force S (n={S.height}): match_date errors={e_m} thr={t_m} | path day errors={e_p} thr={t_p} | build errors={e_b} thr={t_b}")
err_m_rows = [(r["match_key"][:50], r["round"], str(r["t_match"]), r["build_num"], "flip" if r["flip"] else "ok")
              for r in S.iter_rows(named=True) if (r["t_match"] >= t_m) != r["flip"]]
print("match_date threshold error rows:", err_m_rows)

RE = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "winner_team_code"]) for f in sorted(glob.glob(str(BASE / f"basic_round_end{SUF}" / "dataframe_*")))]).collect()
X = pl.read_parquet(str(LOG / "D7_rounds_with_ledger_dp.parquet"), columns=["match_key", "round", "winner_team_code", "w_ok", "ledger_side"])
cmp_ = X.join(RE, on=["match_key", "round"], how="left", suffix="_raw")
print("\nall D7 rows vs raw round_end: rows", cmp_.height, "| missing raw:", cmp_.filter(pl.col("winner_team_code_raw").is_null()).height,
      "| winner mismatch:", cmp_.filter(pl.col("winner_team_code") != pl.col("winner_team_code_raw")).height)

H = pl.read_parquet(str(BASE / "header" / "dataframe_00000"), columns=["key", "build_num"]).rename({"key": "match_key"})
L = pl.read_parquet(str(LOG / "D7_ledger_dp.parquet"))
sw = L.filter(pl.col("swap") == True).select(["match_key", "round"]).with_columns((pl.col("round") + 1).alias("round")).with_columns(pl.lit(True).alias("fas"))
F = X.join(sw, on=["match_key", "round"], how="left").with_columns(pl.col("fas").fill_null(False)).filter(pl.col("w_ok").is_not_null())
F = F.join(H, on="match_key", how="left")
pre_flips = F.filter((~pl.col("w_ok")) & (pl.col("build_num") == 9842))
print("\nbuild 9842 flips (any round): rounds", pre_flips.group_by("round").len().sort("round").rows())
print("build 9842 round-27 flip count:", pre_flips.filter(pl.col("round") == 27).height)
print("build 9842 flips by round 13:", pre_flips.filter(pl.col("round") == 13).height)
M = pl.read_parquet(str(BASE / "header" / "dataframe_00000"), columns=["key", "match_date"]).rename({"key": "match_key"}) \
    .with_columns(pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=True).alias("mdt_h"))
pf = pre_flips.join(M, on="match_key", how="left")
print("9842 round-27 flip match_date range:", pf["mdt_h"].min(), "to", pf["mdt_h"].max(), "| matches:", pf["match_key"].n_unique())
print("9842 round-27 flips in S (fas):", pre_flips.filter((pl.col("round") == 27) & pl.col("fas")).height)
