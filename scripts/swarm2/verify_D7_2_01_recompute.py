"""verify D7-2 (01): recompute the claimed first-after-swap set (rounds 9/13, 3,287 rows) and the best single-threshold error
counts on three axes: header match_date (UTC), csds path day, header build_num.
Also reproduces the claim's own code path (which allows splits inside tied groups) and a realizable version
(splits only at distinct-value boundaries).
ASSUMPTION: w_ok (ledger side == winner_team_code) is taken from the D7 parquet outputs; spot-checked against raw round_end here.
ASSUMPTION: csds path day = YYYY/MM/DD segment of header key csds/YYYY/MM/DD/<match_id>.
ASSUMPTION: first-after-swap round r in {9,13} <=> ledger swap flag at round r-1 (as in the claim).
"""
import sys, glob
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
LOG = ROOT / "logs" / "swarm2"

H = (pl.read_parquet(str(BASE / "header" / "dataframe_00000"),
                     columns=["key", "match_date", "build_num", "map_name", "server_name"])
     .rename({"key": "match_key"})
     .with_columns(
         pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=True).alias("mdt_h"),
         pl.col("match_key").str.extract(r"^csds/(\d{4}/\d{2}/\d{2})/", 1).alias("path_ymd"),
     ))
print("header rows:", H.height, "| null mdt_h:", H["mdt_h"].null_count(), "| null path_ymd:", H["path_ymd"].null_count())

X = pl.read_parquet(str(LOG / "D7_rounds_with_ledger_dp.parquet"))
L = pl.read_parquet(str(LOG / "D7_ledger_dp.parquet"))
print("X rows:", X.height, "| L rows:", L.height)

# consistency of the claim's derived columns with the raw header
XH = X.select(["match_key", "round", "mdt", "key_day"]).join(H.select(["match_key", "mdt_h", "path_ymd"]), on="match_key", how="left")
print("X.mdt != header mdt_h:", XH.filter(pl.col("mdt") != pl.col("mdt_h")).height,
      "| X.key_day != path_ymd:", XH.filter(pl.col("key_day") != pl.col("path_ymd")).height,
      "| rows with no header match:", XH.filter(pl.col("mdt_h").is_null()).height)

fas = (L.filter(pl.col("swap") == True)
         .with_columns((pl.col("round") + 1).alias("round"))
         .select(["match_key", "round"]).unique()
         .with_columns(pl.lit(True).alias("fas")))
Y = (X.join(fas, on=["match_key", "round"], how="left")
       .with_columns(pl.col("fas").fill_null(False))
       .filter(pl.col("w_ok").is_not_null()))
S = (Y.filter(pl.col("fas") & pl.col("round").is_in([9, 13]))
       .join(H.select(["match_key", "build_num", "server_name"]), on="match_key", how="left")
       .with_columns(
           pl.col("mdt").alias("t_match"),
           pl.col("key_day").str.replace_all("/", "-").str.to_date("%Y-%m-%d").cast(pl.Datetime("us")).alias("t_path"),
           (~pl.col("w_ok")).alias("flip"),
       ))
print("\nS rows (fas rounds 9/13, w_ok not null):", S.height, "| distinct matches:", S["match_key"].n_unique())
print("rows by round:", S.group_by("round").len().sort("round").rows())
print("rows by build:", S.group_by("build_num").len().sort("build_num").rows())
print("w_ok by build (True=winner agrees with ledger):", S.group_by(["build_num", "w_ok"]).len().sort(["build_num", "w_ok"]).rows())
print("distinct values: match_date", S["t_match"].n_unique(), "| path day", S["t_path"].n_unique(), "| build", S["build_num"].n_unique())

# raw spot check of the winner code (round_end) and of w_ok definition
RE = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "winner_team_code"]) for f in sorted(glob.glob(str(BASE / f"basic_round_end{SUF}" / "dataframe_*")))]).collect()
chk = S.join(RE, on=["match_key", "round"], how="left", suffix="_raw")
print("\nS rows missing raw round_end:", chk.filter(pl.col("winner_team_code_raw").is_null()).height)
print("S winner_team_code != raw round_end winner:", chk.filter(pl.col("winner_team_code") != pl.col("winner_team_code_raw")).height)
print("S w_ok != (ledger_side == winner_team_code):", S.filter(pl.col("w_ok") != (pl.col("ledger_side") == pl.col("winner_team_code"))).height)
print("S ledger_side null:", S.filter(pl.col("ledger_side").is_null()).height)

# save the recomputed set for the alternatives script
S.select(["match_key", "round", "t_match", "t_path", "build_num", "server_name", "map_name", "w_ok", "flip", "winner_team_code", "ledger_side", "bomb_side"]) \
 .write_parquet(str(Path("C:/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/e0a18ad2-aa9e-4a3c-9946-29641b2cde82/scratchpad") / "verify_D7_2_S.parquet"))

def claim_style(vals, flips):
    """Reproduces the claim's code: sort, split index c anywhere (incl. inside tie groups), predict ok before c, flip after c."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    f = [1 if flips[i] else 0 for i in order]
    n = len(f)
    Fb = [0] * (n + 1)
    for i in range(n):
        Fb[i + 1] = Fb[i] + f[i]
    errs = [Fb[c] + ((n - c) - (Fb[n] - Fb[c])) for c in range(n + 1)]
    # errs[c] = flips before c (predicted ok) + ok after c (predicted flip)
    c = min(range(n + 1), key=lambda i: errs[i])
    return errs[c], (vals[order[c - 1]] if c > 0 else None), (vals[order[c]] if c < n else None)

def realizable(vals, flips):
    """Single threshold on a sorted value; split only where the value changes, both orientations."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    v = [vals[i] for i in order]
    f = [1 if flips[i] else 0 for i in order]
    n = len(f)
    Fb = [0] * (n + 1)
    for i in range(n):
        Fb[i + 1] = Fb[i] + f[i]
    Ft = Fb[n]
    res = {}
    best = {"flip_after": None, "flip_before": None}
    for c in range(n + 1):
        if not (c == 0 or c == n or v[c - 1] < v[c]):
            continue  # not a realizable split: splits a tied group
        a_err = Fb[c] + ((n - c) - (Ft - Fb[c]))     # ok before, flip after
        b_err = (c - Fb[c]) + (Ft - Fb[c])           # flip before, ok after
        lo = v[c - 1] if c > 0 else None
        hi = v[c] if c < n else None
        if best["flip_after"] is None or a_err < best["flip_after"][0]:
            best["flip_after"] = (a_err, lo, hi)
        if best["flip_before"] is None or b_err < best["flip_before"][0]:
            best["flip_before"] = (b_err, lo, hi)
    return best

def report(name, df):
    vals = df[name].to_list() if name != "build_num" else df["build_num"].to_list()
    flips = df["flip"].to_list()
    n = len(flips)
    ce, lo, hi = claim_style(vals, flips)
    rb = realizable(vals, flips)
    print(f"\naxis={name:<10} n={n} | claim-style (may split ties) errors={ce} last-before={lo} first-after={hi}")
    print(f"   realizable flip-after : errors={rb['flip_after'][0]} last-before={rb['flip_after'][1]} first-after={rb['flip_after'][2]}")
    print(f"   realizable flip-before: errors={rb['flip_before'][0]} last-before={rb['flip_before'][1]} first-after={rb['flip_before'][2]}")

print("\n==== single-threshold error counts on S (flip = winner disagrees with ledger) ====")
print("flip rows:", S.filter(pl.col("flip")).height, "| ok rows:", S.filter(~pl.col("flip")).height)
report("t_match", S)
report("t_path", S)
report("build_num", S)
