"""A1 exploration: freeze-end snapshot structure, armor/inventory value counts,
equipment-vs-armor check, and weapon code mapping from weapon_fire. Read-only; prints summaries.
"""
import glob
import os
import time

import numpy as np
import polars as pl

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


SAMPLE = r"G:\pureskill\csds-cs2\data\state_sample\players_1hz.parquet"
ROUNDS = r"G:\pureskill\csds-cs2\data\state_sample\rounds.parquet"
TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(220)

COLS = ["match_key", "round", "tick", "seconds_since_live", "split", "player_id", "team_code",
        "armor", "has_helmet", "has_defuser", "money", "equipment_value_calc", "inv_primary",
        "inv_secondary", "inv_flashbang", "inv_smokegrenade", "inv_hegrenade", "inv_molotov",
        "inv_incgrenade", "inv_decoy", "inv_c4", "weapon_code", "health"]

lf = pl.scan_parquet(SAMPLE).select(COLS)
log("start")

# ---- coverage of snapshots per round
cov = (lf.group_by(["match_key", "round"])
       .agg(pl.len().alias("rows"), pl.col("tick").n_unique().alias("n_ticks"),
            pl.col("tick").min().alias("t0"), pl.col("seconds_since_live").min().alias("ssl_min"),
            pl.col("seconds_since_live").max().alias("ssl_max"))
       .collect())
log("round-level coverage rows", cov.height)
print("n_ticks distribution:")
print(cov["n_ticks"].value_counts().sort("n_ticks"))
print("ssl_min of first (per round) distribution:")
print(cov["ssl_min"].value_counts().sort("ssl_min").head(20))
print("ssl_max distribution (top 10):")
print(cov["ssl_max"].value_counts().sort("count", descending=True).head(10))
print("rounds per match:")
rpm = cov.group_by("match_key").agg(pl.len().alias("n"))
print(rpm["n"].describe())
print("round numbers present (count):")
print(cov.group_by("round").agg(pl.len().alias("n")).sort("round"))

# ---- first snapshot per round (min tick)
fs = (lf.with_columns(pl.col("tick").min().over(["match_key", "round"]).alias("t0"))
      .filter(pl.col("tick") == pl.col("t0"))
      .collect())
log("first-snapshot rows", fs.height)
ppr = fs.group_by(["match_key", "round"]).agg(pl.len().alias("n_players"))
print("players per first snapshot:")
print(ppr["n_players"].value_counts().sort("n_players"))
print("team_code counts (first snapshot):")
print(fs["team_code"].value_counts())
print("split counts (first snapshot, distinct matches):")
print(fs.group_by("split").agg(pl.col("match_key").n_unique().alias("matches")))

# ---- armor / helmet / defuser value counts
print("armor value counts (top 25):")
print(fs["armor"].value_counts().sort("count", descending=True).head(25))
print("armor summary:", fs.select(pl.col("armor").min().alias("min"), pl.col("armor").max().alias("max"),
                                   pl.col("armor").null_count().alias("nulls")))
print("has_helmet x armor>0:")
print(fs.with_columns((pl.col("armor") > 0).alias("arm_pos")).group_by(["has_helmet", "arm_pos"]).agg(pl.len()))
print("has_defuser by team:")
print(fs.group_by(["team_code", "has_defuser"]).agg(pl.len()).sort(["team_code", "has_defuser"]))
print("health summary:", fs.select(pl.col("health").min().alias("min"), pl.col("health").max().alias("max"),
                                   pl.col("health").null_count().alias("nulls")))
print("null counts first snapshot:")
print(fs.null_count())

# ---- inventory codes
print("inv_primary value counts (first snapshot):")
print(fs["inv_primary"].value_counts().sort("count", descending=True))
print("inv_secondary value counts (first snapshot):")
print(fs["inv_secondary"].value_counts().sort("count", descending=True))
print("weapon_code value counts (first snapshot, top 30):")
print(fs["weapon_code"].value_counts().sort("count", descending=True).head(30))
for c in ["inv_flashbang", "inv_smokegrenade", "inv_hegrenade", "inv_molotov", "inv_incgrenade", "inv_decoy", "inv_c4"]:
    print(c, "value counts:", fs[c].value_counts().sort("count", descending=True).head(6).rows())

# ---- equipment vs armor check (does equipment_value_calc include kevlar/helmet/defuser cost?)
d = fs.with_columns(
    ((pl.col("armor") > 0).cast(pl.Float64)).alias("arm_pos"),
    (pl.col("armor") >= 100).cast(pl.Float64).alias("arm100"),
    pl.col("has_helmet").cast(pl.Float64).alias("helm"),
    pl.col("has_defuser").cast(pl.Float64).alias("defu"),
    (pl.col("team_code") == 3).cast(pl.Float64).alias("is_ct"),
)
top_prim = d["inv_primary"].value_counts().sort("count", descending=True).head(12)["inv_primary"].to_list()
top_sec = d["inv_secondary"].value_counts().sort("count", descending=True).head(8)["inv_secondary"].to_list()
X_parts = [np.ones(d.height)]
names = ["const"]
for c in ["arm_pos", "arm100", "helm", "defu", "is_ct", "inv_flashbang", "inv_smokegrenade",
          "inv_hegrenade", "inv_molotov", "inv_incgrenade", "inv_decoy", "inv_c4"]:
    X_parts.append(d[c].cast(pl.Float64).fill_null(0).to_numpy())
    names.append(c)
for v in top_prim:
    X_parts.append((d["inv_primary"] == v).cast(pl.Float64).to_numpy())
    names.append(f"prim_{v}")
for v in top_sec:
    X_parts.append((d["inv_secondary"] == v).cast(pl.Float64).to_numpy())
    names.append(f"sec_{v}")
X = np.column_stack(X_parts)
y = d["equipment_value_calc"].cast(pl.Float64).to_numpy()
ok = ~np.isnan(y)
beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
resid = y[ok] - X[ok] @ beta
r2 = 1 - resid.var() / y[ok].var()
print("OLS equipment_value_calc ~ armor/helmet/defuser + inventory; R2=", round(r2, 4), "n=", int(ok.sum()))
for n_, b in zip(names[:12], beta[:12]):
    print(f"  {n_:18s} {b:10.2f}")

# also: money + equip relation check
print("money vs equipment_value_calc corr:", np.corrcoef(d["money"].cast(pl.Float64).to_numpy(),
                                                          d["equipment_value_calc"].cast(pl.Float64).to_numpy())[0, 1])

# ---- rounds join check
rd = pl.scan_parquet(ROUNDS).select(["match_key", "round", "ct_win", "split"]).collect()
rd = rd.with_columns(pl.col("round").cast(pl.Int32))
sample_keys = fs.select(["match_key", "round"]).unique()
j = sample_keys.join(rd.select(["match_key", "round", "ct_win"]), on=["match_key", "round"], how="left")
print("sample rounds:", sample_keys.height, "matched in rounds.parquet:", j.filter(pl.col("ct_win").is_not_null()).height)
print("ct_win counts in matched sample rounds:")
print(j["ct_win"].value_counts())

# ---- weapon_fire mapping (code -> name), codes used at freeze end
wf_folder = glob.glob(os.path.join(TOME, "basic_weapon_fire.*"))[0]
wf_files = sorted(glob.glob(os.path.join(wf_folder, "dataframe_*")))
wf = pl.scan_parquet(wf_files).select(["player_weapon_code", "weapon_name"])
pairs = wf.group_by(["player_weapon_code", "weapon_name"]).agg(pl.len().alias("rows")).collect()
log("weapon_fire pairs collected", pairs.height)
per_code = (pairs.sort(["player_weapon_code", "rows"], descending=[False, True])
            .group_by("player_weapon_code", maintain_order=True)
            .agg(pl.col("rows").sum().alias("code_rows"),
                 pl.col("weapon_name").first().alias("top_name"),
                 pl.col("rows").first().alias("top_rows"),
                 pl.col("weapon_name").n_unique().alias("n_names"),
                 pl.col("weapon_name").slice(1, 2).alias("other"))
            .with_columns((pl.col("top_rows") / pl.col("code_rows")).round(4).alias("top_share")))
used = sorted(set(top_prim + top_sec + fs["weapon_code"].unique().to_list()))
print("codes used in freeze-end data -> weapon_fire mapping:")
with pl.Config(tbl_rows=100, tbl_width_chars=260, fmt_str_lengths=40):
    print(per_code.filter(pl.col("player_weapon_code").is_in([int(u) for u in used if u is not None]))
          .sort("code_rows", descending=True))
print("codes with more than one name (count):", per_code.filter(pl.col("n_names") > 1).height)
print("total weapon_fire rows (pair sum):", int(pairs["rows"].sum()))
log("done")
