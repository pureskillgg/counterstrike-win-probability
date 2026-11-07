"""Test side-definition variants for the rifle-vs-SMG/pistol round win rate (report n=266, 81.6%),
and whether 'other heavy' in the report includes G3SG1 (11) and SCAR-20 (38)."""
import polars as pl

HZ = r"G:\pureskill\csds-cs2\data\state_sample\players_1hz.parquet"
RD = r"G:\pureskill\csds-cs2\data\state_sample\rounds.parquet"

RIFLE = {7, 8, 10, 13, 16, 39, 60}
SNIPER = {9, 40}
SMG = {17, 19, 23, 24, 26, 33, 34}
HEAVY = {14, 25, 27, 28, 29, 35}
AUTO = {11, 38}

hz = (
    pl.scan_parquet(HZ)
    .select(["match_key", "round", "tick", "team_code", "player_id", "inv_primary", "equipment_value_calc"])
    .with_columns(pl.col("tick").min().over(["match_key", "round"]).alias("t0"))
    .filter(pl.col("tick") == pl.col("t0"))
    .drop("t0")
    .collect()
)

side = hz.group_by(["match_key", "round", "team_code"]).agg(
    pl.col("inv_primary").is_in(list(RIFLE)).any().alias("rifle"),
    pl.col("inv_primary").is_in(list(SNIPER)).any().alias("sniper"),
    pl.col("inv_primary").is_in(list(SMG)).any().alias("smg"),
    pl.col("inv_primary").is_in(list(HEAVY)).any().alias("heavy"),
    pl.col("inv_primary").is_in(list(AUTO)).any().alias("auto"),
    pl.col("inv_primary").is_in(list(RIFLE | SNIPER | HEAVY)).any().alias("big"),
    (pl.col("inv_primary") == 0).all().alias("allpistol"),
    pl.col("equipment_value_calc").sum().alias("eq"),
)
ct = side.filter(pl.col("team_code") == 3).drop("team_code").rename({c: f"ct_{c}" for c in ["rifle", "sniper", "smg", "heavy", "auto", "big", "allpistol", "eq"]})
tt = side.filter(pl.col("team_code") == 2).drop("team_code").rename({c: f"t_{c}" for c in ["rifle", "sniper", "smg", "heavy", "auto", "big", "allpistol", "eq"]})
rd = pl.read_parquet(RD).select(["match_key", "round", "ct_win"])
rr = rd.join(ct, on=["match_key", "round"]).join(tt, on=["match_key", "round"])
np_ = rr.filter(~pl.col("round").is_in([1, 13]))
print("non-pistol rounds:", np_.height)

# holdings with auto snipers folded into heavy
for label, prefix in [("CT", "ct"), ("T", "t")]:
    a = np_[f"{prefix}_heavy"].cast(pl.Float64).mean()
    b = (np_[f"{prefix}_heavy"] | np_[f"{prefix}_auto"]).cast(pl.Float64).mean()
    print(f"{label} other-heavy: base={100*a:.1f}%  with G3SG1/SCAR-20={100*b:.1f}%  (report CT 18.6, T 8.4, both 13.5)")
both_h = (np_["ct_heavy"] | np_["t_heavy"]).cast(pl.Float64).mean()
both_hw = (np_["ct_heavy"] | np_["t_heavy"] | np_["ct_auto"] | np_["t_auto"]).cast(pl.Float64).mean()
print(f"both sides heavy-any (per round) base={100*both_h:.1f}% with autos={100*both_hw:.1f}%")

def run(name, rifle_side_expr, other_side_expr):
    """rifle_side_expr(prefix) and other_side_expr(prefix) give boolean per side."""
    ct_is_rifle = rifle_side_expr("ct") & other_side_expr("t")
    t_is_rifle = rifle_side_expr("t") & other_side_expr("ct")
    sel = np_.filter(ct_is_rifle | t_is_rifle)
    n = sel.height
    # rifle side wins: CT rifle & ct_win==1, or T rifle & ct_win==0
    ct_rifle_mask = sel.select(ct_is_rifle.alias("m"))["m"]
    w = (ct_rifle_mask & (sel["ct_win"] == 1)) | (~ct_rifle_mask & (sel["ct_win"] == 0))
    print(f"{name}: n={n}, rifle-side wins={int(w.sum())}, rate={100*w.mean():.1f}%, "
          f"CT-rifle n={int(ct_rifle_mask.sum())}, T-rifle n={int((~ct_rifle_mask).sum())}")


run("V1 exactly one side has rifle (other side no rifle)",
    lambda p: pl.col(f"{p}_rifle"),
    lambda p: ~pl.col(f"{p}_rifle"))
run("V2 other side has no rifle and no sniper",
    lambda p: pl.col(f"{p}_rifle"),
    lambda p: ~pl.col(f"{p}_rifle") & ~pl.col(f"{p}_sniper"))
run("V3 other side has no rifle/sniper/heavy (SMG-or-pistol only)",
    lambda p: pl.col(f"{p}_rifle"),
    lambda p: ~pl.col(f"{p}_rifle") & ~pl.col(f"{p}_sniper") & ~pl.col(f"{p}_heavy"))
run("V4 other side has no rifle/sniper/heavy/auto",
    lambda p: pl.col(f"{p}_rifle"),
    lambda p: ~pl.col(f"{p}_rifle") & ~pl.col(f"{p}_sniper") & ~pl.col(f"{p}_heavy") & ~pl.col(f"{p}_auto"))
run("V5 other side all pistol-only",
    lambda p: pl.col(f"{p}_rifle"),
    lambda p: pl.col(f"{p}_allpistol"))
run("V6 rifle side has no sniper and no heavy AND other side has no rifle",
    lambda p: pl.col(f"{p}_rifle") & ~pl.col(f"{p}_sniper") & ~pl.col(f"{p}_heavy"),
    lambda p: ~pl.col(f"{p}_rifle"))
