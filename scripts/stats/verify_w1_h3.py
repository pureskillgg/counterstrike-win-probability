"""H3 final: rifle-side vs SMG/pistol-only side win rate (non-pistol rounds, 1 Hz first snapshot).
Definition V3 (reproduces report counts 134/132, n=266): rifle side = team with >=1 rifle (AK/M4/AUG/FAMAS/Galil/SG553);
other side = no rifle, no sniper (AWP/SSG08), no other heavy (incl. G3SG1/SCAR-20 treated as heavy).
Match-cluster bootstrap, 1000 resamples, seed 0. Appends to logs\\stats\\verify_w1.log."""
import numpy as np
import polars as pl

HZ = r"G:\pureskill\csds-cs2\data\state_sample\players_1hz.parquet"
RD = r"G:\pureskill\csds-cs2\data\state_sample\rounds.parquet"
LOG = r"G:\pureskill\csds-cs2\logs\stats\verify_w1.log"
NB = 1000
SEED = 0
out = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)


RIFLE = {7, 8, 10, 13, 16, 39, 60}
SNIPER = {9, 40}
HEAVY = {14, 25, 27, 28, 29, 35, 11, 38}  # other heavy incl. G3SG1 (11) and SCAR-20 (38)

hz = (
    pl.scan_parquet(HZ)
    .select(["match_key", "round", "tick", "team_code", "inv_primary", "equipment_value_calc"])
    .with_columns(pl.col("tick").min().over(["match_key", "round"]).alias("t0"))
    .filter(pl.col("tick") == pl.col("t0"))
    .drop("t0")
    .collect()
)
side = hz.group_by(["match_key", "round", "team_code"]).agg(
    pl.col("inv_primary").is_in(list(RIFLE)).any().alias("rifle"),
    pl.col("inv_primary").is_in(list(SNIPER)).any().alias("sniper"),
    pl.col("inv_primary").is_in(list(HEAVY)).any().alias("heavy"),
    pl.col("equipment_value_calc").sum().alias("eq"),
)
ct = side.filter(pl.col("team_code") == 3).drop("team_code").rename(
    {"rifle": "ct_rifle", "sniper": "ct_sniper", "heavy": "ct_heavy", "eq": "ct_eq"})
tt = side.filter(pl.col("team_code") == 2).drop("team_code").rename(
    {"rifle": "t_rifle", "sniper": "t_sniper", "heavy": "t_heavy", "eq": "t_eq"})
rd = pl.read_parquet(RD).select(["match_key", "round", "ct_win"])
rr = rd.join(ct, on=["match_key", "round"]).join(tt, on=["match_key", "round"])
np_ = rr.filter(~pl.col("round").is_in([1, 13]))
log("== verify_w1_h3 (V3 definition)")
log("non-pistol rounds:", np_.height)

ct_rifle_side = np_["ct_rifle"] & ~np_["t_rifle"] & ~np_["t_sniper"] & ~np_["t_heavy"]
t_rifle_side = np_["t_rifle"] & ~np_["ct_rifle"] & ~np_["ct_sniper"] & ~np_["ct_heavy"]
mask = ct_rifle_side | t_rifle_side
h = np_.filter(mask).with_columns(
    pl.Series("rifle_is_ct", ct_rifle_side.filter(mask).to_list()),
)
h = h.with_columns(
    pl.when(pl.col("rifle_is_ct")).then(pl.col("ct_win") == 1).otherwise(pl.col("ct_win") == 0).alias("rifle_won"),
    pl.when(pl.col("rifle_is_ct")).then(pl.col("ct_eq")).otherwise(pl.col("t_eq")).alias("rifle_eq"),
    pl.when(pl.col("rifle_is_ct")).then(pl.col("t_eq")).otherwise(pl.col("ct_eq")).alias("other_eq"),
)
n = h.height
w = int(h["rifle_won"].sum())
log(f"H3 n={n} rounds, rifle-side wins={w}, rate={100*w/n:.2f}%  (report: n=266, 81.6%)")
log(f"H3 by side: rifle=CT n={int(h['rifle_is_ct'].sum())} won={int(h.filter(pl.col('rifle_is_ct'))['rifle_won'].sum())} "
    f"({100*h.filter(pl.col('rifle_is_ct'))['rifle_won'].mean():.1f}%); rifle=T n={int((~h['rifle_is_ct']).sum())} "
    f"won={int(h.filter(~pl.col('rifle_is_ct'))['rifle_won'].sum())} ({100*h.filter(~pl.col('rifle_is_ct'))['rifle_won'].mean():.1f}%)"
    f"  (report: CT 134 79.9%, T 132 83.3%)")

per = h.group_by("match_key").agg(pl.col("rifle_won").cast(pl.Int64).sum().alias("w"), pl.len().alias("n")).sort("match_key")
W = per["w"].to_numpy().astype(np.float64)
N = per["n"].to_numpy().astype(np.float64)
rng = np.random.default_rng(SEED)
bs = np.empty(NB)
for b in range(NB):
    idx = rng.integers(0, len(W), len(W))
    bs[b] = W[idx].sum() / N[idx].sum()
lo, hi = np.percentile(bs, [2.5, 97.5])
log(f"H3 match-cluster 95% CI: [{100*lo:.1f}, {100*hi:.1f}] over {len(W)} matches (report [76.2, 86.5], 125 matches)")

gap = h["rifle_eq"] - h["other_eq"]
big = h.filter(gap > 5000)
small = h.filter(gap <= 5000)
log(f"H3 equipment: mean rifle side ${h['rifle_eq'].mean()/1000:.1f}k vs other side ${h['other_eq'].mean()/1000:.1f}k "
    f"(report: 21.2k vs 5.9k, all non-pistol side-rounds)")
log(f"H3 gap > $5k: n={big.height}, win={100*big['rifle_won'].mean():.1f}% (report: 248, 83.5%); "
    f"gap <= $5k: n={small.height} (report: 18)")

# all non-pistol side-rounds: rifle-holding sides vs SMG/pistol-only sides
sides = []
for p in ["ct", "t"]:
    sides.append(np_.select([
        pl.col(f"{p}_rifle").alias("rifle"),
        (~pl.col(f"{p}_rifle") & ~pl.col(f"{p}_sniper") & ~pl.col(f"{p}_heavy")).alias("smg_pistol"),
        pl.col(f"{p}_eq").alias("eq"),
    ]))
allsides = pl.concat(sides)
r_eq = allsides.filter(pl.col("rifle"))["eq"].mean()
s_eq = allsides.filter(pl.col("smg_pistol") & ~pl.col("rifle"))["eq"].mean()
log(f"ALL non-pistol side-rounds: rifle-holding side mean eq ${r_eq/1000:.1f}k, SMG/pistol side ${s_eq/1000:.1f}k "
    f"(report: 21.2k vs 5.9k)")

with open(LOG, "a", encoding="utf-8") as fh:
    fh.write("\n".join(out) + "\n")
log("appended", LOG)
