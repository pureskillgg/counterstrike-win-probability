"""Independent recomputation of W1 headline 3 (rifle-side round win rate, non-pistol rounds)
plus freeze-end holdings (Table 3 analogue) and pistol-round CT win rates, from the 1 Hz sample.

Match-cluster bootstrap: 1000 resamples of matches, seed 0.
Appends to logs\\stats\\verify_w1.log (this script writes its own section to that file).
"""
import numpy as np
import polars as pl

HZ = r"G:\pureskill\csds-cs2\data\state_sample\players_1hz.parquet"
RD = r"G:\pureskill\csds-cs2\data\state_sample\rounds.parquet"
BT = r"G:\pureskill\csds-cs2\data\state_sample\round_buy_types.parquet"
LOG = r"G:\pureskill\csds-cs2\logs\stats\verify_w1.log"
NB = 1000
SEED = 0
out = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)


# CS item definition indices (inv_primary). Verified against the code list below in the log.
RIFLE = {7, 8, 10, 13, 16, 39, 60}
SNIPER = {9, 40}
SMG = {17, 19, 23, 24, 26, 33, 34}
HEAVY = {14, 25, 27, 28, 29, 35}
AUTO_SNIPER_UNSURE = {11, 38}  # G3SG1, SCAR-20: check if present

log("== verify_w1_rifle")

hz = (
    pl.scan_parquet(HZ)
    .select(["match_key", "round", "tick", "team_code", "player_id", "inv_primary", "equipment_value_calc"])
    .with_columns(pl.col("tick").min().over(["match_key", "round"]).alias("t0"))
    .filter(pl.col("tick") == pl.col("t0"))
    .drop("t0")
    .collect()
)
log("freeze-end player rows:", hz.height, " rounds:", hz.select(["match_key", "round"]).unique().height,
    " matches:", hz["match_key"].n_unique())

prim_counts = hz["inv_primary"].value_counts(sort=True).rows()
log("inv_primary codes (code, count):", sorted(prim_counts))
seen = set(hz["inv_primary"].unique().to_list())
unmapped = sorted(c for c in seen if c not in RIFLE | SNIPER | SMG | HEAVY | {0} | AUTO_SNIPER_UNSURE)
log("inv_primary codes not in my class map:", unmapped)
log("AUTO_SNIPER_UNSURE codes present:", sorted(seen & AUTO_SNIPER_UNSURE))

hz = hz.with_columns(
    pl.col("inv_primary").is_in(list(RIFLE)).alias("is_rifle"),
    pl.col("inv_primary").is_in(list(SNIPER)).alias("is_sniper"),
    pl.col("inv_primary").is_in(list(SMG)).alias("is_smg"),
    pl.col("inv_primary").is_in(list(HEAVY)).alias("is_heavy"),
    (pl.col("inv_primary") == 0).alias("is_nopri"),
)

side = (
    hz.group_by(["match_key", "round", "team_code"])
    .agg(
        pl.len().alias("n_players"),
        pl.col("is_rifle").any().alias("has_rifle"),
        pl.col("is_sniper").any().alias("has_sniper"),
        pl.col("is_smg").any().alias("has_smg"),
        pl.col("is_heavy").any().alias("has_heavy"),
        pl.col("is_nopri").any().alias("has_pistol_only"),
        pl.col("is_nopri").all().alias("all_pistol_only"),
        pl.col("equipment_value_calc").sum().alias("team_eq"),
    )
)

rd = pl.read_parquet(RD).select(["match_key", "round", "ct_win", "round_number"])
log("rounds.parquet rows:", rd.height, " round values:", sorted(rd["round"].unique().to_list())[:3], "...",
    " round_number range:", rd["round_number"].min(), rd["round_number"].max())

ct = side.filter(pl.col("team_code") == 3).select(["match_key", "round"] + [
    pl.col(c).alias(f"ct_{c}") for c in ["n_players", "has_rifle", "has_sniper", "has_smg", "has_heavy",
                                          "has_pistol_only", "all_pistol_only", "team_eq"]])
tt = side.filter(pl.col("team_code") == 2).select(["match_key", "round"] + [
    pl.col(c).alias(f"t_{c}") for c in ["n_players", "has_rifle", "has_sniper", "has_smg", "has_heavy",
                                         "has_pistol_only", "all_pistol_only", "team_eq"]])
rr = rd.join(ct, on=["match_key", "round"], how="left").join(tt, on=["match_key", "round"], how="left")
log("rounds joined:", rr.height, " with CT side:", rr["ct_has_rifle"].is_not_null().sum(),
    " with T side:", rr["t_has_rifle"].is_not_null().sum())

# --- pistol rounds: rounds 1 and 13 (MR12)
pist = rr.filter(pl.col("round").is_in([1, 13]))
for r in [1, 13]:
    p = pist.filter(pl.col("round") == r)
    n = p.height
    ctw = p["ct_win"].sum()
    rng = np.random.default_rng(SEED)
    mk = p["match_key"].unique().to_list()
    per = p.group_by("match_key").agg(pl.col("ct_win").sum().alias("w"), pl.len().alias("n")).sort("match_key")
    W = per["w"].to_numpy()
    N = per["n"].to_numpy()
    bs = []
    for _ in range(NB):
        idx = rng.integers(0, len(W), len(W))
        bs.append(W[idx].sum() / N[idx].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    log(f"pistol round {r}: rounds={n} matches={len(mk)} CT win={100*ctw/n:.1f}% [{100*lo:.1f}, {100*hi:.1f}] "
        f"(report: {'50.0 [41.6, 57.9], 144 matches' if r == 1 else '42.1 [33.6, 50.7], 126 matches'})")

# freeze-end helmet/armor for pistol rounds is not in side table; armor needs player-level, handled below.

# --- non-pistol rounds
np_rounds = rr.filter(~pl.col("round").is_in([1, 13]))
log("non-pistol rounds in sample:", np_rounds.height, "(report: 2,403)")
log("non-pistol rounds with both sides present:", np_rounds.filter(pl.col("ct_n_players").is_not_null() & pl.col("t_n_players").is_not_null()).height)

# holdings (side-rounds), Table 3 analogue
sides_long = []
for prefix, label in [("ct", "CT"), ("t", "T")]:
    sides_long.append(np_rounds.select([
        pl.lit(label).alias("side"),
        "match_key",
        "round",
        pl.col(f"{prefix}_has_rifle").alias("rifle"),
        pl.col(f"{prefix}_has_smg").alias("smg"),
        pl.col(f"{prefix}_has_sniper").alias("sniper"),
        pl.col(f"{prefix}_has_heavy").alias("heavy"),
        pl.col(f"{prefix}_has_pistol_only").alias("pistol_only"),
        pl.col(f"{prefix}_all_pistol_only").alias("all_pistol_only"),
        pl.col(f"{prefix}_team_eq").alias("team_eq"),
        pl.col(f"{prefix}_n_players").alias("n_players"),
    ]))
sl = pl.concat(sides_long).drop_nulls(["rifle"])
log("side-rounds for holdings:", sl.height)
for side_label in ["CT", "T", "both"]:
    s = sl if side_label == "both" else sl.filter(pl.col("side") == side_label)
    parts = []
    for col, nm in [("rifle", "at least one rifle"), ("smg", "at least one SMG"), ("sniper", "at least one sniper (AWP/SSG 08)"),
                    ("heavy", "at least one other heavy"), ("pistol_only", "at least one pistol-only player"),
                    ("all_pistol_only", "all players pistol-only")]:
        v = s[col].cast(pl.Float64).to_numpy()
        mv = s["match_key"].to_numpy()
        p = v.mean()
        # match-cluster bootstrap over matches
        uniq, inv = np.unique(mv, return_inverse=True)
        sums = np.bincount(inv, weights=v, minlength=len(uniq))
        cnts = np.bincount(inv, minlength=len(uniq))
        rng = np.random.default_rng(SEED)
        bs = []
        for _ in range(NB):
            idx = rng.integers(0, len(uniq), len(uniq))
            bs.append(sums[idx].sum() / cnts[idx].sum())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        parts.append(f"{nm}: {100*p:.1f} [{100*lo:.1f}, {100*hi:.1f}]")
    log(f"HOLDINGS {side_label} (n={s.height}): " + " | ".join(parts))

# --- rifle side vs non-rifle side (H3)
ct_r = np_rounds["ct_has_rifle"].cast(pl.Boolean)
t_r = np_rounds["t_has_rifle"].cast(pl.Boolean)
rifle_side = np_rounds.with_columns(
    pl.when(ct_r & ~t_r).then(pl.lit("CT"))
    .when(t_r & ~ct_r).then(pl.lit("T"))
    .when(ct_r & t_r).then(pl.lit("both"))
    .otherwise(pl.lit("neither"))
    .alias("rifle_side_class")
)
log("rifle-side classes (non-pistol rounds):", rifle_side["rifle_side_class"].value_counts(sort=True).rows())

h3 = rifle_side.filter(pl.col("rifle_side_class").is_in(["CT", "T"])).with_columns(
    pl.when(pl.col("rifle_side_class") == "CT").then(pl.col("ct_win") == 1).otherwise(pl.col("ct_win") == 0).alias("rifle_side_won"),
    pl.when(pl.col("rifle_side_class") == "CT").then(pl.col("ct_team_eq")).otherwise(pl.col("t_team_eq")).alias("rifle_eq"),
    pl.when(pl.col("rifle_side_class") == "CT").then(pl.col("t_team_eq")).otherwise(pl.col("ct_team_eq")).alias("other_eq"),
)
n3 = h3.height
w3 = int(h3["rifle_side_won"].sum())
log(f"H3 rifle-side vs non-rifle-side rounds: n={n3}, rifle-side wins={w3}, rate={100*w3/n3:.2f}% (report: 266 rounds, 81.6%)")
per = h3.group_by("match_key").agg(pl.col("rifle_side_won").cast(pl.Int64).sum().alias("w"), pl.len().alias("n")).sort("match_key")
W = per["w"].to_numpy()
N = per["n"].to_numpy()
rng = np.random.default_rng(SEED)
bs = []
for _ in range(NB):
    idx = rng.integers(0, len(W), len(W))
    bs.append(W[idx].sum() / N[idx].sum())
lo, hi = np.percentile(bs, [2.5, 97.5])
log(f"H3 match-cluster 95% CI: [{100*lo:.1f}, {100*hi:.1f}] over {len(W)} matches (report CI [76.2, 86.5])")

ct_side = h3.filter(pl.col("rifle_side_class") == "CT")
t_side = h3.filter(pl.col("rifle_side_class") == "T")
log(f"H3 by side: rifle=CT n={ct_side.height} win={100*ct_side['rifle_side_won'].mean():.1f}% ; "
    f"rifle=T n={t_side.height} win={100*t_side['rifle_side_won'].mean():.1f}%  (report: CT 134/79.9, T 132/83.3)")
gap = h3["rifle_eq"] - h3["other_eq"]
log(f"H3 mean team equipment: rifle side ${h3['rifle_eq'].mean()/1000:.1f}k vs other side ${h3['other_eq'].mean()/1000:.1f}k (report: 21.2k vs 5.9k)")
big = h3.filter(gap > 5000)
log(f"H3 rifle side ahead by > $5k: n={big.height} win={100*big['rifle_side_won'].mean():.1f}% (report: 248, 83.5)")

with open(LOG, "a", encoding="utf-8") as fh:
    fh.write("\n".join(out) + "\n")
log("appended", LOG)
