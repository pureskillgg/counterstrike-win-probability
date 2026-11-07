"""Armor/helmet check at freeze end (1 Hz sample, first snapshot per match-round). Appends to logs\\stats\\verify_w1.log.
Compares with report Table 6 (pistol rounds) and Table 7 (non-pistol, by primary class)."""
import polars as pl

HZ = r"G:\pureskill\csds-cs2\data\state_sample\players_1hz.parquet"
LOG = r"G:\pureskill\csds-cs2\logs\stats\verify_w1.log"
out = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)


RIFLE = {7, 8, 10, 13, 16, 39, 60}
SNIPER = {9, 40}
SMG = {17, 19, 23, 24, 26, 33, 34}
HEAVY = {14, 25, 27, 28, 29, 35, 11, 38}

fs = (
    pl.scan_parquet(HZ)
    .select(["match_key", "round", "tick", "team_code", "inv_primary", "armor", "has_helmet"])
    .with_columns(pl.col("tick").min().over(["match_key", "round"]).alias("t0"))
    .filter(pl.col("tick") == pl.col("t0"))
    .drop("t0")
    .collect()
)
log("== verify_w1_armor (freeze-end, first snapshot)")

pist = fs.filter(pl.col("round").is_in([1, 13]))
pt = (
    pist.group_by(["round", "team_code"])
    .agg(
        pl.len().alias("players"),
        (pl.col("armor") > 0).mean().mul(100).round(1).alias("armor_gt0_pct"),
        pl.col("has_helmet").mean().mul(100).round(1).alias("helmet_pct"),
    )
    .sort(["round", "team_code"])
)
log("PISTOL ROUNDS (team 2=T, 3=CT); report: r1 T 42.6 / CT 45.3 armor>0, r13 T 48.6 / CT 44.4; helmet 0% both")
for r in pt.iter_rows(named=True):
    log(f"  round {r['round']} team {r['team_code']}: players={r['players']} armor>0={r['armor_gt0_pct']}% helmet={r['helmet_pct']}%")

np_ = fs.filter(~pl.col("round").is_in([1, 13]))
cls = (
    np_.with_columns(
        pl.when(pl.col("inv_primary").is_in(list(RIFLE))).then(pl.lit("RIFLE"))
        .when(pl.col("inv_primary").is_in(list(SNIPER))).then(pl.lit("SNIPER"))
        .when(pl.col("inv_primary").is_in(list(SMG))).then(pl.lit("SMG"))
        .when(pl.col("inv_primary").is_in(list(HEAVY))).then(pl.lit("HEAVY"))
        .when(pl.col("inv_primary") == 0).then(pl.lit("NONE"))
        .otherwise(pl.lit("OTHER")).alias("cls")
    )
    .group_by("cls")
    .agg(
        pl.len().alias("players"),
        pl.col("has_helmet").mean().mul(100).round(1).alias("helmet_pct"),
        (pl.col("armor") > 0).mean().mul(100).round(1).alias("armor_gt0_pct"),
    )
    .sort("cls")
)
log("NON-PISTOL PLAYERS by primary class; report Table 7: RIFLE 13,572 88.2/97.2; SMG 2,695 79.0/94.7; SNIPER 2,745 83.5/92.4; NONE 4,264 30.0/39.9")
for r in cls.iter_rows(named=True):
    log(f"  {r['cls']}: players={r['players']} helmet={r['helmet_pct']}% armor>0={r['armor_gt0_pct']}%")

with open(LOG, "a", encoding="utf-8") as fh:
    fh.write("\n".join(out) + "\n")
log("appended", LOG)
