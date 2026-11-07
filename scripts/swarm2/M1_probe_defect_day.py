import glob
import math
import collections
import polars as pl

BASE = "G:/pureskill/csds-cs2/tomes/2024-02-g/tome/csds/"
SUF = ".2023-12-28,2024-02-25"


def read(ch, cols):
    folder = BASE + ("header" if ch == "header" else f"basic_{ch}{SUF}")
    files = sorted(glob.glob(folder + "/dataframe_*"))
    frames = [pl.read_parquet(f, columns=cols) for f in files if len(pl.read_parquet_schema(f)) > 0]
    return pl.concat(frames, how="vertical_relaxed")


H = read("header", ["key", "map_name", "match_date"]).rename({"key": "match_key"})
RE = read("round_end", ["match_key", "round", "winner_team_code", "player_count"])
BS = read("bomb_state", ["match_key", "round", "event_type"])
outcome = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
           .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
j = (RE.filter(pl.col("player_count") == 20)
     .join(outcome, on=["match_key", "round"], how="inner")
     .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3))
                   .otherwise(pl.lit(2)).alias("bomb_side"))
     .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"))
     .join(H, on="match_key", how="left")
     .with_columns(pl.col("match_date").str.slice(0, 10).alias("date"),
                   pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")))
r13 = j.filter(pl.col("round") == 13)
print("r13 rows:", r13.height, "mismatches:", int(r13["mm"].sum()))
# day-level rates
dr = r13.group_by("day").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"))
rate = {r["day"]: r["mm"] / r["n"] for r in dr.iter_rows(named=True)}
print("day rates:", {k: round(v, 3) for k, v in sorted(rate.items())})
# per-map observed vs expected under day-only model (indirect standardization)
rows = []
tot_o = tot_e = 0.0
by_map = collections.defaultdict(lambda: [0, 0, 0.0])  # n, observed, expected
for x in r13.iter_rows(named=True):
    m = x["map_name"]
    by_map[m][0] += 1
    by_map[m][1] += int(bool(x["mm"]))
    by_map[m][2] += rate[x["day"]]
chi = 0.0
print("map: n, observed mismatches, day-expected, O-E, z (day-standardized)")
for m, (n, o, e) in sorted(by_map.items(), key=lambda kv: -kv[1][0]):
    var = sum(rate[x["day"]] * (1 - rate[x["day"]]) for x in r13.filter(pl.col("map_name") == m).iter_rows(named=True))
    z = (o - e) / math.sqrt(var) if var > 0 else float("nan")
    print(f"  {m:<12} n={n:>4} O={o:>4} E={e:>7.1f} O-E={o-e:+7.1f} z={z:+.2f}")
    if var > 0:
        chi += (o - e) ** 2 / var
    tot_o += o
    tot_e += e
print(f"sum chi2 (day-standardized, df<= {len(by_map)-1}) = {chi:.2f}; total O={tot_o:.0f} E={tot_e:.1f}")
# day 12-25 only, other rounds (non-13) for comparison: mismatch by map
oth = j.filter((pl.col("round") != 13) & (pl.col("round") != 25))
print("non-13/25 rows by map: n, mismatches")
print(oth.group_by("map_name").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("n", descending=True))
