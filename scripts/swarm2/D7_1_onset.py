"""D7-1/D7-2: exact onset of 2024 round-13 side mismatch, per day and per match.

Mismatch definition (same as scripts/swarm/e2_side_check_r13.py): a round with a bomb_defused or
bomb_exploded event; bomb side = 3 (CT) if defused else 2 (T); mismatch = bomb side != winner_team_code.
ASSUMPTION: player_count == 20 is the 2024 full-match convention (as in e2); other counts reported separately.
ASSUMPTION: match time = header match_date (UTC, as recorded); key-path day reported as a cross-check.
"""
import glob, sys, datetime as dt
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(220)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"

def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))

def scan(ch, cols, pred=None):
    lf = pl.concat([pl.scan_parquet(f) for f in files(BASE / f"basic_{ch}{SUF}")])
    if pred is not None:
        lf = lf.filter(pred)
    return lf.select(cols)

# ---- header: one row per match
H = pl.concat([pl.read_parquet(f, columns=["key", "match_id", "match_date", "map_name", "tick_rate",
                                             "t_starters_score_final", "ct_starters_score_final"])
               for f in files(BASE / "header")])
H = H.with_columns(
    pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt"),
    pl.col("key").str.extract(r"csds/(\d{4}/\d\d/\d\d)/", 1).alias("key_day"),
)
print("header matches:", H.height, "unique keys:", H["key"].n_unique(), "null mdt:", H["mdt"].null_count())
print("tick_rate:", H["tick_rate"].value_counts().sort("tick_rate").rows())
H = H.with_columns(pl.col("mdt").dt.strftime("%Y/%m/%d").alias("md_day"))
xd = H.filter(pl.col("md_day").str.replace_all("/", "/") != pl.col("key_day")).height
print("matches where match_date UTC day != key path day:", xd, "of", H.height)
print("\nmatch_date UTC day counts (all 2024 tome matches):")
print(H.group_by("md_day").len().sort("md_day").rows())
print("\nmap_name counts:")
print(H.group_by("map_name").len().sort("len", descending=True).rows())

# ---- rounds and bomb outcomes
RE = scan("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"]).collect()
print("\nround_end rows:", RE.height, "unique (match,round):", RE.select(["match_key", "round"]).unique().height)
print("round_end keys not in header:", RE.filter(~pl.col("match_key").is_in(H["key"].implode())).height)
print("player_count dist:", RE["player_count"].value_counts().sort("player_count").rows())
print("winner_team_code dist:", RE["winner_team_code"].value_counts().sort("winner_team_code").rows())

BS_types = scan("bomb_state", ["event_type"]).group_by("event_type").len().collect()
print("\nbomb_state event_type counts:", BS_types.sort("event_type").rows())
BS = scan("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"],
          pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"])).collect()

outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"])
        .agg(pl.col("event_type").unique().alias("evs"), pl.len().alias("n_out"),
             pl.col("tick").min().alias("out_tick")))
print("rounds with >1 outcome event:", outc.filter(pl.col("n_out") > 1).height,
      "| with both defused and exploded:", outc.filter(pl.col("evs").list.len() > 1).height)
plant = (BS.filter(pl.col("event_type") == "bomb_planted")
         .group_by(["match_key", "round"])
         .agg(pl.col("player_team_code").first().alias("planter_side"), pl.col("tick").min().alias("plant_tick")))

R = (RE.join(outc, on=["match_key", "round"], how="left")
       .join(plant, on=["match_key", "round"], how="left")
       .join(H.select(["key", "mdt", "md_day", "key_day", "map_name", "t_starters_score_final",
                       "ct_starters_score_final"]).rename({"key": "match_key"}),
             on="match_key", how="left")
       .with_columns(
           pl.when(pl.col("evs").list.contains("bomb_defused")).then(3)
             .when(pl.col("evs").list.contains("bomb_exploded")).then(2).otherwise(None).alias("bomb_side"),
       )
       .with_columns(
           pl.when(pl.col("bomb_side").is_not_null()).then(pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"),
           pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
             .otherwise(pl.lit("other")).alias("rbin"),
       ))
R = R.with_columns(
    pl.when(pl.col("bomb_side").is_not_null()).then(pl.col("out_tick") - pl.col("tick")).otherwise(None).alias("out_minus_end_tick"))
R.write_parquet(str(ROOT / "logs" / "swarm2" / "D7_rounds_all.parquet"))  # own scratch output
print("\nrounds total:", R.height, "| with outcome:", R.filter(pl.col("bomb_side").is_not_null()).height)

OUT = R.filter((pl.col("bomb_side").is_not_null()) & (pl.col("player_count") == 20))
print("ASSUMPTION player_count==20 outcome rounds:", OUT.height, "mismatches:", int(OUT["mm"].sum()))
print("\nOutcome rounds with player_count != 20 (count, mismatches):",
      R.filter((pl.col("bomb_side").is_not_null()) & (pl.col("player_count") != 20)).select(
          pl.len(), pl.col("mm").sum()).rows())
print("\nmismatch by rbin (player_count==20):")
print(OUT.group_by("rbin").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"),
                               pl.col("mm").mean().alias("rate")).sort("rbin"))

# ---- per match-day (match_date UTC) for r13, r25, other
print("\nper match_date UTC day (player_count==20):")
d = (OUT.group_by(["md_day", "rbin"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"))
       .with_columns((pl.col("mm") / pl.col("n")).round(3).alias("rate"))
       .pivot(on="rbin", index="md_day", values=["n", "mm", "rate"]).sort("md_day"))
print(d)

print("\nper key-path day (e2 convention) for r13:")
print(OUT.filter(pl.col("round") == 13).group_by("key_day").agg(pl.len().alias("n"),
      pl.col("mm").sum().alias("mm"), pl.col("mm").mean().round(3).alias("rate")).sort("key_day"))
