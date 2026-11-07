"""M4 round-13 check by map: is the 2024 bomb-side mismatch map-specific?

Definition follows scripts/swarm/e2_side_check_r13.py: a round with a bomb_defused or bomb_exploded
row in bomb_state. Bomb side: defused -> 3 (CT), exploded -> 2 (T). Mismatch: bomb side differs from
round_end.winner_team_code. Build comes from header.build_num; day from header.match_date.
2026 tome is the control. Read-only.
"""
import glob
import os

import polars as pl

T2024 = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
T2026 = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"


def read(base, channel, cols, suffix):
    folder = os.path.join(base, "header") if channel == "header" else os.path.join(base, f"basic_{channel}{suffix}")
    files = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def run(label, base, suffix, hdr_key_fallback=True):
    hcols = pl.read_parquet(sorted(glob.glob(os.path.join(base, "header", "dataframe_*")))[0]).columns
    hk = "key" if "key" in hcols else "match_key"
    hdr_cols = [hk, "map_name", "match_date"] + (["build_num"] if "build_num" in hcols else [])
    H = read(base, "header", hdr_cols, suffix).rename({hk: "match_key"})
    BS = read(base, "bomb_state", ["match_key", "round", "event_type"], suffix)
    RE = read(base, "round_end", ["match_key", "round", "winner_team_code", "player_count"], suffix)
    ev = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
          .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
    j = (ev.join(RE, on=["match_key", "round"], how="inner")
         .join(H, on="match_key", how="left"))
    j = j.with_columns(
        pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"),
    ).with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"))
    print(f"\n===== {label}: rounds with a bomb outcome: {j.height}; mismatches: {int(j['mismatch'].sum())}")
    r13 = j.filter(pl.col("round") == 13)
    print(f"round 13 rows: {r13.height}; mismatches: {int(r13['mismatch'].sum())}")
    if "build_num" in r13.columns:
        print("round 13 by build x map (n, mismatches):")
        print(r13.group_by(["build_num", "map_name"]).agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"))
              .sort(["build_num", "map_name"]))
    print("round 13 by map (n, mismatches, rate):")
    g = (r13.group_by("map_name").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"))
         .with_columns((pl.col("mm") / pl.col("n")).round(3).alias("rate")).sort("map_name"))
    print(g)
    if "build_num" in r13.columns:
        b = r13.filter(pl.col("build_num") == 9961)
        if b.height:
            gg = (b.group_by("map_name").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"))
                  .with_columns((pl.col("mm") / pl.col("n")).round(3).alias("rate")).sort("map_name"))
            print("build 9961 round 13 by map:")
            print(gg)
    # other rounds by map (all rounds except 13)
    o = j.filter(pl.col("round") != 13)
    og = (o.group_by("map_name").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"))
          .with_columns((pl.col("mm") / pl.col("n")).round(4).alias("rate")).sort("map_name"))
    print("other rounds by map:")
    print(og)


def main():
    run("2024 tome (2024-02-g)", T2024, ".2023-12-28,2024-02-25")
    run("2026 tome (control)", T2026, ".2026-09-18,2026-10-08")


if __name__ == "__main__":
    main()
