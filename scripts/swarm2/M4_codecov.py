"""M4 code coverage: how much of 2026 planted-round volume would a per-map site-code lookup cover?

Lookup built from the 1 Hz place-name sample (150 matches, bomb-zone place names BombsiteA/B at the
plant, same method as M4_sites2.py): per map, a site code whose named rounds agree on one place with
purity >= 0.95 and at least 2 named rounds. Coverage is then measured on all kept 2026 planted rounds.
Also: share of planted rounds with missing planter coordinates (2026), which the cluster method drops.
"""
import collections
import os

import polars as pl

SCR = (r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
       r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\M4")
PN = r"G:\pureskill\csds-cs2\data\state_sample\place_names.parquet"


def main():
    P = pl.read_parquet(os.path.join(SCR, "plants_all_2026.parquet"))
    K = P.filter(pl.col("kept")).unique(subset=["match_key", "round"], keep="first")
    print(f"kept planted rounds (2026): {K.height}")
    miss = K.filter(pl.col("x").is_null() | pl.col("y").is_null() | pl.col("z").is_null())
    print(f"kept planted rounds with a null planter coordinate: {miss.height} "
          f"({miss.height / K.height:.4f}); by map: {miss.group_by('map_name').len().sort('map_name').iter_rows().__class__.__name__}")
    print(miss.group_by("map_name").len().sort("map_name"))

    pn = pl.read_parquet(PN, columns=["match_key", "round", "tick", "place_name", "is_in_bomb_zone"])
    bz = pn.filter(pl.col("is_in_bomb_zone"))
    grp = bz.group_by(["match_key", "round"]).agg(pl.col("tick"), pl.col("place_name"))
    gd = {(r["match_key"], r["round"]): (r["tick"], r["place_name"]) for r in grp.iter_rows(named=True)}
    sample = set(pn["match_key"].unique().to_list())
    Ks = K.filter(pl.col("match_key").is_in(list(sample)))
    rows = []
    for r in Ks.iter_rows(named=True):
        v = gd.get((r["match_key"], r["round"]))
        if v is None:
            continue
        cand = [p for t, p in zip(v[0], v[1]) if abs(t - r["tick"]) <= 64]
        if cand:
            rows.append((r["map_name"], int(r["site_code"]), collections.Counter(cand).most_common(1)[0][0]))
    named = pl.DataFrame(rows, schema=["map_name", "site_code", "place"], orient="row")
    named = named.filter(pl.col("place").is_in(["BombsiteA", "BombsiteB"]))
    print(f"\nnamed sample planted rounds: {named.height}")
    tot = named.group_by(["map_name", "site_code"]).agg(pl.len().alias("n"),
                                                      (pl.col("place") == "BombsiteA").sum().alias("a"))
    tot = tot.with_columns((pl.max_horizontal("a", pl.col("n") - pl.col("a")) / pl.col("n")).alias("purity"))
    look = tot.filter((pl.col("n") >= 2) & (pl.col("purity") >= 0.95))
    print(f"lookup entries (map, code) with n>=2 and purity>=0.95: {look.height}")
    cov_rows = K.join(look.select(["map_name", "site_code"]), on=["map_name", "site_code"], how="inner")
    print(f"kept 2026 planted rounds whose (map, code) is in the lookup: {cov_rows.height} of {K.height} "
          f"({cov_rows.height / K.height:.3f})")
    print("coverage by map (covered / total):")
    cov = K.group_by("map_name").len().rename({"len": "total"}).join(
        cov_rows.group_by("map_name").len().rename({"len": "covered"}), on="map_name", how="left"
    ).with_columns(pl.col("covered").fill_null(0)).with_columns(
        (pl.col("covered") / pl.col("total")).round(3).alias("share")).sort("map_name")
    print(cov)


if __name__ == "__main__":
    main()
