"""Intern-grader rows for the validation matches of the positions runs (pin 594c2815).

Takes the validation predictions of experiments/positions/positions_validation.py, one file per arm, applies the intern's filters
(README step 1) and joins the predictions onto every second of the kept 5v5 rounds. Matches on the upload days the table was built
on (Oct 9 to 13, 2026) are dropped. The validation keys come from the prediction files, and the script stops if the arms disagree.

    python evaluation/build_validation_gdt_rows.py --model p_full_trainctr=<preds.parquet> --model p_full_oldctr=<preds.parquet> \
        --model p_base=<preds.parquet> --out <rows.parquet>
"""
import argparse
import json

import polars as pl

ROOT = "G:/pureskill/csds-cs2"
MAPS = ["de_ancient", "de_anubis", "de_cache", "de_dust2", "de_inferno", "de_mirage", "de_nuke"]
TABLE_DAYS = ("2026/10/09", "2026/10/13")
KEY = ["match_key", "round", "second"]
FROZEN = "C:/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/frozen_test_keys.json"


def main():
    ap = argparse.ArgumentParser(description="Intern-grader rows for the validation matches of the positions runs")
    ap.add_argument("--cache", default=ROOT + "/data/combined/cache_snapshot_2026.parquet")
    ap.add_argument("--frozen", default=FROZEN, help="the frozen test match keys, a JSON list; checked against, never used")
    ap.add_argument("--model", action="append", required=True, help="name=preds.parquet; every file must cover the same rows")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    names = [spec.split("=", 1)[0] for spec in a.model]

    # 1. the predictions: one column per arm, inner-joined; each arm's own row count must survive the join
    preds, heights = None, []
    for spec in a.model:
        name, path = spec.split("=", 1)
        p = (pl.read_parquet(path)
             .select(["match_key", "round", "second", "p"])
             .with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int32))
             .rename({"p": name}))
        assert p.select(KEY).is_duplicated().sum() == 0, f"{name}: duplicate keys"
        heights.append(p.height)
        preds = p if preds is None else preds.join(p, on=KEY, how="inner")
    assert all(h == preds.height for h in heights), f"arms cover different rows: {heights} vs joined {preds.height}"
    vkeys = preds["match_key"].unique().to_list()
    holdout = set(json.load(open(ROOT + "/data/combined/holdout_22_25_test_keys.json"))["match_keys"])
    frozen = set(json.load(open(a.frozen)))
    assert not (set(vkeys) & (holdout | frozen)), "a holdout B or frozen test match is in the validation keys"
    print(f"validation matches with predictions: {len(vkeys):,}; prediction rows: {preds.height:,}; none in holdout B or the frozen test",
          flush=True)

    # 2. the intern's match filters: in scope, max_rounds 24, the seven maps, not the table-build days
    mt = pl.read_parquet(ROOT + "/data/combined/matches.parquet", columns=["match_key", "in_scope", "max_rounds_final"])
    mt = mt.filter(pl.col("match_key").is_in(vkeys) & (pl.col("in_scope") == True) & (pl.col("max_rounds_final") == 24))  # noqa: E712
    print(f"in scope, max_rounds 24: {mt.height:,} matches", flush=True)
    mk = (pl.scan_parquet(a.cache).filter(pl.col("match_key").is_in(mt["match_key"].to_list()))
          .select(["match_key", "map_name"]).unique().collect()
          .with_columns(pl.col("match_key").str.extract(r"csds/(\d{4}/\d{2}/\d{2})", 1).alias("day")))
    assert mk.height == mt.height and mk["day"].null_count() == 0, "match keys or upload days did not parse"
    mk = mk.filter(pl.col("map_name").is_in(MAPS))
    print(f"on the seven maps: {mk.height:,} matches", flush=True)
    mk = mk.filter(~((pl.col("day") >= TABLE_DAYS[0]) & (pl.col("day") <= TABLE_DAYS[1])))
    print(f"not on the table-build days {TABLE_DAYS[0]} to {TABLE_DAYS[1]}: {mk.height:,} matches", flush=True)
    assert mk.height > 0, "no matches left"
    final_keys = mk["match_key"].to_list()

    # 3. every second of those matches; 5v5 round starts only
    need = ["match_key", "round", "second", "alive_ct", "alive_t", "st_equip_ct", "st_equip_t", "bomb_planted",
            "seconds_since_plant", "label_ct_win"]
    df = (pl.scan_parquet(a.cache).filter(pl.col("match_key").is_in(final_keys)).select(need).collect()
          .with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int32)))
    starts = (df.filter((pl.col("second") == 0) & (pl.col("alive_ct") == 5) & (pl.col("alive_t") == 5))
              .select(["match_key", "round"]).unique())
    df = df.join(starts, on=["match_key", "round"], how="inner")
    assert df.select(KEY).is_duplicated().sum() == 0, "cache rows repeat a (match, round, second)"
    print(f"5v5 round starts: {starts.height:,} rounds; qualifying seconds: {df.height:,}", flush=True)

    # 4. the predictions onto those seconds; the arms predicted only the seconds the position table has
    before = df.height
    df = df.join(preds.filter(pl.col("match_key").is_in(final_keys)), on=KEY, how="inner")
    print(f"qualifying seconds with predictions from every arm: {df.height:,} of {before:,} ({df.height / before:.1%})", flush=True)

    # 5. the grader's columns; since_plant is 0 before the plant
    out = df.select([
        pl.col("match_key").alias("match"), "round", "second", "alive_ct", "alive_t",
        pl.col("st_equip_ct").alias("equip_ct"), pl.col("st_equip_t").alias("equip_t"),
        pl.col("bomb_planted").cast(pl.Int64).alias("planted"),
        pl.when(pl.col("bomb_planted") == 1).then(pl.col("seconds_since_plant")).otherwise(0.0).alias("since_plant"),
        pl.col("label_ct_win").cast(pl.Int64).alias("y"),
        *names,
    ])
    out.write_parquet(a.out)
    print(f"WROTE {a.out} rows {out.height:,} matches {out['match'].n_unique():,} rounds {out.select(['match', 'round']).n_unique():,}",
          flush=True)


if __name__ == "__main__":
    main()
