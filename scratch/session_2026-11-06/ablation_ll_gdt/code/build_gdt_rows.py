"""GDT rows for the ablation arms, one parquet per model seed, same population as the validation GDT report.

Mirrors evaluation/build_validation_gdt_rows.py steps 2 to 5 (in scope, max_rounds 24, the seven maps, not the table-build days
2026/10/09 to 2026/10/13, 5v5 round starts, every second, inner join on the predictions) so that the population is the one the
report used. It does NOT open the frozen-test key file or the holdout B key file, which that script reads for its overlap assertion;
validation-only zero overlap is taken from the sprint matrix build (sprint_prepare.py asserts it) and reported as such.

Output columns follow the README "The rows": match, round, second, alive_ct, alive_t, equip_ct, equip_t, planted, since_plant, y,
then one column per arm: p_base (59 inputs), p_geo (75), p_full (83).
"""
import json
import os

import polars as pl

SCRATCH = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/ablation_ll_gdt"
ROOT = "G:/pureskill/csds-cs2"
CACHE = ROOT + "/data/combined/cache_snapshot_2026.parquet"
MATCHES = ROOT + "/data/combined/matches.parquet"
PRED_DIR = SCRATCH + "/out/fits/preds"
OUT_DIR = SCRATCH + "/out/gdt_rows"
MAPS = ["de_ancient", "de_anubis", "de_cache", "de_dust2", "de_inferno", "de_mirage", "de_nuke"]
TABLE_DAYS = ("2026/10/09", "2026/10/13")
DAY_RE = r"csds/(\d{4}/\d{2}/\d{2})"
KEY = ["match_key", "round", "second"]
ARMS = [("base", "p_base"), ("base-GEO", "p_geo"), ("full", "p_full")]
SEEDS = [0, 1, 2]
EXPECTED_VAL_ROWS = 1515228
EXPECTED_VAL_MATCHES = 1168
EXPECTED_GDT_MATCHES = 297  # docs/eval_reports/positions_validation_gdt/summary.json, counts.matches


def pred_path(arm, seed):
    return f"{PRED_DIR}/{arm}_d4_e0.03_m20_s{seed}.parquet"


def load_preds():
    """Every arm and seed on the same validation rows, one column per arm and seed, labels checked to agree."""
    preds = None
    for seed in SEEDS:
        for arm, col in ARMS:
            p = (pl.read_parquet(pred_path(arm, seed))
                 .select(["match_key", "round", "second", "label_ct_win", "p"])
                 .with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int32))
                 .rename({"p": f"{col}_s{seed}"}))
            assert p.height == EXPECTED_VAL_ROWS, f"{arm} seed {seed}: {p.height} rows"
            assert p.select(KEY).is_duplicated().sum() == 0, f"{arm} seed {seed}: duplicate keys"
            if preds is None:
                preds = p
                continue
            joined = preds.join(p.select(KEY + ["label_ct_win", f"{col}_s{seed}"]), on=KEY, how="inner", suffix="_r")
            assert joined.height == EXPECTED_VAL_ROWS, f"{arm} seed {seed}: join gives {joined.height} rows"
            assert (joined["label_ct_win"] == joined["label_ct_win_r"]).all(), f"{arm} seed {seed}: labels disagree"
            preds = joined.drop("label_ct_win_r")
    return preds


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    preds = load_preds()
    vkeys = preds["match_key"].unique().to_list()
    assert len(vkeys) == EXPECTED_VAL_MATCHES, len(vkeys)
    print(f"validation prediction rows {preds.height:,}; validation matches {len(vkeys):,}; all 9 arm-seed files agree on keys "
          f"and labels", flush=True)

    # 2. the intern's match filters
    mt = pl.read_parquet(MATCHES, columns=["match_key", "in_scope", "max_rounds_final"])
    mt = mt.filter(pl.col("match_key").is_in(vkeys) & (pl.col("in_scope") == True) & (pl.col("max_rounds_final") == 24))  # noqa: E712
    print(f"in scope, max_rounds 24: {mt.height:,} matches", flush=True)
    mk = (pl.scan_parquet(CACHE).filter(pl.col("match_key").is_in(mt["match_key"].to_list()))
          .select(["match_key", "map_name"]).unique().collect()
          .with_columns(pl.col("match_key").str.extract(DAY_RE, 1).alias("day")))
    assert mk.height == mt.height and mk["day"].null_count() == 0, "match keys or upload days did not parse"
    mk = mk.filter(pl.col("map_name").is_in(MAPS))
    on_seven_maps = mk.height
    print(f"on the seven maps: {on_seven_maps:,} matches", flush=True)
    mk = mk.filter(~((pl.col("day") >= TABLE_DAYS[0]) & (pl.col("day") <= TABLE_DAYS[1])))
    print(f"not on the table-build days {TABLE_DAYS[0]} to {TABLE_DAYS[1]}: {mk.height:,} matches", flush=True)
    final_keys = mk["match_key"].to_list()
    assert len(final_keys) == EXPECTED_GDT_MATCHES, f"GDT population {len(final_keys)} matches, report has {EXPECTED_GDT_MATCHES}"
    print(f"GDT population matches {len(final_keys)} == report {EXPECTED_GDT_MATCHES}", flush=True)
    all_days = preds.select("match_key").unique().with_columns(pl.col("match_key").str.extract(DAY_RE, 1).alias("day"))
    print(f"upload days, all 1,168 validation matches: {all_days['day'].min()} to {all_days['day'].max()}; "
          f"GDT population: {mk['day'].min()} to {mk['day'].max()}", flush=True)

    # 3. every second of those matches; 5v5 round starts only
    need = ["match_key", "round", "second", "alive_ct", "alive_t", "st_equip_ct", "st_equip_t", "bomb_planted",
            "seconds_since_plant", "label_ct_win"]
    df = (pl.scan_parquet(CACHE).filter(pl.col("match_key").is_in(final_keys)).select(need).collect()
          .with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int32)))
    starts = (df.filter((pl.col("second") == 0) & (pl.col("alive_ct") == 5) & (pl.col("alive_t") == 5))
              .select(["match_key", "round"]).unique())
    df = df.join(starts, on=["match_key", "round"], how="inner")
    assert df.select(KEY).is_duplicated().sum() == 0, "cache rows repeat a (match, round, second)"
    print(f"5v5 round starts: {starts.height:,} rounds; qualifying seconds: {df.height:,}", flush=True)
    before = df.height

    # 4. predictions onto those seconds, per seed; the rows are the seconds every arm predicted
    final_preds = preds.filter(pl.col("match_key").is_in(final_keys))
    for seed in SEEDS:
        cols = [f"{col}_s{seed}" for _, col in ARMS]
        d = df.join(final_preds.select(KEY + cols), on=KEY, how="inner")
        print(f"seed {seed}: qualifying seconds with predictions from every arm: {d.height:,} of {before:,} "
              f"({d.height / before:.1%})", flush=True)
        # 5. the grader's columns; since_plant is 0 before the plant
        out = d.select([
            pl.col("match_key").alias("match"), "round", "second", "alive_ct", "alive_t",
            pl.col("st_equip_ct").alias("equip_ct"), pl.col("st_equip_t").alias("equip_t"),
            pl.col("bomb_planted").cast(pl.Int64).alias("planted"),
            pl.when(pl.col("bomb_planted") == 1).then(pl.col("seconds_since_plant")).otherwise(0.0).alias("since_plant"),
            pl.col("label_ct_win").cast(pl.Int64).alias("y"),
            *[pl.col(f"{col}_s{seed}").alias(col) for _, col in ARMS],
        ])
        path = f"{OUT_DIR}/rows_s{seed}.parquet"
        out.write_parquet(path)
        print(f"WROTE {path} rows {out.height:,} matches {out['match'].n_unique():,} "
              f"rounds {out.select(['match', 'round']).n_unique():,}", flush=True)

    summary = {"validation_matches": len(vkeys), "validation_prediction_rows": preds.height,
               "in_scope_max_rounds_24": mt.height, "on_seven_maps": on_seven_maps,
               "gdt_population_matches": len(final_keys),
               "gdt_population_upload_days": [str(mk["day"].min()), str(mk["day"].max())],
               "all_validation_upload_days": [str(all_days["day"].min()), str(all_days["day"].max())]}
    with open(f"{OUT_DIR}/population.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print("population written", flush=True)


if __name__ == "__main__":
    main()
