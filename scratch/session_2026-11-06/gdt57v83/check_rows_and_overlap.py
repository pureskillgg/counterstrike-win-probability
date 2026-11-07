import json, os
import numpy as np
import polars as pl
S = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/gdt57v83"
P = "C:/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/positions_validation"
new = pl.read_parquet(f"{S}/validation_rows_297_five_arms.parquet")
old = pl.read_parquet(f"{P}/validation_gdt_rows.parquet")
print("new rows", new.height, new.columns)
key = ["match", "round", "second"]
j = old.join(new, on=key, how="full", suffix="_new", coalesce=False)
print("old-only", j.filter(pl.col("y_new").is_null()).height, "new-only", j.filter(pl.col("y").is_null()).height)
for c in ["alive_ct", "alive_t", "equip_ct", "equip_t", "planted", "since_plant", "y"]:
    d = j.filter(pl.col(c).cast(pl.Float64) != pl.col(f"{c}_new").cast(pl.Float64)).height if f"{c}_new" in j.columns else -1
    print("  field", c, "mismatches", d)
for c in ["p_base", "p_full_trainctr", "p_full_oldctr"]:
    mx = j.select((pl.col(c).cast(pl.Float64) - pl.col(f"{c}_new").cast(pl.Float64)).abs().max()).item()
    print("  pred", c, "max abs diff vs existing", mx)
# key sets of the 57 and sprint 83 arms against the positions keys
p57 = new.select(["match", "round", "second", "p_57"]).drop_nulls()
print("p_57 non-null rows", p57.height, "of", new.height)
print("p_sprint83 non-null", new["p_sprint83"].null_count(), "p_57 null", new["p_57"].null_count())
print("p_57 range", new["p_57"].min(), new["p_57"].max(), "p_sprint83 range", new["p_sprint83"].min(), new["p_sprint83"].max())
# ---- overlap with holdout B, frozen test, Oct 26-31 keys (match identifiers and round/second only)
used_matches = set(new["match"].unique().to_list())
used_rows = new.select(key)
def key_report(name, matches=None, rows_df=None):
    m_ov = len(used_matches & matches) if matches is not None else None
    r_ov = int(new.filter(pl.col("match").is_in(list(matches))).height) if matches is not None else None
    rows_ov = None
    if rows_df is not None:
        rows_ov = int(used_rows.join(rows_df, on=key, how="inner").height)
    return {"key_set": name, "key_matches_in_set": None if matches is None else len(matches), "used_matches_overlapping": m_ov,
            "used_rows_overlapping_by_match": r_ov, "used_rows_overlapping_by_exact_key": rows_ov}
results = []
hb = json.load(open("G:/pureskill/csds-cs2/data/combined/holdout_22_25_test_keys.json", encoding="utf-8"))
print("holdout B top-level keys:", list(hb.keys()) if isinstance(hb, dict) else type(hb).__name__)
hb_matches = set(hb["match_keys"]) if isinstance(hb, dict) else set(hb)
results.append(key_report("holdout_B_22_25 (data/combined/holdout_22_25_test_keys.json)", hb_matches))
fz = json.load(open(f"{P}/../frozen_test_keys.json", encoding="utf-8"))
fz_matches = set(fz) if isinstance(fz, list) else set(fz.get("match_keys", []))
results.append(key_report("frozen_test (scratchpad frozen_test_keys.json)", fz_matches))
for path in ["C:/Users/<user>/sprint/holdout_oct26_31_keys.parquet", "C:/Users/<user>/sprint/holdout_oct26_31/keys.parquet"]:
    if not os.path.exists(path):
        results.append({"key_set": path, "missing": True}); continue
    cols = pl.read_parquet_schema(path).keys()
    use = [c for c in ["match_key", "round", "second"] if c in cols]
    kd = pl.read_parquet(path, columns=use)
    kd = kd.rename({"match_key": "match"}) if "match_key" in use else kd
    ms = set(kd["match"].unique().to_list())
    rd = kd.select([c for c in key if c in kd.columns]).with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int32)) if all(c in kd.columns for c in key) else None
    rep = key_report(f"{path} (key columns {use})", ms, rd)
    rep["key_rows"] = kd.height
    results.append(rep)
for r in results:
    print(r)
assert all((r.get("used_matches_overlapping") in (0, None)) and (r.get("used_rows_overlapping_by_exact_key") in (0, None)) for r in results), "OVERLAP FOUND"
json.dump({"rows_used": new.height, "matches_used": len(used_matches), "overlap_checks": results}, open(f"{S}/overlap_checks_final.json", "w", encoding="utf-8"), indent=2, default=str)
print("overlap checks written")
