# Assembles the v4 status CSV and built-key list. Reads the v3 status (unchanged) and the two v4 coverage outputs.
import json
import polars as pl

V3 = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout/"
V4 = "/home/sagemaker-user/adam/builder/final/export_clean_v4/"
V4S = V4 + "scripts/"

st = pl.read_csv(V3 + "match_loadout_status.csv", schema_overrides={"match_key": pl.String})
keys = json.load(open(V3 + "match_keys.json"))
assert st["match_key"].to_list() == keys and st.height == 57811
s3_t = set(json.load(open(V4S + "s3_target_v4.json")))
loc_t = set(json.load(open(V4S + "local_target_v4.json")))
assert len(s3_t) == 7361 and len(loc_t) == 1164 and not (s3_t & loc_t)

cs = pl.read_csv(V4S + "s3_cov_v4.csv", schema_overrides={"match_key": pl.String}).filter(pl.col("set") == "target")
assert cs["match_key"].n_unique() == cs.height
cl = pl.read_parquet(V4S + "coverage_local_v4.parquet")
assert cl["match_key"].n_unique() == cl.height
COV = ["rows", "rounds_with_rows", "steps_with_rows", "freeze_rows"]
cov = {}
for r in cs.iter_rows(named=True):
    cov[r["match_key"]] = ("s3_channel", {c: r[c] for c in COV})
for r in cl.iter_rows(named=True):
    assert r["match_key"] not in cov
    cov[r["match_key"]] = ("local_1hz", {c: r[c] for c in COV})
print("coverage rows: s3", cs.height, "local", cl.height, "| targets s3", len(s3_t), "local", len(loc_t))

batch_n = {"v4_s3_test": 0, "v4_s3_protected": 0, "v4_local_test": 0, "target_no_rows": 0}
recs = []
for r in st.iter_rows(named=True):
    rec = dict(r)
    rec["v4_build_scope"] = None
    rec["v4_batch"] = None
    mk = r["match_key"]
    if mk in s3_t or mk in loc_t:
        assert r["build_status"] == "not_built"
        if mk not in cov:
            batch_n["target_no_rows"] += 1
        else:
            kind, cv = cov[mk]
            rec.update(cv)
            if kind == "s3_channel":
                rec["build_status"] = "s3_channel"
                if r["split"] == 2:
                    rec["reason"] = "test_split_coverage_only_v4_s3_channel"
                    rec["v4_batch"] = "v4_s3_test"
                    rec["v4_build_scope"] = "test_coverage_only"
                else:
                    rec["reason"] = "protected_date_2026-10-21_onward_v4_s3_channel"
                    rec["v4_batch"] = "v4_s3_protected"
                    rec["v4_build_scope"] = "train_val_protected_date_owner_check_open"
            else:
                rec["build_status"] = "local_1hz"
                rec["src"] = "local_1hz"
                rec["reason"] = "test_split_coverage_only_v4_local_1hz"
                rec["v4_batch"] = "v4_local_test"
                rec["v4_build_scope"] = "test_coverage_only"
            batch_n[rec["v4_batch"]] += 1
    recs.append(rec)
out = pl.from_dicts(recs, infer_schema_length=None).select(list(st.columns) + ["v4_build_scope", "v4_batch"])
out.write_csv(V4 + "match_loadout_status.csv")
print("batch counts", batch_n)
status_counts = out.group_by(["build_status", "reason"]).agg(pl.len().alias("matches")).sort(["build_status", "reason"])
print(status_counts)
built = out.filter(pl.col("build_status").is_in(["s3_channel", "local_1hz"]))["match_key"].to_list()
json.dump(built, open(V4 + "built_match_keys.json", "w"))
not_built = out.filter(pl.col("build_status") == "not_built").height
print("built matches v4", len(built), "| not_built", not_built)
summary = {"v3_built": 49286, "v4_built": len(built), "v4_not_built": not_built, "batch_counts": batch_n,
           "status_counts": status_counts.to_dicts()}
json.dump(summary, open(V4S + "assemble_summary_v4.json", "w"), indent=1, default=str)
