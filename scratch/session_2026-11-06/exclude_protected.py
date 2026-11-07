import json
import polars as pl

V4 = "/home/sagemaker-user/adam/builder/final/export_clean_v4/"
st = pl.read_csv(V4 + "match_loadout_status.csv", infer_schema_length=0)
print("columns:", st.columns)
batch_col = "v4_batch"
prot = st.filter(pl.col(batch_col) == "v4_s3_protected")
keys = prot["match_key"].to_list()
split_counts = prot.group_by("split").len().sort("split").to_dicts() if "split" in prot.columns else []
print("protected matches:", len(keys), "unique:", len(set(keys)))
print("split counts:", split_counts)
out = {
    "purpose": "Default training exclusion. Matches in batch v4_s3_protected (dated 21 to 25 Oct 2026). Owner confirmation is open in export_clean_v4/README.json (owner_checks_open). Use this list to skip the batch in any loader until the owner confirms in writing.",
    "batch": "v4_s3_protected",
    "n_matches": len(set(keys)),
    "split_counts": split_counts,
    "lift_when": "Written owner confirmation filed (Meridian pin 425fca91 cites the chat confirmation only).",
    "match_keys": sorted(set(keys)),
}
with open(V4 + "TRAINING_EXCLUDE.json", "w") as f:
    json.dump(out, f, indent=1)
print("written", V4 + "TRAINING_EXCLUDE.json", len(out["match_keys"]), "keys")
