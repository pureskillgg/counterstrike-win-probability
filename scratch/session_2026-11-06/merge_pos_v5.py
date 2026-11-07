"""Merge the rebuilt position rows (affected matches) into full v5 chunk files.

For each chunk: keep every row of unaffected matches from the v4 chunk unchanged; replace affected matches with the rebuilt rows.
Checks, before writing:
  - the rebuilt set of (match, round, second) keys equals the v4 set for the affected matches;
  - every NON-target round of an affected match is identical to v4 (all feature columns);
  - the schema is the same as v4.
Writes only under fixes/fix332/v5_build/pos_all_seconds_v5/ (a new folder). v4 chunks are read only.
"""
import glob
import json
import os
import shutil
import polars as pl

BUILDER = "/home/sagemaker-user/adam/builder/"
OLD = BUILDER + "final/pos_all_seconds/"
PART = BUILDER + "fixes/fix332/v5_build/pos_partial/"
NEW = BUILDER + "fixes/fix332/v5_build/pos_all_seconds_v5/"
os.makedirs(NEW, exist_ok=True)
aff = set(l.strip() for l in open(BUILDER + "fixes/fix332/v5_build/affected_matches.txt") if l.strip())
ovr = pl.read_parquet(BUILDER + "fixes/fix332/v5_build/ovr_all.parquet").select("match_key", "round").unique()
target = set(zip(ovr["match_key"].to_list(), ovr["round"].to_list()))

report = {}
for old_path in sorted(glob.glob(OLD + "bucket_c*.parquet")):
    name = os.path.basename(old_path)
    c = name[len("bucket_"):-len(".parquet")]
    part_path = PART + name
    if not os.path.exists(part_path):
        shutil.copy2(old_path, NEW + name)
        report[c] = {"action": "copied unchanged"}
        continue
    old = pl.read_parquet(old_path)
    new = pl.read_parquet(part_path)
    assert old.schema == new.schema, (c, old.schema, new.schema)
    keycols = ["match_key", "round", "second"]
    old_aff = old.filter(pl.col("match_key").is_in(list(aff)))
    k_old = set(map(tuple, old_aff.select(keycols).unique().iter_rows()))
    k_new = set(map(tuple, new.select(keycols).unique().iter_rows()))
    assert k_old == k_new, (c, len(k_old), len(k_new), len(k_old ^ k_new))
    # non-target rounds must be identical to v4
    feat = [x for x in old.columns if x not in keycols]
    j = old_aff.join(new, on=keycols, how="inner", suffix="_new")
    nontarget = j.filter(~pl.struct(["match_key", "round"]).map_elements(
        lambda s: (s["match_key"], s["round"]) in target, return_dtype=pl.Boolean))
    diff_nontarget = 0
    for f in feat:
        a, b = nontarget[f], nontarget[f + "_new"]
        d = ((a != b) & ~(a.is_null() & b.is_null())).sum()
        diff_nontarget += int(d or 0)
    assert diff_nontarget == 0, (c, "non-target rounds changed", diff_nontarget)
    changed_target = 0
    tgt = j.filter(pl.struct(["match_key", "round"]).map_elements(
        lambda s: (s["match_key"], s["round"]) in target, return_dtype=pl.Boolean))
    for f in feat:
        a, b = tgt[f], tgt[f + "_new"]
        changed_target += int(((a != b) & ~(a.is_null() & b.is_null())).sum() or 0)
    keep = old.filter(~pl.col("match_key").is_in(list(aff)))
    out = pl.concat([keep, new.select(old.columns)], how="vertical")
    out.write_parquet(NEW + name + ".tmp")
    os.replace(NEW + name + ".tmp", NEW + name)
    report[c] = {"action": "merged", "affected_rows": old_aff.height, "rebuilt_rows": new.height,
                 "nontarget_rows_changed": diff_nontarget, "target_feature_cells_changed": changed_target,
                 "rows_out": out.height}
    print(c, report[c], flush=True)

json.dump(report, open(BUILDER + "fixes/fix332/v5_build/merge_pos_v5_report.json", "w"), indent=1)
print("MERGE_DONE", len(report), flush=True)
