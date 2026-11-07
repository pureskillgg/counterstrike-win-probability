#!/bin/bash
D=~/adam/builder/fixes/loadout_chunk_locate
cd "$D" || exit 1
~/venv-dsdk/bin/python - <<'PY'
import polars as pl, json
cg = pl.read_csv("chunk_by_reason_counts.csv")
t = cg.filter(pl.col("test_split_coverage_only") > 0)
print("chunks with test keys:", t.height, "first", t["chunk"][0], "last", t["chunk"][-1], "sum test keys in chunks:", int(t["test_split_coverage_only"].sum()))
p = cg.filter(pl.col("protected_date_2026-10-21_onward") > 0)
print("chunks with protected keys:", p.select(["chunk", "protected_date_2026-10-21_onward"]).rows(), "sum", int(p["protected_date_2026-10-21_onward"].sum()))
s = json.load(open("summary.json"))
print("status-vs-info disagree keys:", s["status_vs_info_chunk_count_disagree_keys"], "| export keys in no chunk:", s["export_keys_in_no_chunk_status"])
PY
echo "--- disk ---"; df -h ~ | tail -1
echo "--- v4 exists? ---"; ls -d ~/adam/builder/final/export_clean_v4 2>&1
echo "--- v3 array sizes ---"; ls -l ~/adam/builder/final/export_clean_v3_loadout/loadout_step_i16.npy ~/adam/builder/final/export_clean_v3_loadout/loadout_equip_i32.npy
