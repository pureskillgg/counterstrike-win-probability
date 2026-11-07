# A2: duplicate handling and header consistency for the 2024-02-g tome; day-01 repaired-file byte check.
import os, sys, hashlib, collections, csv
import polars as pl
import pyarrow.parquet as pq
BASE = r"G:\pureskill\csds-cs2"
TOME = os.path.join(BASE, "tomes", "2024-02-g", "tome", "csds")
hdr = pl.read_parquet(os.path.join(TOME, "header", "dataframe_00000"))
print("header rows", hdr.height, "distinct key", hdr["key"].n_unique(), "distinct match_id", hdr["match_id"].n_unique(), "distinct demo_id", hdr["demo_id"].n_unique(), "distinct sharecode", hdr["sharecode"].n_unique(), flush=True)
hdr = hdr.with_columns(pl.col("key").str.split("/").list.get(3).alias("day"), pl.col("key").str.split("/").list.get(4).alias("mid"))
print("key folder id == match_id:", (hdr["mid"] == hdr["match_id"]).sum(), "of", hdr.height, flush=True)
hdr = hdr.with_columns(pl.col("match_date").str.slice(0, 10).alias("mdate"))
print("match_date day == folder day:", hdr.filter(pl.col("mdate").str.slice(8, 2) == pl.col("day")).height, "of", hdr.height, flush=True)
print("match_date month/year distribution:", hdr.group_by(pl.col("mdate").str.slice(0, 7)).len().sort("mdate").rows(), flush=True)
print("map_name:", hdr.group_by("map_name").len().sort("len", descending=True).rows(), flush=True)
print("build_num:", hdr.group_by("build_num").len().sort("build_num").rows(), flush=True)
print("tick_rate:", hdr.group_by("tick_rate").len().rows(), " match_type:", hdr.group_by("match_type").len().rows(), " platform:", hdr.group_by("platform").len().rows(), flush=True)
print("demo_id values:", hdr["demo_id"].unique().head(3).to_list(), " sharecode unique:", hdr["sharecode"].n_unique(), flush=True)
sig = hdr.with_columns(pl.concat_str([pl.col("match_date"), pl.col("map_name"), pl.col("t_starters_score_final").cast(pl.Utf8), pl.col("ct_starters_score_final").cast(pl.Utf8), pl.col("number_of_points").cast(pl.Utf8)], separator="|").alias("sig"))
dups = sig.group_by("sig").len().filter(pl.col("len") > 1)
print("header signature duplicates (match_date|map|final scores|number_of_points): groups", dups.height, "rows", int(dups["len"].sum()) if dups.height else 0, flush=True)
np_dup = hdr.group_by("number_of_points").len().filter(pl.col("len") > 1)
print("number_of_points duplicated values:", np_dup.height, flush=True)
# day-01 repaired files: the 7 same-size pairs, and sample of 100 add_from_C by sha
import csv as _c
man = list(_c.DictReader(open(os.path.join(BASE, "notes", "spec", "merge_manifest_day01.csv"), encoding="utf-8-sig")))
A_root = r"E:\csds_extracted\csds\2024\02\01"; C_root = r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\01"
G_root = os.path.join(BASE, "merged_2024", "csds", "2024", "02", "01")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
res = collections.Counter(); details = []
for r in man:
    if r["action"] == "skip_duplicate":
        cat = "skip_dup"
    elif "repaired" in r["note"]:
        cat = "repaired_copy_A"
    elif r["action"] == "add_from_C":
        cat = "add_from_C"
    else:
        continue
    g = os.path.join(G_root, r["relative_path"])
    if not os.path.isfile(g):
        res[cat + ":G_missing"] += 1; continue
    src = (A_root if r["source_copy"] == "A" else C_root)
    sp = os.path.join(src, r["relative_path"])
    if not os.path.isfile(sp):
        res[cat + ":src_missing"] += 1; details.append((cat, r["relative_path"], "src_missing")); continue
    if cat == "repaired_copy_A" or cat == "skip_dup":
        other = os.path.join(C_root if r["source_copy"] == "A" else A_root, r["relative_path"])
    if cat == "skip_dup":
        # manifest says C duplicate identical to A; compare A vs C directly
        ap = os.path.join(A_root, r["relative_path"])
        ok = os.path.isfile(ap) and sha(ap) == sha(sp)
        res[cat + (":A==C" if ok else ":A!=C")] += 1
        details.append((cat, r["relative_path"], "A==C" if ok else "A!=C")); continue
    if cat == "repaired_copy_A":
        cp = os.path.join(C_root, r["relative_path"])
        ok_gc = os.path.isfile(cp) and sha(g) == sha(cp)
        res[cat + (":G==C" if ok_gc else ":G!=C_or_C_absent")] += 1
        details.append((cat, r["relative_path"], "G==C" if ok_gc else "G!=C"))
        continue
    ok = sha(g) == sha(sp)
    res[cat + (":G==src" if ok else ":G!=src")] += 1
    details.append((cat, r["relative_path"], "G==src" if ok else "G!=src"))
print("day01 merge byte checks:", dict(res), flush=True)
bad = [d for d in details if "!=" in d[2] or "missing" in d[2]]
print("day01 byte-check exceptions:", len(bad), bad[:10], flush=True)
print("DONE", flush=True)
