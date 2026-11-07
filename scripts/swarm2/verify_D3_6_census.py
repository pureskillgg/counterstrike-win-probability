"""D3-6 independent verification, part 1 (read-only).

Recomputes, without importing or reusing D3_census_e_coverage.py or D3_coverage_gaps.py:
  1. E: tail-copy coverage of all header matches in the 2024-02-g tome, classified by file presence AND size
     (the claimed census only checks name presence, so zero-byte or non-file entries would count as present).
  2. Day-01 provenance: E: copy A (E:\\csds_extracted\\csds\\2024\\02\\01), E: tail copy C, the day-01 merge
     manifest (notes\\spec\\merge_manifest_day01.csv) and the merged_2024 folder on G: that feeds the -g tome.
  3. The RAR salvage index (E:\\csds_rar5_tail_salvage_20261011.tsv) entries for the day-01 keys.
  4. Key-day vs header match_date (UTC) consistency.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D3_6_census.py
"""
import collections
import csv
import glob
import os
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
E_TAIL = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds")
E_A = Path(r"E:\csds_extracted\csds")
MERGED = ROOT / "merged_2024" / "csds"
MANIFEST = ROOT / "notes" / "spec" / "merge_manifest_day01.csv"
TSV = Path(r"E:\csds_rar5_tail_salvage_20261011.tsv")
CH4 = ["round_end", "round_state", "bomb_state", "bomb_action"]


def g_read(folder_name, cols):
    files = sorted(glob.glob(str(TOME / folder_name / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def scan_dir(base):
    """None if the directory is missing, else {name: (is_dir, size_bytes)}."""
    if not os.path.isdir(base):
        return None
    out = {}
    with os.scandir(base) as it:
        for e in it:
            try:
                is_dir = e.is_dir(follow_symlinks=False)
                size = -1 if is_dir else e.stat(follow_symlinks=False).st_size
            except OSError:
                is_dir, size = None, -1
            out[e.name] = (is_dir, size)
    return out


def classify(ents):
    """File-based classification. Returns (status, name_only_status)."""
    if ents is None:
        return "dir_missing", "dir_missing"
    if not ents:
        return "dir_empty", "dir_empty"
    present = [c for c in CH4 if c in ents and ents[c][0] is False]
    nonzero = [c for c in present if ents[c][1] > 0]
    named = [c for c in CH4 if c in ents]
    if len(nonzero) == 4:
        st = "all4"
    elif nonzero:
        st = f"partial({len(nonzero)}/4)"
    elif present:
        st = "zero_byte_only"
    else:
        st = "no_compare_channels"
    name_st = {4: "all4", 0: "no_compare_channels"}.get(len(named), f"partial({len(named)}/4)")
    return st, name_st


# ---------------------------------------------------------------- 4. header and key-day consistency
H = g_read("header", ["key", "match_date"])
keys = H["key"].to_list()
print(f"header rows {H.height}; unique keys {H['key'].n_unique()}")
bad_fmt = [k for k in keys if len(k.split("/")) != 6 or k.split("/")[0] != "csds"]
print(f"keys not in csds/YYYY/MM/DD/uuid/csds form: {len(bad_fmt)}")
day_of = {k: k.split("/")[3] for k in keys}
print("header matches per key-day:", dict(sorted(collections.Counter(day_of.values()).items())))
mdate = dict(H.select(["key", "match_date"]).iter_rows())
kd_mis = collections.Counter()
kd_mis_ex = []
for k in keys:
    md = mdate.get(k)
    if md is None:
        kd_mis[(day_of[k], "NULL")] += 1
        continue
    if md[:10] != "2024-02-" + day_of[k]:
        kd_mis[(day_of[k], md[:10])] += 1
        if len(kd_mis_ex) < 5:
            kd_mis_ex.append((k, md))
print(f"key-day vs UTC match_date mismatches: {sum(kd_mis.values())}; by (key day, match date): {dict(kd_mis)}")
print("  sample:", kd_mis_ex)

# ---------------------------------------------------------------- 1. E: tail census (file-based and name-based)
tail_ents = {}
tail_status = {}
tail_name_status = {}
for k in keys:
    p = k.split("/")
    ents = scan_dir(str(E_TAIL / p[1] / p[2] / p[3] / p[4]))
    tail_ents[k] = ents
    tail_status[k], tail_name_status[k] = classify(ents)
print("\nE: tail coverage, file-based (present, non-zero size), all header matches:")
for st, n in sorted(collections.Counter(tail_status.values()).items(), key=lambda x: -x[1]):
    print(f"  {st:<22} {n}")
diff_name_vs_file = [k for k in keys if tail_status[k] != tail_name_status[k]]
print(f"matches where name-only and file-based classification differ: {len(diff_name_vs_file)}")
print("name-only (claim-style) coverage:", dict(collections.Counter(tail_name_status.values())))

by_day = collections.defaultdict(collections.Counter)
for k in keys:
    by_day[day_of[k]][tail_status[k]] += 1
print("E: tail coverage by key-day (file-based):")
for d in sorted(by_day):
    print(f"  {d}: " + ", ".join(f"{s}={n}" for s, n in sorted(by_day[d].items())))

# ---------------------------------------------------------------- 2. day-01 zoom
d01 = [k for k in keys if day_of[k] == "01"]
d01_uuid = {k: k.split("/")[4] for k in d01}
print(f"\nday-01 header matches: {len(d01)}")
no_tail = [k for k in d01 if tail_status[k] in ("dir_empty", "no_compare_channels", "zero_byte_only")]
print(f"day-01 with no non-zero compare-channel file in E: tail copy: {len(no_tail)} "
      f"(dir_empty {sum(tail_status[k] == 'dir_empty' for k in no_tail)}, "
      f"no_compare {sum(tail_status[k] == 'no_compare_channels' for k in no_tail)}, "
      f"zero_byte_only {sum(tail_status[k] == 'zero_byte_only' for k in no_tail)})")
print("day-01 tail status:", dict(collections.Counter(tail_status[k] for k in d01)))
print("day-01 partial tail matches:", [(d01_uuid[k], tail_ents[k] and sorted(n for n in tail_ents[k])) for k in d01
                                       if tail_status[k].startswith("partial")])

a_ents = {k: scan_dir(str(E_A / "2024" / "02" / "01" / d01_uuid[k])) for k in d01}
a_status = {k: classify(a_ents[k])[0] for k in d01}
print("\ncopy A (E:\\csds_extracted) status, all day-01 header matches:", dict(collections.Counter(a_status.values())))
print("copy A status for the tail-empty day-01 matches:", dict(collections.Counter(a_status[k] for k in no_tail)))
a_dir = scan_dir(str(E_A / "2024" / "02" / "01"))
a_uuids = set(a_dir.keys()) if a_dir else set()
d01_set = set(d01_uuid.values())
print(f"copy A day-01 folder entries: {len(a_uuids)}; header day-01 uuids present in copy A listing: "
      f"{len(a_uuids & d01_set)}; header day-01 absent from copy A listing: {len(d01_set - a_uuids)}; "
      f"copy A entries not in header day-01: {len(a_uuids - d01_set)}")
for k in no_tail:
    if a_status[k] == "dir_missing" or (a_status[k] != "all4"):
        pass
a_no_tail_round_end = [k for k in no_tail if a_ents[k] and "round_end" in a_ents[k] and a_ents[k]["round_end"][1] > 0]
print(f"tail-empty day-01 matches with a non-zero round_end file in copy A: {len(a_no_tail_round_end)} of {len(no_tail)}")
print(f"tail-empty day-01 matches with copy A status all4: {sum(a_status[k] == 'all4' for k in no_tail)}")

# ---------------------------------------------------------------- 3. merge manifest and merged_2024 (G:)
rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig", newline="")))
print(f"\nmerge manifest rows: {len(rows)}")
print("  actions:", dict(collections.Counter(r["action"] for r in rows)))
print("  source_copy:", dict(collections.Counter(r["source_copy"] for r in rows)))
depth = collections.Counter(r["relative_path"].count("/") for r in rows)
print("  relative_path separator counts:", dict(depth))
man_by = collections.defaultdict(list)
for r in rows:
    u, _, ch = r["relative_path"].partition("/")
    man_by[u].append((ch, r))
man_uuids = set(man_by)
print(f"manifest uuids {len(man_uuids)}; header day-01 uuids {len(d01_set)}; "
      f"manifest-only {len(man_uuids - d01_set)}; header-only {len(d01_set - man_uuids)}")

# manifest sizes vs the copy the manifest names
size_mis = collections.Counter()
size_ex = []
for u, lst in man_by.items():
    for ch, r in lst:
        if r["action"] == "skip_duplicate":
            size_mis["skip_duplicate"] += 1
            continue
        src_root = E_A if r["source_copy"] == "A" else E_TAIL
        p = src_root / "2024" / "02" / "01" / u / ch
        try:
            sz = os.stat(p).st_size
        except OSError:
            size_mis[f"missing_in_{r['source_copy']}"] += 1
            continue
        if sz == int(r["size_bytes"]):
            size_mis["size_ok"] += 1
        else:
            size_mis["size_mismatch"] += 1
            if len(size_ex) < 3:
                size_ex.append((u, ch, sz, r["size_bytes"], r["source_copy"]))
print("manifest rows vs named source copy on E::", dict(size_mis), "examples:", size_ex)

# merged_2024 (G:) sizes vs manifest
mg_mis = collections.Counter()
for u, lst in man_by.items():
    for ch, r in lst:
        if r["action"] == "skip_duplicate":
            continue
        p = MERGED / "2024" / "02" / "01" / u / ch
        try:
            sz = os.stat(p).st_size
        except OSError:
            mg_mis["missing_in_merged"] += 1
            continue
        mg_mis["ok" if sz == int(r["size_bytes"]) else "size_mismatch"] += 1
print("merged_2024 (G:) sizes vs manifest:", dict(mg_mis))

rc_src = collections.Counter()
for k in no_tail:
    lst = man_by.get(d01_uuid[k], [])
    rc = [r for ch, r in lst if ch == "round_end"]
    if not rc:
        rc_src["no_round_end_row_in_manifest"] += 1
    else:
        rc_src[f"round_end_source_{rc[0]['source_copy']}_{rc[0]['action']}"] += 1
print("tail-empty day-01 matches: round_end provenance in manifest:", dict(rc_src))
rc_src_all = collections.Counter()
for k in d01:
    lst = man_by.get(d01_uuid[k], [])
    rc = [r for ch, r in lst if ch == "round_end"]
    rc_src_all[rc[0]["source_copy"] if rc else "none"] += 1
print("all day-01 matches: round_end source copy in manifest:", dict(rc_src_all))

# ---------------------------------------------------------------- 5. RAR salvage index
idx_present = collections.defaultdict(dict)
idx_days = collections.defaultdict(set)
other_year = collections.Counter()
n_lines = 0
with open(TSV, encoding="utf-8", errors="replace", newline="") as f:
    f.readline()
    for line in f:
        n_lines += 1
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) < 4:
            continue
        segs = parts[3].split("/")
        if len(segs) != 6 or segs[0] != "csds":
            other_year["bad_shape"] += 1
            continue
        if segs[1] != "2024" or segs[2] != "02":
            other_year[f"{segs[1]}/{segs[2]}"] += 1
            continue
        idx_days[segs[3]].add(segs[4])
        idx_present[segs[4]][segs[5]] = int(parts[2])
print(f"\nRAR index lines {n_lines}; non-2024-02 or bad entries: {dict(other_year)}")
print("index unique match uuids per 2024-02 day:", {d: len(s) for d, s in sorted(idx_days.items())})
d01_in_idx = [k for k in d01 if d01_uuid[k] in idx_present]
no_tail_in_idx = [k for k in no_tail if d01_uuid[k] in idx_present]
no_tail_idx_rc = [k for k in no_tail if "round_end" in idx_present.get(d01_uuid[k], {})]
print(f"day-01 header matches with any index entry: {len(d01_in_idx)} of {len(d01)}")
print(f"tail-empty day-01 matches with any index entry: {len(no_tail_in_idx)} of {len(no_tail)}; "
      f"with round_end entry: {len(no_tail_idx_rc)}")
if no_tail_idx_rc:
    ex = no_tail_idx_rc[0]
    print("  example tail-empty uuid in index:", d01_uuid[ex], dict(idx_present[d01_uuid[ex]]))
print("index entries per day-01 uuid (all header day-01):", dict(collections.Counter(len(idx_present.get(d01_uuid[k], {})) for k in d01)))
print("\ndone part 1")
