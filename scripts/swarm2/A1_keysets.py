# A1 audit: header key count, channel keysets vs header, page/row counts per channel.
# Reads only metadata + single key columns (column projection), one channel at a time.
import glob, os, sys, json, hashlib, time
import pyarrow.parquet as pq
import pyarrow.compute as pc
T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
t0 = time.time()

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# 1. header
hdr_files = sorted(glob.glob(os.path.join(T, "header", "dataframe_*")))
hdr_tab = pq.read_table(hdr_files[0], columns=["key", "match_id", "match_date", "t_starters_score_final", "ct_starters_score_final", "max_rounds"])
print("header dataframe files:", len(hdr_files), "rows:", hdr_tab.num_rows)
keys_col = hdr_tab.column("key").to_pylist()
print("header key rows:", len(keys_col), "unique keys:", len(set(keys_col)), "null keys:", sum(1 for k in keys_col if k is None))
hdr_keys = set(keys_col)
mid = hdr_tab.column("match_id").to_pylist()
print("header match_id unique:", len(set(mid)), "null:", sum(1 for m in mid if m is None))
hk_keyset = pq.read_table(os.path.join(T, "header", "keyset_00000"))
kv = hk_keyset.column(0).to_pylist()
print("header keyset rows:", len(kv), "unique:", len(set(kv)), "equal to header keys:", set(kv) == hdr_keys)
print("header keyset sample:", kv[:2])
print("header key sample:", keys_col[:2])
print("header sha256 dataframe:", sha(hdr_files[0]))
print("header sha256 tome:", sha(os.path.join(T, "header", "tome")))

# 2. channels
chans = sorted(d for d in os.listdir(T) if d != "header" and not d.startswith("."))
print("channel folders:", len(chans))
report = []
for c in chans:
    folder = os.path.join(T, c)
    dfs = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
    kss = sorted(glob.glob(os.path.join(folder, "keyset_*")))
    # page numbering check
    dnums = [os.path.basename(p).split("_")[1] for p in dfs]
    knums = [os.path.basename(p).split("_")[1] for p in kss]
    contiguous = dnums == knums and dnums == [f"{i:05d}" for i in range(len(dnums))]
    schema_names = pq.read_schema(dfs[0]).names
    mk = "match_key" if "match_key" in schema_names else ("key" if "key" in schema_names else None)
    rows_total = 0
    union_keys = set()
    page_detail = []
    per_page_match_ok = True
    dup_across_pages = 0
    for i, (dp, kp) in enumerate(zip(dfs, kss)):
        md = pq.ParquetFile(dp).metadata
        nrows = md.num_rows
        rows_total += nrows
        kset_vals = pq.read_table(kp).column(0).to_pylist()
        pk = set(kset_vals)
        if mk is not None:
            pg_keys = pq.read_table(dp, columns=[mk]).column(mk).to_pylist()
            pgs = set(pg_keys)
            ok = pgs <= pk and pk <= hdr_keys
            if pgs != pk:
                per_page_match_ok = False
            page_detail.append((i, nrows, len(kset_vals), len(pk), len(pgs), ok))
            dup_across_pages += len(pk & union_keys)
            union_keys |= pk
        else:
            page_detail.append((i, nrows, len(kset_vals), len(pk), None, None))
            union_keys |= pk
    keyset_equal = union_keys == hdr_keys
    # nested header copy check
    nested = os.path.join(folder, "header", "dataframe_00000")
    nested_ok = None
    if os.path.exists(nested):
        nested_ok = sha(nested) == sha(hdr_files[0])
    report.append(dict(channel=c, pages=len(dfs), contiguous=contiguous, match_key_col=mk,
                       rows=rows_total, union_keyset=len(union_keys), keyset_equal_header=keyset_equal,
                       page_keyset_matches_page_data=per_page_match_ok, cross_page_keyset_overlap=dup_across_pages,
                       nested_header_dataframe_equal=nested_ok, schema=schema_names))
    print(json.dumps({k: v for k, v in report[-1].items() if k != "schema"}), flush=True)
    if not keyset_equal or not contiguous or per_page_match_ok is False or dup_across_pages or nested_ok is False:
        print("  page detail:", page_detail)
json.dump(report, open(r"G:\pureskill\csds-cs2\logs\swarm2\A1_keysets.json", "w"), indent=1, default=str)
print("elapsed s", round(time.time() - t0, 1))
