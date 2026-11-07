import json
import collections
import pyarrow.parquet as pq
import pyarrow.compute as pc

MANIFEST = r"C:\Users\<user>\AppData\Local\Temp\claude\44708acea14da7d4\047ee5fa-b35b-4b25-97b1-beaf96834ff2\scratchpad\split_manifest.json"
HDR = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds\header\dataframe_00000"

m = json.load(open(MANIFEST, encoding="utf-8"))
assign = m["assign"]
label_by_key = assign  # manifest keys are csds/YYYY/MM/DD/<uuid>/csds

f = pq.ParquetFile(HDR)
print("rows:", f.metadata.num_rows)
print("schema types for PII-relevant cols:")
s = f.schema_arrow
for n in ["client_name", "server_name", "key", "match_id", "demo_id", "sharecode", "match_date",
          "platform", "is_gotv_recording", "unique_steamids", "t_starters_avg_rank", "match_type"]:
    print("  ", n, s.field(n).type if n in s.names else "MISSING")

# key shape only (no key values printed)
keys = pq.read_table(HDR, columns=["key"]).column("key").to_pylist()
shape = collections.Counter()
import re
for k in keys:
    if k is None:
        shape["None"] += 1
    elif k in label_by_key:
        shape["exact_manifest_path"] += 1
    elif re.fullmatch(r"csds/\d{4}/\d{2}/\d{2}/[0-9a-f\-]{36}/csds", k):
        shape["path_form_not_in_manifest"] += 1
    elif re.fullmatch(r"[0-9a-f\-]{36}", k):
        shape["bare_uuid"] += 1
    else:
        shape["other_form_len%d" % len(k)] += 1
print("key shape counts:", dict(shape))

labels = collections.Counter()
for k in keys:
    labels[label_by_key.get(k, "NOT_IN_MANIFEST")] += 1
print("split label counts for header rows:", dict(labels))

# match_date: prefix shape only
dates = pq.read_table(HDR, columns=["match_date"]).column("match_date").to_pylist()
ymd = collections.Counter()
for d in dates:
    if d is None:
        ymd["None"] += 1
    else:
        ymd[d[:7]] += 1
print("match_date month counts:", dict(sorted(ymd.items())))
