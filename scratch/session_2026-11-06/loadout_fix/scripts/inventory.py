import io, json, struct, hashlib, collections, sys, time
import boto3
import pyarrow.parquet as pq
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor

BUCKET = "pureskillgg-csgo-production-sagemaker-0"
PREFIX = "adam/tomes/chunks"
F = "/home/sagemaker-user/adam/builder/final"
s3 = boto3.client("s3", config=Config(max_pool_connections=64, retries={"max_attempts": 8, "mode": "standard"}))
EX = ThreadPoolExecutor(24)
T0 = time.time()


def log(*a):
    print("[%5.0fs]" % (time.time() - T0), *a, file=sys.stderr, flush=True)


def list_objs(prefix):
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            out.append((o["Key"], o["Size"]))
    return out


def body(key, rng=None):
    kw = {"Bucket": BUCKET, "Key": key}
    if rng:
        kw["Range"] = rng
    return s3.get_object(**kw)["Body"].read()


def keyset_keys(key):
    t = pq.read_table(io.BytesIO(body(key)))
    col = "_" if "_" in t.column_names else t.column_names[0]
    return t.column(col).to_pylist(), t.column_names


def footer(key, size):
    tail = body(key, f"bytes={size-8}-{size-1}")
    if tail[4:] != b"PAR1":
        raise ValueError("not parquet: " + key)
    flen = struct.unpack("<I", tail[:4])[0]
    start = size - 8 - flen
    part = body(key, f"bytes={start}-{size-1}")
    buf = b"PAR1" + b"\0" * (start - 4) + part  # zero gap; only the footer is parsed
    md = pq.read_metadata(io.BytesIO(buf))
    sch = pq.ParquetFile(io.BytesIO(buf)).schema_arrow
    return {"rows": md.num_rows, "row_groups": md.num_row_groups,
            "cols": [[f.name, str(f.type)] for f in sch], "created_by": md.created_by}


def sha(x):
    return hashlib.sha1(json.dumps(x, sort_keys=True).encode()).hexdigest()[:10]


def jget(key):
    try:
        return json.loads(body(key))
    except Exception as e:
        return {"error": str(e)[:160]}


chunks = []
for i in range(68):
    c = f"c{i:03d}"
    base = f"{PREFIX}/{c}/"
    objs = list_objs(base + "tome/csds/basic_player_status.")
    folders = collections.defaultdict(list)
    for k, s in objs:
        d, n = k.rsplit("/", 1)
        folders[d].append((n, k, s))
    info = {"chunk": i, "n_folders": len(folders), "folders": sorted(folders)}
    if len(folders) != 1:
        info["flag"] = "folder count != 1"
        chunks.append(info)
        continue
    folder = list(folders)[0]
    files = folders[folder]
    dfs = sorted([x for x in files if x[0].startswith("dataframe_")])
    kss = sorted([x for x in files if x[0].startswith("keyset_")])
    other = sorted(x[0] for x in files if not x[0].startswith(("dataframe_", "keyset_")))
    info.update({
        "suffix": folder.rsplit("/", 1)[1].split(".", 1)[1],
        "n_df": len(dfs), "n_ks": len(kss), "other_files": other,
        "df_bytes": sum(s for _, _, s in dfs), "ks_bytes": sum(s for _, _, s in kss),
        "df_names_match_ks": [x[0][-5:] for x in dfs] == [x[0][-5:] for x in kss],
        "status_done": jget(base + "STATUS_DONE.json"),
        "done": jget(base + "DONE.json"),
    })
    ksr = list(EX.map(lambda x: keyset_keys(x[1]), kss))
    info["ks_cols"] = sorted(set(tuple(c_) for _, c_ in ksr))
    pages = []
    for (n, k, s), (vals, _) in zip(kss, ksr):
        pages.append({"ks": n, "keys": vals})
    fo = list(EX.map(lambda x: footer(x[1], x[2]), dfs))
    for p, (n, k, s), f in zip(pages, dfs, fo):
        p["df"] = n
        p["df_bytes"] = s
        p["rows"] = f["rows"]
        p["row_groups"] = f["row_groups"]
    info["cols"] = fo[0]["cols"] if fo else None
    info["col_sig"] = sha(fo[0]["cols"]) if fo else None
    info["schema_sigs_all_pages"] = sorted(set(sha(f["cols"]) for f in fo))
    info["created_by"] = sorted(set(f["created_by"] for f in fo))
    info["rows_sum"] = sum(f["rows"] for f in fo)
    info["keys_per_page"] = [len(p["keys"]) for p in pages]
    info["keys_total"] = sum(len(p["keys"]) for p in pages)
    info["keys_distinct"] = len(set(k for p in pages for k in p["keys"]))
    info["pages"] = pages
    chunks.append(info)
    log(c, info["suffix"], "df", info["n_df"], "ks", info["n_ks"], "keys", info["keys_total"], "rows", info["rows_sum"], "sig", info["col_sig"])

# export side: bucket-sourced keys (clean_matches src == bucket, within export_clean_v2)
ek = set(json.load(open(F + "/export_clean_v2/match_keys.json")))
cm = pq.read_table(F + "/clean_matches.parquet", columns=["match_key", "src"]).to_pylist()
src = {r["match_key"]: r["src"] for r in cm}
export_all = ek
export_bucket = {k for k in ek if src.get(k) == "bucket"}
export_laptop = {k for k in ek if src.get(k) == "laptop"}
log("export keys", len(export_all), "bucket", len(export_bucket), "laptop", len(export_laptop), "no src", len(export_all - set(src)))

chan_count = collections.Counter()
for ch in chunks:
    for p in ch.get("pages", []):
        for k in p["keys"]:
            chan_count[k] += 1
chan_keys = set(chan_count)
covered = export_bucket & chan_keys
missing = export_bucket - chan_keys
chan_only = chan_keys - export_bucket
chan_in_export_laptop = chan_only & export_laptop
chan_not_export = chan_only - export_laptop
dup_across = sum(1 for k, v in chan_count.items() if v > 1)

tot_df = 0
need_whole = 0
need_prop = 0.0
pages_need = 0
for ch in chunks:
    ch_cov = 0
    ch_pages_need = 0
    ch_need_bytes = 0
    for p in ch.get("pages", []):
        cov_p = sum(1 for k in p["keys"] if k in export_bucket)
        ch_cov += cov_p
        tot_df += p["df_bytes"]
        if cov_p:
            ch_pages_need += 1
            ch_need_bytes += p["df_bytes"]
            need_prop += p["df_bytes"] * cov_p / max(1, len(p["keys"]))
    need_whole += ch_need_bytes
    pages_need += ch_pages_need
    ch["covered_export_bucket"] = ch_cov
    ch["pages_needed"] = ch_pages_need
    ch["need_whole_page_bytes"] = ch_need_bytes


def version(ch):
    sd = ch.get("status_done", {})
    if "error" in sd:
        return "no summary"
    if "reduce_secs" in sd or "reduce_failed" in sd:
        return "v3-style summary"
    return "v4-style summary"


by_ver = collections.defaultdict(collections.Counter)
for ch in chunks:
    by_ver[version(ch)][ch.get("col_sig")] += 1
    ch["version"] = version(ch)

summary = {
    "chunks": len(chunks),
    "channel_keys_total_listed": sum(ch.get("keys_total", 0) for ch in chunks),
    "channel_keys_distinct": len(chan_keys),
    "channel_keys_listed_more_than_once": dup_across,
    "export_keys": len(export_all),
    "export_bucket_keys": len(export_bucket),
    "export_laptop_keys": len(export_laptop),
    "export_bucket_covered_by_channel": len(covered),
    "export_bucket_missing_from_channel": len(missing),
    "channel_only_keys_in_export_laptop": len(chan_in_export_laptop),
    "channel_keys_not_in_export": len(chan_not_export),
    "df_bytes_all_chunks": tot_df,
    "download_bytes_whole_pages_needed": need_whole,
    "download_bytes_proportional_estimate": round(need_prop),
    "pages_needed": pages_need,
    "by_version_col_sig": {k: dict(v) for k, v in by_ver.items()},
    "missing_examples": sorted(missing)[:5],
    "covered_by_month": dict(sorted(collections.Counter(k[4:11] for k in covered).items())),
    "missing_by_month": dict(sorted(collections.Counter(k[4:11] for k in missing).items())),
    "export_bucket_by_month": dict(sorted(collections.Counter(k[4:11] for k in export_bucket).items())),
}
print("=== SUMMARY ===")
print(json.dumps(summary, indent=1))
print("=== PER CHUNK ===")
for ch in chunks:
    sd = ch.get("status_done", {})
    dn = ch.get("done", {})
    if not isinstance(sd, dict):
        sd = {}
    print(json.dumps({
        "c": ch["chunk"], "suffix": ch.get("suffix"), "folders": ch.get("n_folders"), "df": ch.get("n_df"), "ks": ch.get("n_ks"),
        "other": ch.get("other_files"), "df_MB": round(ch.get("df_bytes", 0) / 1e6, 2), "ks_KB": round(ch.get("ks_bytes", 0) / 1e3, 1),
        "names_align": ch.get("df_names_match_ks"), "keys": ch.get("keys_total"), "keys_distinct": ch.get("keys_distinct"),
        "rows_sum": ch.get("rows_sum"), "status_rows": sd.get("reduced_rows"), "status_matches": sd.get("matches"),
        "status_missing": sd.get("missing"), "status_failed": sd.get("failed"), "done_matches": dn.get("matches"),
        "version": ch.get("version"), "col_sig": ch.get("col_sig"), "sigs_all_pages": ch.get("schema_sigs_all_pages"),
        "created_by": ch.get("created_by"), "covered": ch.get("covered_export_bucket"), "pages_needed": ch.get("pages_needed"),
        "need_whole_MB": round(ch.get("need_whole_page_bytes", 0) / 1e6, 2), "status_keys": sorted(sd.keys()),
    }))
print("=== COLS ===")
c0 = next(ch for ch in chunks if ch.get("cols"))
print(json.dumps(c0["cols"]))
print("=== KEYS JSON ===")
rec = {"chunk_keys": {ch["chunk"]: [k for p in ch.get("pages", []) for k in p["keys"]] for ch in chunks if ch.get("pages")},
       "page_bytes": {ch["chunk"]: [p["df_bytes"] for p in ch.get("pages", [])] for ch in chunks if ch.get("pages")}}
print(json.dumps(rec))
log("done")
