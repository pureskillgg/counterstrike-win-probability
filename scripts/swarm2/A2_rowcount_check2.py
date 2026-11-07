# A2 (v2): per-match, per-channel row-count reconciliation. Reads ALL tome pages
# (dataframe_000NN listed in each channel's `tome` JSON), not only page 0.
# Sources: staged 2024 match folders (staging_2024_g junctions -> merged_2024 / source_2024), parquet footers.
import os, sys, csv, json, collections, time
import polars as pl, pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
BASE = r"G:\pureskill\csds-cs2"
STAGE = os.path.join(BASE, "staging_2024_g", "csds", "2024", "02")
TOME = os.path.join(BASE, "tomes", "2024-02-g", "tome", "csds")
SCR = sys.argv[1]
def tome_dirs():
    out = {}
    for d in sorted(os.listdir(TOME)):
        dp = os.path.join(TOME, d)
        if not os.path.isdir(dp): continue
        if d == "header":
            out["header"] = dp
        elif d.startswith("basic_"):
            out[d.split(".", 1)[0][len("basic_"):]] = dp
    return out
dirs = tome_dirs()
chans = sorted(c for c in dirs if c != "header")
print("tome channel dirs:", len(dirs), flush=True)
tome_counts, tome_pages, tome_rows = {}, {}, {}
t0 = time.time()
for ch, dp in sorted(dirs.items()):
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    pages = [p["dataframe"]["dataframe"] for p in meta["pages"]]
    tome_pages[ch] = len(pages)
    counts = collections.Counter(); rows = 0; cols = None
    for pg in pages:
        pth = os.path.join(dp, pg)
        names = pq.read_schema(pth).names
        if not names:
            continue
        kcol = "match_key" if "match_key" in names else "key"
        cols = len(names)
        df = pl.read_parquet(pth, columns=[kcol])
        rows += df.height
        g = df.group_by(kcol).len()
        for k, n in zip(g[kcol].to_list(), g["len"].to_list()):
            counts[k] += n
    tome_counts[ch] = counts; tome_rows[ch] = rows
    print(f"  {ch}: pages={len(pages)} tome_rows={rows} distinct_keys={len(counts)} cols={cols}", flush=True)
print("tome read", f"{time.time()-t0:.0f}s", flush=True)
jobs = []
for day in sorted(os.listdir(STAGE)):
    dp = os.path.join(STAGE, day)
    for mid in sorted(os.listdir(dp)):
        jobs.append((day, mid, os.path.join(dp, mid)))
chan_files = chans + ["header"]
def footers(job):
    day, mid, mp = job
    out = {}
    for ch in chan_files:
        p = os.path.join(mp, ch)
        if os.path.isfile(p):
            try:
                out[ch] = pq.ParquetFile(p).metadata.num_rows
            except Exception as e:
                out[ch] = "ERR:" + str(e)[:80]
        else:
            out[ch] = None
    return day, mid, out
t0 = time.time()
with ThreadPoolExecutor(12) as ex:
    res = list(ex.map(footers, jobs))
print("footers read", len(res), f"{time.time()-t0:.0f}s", flush=True)
src_tot = collections.Counter(); tome_tot = collections.Counter()
mism = []
for day, mid, out in res:
    key = f"csds/2024/02/{day}/{mid}/csds"
    for ch in chan_files:
        s = out.get(ch)
        if isinstance(s, str):
            mism.append((day, mid, ch, "source_err", s, None)); continue
        s = 0 if s is None else s
        src_tot[ch] += s
        t = tome_counts.get(ch, {}).get(key, 0)
        tome_tot[ch] += t
        if s != t:
            mism.append((day, mid, ch, "diff", s, t))
print("per-channel totals (source footer rows / tome rows / diff) [tome pages]:")
for ch in chan_files:
    print(f"  {ch}: {src_tot[ch]} / {tome_tot[ch]} / {tome_tot[ch]-src_tot[ch]}  [pages={tome_pages.get(ch)}]")
print("mismatched (match, channel) pairs:", len(mism))
mc = collections.Counter((m[2], m[3]) for m in mism)
print("mismatch by channel:", dict(mc.most_common(40)))
with open(os.path.join(SCR, "A2_rowcount_mismatch2.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["day", "match", "channel", "kind", "source_rows", "tome_rows"]); w.writerows(mism)
staged_keys = set(f"csds/2024/02/{d}/{m}/csds" for d, m, _ in jobs)
for ch in chan_files:
    ks = set(tome_counts.get(ch, {}).keys())
    extra = ks - staged_keys
    if extra: print("keys in tome not staged:", ch, len(extra), list(extra)[:3], flush=True)
print("DONE", flush=True)
