# A2: per-match, per-channel row-count reconciliation: staged source parquet footers (G:) vs tome dataframe rows.
# Read-only. Uses parquet footer metadata for sources and column projection for the tome.
import os, sys, csv, collections, time
import polars as pl, pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
BASE = r"G:\pureskill\csds-cs2"
STAGE = os.path.join(BASE, "staging_2024_g", "csds", "2024", "02")
TOME = os.path.join(BASE, "tomes", "2024-02-g", "tome", "csds")
SCR = sys.argv[1]
chans = sorted(d.rsplit(".", 1)[0] for d in os.listdir(TOME) if d.startswith("basic_"))
chans = [c.replace("basic_", "", 1) for c in chans]
print("tome channels with dataframe:", len(chans), flush=True)
# 1. tome row counts per match key per channel
tome_counts = {}
tome_cols = {}
t0 = time.time()
for ch in chans + ["header"]:
    folder = [d for d in os.listdir(TOME) if d == ("header" if ch == "header" else "basic_" + ch + ".2023-12-28,2024-02-25")]
    if not folder:
        print("MISSING FOLDER", ch); continue
    p = os.path.join(TOME, folder[0], "dataframe_00000")
    schema_names = pq.read_schema(p).names
    if not schema_names:
        tome_counts[ch] = {}; tome_cols[ch] = 0
        print(f"  {ch}: 0 columns, rows={pq.ParquetFile(p).metadata.num_rows}", flush=True)
        continue
    kcol = "match_key" if "match_key" in schema_names else "key"
    df = pl.read_parquet(p, columns=[kcol])
    g = df.group_by(kcol).len()
    tome_counts[ch] = dict(zip(g[kcol].to_list(), g["len"].to_list()))
    tome_cols[ch] = len(schema_names)
    print(f"  {ch}: tome rows={df.height} distinct_keys={len(tome_counts[ch])} cols={len(schema_names)} key={kcol}", flush=True)
print("tome read", f"{time.time()-t0:.0f}s", flush=True)
# 2. source footers per staged match folder
jobs = []
for day in sorted(os.listdir(STAGE)):
    dp = os.path.join(STAGE, day)
    for mid in sorted(os.listdir(dp)):
        jobs.append((day, mid, os.path.join(dp, mid)))
def footers(job):
    day, mid, mp = job
    out = {}
    for ch in chans + ["header"]:
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
# 3. reconcile
src_tot = collections.Counter(); tome_tot = collections.Counter()
mism = []
for day, mid, out in res:
    key = f"csds/2024/02/{day}/{mid}/csds"
    for ch in chans + ["header"]:
        s = out.get(ch)
        if isinstance(s, str):
            mism.append((day, mid, ch, "source_err", s, None)); continue
        s = 0 if s is None else s
        src_tot[ch] += s
        t = tome_counts.get(ch, {}).get(key, 0)
        tome_tot[ch] += t
        if s != t:
            mism.append((day, mid, ch, "diff", s, t))
print("per-channel totals (source footer rows / tome rows / diff):")
for ch in chans + ["header"]:
    print(f"  {ch}: {src_tot[ch]} / {tome_tot[ch]} / {tome_tot[ch]-src_tot[ch]}")
print("mismatched (match, channel) pairs:", len(mism))
mc = collections.Counter((m[2], m[3]) for m in mism)
print("mismatch by channel:", dict(mc.most_common(40)))
os.makedirs(SCR, exist_ok=True)
with open(os.path.join(SCR, "A2_rowcount_mismatch.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["day", "match", "channel", "kind", "source_rows", "tome_rows"]); w.writerows(mism)
# 4. keys in tome not in staged folders
staged_keys = set(f"csds/2024/02/{d}/{m}/csds" for d, m, _ in jobs)
for ch in ["header"] + chans:
    ks = set(tome_counts.get(ch, {}).keys())
    extra = ks - staged_keys
    if extra: print("keys in tome not staged:", ch, len(extra), list(extra)[:3])
