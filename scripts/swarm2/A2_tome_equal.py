# A2: row-for-row equality of tomes/2024-02-g vs tomes/2024-02-all, channel by channel, page by page.
# Order-sensitive (DataFrame.equals) plus per-channel row counts. Reads only; one page pair in memory at a time.
import os, sys, json, time
import polars as pl
BASE = r"G:\pureskill\csds-cs2\tomes"
A = os.path.join(BASE, "2024-02-g", "tome", "csds")
B = os.path.join(BASE, "2024-02-all", "tome", "csds")
def dirs(root):
    out = {}
    for d in sorted(os.listdir(root)):
        dp = os.path.join(root, d)
        if not os.path.isdir(dp): continue
        key = "header" if d == "header" else d.split(".", 1)[0]
        out[key] = dp
    return out
da, db = dirs(A), dirs(B)
print("channels A:", len(da), "B:", len(db), "only_A:", sorted(set(da)-set(db)), "only_B:", sorted(set(db)-set(da)), flush=True)
t0 = time.time(); allok = True
for ch in sorted(set(da) & set(db)):
    ma = json.load(open(os.path.join(da[ch], "tome"), encoding="utf8"))
    mb = json.load(open(os.path.join(db[ch], "tome"), encoding="utf8"))
    pa = [p["dataframe"]["dataframe"] for p in ma["pages"]]
    pb = [p["dataframe"]["dataframe"] for p in mb["pages"]]
    if len(pa) != len(pb):
        allok = False; print(f"{ch}: PAGE COUNT DIFF {len(pa)} vs {len(pb)}", flush=True); continue
    rows_a = rows_b = 0; page_eq = []
    for xa, xb in zip(pa, pb):
        fa = pl.read_parquet(os.path.join(da[ch], xa)); fb = pl.read_parquet(os.path.join(db[ch], xb))
        rows_a += fa.height; rows_b += fb.height
        eq = fa.equals(fb)
        page_eq.append(eq)
        if not eq:
            allok = False
            # diagnose: schema, then multiset of row hashes
            same_schema = fa.schema == fb.schema
            ha = sorted(fa.hash_rows().to_list()) if fa.height and same_schema else []
            hb = sorted(fb.hash_rows().to_list()) if fb.height and same_schema else []
            print(f"  {ch} page {xa}: NOT EQUAL rows {fa.height} vs {fb.height} schema_equal={same_schema} multiset_equal={ha==hb}", flush=True)
            del ha, hb
        del fa, fb
    print(f"{ch}: pages={len(pa)} rows_A={rows_a} rows_B={rows_b} pages_equal={sum(page_eq)}/{len(page_eq)}", flush=True)
print("ALL EQUAL" if allok else "DIFFERENCES FOUND", f"{time.time()-t0:.0f}s", flush=True)
print("DONE", flush=True)
