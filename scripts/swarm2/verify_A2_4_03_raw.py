# verify_A2_4 step 03: does any raw copy of a build-9842 match contain a round 1 round_end / round_state row?
# Read-only. Sources: (a) E: 2024 tail copy (read-only, slow; parquet per channel, no extension),
# (b) G: staging_2024_full copy, (c) the 2024 tome (round_end labels). Output: stdout only.
import os, json, collections
import polars as pl

TOME = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
E_ROOT = r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02"
G_ROOT = r"G:\pureskill\csds-cs2\staging_2024_full\csds\2024\02"


def chan_frame(name, cols):
    d = [x for x in os.listdir(TOME) if x.split(".", 1)[0] == "basic_" + name]
    assert len(d) == 1, (name, d)
    dp = os.path.join(TOME, d[0])
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    parts = [pl.scan_parquet(os.path.join(dp, p["dataframe"]["dataframe"])).select(cols).collect()
             for p in meta["pages"]]
    return pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]


hdir = os.path.join(TOME, "header")
hm = json.load(open(os.path.join(hdir, "tome"), encoding="utf8"))
H = pl.concat([pl.read_parquet(os.path.join(hdir, p["dataframe"]["dataframe"])).select(["key", "build_num"])
               for p in hm["pages"]], how="vertical_relaxed")
re_t = chan_frame("round_end", ["round", "match_key"]).unique(["match_key", "round"])
TOME_R = collections.defaultdict(set)
for m, r in zip(re_t["match_key"].to_list(), re_t["round"].to_list()):
    TOME_R[m].add(r)

rows = H.to_dicts()
print("header matches:", len(rows))
stat = collections.Counter()
per = collections.defaultdict(collections.Counter)
examples = []
for row in rows:
    key = row["key"]
    b = row["build_num"]
    parts = key.split("/")  # csds / 2024 / 02 / DD / match_id / csds
    dd, mid = parts[3], parts[4]
    for src_name, root in [("E_tail", E_ROOT), ("G_staging_full", G_ROOT)]:
        f = os.path.join(root, dd, mid, "round_end")
        if not os.path.isfile(f):
            per[(b, src_name)]["no round_end file"] += 1
            continue
        try:
            df = pl.read_parquet(f, columns=["round"])
        except Exception as e:  # keep going, record the failure type
            per[(b, src_name)]["read error " + type(e).__name__] += 1
            continue
        labs = set(df["round"].drop_nulls().to_list())
        per[(b, src_name)]["file ok"] += 1
        per[(b, src_name)]["raw has round 1" if 1 in labs else "raw has no round 1"] += 1
        tl = TOME_R.get(key, set())
        per[(b, src_name)]["raw labels == tome labels" if labs == tl else "raw labels != tome labels"] += 1
        if 1 in labs and 1 not in tl and len(examples) < 5:
            examples.append((b, src_name, key, sorted(labs)[:5], sorted(tl)[:5]))
    per[(b, "tome")]["has round 1" if 1 in TOME_R.get(key, set()) else "no round 1"] += 1

for k in sorted(per, key=lambda x: (str(x[0]), x[1])):
    print(f"build {k[0]} source {k[1]}: {dict(per[k])}")
print("examples (raw has round 1 but tome does not):", examples)

# round_state round-1 rows in raw for a sample of build 9842 matches with a raw round_end file
samp = collections.Counter()
n = 0
for row in rows:
    if row["build_num"] != 9842:
        continue
    key = row["key"]
    dd, mid = key.split("/")[3], key.split("/")[4]
    f_rs = os.path.join(E_ROOT, dd, mid, "round_state")
    if not os.path.isfile(f_rs):
        continue
    n += 1
    if n > 60:
        break
    rs = pl.read_parquet(f_rs, columns=["round", "event_type"])
    ev1 = tuple(sorted(set(rs.filter(pl.col("round") == 1)["event_type"].to_list())))
    samp[ev1] += 1
print("raw round_state round-1 event sets (build 9842, E_tail sample of up to 60):", dict(samp), "n=", n)
print("DONE raw")
