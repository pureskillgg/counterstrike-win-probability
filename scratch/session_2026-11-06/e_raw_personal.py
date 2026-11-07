import os, re, json, random, collections
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
frozen = set(json.load(open("C:/Users/<user>/Documents/pureskill_internship/csds-cs2/data/benchmark/frozen_test_keys.json", encoding="utf-8")))
hd = json.load(open("C:/Users/<user>/Documents/pureskill_internship/csds-cs2/data/combined/holdout_22_25_test_keys.json", encoding="utf-8"))
excl = frozen | set(hd["match_keys"])
t = pq.read_table("G:/pureskill/csds-cs2/data/combined/cache_snapshot_2026.parquet", columns=["match_key"])
keys = pc.unique(t.column("match_key")).to_pylist()
ok = sorted(k for k in keys if k not in excl and "/2026/09/" in k and "/2026/10/" not in k)
random.seed(7)
sample = random.sample(ok, 20)
TABLES = ["player_personal", "player_name", "player_chat", "header", "player_info", "player_connect"]
PATS = {"steamid64": r"7656119[0-9]{10}", "steam2": r"STEAM_[0-5]:[01]:[0-9]+", "steam3": r"\[U:1:[0-9]+\]",
        "ipv4": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "win_path": r"[A-Za-z]:\+Users\+", "steamcommunity": r"steamcommunity", "url": r"https?://"}
def shape(v):
    s = re.sub(r"[0-9]", "9", v); s = re.sub(r"[a-z]", "a", s); s = re.sub(r"[A-Z]", "A", s)
    return re.sub(r"(.)\1{3,}", r"\1{4+}", s)[:40]
agg = collections.defaultdict(lambda: {"folders": 0, "rows": 0, "cols": {}})
missing = collections.Counter(); magic = collections.Counter()
for k in sample:
    p = k.split("/")
    d = os.path.join("E:/csds_final", p[0], p[1], p[2], p[3], p[4])
    for tb in TABLES:
        f = os.path.join(d, tb)
        if not os.path.isfile(f):
            missing[tb] += 1; continue
        with open(f, "rb") as fh:
            magic[(tb, fh.read(4))] += 1
        try:
            tab = pq.read_table(f)
        except Exception as e:
            missing[f"{tb}_read_error"] += 1; continue
        a = agg[tb]; a["folders"] += 1; a["rows"] += tab.num_rows
        for name in tab.column_names:
            typ = tab.schema.field(name).type
            if not (pa.types.is_string(typ) or pa.types.is_large_string(typ)):
                continue
            col = tab.column(name).combine_chunks()
            c = a["cols"].setdefault(name, {"non_empty": 0, "distinct": set(), "pats": collections.Counter()})
            c["non_empty"] += int(pc.sum(pc.fill_null(pc.greater(pc.utf8_length(col), 0), False)).as_py() or 0)
            vals = pc.unique(col).drop_null().to_pylist()
            c["distinct"].update(vals)
            for pn, pat in PATS.items():
                c["pats"][pn] += int(pc.sum(pc.fill_null(pc.match_substring_regex(col, pat), False)).as_py() or 0)
print("sampled eligible matches:", len(sample), "| missing or unreadable (table: count):", dict(missing))
print("file magic bytes (table, magic: count):", {f"{k[0]}:{k[1]}": v for k, v in magic.items()})
for tb, a in agg.items():
    print(f"== {tb}: folders={a['folders']} rows={a['rows']}")
    for name, c in a["cols"].items():
        shp = collections.Counter(shape(v) for v in c["distinct"])
        print(f"   str {name}: non_empty={c['non_empty']} distinct={len(c['distinct'])} patterns={ {k: v for k, v in c['pats'].items() if v} }")
        for s_, n in shp.most_common(2):
            print(f"        shape(n={n}): {s_}")
