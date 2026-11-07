# Verifier for A1-3, alternative explanations:
#  (1) is the int64/double class a per-match property shared across channels (batch/writer effect) or per-page?
#  (2) do null fractions separate the two storage classes? (claim: they do not)
#  (3) is def_index degenerate (only 0/null) in item_equip, and is it present for day-08 rows in the double page?
#  (4) do the id values differ in distribution between classes (data difference vs storage difference)?
#  (5) do all pages of the three channels fall inside one build window (tome createdAt)?
import os, glob, json, collections
import pyarrow.parquet as pq
import pyarrow.compute as pc

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
CH = {"basic_item_equip": "player_id_fixed", "basic_item_pickup": "player_id_fixed",
      "basic_player_hurt": "attacker_id_fixed"}

def folder_of(ch):
    return os.path.join(T, ch + ".2026-09-18,2026-10-08")

def uuid_of(k):
    return k.split("/")[4]

# ---------- (1) per-match class map, cross-channel ----------
cls = {}      # ch -> {uuid: 'double'|'int64'}
for ch, idc in CH.items():
    f = folder_of(ch)
    m = {}
    for dp in sorted(glob.glob(os.path.join(f, "dataframe_*"))):
        n = os.path.basename(dp).split("_")[1]
        kp = os.path.join(f, "keyset_" + n)
        keys = pq.read_table(kp).column(0).to_pylist()
        typ = str(pq.ParquetFile(dp).schema_arrow.field(idc).type)
        for k in keys:
            m[uuid_of(k)] = typ
    cls[ch] = m
    print(ch, "matches mapped:", len(m), "class counts:", dict(collections.Counter(m.values())))

allm = set(cls["basic_item_equip"]) & set(cls["basic_item_pickup"]) & set(cls["basic_player_hurt"])
print("matches common to all three channels:", len(allm))
for a, b in [("basic_item_equip", "basic_item_pickup"), ("basic_item_equip", "basic_player_hurt"),
             ("basic_item_pickup", "basic_player_hurt")]:
    ct = collections.Counter((cls[a][u], cls[b][u]) for u in allm)
    print(f"cross-tab {a} x {b}:", dict(ct))

# day of each match (from keyset), for the day-08 breakdown
day_of = {}
f = folder_of("basic_item_equip")
for kp in sorted(glob.glob(os.path.join(f, "keyset_*"))):
    for k in pq.read_table(kp).column(0).to_pylist():
        day_of[uuid_of(k)] = k.split("/")[3]
d8 = [u for u in allm if day_of.get(u) == "08"]
ct8 = collections.Counter((cls["basic_item_equip"][u], cls["basic_item_pickup"][u], cls["basic_player_hurt"][u]) for u in d8)
print("day-08 common matches:", len(d8), "(equip,pickup,hurt) class triples:", dict(ct8))

# ---------- (2) null fraction by class ----------
print("=" * 100)
for ch, idc in CH.items():
    f = folder_of(ch)
    agg = collections.defaultdict(lambda: [0, 0, 0])  # class -> [rows, nulls, pages]
    for dp in sorted(glob.glob(os.path.join(f, "dataframe_*"))):
        typ = str(pq.ParquetFile(dp).schema_arrow.field(idc).type)
        t = pq.read_table(dp, columns=[idc])
        a = agg[typ]
        a[0] += t.num_rows; a[1] += t.column(0).null_count; a[2] += 1
    for typ, (r, nn, p) in agg.items():
        print(f"{ch} {idc} class={typ} pages={p} rows={r} nulls={nn} null_frac={nn / r:.5%}")

# ---------- (3) def_index degeneracy and day-08 presence ----------
print("=" * 100)
fe = folder_of("basic_item_equip")
p1 = os.path.join(fe, "dataframe_00001")
k1 = set(uuid_of(k) for k in pq.read_table(os.path.join(fe, "keyset_00001")).column(0).to_pylist())
d8_p1 = set(u for u in k1 if day_of.get(u) == "08")
t1 = pq.read_table(p1, columns=["match_key", "def_index"])
mk = [uuid_of(x) for x in t1.column("match_key").to_pylist()]
di = t1.column("def_index")
print("item_equip page1 match_key sample:", mk[:2])
print("item_equip page1: distinct match_key uuids in keyset:", len(set(mk) & k1), "of", len(k1))
mask8 = [m in d8_p1 for m in mk]
import numpy as np
mask8 = np.array(mask8)
di_np = di.to_numpy(zero_copy_only=False)
print("item_equip page1 day-08 rows:", int(mask8.sum()), "def_index non-null on those rows:",
      int(np.sum(~np.isnan(di_np[mask8].astype(float)))) if mask8.any() else 0)
for ch in ["basic_item_equip", "basic_item_pickup"]:
    f = folder_of(ch)
    vals = set()
    for dp in sorted(glob.glob(os.path.join(f, "dataframe_*"))):
        if "def_index" not in pq.ParquetFile(dp).schema_arrow.names:
            continue
        v = pq.read_table(dp, columns=["def_index"]).column(0)
        u = pc.unique(pc.drop_null(v)).to_pylist()
        vals |= set(u)
    print(ch, "distinct non-null def_index values across pages that have it:", sorted(vals)[:20], "count", len(vals))

# ---------- (4) value distribution per class ----------
print("=" * 100)
for ch, idc in CH.items():
    f = folder_of(ch)
    per = collections.defaultdict(collections.Counter)
    for dp in sorted(glob.glob(os.path.join(f, "dataframe_*"))):
        typ = str(pq.ParquetFile(dp).schema_arrow.field(idc).type)
        v = pq.read_table(dp, columns=[idc]).column(0)
        vc = pc.value_counts(v).to_pylist()
        for row in vc:
            if row["values"] is not None:
                per[typ][int(row["values"])] += row["counts"]
    for typ in per:
        keys = sorted(per[typ])
        print(f"{ch} {idc} class={typ} value_range=({keys[0]}..{keys[-1]}) distinct={len(keys)} top5={per[typ].most_common(5)}")

# ---------- (5) build window from tome createdAt ----------
print("=" * 100)
stamps = []
for ch in sorted(os.listdir(T)):
    if ch == "header" or not os.path.isdir(os.path.join(T, ch)):
        continue
    tp = os.path.join(T, ch, "tome")
    if not os.path.exists(tp):
        continue
    tj = json.load(open(tp, encoding="utf8"))
    for p in tj.get("pages", []):
        stamps.append((p["createdAt"], ch, p["number"]))
stamps.sort()
print("tome pages total:", len(stamps))
print("earliest page createdAt:", stamps[0])
print("latest page createdAt:", stamps[-1])
for ch in CH:
    s = [x for x in stamps if x[1].startswith(ch + ".")]
    print(ch, "pages:", len(s), "createdAt range:", s[0][0], "->", s[-1][0])
print("done")
