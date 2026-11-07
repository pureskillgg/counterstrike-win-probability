# A1: registry record checks and per-channel tome descriptor checks for the 2026 tome.
import os, glob, json, hashlib
import polars as pl
import pyarrow.parquet as pq
TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08"
T = os.path.join(TOME, "tome", "csds")
REG = os.path.join(r"G:\pureskill\csds-cs2\tomes\.meridian-outputs-cache", "artifact_registry.json")
reg = json.load(open(REG, encoding="utf-8"))
arts = list(reg["artifacts"].values()) if isinstance(reg["artifacts"], dict) else reg["artifacts"]
def lpath(lp):
    return lp if isinstance(lp, str) else (lp.get("local_only_path") or "")
mine = [a for a in arts if any("2026-10-01_to_08" in lpath(lp) for lp in a.get("local_paths", []))]
print("registry artifacts total:", len(arts), "| for 2026 tome:", len(mine))
paths = []
for a in mine:
    for lp in a["local_paths"]:
        paths.append((lpath(lp), (a.get("metadata") or {}).get("channel") or ((a.get("metadata") or {}).get("build") or "")[:30], a["content_hash"] is not None, a["hash_verified"], a["lifecycle_state"], a["role"], a["metadata"].get("matches")))
print("registry path records:", len(paths))
exists = [p for p in paths if os.path.exists(p[0])]
print("registry paths that exist on disk:", len(exists))
missing_in_reg = []
chan_dirs = sorted(d for d in os.listdir(T) if not d.startswith("."))
reg_names = set(os.path.basename(os.path.dirname(p[0])) if os.path.basename(p[0]) == "tome" and "header" in p[0] else os.path.basename(p[0]) for p in paths)
print("on-disk tome dirs not in registry:", [d for d in chan_dirs if d not in reg_names])
print("registry dirs not on disk:", [p[0] for p in paths if not os.path.exists(p[0])])
print("registry records with channel metadata counts:", pl.DataFrame([{"n": 1, "hash": p[2], "verified": p[3], "state": p[4], "role": p[5], "matches": p[6]} for p in paths]).group_by(["hash", "verified", "state", "role", "matches"]).len().rows())
hdr = [a for a in mine if a["metadata"].get("build", "").startswith("full")]
print("header record build:", hdr[0]["metadata"]["build"] if hdr else None, "| verified:", hdr[0]["metadata"].get("verified") if hdr else None)
print("header record content_hash:", hdr[0]["content_hash"] if hdr else None, "| hash_verified:", hdr[0]["hash_verified"] if hdr else None)
ch_hash = {}
for a in mine:
    lp = lpath(a["local_paths"][0])
    ch_hash[lp] = a["content_hash"]
# recompute header tome descriptor sha256
h = hashlib.sha256(open(os.path.join(T, "header", "tome"), "rb").read()).hexdigest()
print("recomputed sha256(header/tome) == registry content_hash:", h == (hdr[0]["content_hash"] if hdr else None))
print("source_locator variants:", sorted(set(a["source_locator"] for a in mine))[:6])
# per-channel tome descriptors
probs = []
for c in chan_dirs:
    tp = os.path.join(T, c, "tome")
    if not os.path.exists(tp):
        probs.append((c, "no tome descriptor")); continue
    tj = json.load(open(tp, encoding="utf-8"))
    pages = tj.get("pages", [])
    ondisk_df = sorted(glob.glob(os.path.join(T, c, "dataframe_*")))
    ondisk_ks = sorted(glob.glob(os.path.join(T, c, "keyset_*")))
    if not tj.get("isComplete"): probs.append((c, "isComplete false"))
    if len(pages) != len(ondisk_df): probs.append((c, f"descriptor pages {len(pages)} != on-disk {len(ondisk_df)}"))
    for pg in pages:
        dfk = os.path.join(TOME, pg["dataframe"]["key"].replace("/", os.sep))
        if not os.path.exists(dfk): probs.append((c, "missing " + pg["dataframe"]["key"]))
    if tj.get("headerTomeName") != "header" or tj.get("isHeader") is not False: probs.append((c, "header flags"))
    if not os.path.exists(os.path.join(T, c, "header", "tome")): probs.append((c, "no nested header"))
    else:
        nt = json.load(open(os.path.join(T, c, "header", "tome"), encoding="utf-8"))
        if nt.get("sourceId") != json.load(open(os.path.join(T, "header", "tome"), encoding="utf-8"))["id"]:
            probs.append((c, "nested header sourceId != header id"))
print("descriptor problems:", probs or "none")
# network_protocol
hdf = pl.read_parquet(os.path.join(T, "header", "dataframe_00000"), columns=["network_protocol", "rank_type", "bomb_time", "rank_type"])
print("network_protocol distinct:", hdf.get_column("network_protocol").n_unique(), "nulls:", hdf.get_column("network_protocol").null_count(), "values:", hdf.get_column("network_protocol").unique().to_list()[:5])
print("bomb_time dtype/null:", hdf.get_column("bomb_time").dtype, hdf.get_column("bomb_time").null_count())
