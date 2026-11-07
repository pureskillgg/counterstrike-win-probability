import csv
import hashlib
import os
from collections import defaultdict

SCR = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
LOCAL = "G:/pureskill/csds-cs2/models/seq_gdt_weights"
MANIFEST = "C:/Users/<user>/Documents/pureskill/counterstrike-win-probability/sequence/results/weights_manifest.csv"
DL_VOL = SCR + "/dl_models_gdt_objective"
DL_RES = SCR + "/dl_results_pt"
S3_MODELS_LIST = SCR + "/s3inv/models_recursive.txt"
S3_RESULTS_LIST = SCR + "/s3inv/results_recursive.txt"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def walk(root):
    out = {}
    for dp, dn, fn in os.walk(root):
        for n in fn:
            full = os.path.join(dp, n)
            rel = os.path.relpath(full, root).replace("\\", "/")
            out[rel] = full
    return out


def s3_sizes(listing, prefix):
    out = {}
    with open(listing) as f:
        for line in f:
            parts = line.split(None, 3)
            if len(parts) < 4:
                continue
            size = int(parts[2])
            key = parts[3].strip()
            if key.startswith(prefix):
                out[key[len(prefix):]] = size
    return out


# ---------- Task 2: manifest vs local master ----------
print("=== TASK 2: MANIFEST vs LOCAL MASTER ===")
with open(MANIFEST, newline="") as f:
    rd = csv.DictReader(f)
    header = rd.fieldnames
    rows = list(rd)
print("manifest header:", header)
print("manifest data rows:", len(rows))
man = {}
dup_paths = 0
for r in rows:
    p = r["file"].strip().replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    if p in man:
        dup_paths += 1
    man[p] = r
print("duplicate manifest paths:", dup_paths)

local = walk(LOCAL)
local_pt = {k: v for k, v in local.items() if k.endswith(".pt")}
local_nonpt = sorted(k for k in local if not k.endswith(".pt"))
print("local files total:", len(local), "| .pt:", len(local_pt), "| non-.pt:", len(local_nonpt))
print("local .pt total bytes:", sum(os.path.getsize(v) for v in local_pt.values()))
print("manifest paths with no local .pt:", sorted(set(man) - set(local_pt)))
print("local .pt with no manifest row:", sorted(set(local_pt) - set(man)))

local_pt_sha = {}
for k in sorted(local_pt):
    local_pt_sha[k] = sha256(local_pt[k])

size_mm = []
hash_mm = []
for k in sorted(set(man) & set(local_pt)):
    sz = os.path.getsize(local_pt[k])
    if str(sz) != man[k]["bytes"].strip():
        size_mm.append((k, sz, man[k]["bytes"]))
    if local_pt_sha[k] != man[k]["sha256"].strip().lower():
        hash_mm.append(k)
print("manifest vs local size mismatches:", len(size_mm), size_mm[:5])
print("manifest vs local sha256 mismatches:", len(hash_mm), hash_mm[:5])

# basename uniqueness in local master (needed for cross-location matching)
by_base_local = defaultdict(list)
for k in local_pt:
    by_base_local[os.path.basename(k)].append(k)
print("local basename collisions:", {b: v for b, v in by_base_local.items() if len(v) > 1})

# ---------- Task 3: volume vs local master ----------
print("=== TASK 3: VOLUME models/gdt_objective vs LOCAL MASTER ===")
vol = walk(DL_VOL)
vol_list = s3_sizes(S3_MODELS_LIST, "models/gdt_objective/")
print("volume files downloaded:", len(vol))
print("S3 listing objects under models/gdt_objective/:", len(vol_list))
print("S3 listing vs downloaded name differences:", sorted(set(vol_list) ^ set(vol)))
print("only in volume:", sorted(set(vol) - set(local)))
print("only in local master:", sorted(set(local) - set(vol)))

vol_size_mm = []
vol_hash_mm = []
vol_sha = {}
for k in sorted(set(vol) & set(local)):
    a = os.path.getsize(vol[k])
    b = os.path.getsize(local[k])
    if a != b:
        vol_size_mm.append((k, a, b))
    if k in vol_list and vol_list[k] != a:
        vol_size_mm.append((k, "s3list", vol_list[k], "downloaded", a))
    ha = sha256(vol[k])
    vol_sha[k] = ha
    hb = sha256(local[k])
    if ha != hb:
        vol_hash_mm.append(k)
print("volume vs local size mismatches:", len(vol_size_mm), vol_size_mm[:5])
print("volume vs local sha256 mismatches (all 94 files):", len(vol_hash_mm), vol_hash_mm[:5])

vol_pt = [k for k in vol if k.endswith(".pt")]
print("volume .pt count:", len(vol_pt), "| bytes:", sum(os.path.getsize(vol[k]) for k in vol_pt))
vol_pt_man_mm = []
for k in vol_pt:
    mk = k
    if mk not in man:
        vol_pt_man_mm.append((k, "not in manifest"))
        continue
    if vol_sha[k] != man[mk]["sha256"].strip().lower():
        vol_pt_man_mm.append((k, "sha differs from manifest"))
print("volume .pt vs manifest sha256 problems:", len(vol_pt_man_mm), vol_pt_man_mm[:5])

# ---------- Task 1 / 5 extra: S3 results .pt copies ----------
print("=== RESULTS .pt COPIES (results/weights_*) ===")
res = walk(DL_RES)
res_list = s3_sizes(S3_RESULTS_LIST, "results/")
print("results .pt downloaded:", len(res))
res_size_bad = [k for k in res if res_list.get(k) != os.path.getsize(res[k])]
print("results downloaded vs S3 listing size mismatches:", res_size_bad)

res_match_vol = []
res_mismatch_vol = []
res_no_vol = []
res_match_local = []
res_mismatch_local = []
res_no_local = []
for k in sorted(res):
    h = sha256(res[k])
    if k in vol_sha:
        if h == vol_sha[k]:
            res_match_vol.append(k)
        else:
            res_mismatch_vol.append(k)
    else:
        res_no_vol.append(k)
    base = os.path.basename(k)
    if base in by_base_local and len(by_base_local[base]) == 1:
        lk = by_base_local[base][0]
        if h == local_pt_sha[lk]:
            res_match_local.append((k, lk))
        else:
            res_mismatch_local.append((k, lk))
    else:
        res_no_local.append(k)
print("results copies with same path in volume:", len(res_match_vol) + len(res_mismatch_vol))
print("  sha256 equal to volume copy:", len(res_match_vol), "| differs:", res_mismatch_vol)
print("results copies with no same-path volume file:", res_no_vol)
print("results copies sha256 equal to local master (by basename):", len(res_match_local))
print("results copies sha256 differs from local master:", res_mismatch_local)
print("results copies with no local counterpart:", res_no_local)

print("=== END ===")
