import json, sys, os
import numpy as np
F = "/home/sagemaker-user/adam/builder/final/"
EXP = F + "export_clean_v4/"
out = {}
ex = json.load(open(EXP + "TRAINING_EXCLUDE.json"))
ex_keys = ex["match_keys"]
print("TRAINING_EXCLUDE n", len(ex_keys), "unique", len(set(ex_keys)), "split_counts", ex["split_counts"], flush=True)
mk = json.load(open(EXP + "match_keys.json"))
print("match_keys.json type", type(mk).__name__, "len", len(mk), flush=True)
if isinstance(mk, dict):
    print("dict keys sample", list(mk.keys())[:5], flush=True)
    mk_list = list(mk.keys()) if not isinstance(list(mk.values())[0], str) else list(mk.values())
else:
    mk_list = mk
split = np.load(EXP + "match_split.npy")
print("match_split shape", split.shape, "counts", {int(v): int((split == v).sum()) for v in np.unique(split)}, flush=True)
idx = {k: i for i, k in enumerate(mk_list)}
batch_idx = [idx[k] for k in ex_keys if k in idx]
print("batch keys found in match_keys", len(batch_idx), flush=True)
print("batch split counts from match_split.npy", {int(v): int((split[batch_idx] == v).sum()) for v in np.unique(split[batch_idx])}, flush=True)
bset = set(ex_keys)
# 2026/10/21..25 keys in full export
day = [k for k in mk_list if any(k.startswith(f"csds/2026/10/{d}/") for d in ["21","22","23","24","25"])]
print("export keys dated 10/21..25", len(day), flush=True)
print("export 10/21..25 not in batch", len([k for k in day if k not in bset]), flush=True)
hb = [l.strip() for l in open(F + "holdout_b_keys.txt") if l.strip()]
print("holdout_b_keys.txt lines", len(hb), "unique", len(set(hb)), flush=True)
print("holdout_b in batch", len(set(hb) & bset), flush=True)
print("holdout_b in export", len(set(hb) & set(mk_list)), flush=True)
fz = json.load(open("/home/sagemaker-user/dup_check/frozen_test_keys.json"))
if isinstance(fz, dict):
    print("frozen dict keys", list(fz.keys())[:10], flush=True)
    fz_list = fz.get("keys") or fz.get("match_keys") or []
else:
    fz_list = fz
fz_set = set(fz_list)
print("frozen test keys", len(fz_list), "unique", len(fz_set), flush=True)
print("frozen in batch", len(fz_set & bset), "frozen in holdout_b", len(fz_set & set(hb)), flush=True)
for name, path in [("final_split_manifest", F + "split_manifest.json"), ("final_split_manifest_v2", F + "split_manifest_v2.json"), ("dup_check_split_manifest", "/home/sagemaker-user/dup_check/split_manifest.json"), ("csds_cs2_combined_manifest", "/home/sagemaker-user/csds-cs2/data/combined/split_manifest.json")]:
    try:
        m = json.load(open(path))
    except Exception as e:
        print(name, "ERR", e, flush=True); continue
    if isinstance(m, dict):
        ks = list(m.keys())
        print(name, "dict keys", ks[:8], flush=True)
        ents = m.get("entries") or m.get("matches") or m.get("splits") or m
    else:
        ents = m
    if isinstance(ents, dict):
        items = list(ents.items())
        print(name, "n entries", len(items), "sample", str(items[0])[:200], flush=True)
        lab = {k: (v.get("split") if isinstance(v, dict) else v) for k, v in items}
    else:
        print(name, "n entries", len(ents), "sample", str(ents[0])[:200], flush=True)
        lab = {}
        for e in ents:
            if isinstance(e, dict):
                k = e.get("match_key") or e.get("key")
                lab[k] = e.get("split") or e.get("label")
    from collections import Counter
    bc = Counter(str(lab.get(k)) for k in ex_keys)
    print(name, "batch labels", dict(bc), flush=True)
    d21 = [k for k in lab if isinstance(k, str) and any(k.startswith(f"csds/2026/10/{d}/") for d in ["21","22","23","24","25"])]
    print(name, "10/21-25 label counts", dict(Counter(str(lab[k]) for k in d21)), "n", len(d21), flush=True)
    t_hb = Counter(str(lab[k]) for k in hb if k in lab)
    print(name, "holdout_b labels", dict(t_hb), flush=True)
