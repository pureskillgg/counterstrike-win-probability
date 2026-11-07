import json
from collections import Counter
F = "/home/sagemaker-user/adam/builder/final/"
ex = json.load(open(F + "export_clean_v4/TRAINING_EXCLUDE.json"))
bset = set(ex["match_keys"])
hb = set(l.strip() for l in open(F + "holdout_b_keys.txt") if l.strip())
def day_ok(k, days=("21","22","23","24","25")):
    return any(k.startswith("csds/2026/10/%s/" % d) for d in days)
for name, path in [("final/split_manifest.json", F + "split_manifest.json"), ("final/split_manifest_v2.json", F + "split_manifest_v2.json"), ("dup_check/split_manifest.json", "/home/sagemaker-user/dup_check/split_manifest.json"), ("csds-cs2 combined split_manifest.json", "/home/sagemaker-user/csds-cs2/data/combined/split_manifest.json"), ("posport combined split_manifest.json", "/home/sagemaker-user/adam/builder/posport/data/combined/split_manifest.json")]:
    m = json.load(open(path))
    a = m.get("assign", {})
    print(name, "n assign", len(a), "labels", dict(Counter(a.values())), "rule", str(m.get("rule"))[:80], "created", m.get("created_at"), flush=True)
    print("  batch labels", dict(Counter(str(a.get(k)) for k in bset)), flush=True)
    print("  10/21-25 keys in manifest", sum(1 for k in a if day_ok(k)), "labels", dict(Counter(v for k, v in a.items() if day_ok(k))), flush=True)
    print("  holdout_b labels", dict(Counter(str(a.get(k)) for k in hb)), flush=True)
    print("  frozen_from", str(m.get("frozen_from"))[:120], flush=True)
