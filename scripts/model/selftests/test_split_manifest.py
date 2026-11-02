"""Self-test of the frozen split (wp_common.hash_assign / frozen_split / make_split): seconds, synthetic data, no real store needed.

  python -I -X utf8 scripts\\model\\selftests\\test_split_manifest.py
Exit code 1 on any failure.
"""
import os
import sys

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import polars as pl

import wp_common as W

fails = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name, detail, flush=True)
    if not ok:
        fails.append(name)


keys = [f"m{i:06d}" for i in range(20000)]
asg = {k: W.hash_assign(k, 0) for k in keys}
n = len(keys)
share = {s: sum(1 for v in asg.values() if v == s) / n for s in ("train", "val", "test")}
check("hash rule gives about 72/8/20 % train/val/test", abs(share["test"] - 0.20) < 0.01 and abs(share["val"] - 0.08) < 0.01 and abs(share["train"] - 0.72) < 0.01, str({k: round(v, 3) for k, v in share.items()}))
check("hash rule is deterministic", all(W.hash_assign(k, 0) == v for k, v in asg.items()))
check("hash rule depends on the seed", sum(W.hash_assign(k, 1) != v for k, v in asg.items()) > n * 0.2)
sub = keys[::7]
check("an assignment does not depend on which other matches are present", all(W.hash_assign(k, 0) == asg[k] for k in sub))

# frozen_split: manifest wins over the hash rule; unknown keys fall back to the hash rule; the three sets partition the keys
man = {"seed": 0, "assign": {k: "test" for k in keys[:100]}}  # deliberately the opposite of the hash rule for many of them
sp = W.frozen_split(keys[:200], man)
check("manifest entries are honoured", all(k in sp["test"] for k in keys[:100]))
check("unlisted matches get the hash rule", all(k in sp[asg[k]] for k in keys[100:200]))
check("train / val / test partition the keys", sp["train"] | sp["val"] | sp["test"] == set(keys[:200]) and not (sp["train"] & sp["val"]) and not (sp["train"] & sp["test"]) and not (sp["val"] & sp["test"]))

# make_split("random") with a manifest file: unchanged when the match table grows; legacy rule when there is no manifest
import json
import tempfile

tmp = tempfile.mkdtemp(prefix="wp_split_")
path = os.path.join(tmp, "split_manifest.json")
G1 = pl.DataFrame({"match_key": keys[:3000], "era": ["2026"] * 3000, "match_date": ["2026-10-01"] * 3000})
legacy = W.make_split("random_legacy", G1, seed=0)
json.dump({"seed": 0, "assign": {k: s for s in legacy for k in legacy[s]}}, open(path, "w"))
old = W.SPLIT_MANIFEST
W.SPLIT_MANIFEST = path
try:
    s1 = W.make_split("random", G1, seed=0)
    G2 = pl.DataFrame({"match_key": keys[:6000], "era": ["2026"] * 6000, "match_date": ["2026-10-01"] * 6000})
    s2 = W.make_split("random", G2, seed=0)
    check("make_split('random') equals the legacy split for the frozen matches", s1 == legacy)
    check("adding matches does not move any frozen match to another set", all((k in s2[s]) for s in s1 for k in s1[s]))
    check("the new matches are split by the hash rule", all(k in s2[asg[k]] for k in keys[3000:6000]))
    W.SPLIT_MANIFEST = os.path.join(tmp, "does_not_exist.json")
    s3 = W.make_split("random", G1, seed=0)
    check("without a manifest make_split('random') is the legacy split", s3 == legacy)
    legacy2 = W.make_split("random_legacy", G2, seed=0)
    moved = sum(1 for s in legacy for k in legacy[s] if k not in legacy2[s])
    check("(why the manifest exists) the legacy rule moves frozen matches when matches are added", moved > 100, f"{moved} of {len(keys[:3000])} moved")
finally:
    W.SPLIT_MANIFEST = old
print("RESULT:", "ALL PASS" if not fails else f"FAILED {fails}")
sys.exit(1 if fails else 0)
