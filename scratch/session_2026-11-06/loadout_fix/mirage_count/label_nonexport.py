import json, os
from collections import Counter
import numpy as np
D = os.path.dirname(os.path.abspath(__file__))
G = "G:/pureskill/csds-cs2/data/combined"
keys = json.load(open(os.path.join(D, "match_keys.json")))
ks = set(keys)
v1 = json.load(open(os.path.join(G, "split_manifest_v1_archive.json")))["assign"]
v2 = json.load(open(os.path.join(G, "split_manifest_v2.json")))["assign"]
print("v1 labels all:", dict(Counter(v1.values())))
print("v1 labels, keys in export:", dict(Counter(v for k, v in v1.items() if k in ks)))
print("v1 labels, keys NOT in export:", dict(Counter(v for k, v in v1.items() if k not in ks)))
print("v2 labels all:", dict(Counter(v2.values())))
print("v2 labels, keys NOT in export:", dict(Counter(v for k, v in v2.items() if k not in ks)))
print("v2 keys not in export:", sum(1 for k in v2 if k not in ks), "| export keys not in v2:", sum(1 for k in keys if k not in v2))
print("v1 keys subset of v2 keys:", all(k in v2 for k in v1), "| v1 labels equal v2 labels on shared keys:", all(v2[k] == v for k, v in v1.items()))
