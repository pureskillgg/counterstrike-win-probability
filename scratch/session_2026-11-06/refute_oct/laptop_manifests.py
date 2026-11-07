import json, sys, csv
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
S = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct/"
bk = set(r["match_key"] for r in csv.DictReader(open(S + "batch_v4_per_match.csv", encoding="utf-8")))
G = "G:/pureskill/csds-cs2/data/combined/"
for name in ["split_manifest.json", "split_manifest_v2.json", "split_manifest_v1_archive.json", "_backup_pre_22_26_20261027/split_manifest.json", "_failed_extension_22_25_20261027/split_manifest.json", "_backup_pre_1021/split_manifest.json"]:
    try:
        m = json.load(open(G + name, encoding="utf-8"))
    except Exception as e:
        print(name, "ERR", str(e)[:120], flush=True); continue
    a = m.get("assign", m)
    print(name, "entries", len(a), "labels", dict(Counter(a.values())) if isinstance(a, dict) else "list", flush=True)
    if isinstance(a, dict):
        print("   batch labels", dict(Counter(str(a.get(k)) for k in bk)), flush=True)
