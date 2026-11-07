import glob, json, os
from collections import Counter
B = "/home/sagemaker-user/adam/builder/data"
tot = Counter()
for d in sorted(glob.glob(B + "/c[0-9][0-9][0-9]")):
    r = json.load(open(os.path.join(d, "prep_report.json")))
    for k, v in r.get("round_status", {}).items():
        tot[k.split("|", 1)[1]] += v
    tot["_rounds_total"] += r.get("rounds_total", 0)
for k, v in sorted(tot.items(), key=lambda x: -x[1]):
    print(f"{k:32s} {v}")
ok = tot.get("ok", 0); allr = tot["_rounds_total"]
print("ok share of all prep rounds: %.4f" % (ok / allr))
