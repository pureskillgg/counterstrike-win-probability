import json, os, re, sys, collections

ROOTS = [
    "G:/pureskill/csds-cs2/data/diagnostics",
    "G:/pureskill/csds-cs2/data/sprint",
    "G:/pureskill/csds-cs2/data/holdout_b",
    "G:/pureskill/csds-cs2/data/combined",
    "C:/Users/<user>/Documents/pureskill/counterstrike-win-probability/docs/sprint",
    "C:/Users/<user>/Documents/pureskill/counterstrike-win-probability/docs/eval_reports",
    "C:/Users/<user>/Documents/pureskill/counterstrike-win-probability/docs/positions_validation",
]

PAT = re.compile(r"(val.*loss|log_?loss|best_val|test_rows)", re.I)

def load(path):
    with open(path, "rb") as f:
        raw = f.read()
    txt = raw.decode("utf-8-sig", errors="replace")
    return json.loads(txt, parse_constant=lambda c: None)

def walk(obj, prefix, out, depth=0):
    if depth > 6:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if PAT.search(str(k)):
                out.add(p)
            walk(v, p, out, depth + 1)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:3]):
            walk(v, f"{prefix}[]", out, depth + 1)

keycount = collections.Counter()
files_with = []
errors = []
total = 0
for root in ROOTS:
    for dp, dn, fn in os.walk(root):
        for f in fn:
            if not f.lower().endswith(".json"):
                continue
            total += 1
            p = os.path.join(dp, f).replace("\\", "/")
            try:
                obj = load(p)
            except Exception as e:
                errors.append((p, str(e)[:120]))
                continue
            out = set()
            walk(obj, "", out)
            if out:
                files_with.append(p)
                for k in out:
                    keycount[k] += 1

print("total json", total, "files with val/loss keys", len(files_with), "errors", len(errors))
for k, c in sorted(keycount.items(), key=lambda x: -x[1])[:200]:
    print(c, k)
print("--- errors")
for p, e in errors:
    print(p, e)
