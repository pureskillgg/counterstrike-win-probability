import json, os, re, sys, collections

ROOTS = [
    "G:/pureskill/csds-cs2/data/diagnostics",
    "G:/pureskill/csds-cs2/data/sprint",
    "G:/pureskill/csds-cs2/data/holdout_b",
    "G:/pureskill/csds-cs2/data/combined",
    "C:/Users/13144/Documents/pureskill/counterstrike-win-probability/docs/sprint",
    "C:/Users/13144/Documents/pureskill/counterstrike-win-probability/docs/eval_reports",
    "C:/Users/13144/Documents/pureskill/counterstrike-win-probability/docs/positions_validation",
]

STRICT = re.compile(r"^(best_val_log_loss|final_val_log_loss|final_raw_val_log_loss|final_ema_val_log_loss|val_log_loss|val_logloss|validation_log_loss|validation_logloss|best_val_logloss)$", re.I)

def load(path):
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw.decode("utf-8-sig", errors="replace"), parse_constant=lambda c: None)

def find_keys(obj, prefix, out, depth=0):
    if depth > 8:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if STRICT.match(str(k)):
                out.append((p, v))
            find_keys(v, p, out, depth + 1)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_keys(v, f"{prefix}[{i}]", out, depth + 1)

rows = []
for root in ROOTS:
    for dp, dn, fn in os.walk(root):
        for f in fn:
            if not f.lower().endswith(".json"):
                continue
            p = os.path.join(dp, f).replace("\\", "/")
            try:
                obj = load(p)
            except Exception as e:
                print("ERR", p, e)
                continue
            found = []
            find_keys(obj, "", found)
            if found:
                top = list(obj.keys()) if isinstance(obj, dict) else type(obj).__name__
                rows.append((p, found, top))

print("strict files:", len(rows))
for p, found, top in rows:
    short = [(k, (v if not isinstance(v, (dict, list)) else type(v).__name__)) for k, v in found[:6]]
    print("\n##", p)
    print("  n_hits", len(found), "top:", (top if not isinstance(top, list) else top[:40]))
    print("  hits:", short)
