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

STRICT = re.compile(r"^(best_val_log_loss|final_val_log_loss|final_raw_val_log_loss|final_ema_val_log_loss|val_log_loss|val_logloss|validation_log_loss|validation_logloss|best_val_logloss)$", re.I)

def load(path):
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw.decode("utf-8-sig", errors="replace"), parse_constant=lambda c: None)

def has_strict(obj, depth=0, skip_hist=True):
    if depth > 8:
        return False
    if isinstance(obj, dict):
        for k, v in obj.items():
            if STRICT.match(str(k)):
                return True
            if skip_hist and k == "history":
                continue
            if has_strict(v, depth + 1, skip_hist):
                return True
    elif isinstance(obj, list):
        for v in obj:
            if has_strict(v, depth + 1, skip_hist):
                return True
    return False

def key_census(obj, prefix, cnt, depth=0):
    if depth > 6:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict,)):
                cnt[p.split(".")[-1] if depth > 0 else p] += 0
            key = str(k)
            cnt[key] += 1
            if k == "history":
                continue
            key_census(v, p, cnt, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:50]:
            key_census(v, prefix + "[]", cnt, depth + 1)

qual = []
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
            if has_strict(obj):
                qual.append((p, obj))

print("qualifying files (strict, history skipped):", len(qual))
cnt = collections.Counter()
for p, obj in qual:
    key_census(obj, "", cnt)
keys_sorted = sorted(cnt.items(), key=lambda x: -x[1])
print("KEY CENSUS (top 400):")
for k, c in keys_sorted[:400]:
    print(c, k)
