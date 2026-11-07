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
WATCH = ["test_rows_in_memory", "test_rows_used", "test_rows_in_arrays", "test_rows_read", "test_rows", "undertrained_flag", "best_step", "layers", "width", "hidden", "gru_hidden", "d_model", "units", "hidden_size", "num_layers", "n_layers", "arm", "lr", "learning_rate", "eta", "steps", "lookback_s", "seed", "undertrained"]
STRICT = re.compile(r"^(best_val_log_loss|final_val_log_loss|final_raw_val_log_loss|final_ema_val_log_loss|val_log_loss|val_logloss|validation_log_loss|validation_logloss|best_val_logloss)$", re.I)

def load(path):
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw.decode("utf-8-sig", errors="replace"), parse_constant=lambda c: None)

vals = collections.defaultdict(collections.Counter)
examples = {}
def scan(obj, path, depth=0):
    if depth > 8:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in WATCH:
                if isinstance(v, (dict, list)):
                    tv = "container:" + type(v).__name__
                    vals[k][tv] += 1
                else:
                    tv = repr(v) if not isinstance(v, float) else "float"
                    vals[k][tv] += 1
                    if (k, tv) not in examples:
                        examples[(k, tv)] = path
            if k == "history":
                continue
            scan(v, f"{path}.{k}" if path else k, depth + 1)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:200]):
            scan(v, f"{path}[{i}]", depth + 1)

for root in ROOTS:
    for dp, dn, fn in os.walk(root):
        for f in fn:
            if not f.lower().endswith(".json"):
                continue
            p = os.path.join(dp, f).replace("\\", "/")
            try:
                obj = load(p)
            except Exception as e:
                continue
            scan(obj, p)

for k in WATCH:
    if k in vals:
        print("==", k, vals[k].most_common(12))
        for (kk, tv), ex in examples.items():
            if kk == k and tv.startswith("container") is False:
                pass
