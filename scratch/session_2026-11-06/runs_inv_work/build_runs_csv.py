import csv
import json
import math
import os
import re

WORK_CSV = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/runs_inv_work/runs_inventory.csv"
OUT_CSV = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/runs_inventory.csv"

ROOTS = [
    "G:/pureskill/csds-cs2/data/diagnostics",
    "G:/pureskill/csds-cs2/data/sprint",
    "G:/pureskill/csds-cs2/data/holdout_b",
    "G:/pureskill/csds-cs2/data/combined",
    "C:/Users/13144/Documents/pureskill/counterstrike-win-probability/docs/sprint",
    "C:/Users/13144/Documents/pureskill/counterstrike-win-probability/docs/eval_reports",
    "C:/Users/13144/Documents/pureskill/counterstrike-win-probability/docs/positions_validation",
]

# Standard validation log-loss key names.
GENERIC_VAL = ["val_log_loss", "val_logloss", "validation_log_loss", "validation_logloss"]
BEST_VAL = ["best_val_log_loss", "best_val_logloss"]
FINAL_VAL = ["final_val_log_loss"]
FINAL_RAW = ["final_raw_val_log_loss"]
FINAL_EMA = ["final_ema_val_log_loss"]
ALL_VAL_KEYS = set(GENERIC_VAL + BEST_VAL + FINAL_VAL + FINAL_RAW + FINAL_EMA)

# Broad family: any key naming both a validation part and a log loss (e.g. validation_logloss_no_mono,
# val_logloss_new, logreg_validation_log_loss). Used only when no standard key sits on the same record.
BROAD = re.compile(r"((val|validation).*log_?loss|log_?loss.*(val|validation))", re.I)

SEED_K = ["seed"]
STEPS_K = ["steps"]
WIDTH_K = ["width", "hidden_size", "hidden", "gru_hidden", "d_model", "units", "hidden_dim"]
LOOKBACK_K = ["lookback_s", "lookback"]
LAYERS_K = ["layers", "num_layers", "n_layers"]
LR_K = ["lr", "learning_rate", "eta"]
ARM_K = ["arm", "model", "name", "id", "tag"]
UNDER_K = ["undertrained_flag", "undertrained"]
BEST_ITER_K = ["best_iteration"]
SUB_CONTAINERS = ["config", "architecture", "params", "hyperparams"]

COLUMNS = [
    "folder", "file_name", "run_path", "arm_or_model", "seed", "steps", "width",
    "lookback_s", "layers", "learning_rate", "final_val_log_loss",
    "final_raw_val_log_loss", "final_ema_val_log_loss", "validation_log_loss",
    "validation_log_loss_key", "best_val_log_loss", "best_step",
    "test_rows_in_arrays", "test_rows_read", "test_rows_in_memory", "test_rows_used",
    "undertrained_flag", "best_iteration", "field_sources", "record_note",
]


def load_json(path):
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw.decode("utf-8-sig", errors="replace"), parse_constant=lambda c: None)


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def lookup(record, keys):
    """Scalar lookup: record first, then config-like sub-dicts. Returns (value, source key)."""
    for k in keys:
        if k in record and not isinstance(record[k], (dict, list)) and record[k] is not None:
            return record[k], k
    for sub in SUB_CONTAINERS:
        d = record.get(sub)
        if isinstance(d, dict):
            for k in keys:
                if k in d and not isinstance(d[k], (dict, list)) and d[k] is not None:
                    return d[k], f"{sub}.{k}"
    return None, None


def scalar_strict(node):
    return [k for k in node if k in ALL_VAL_KEYS and is_num(node[k])]


def dict_strict(node):
    return [k for k in node if k in ALL_VAL_KEYS and isinstance(node[k], dict)]


def broad_scalar(node):
    return [k for k in node if k not in ALL_VAL_KEYS and BROAD.search(str(k)) and is_num(node[k])]


def history_has_val(node):
    h = node.get("history")
    if not isinstance(h, list):
        return False
    for item in h:
        if isinstance(item, dict) and any((k in ALL_VAL_KEYS or BROAD.search(str(k))) for k in item):
            return True
    return False


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def build_rows_for_file(file_path, obj):
    folder = os.path.dirname(file_path)
    fname = os.path.basename(file_path)
    rows = []

    def make_row(node, run_path, arm_fallback, notes, validation_override=None, arm_forced=False):
        srcs = []
        arm, ak = lookup(node, ARM_K)
        if arm_forced and arm_fallback is not None:
            arm, ak = arm_fallback, "dict_key"
        elif arm is None and arm_fallback is not None:
            arm, ak = arm_fallback, "dict_key"
        seed, sk = lookup(node, SEED_K)
        steps, stk = lookup(node, STEPS_K)
        width, wk = lookup(node, WIDTH_K)
        lookback, lbk = lookup(node, LOOKBACK_K)
        layers, lyk = lookup(node, LAYERS_K)
        lr, lrk = lookup(node, LR_K)
        best_iter, _ = lookup(node, BEST_ITER_K)
        under, _ = lookup(node, UNDER_K)

        def numval(keys):
            for k in keys:
                if k in node and is_num(node[k]):
                    return node[k], k
            return None, None

        final_v, _ = numval(FINAL_VAL)
        raw_v, _ = numval(FINAL_RAW)
        ema_v, _ = numval(FINAL_EMA)
        best_v, _ = numval(BEST_VAL)
        best_step, _ = lookup(node, ["best_step"])
        if validation_override is not None:
            gen_v, gen_k = validation_override
        else:
            gen_v, gen_k = numval(GENERIC_VAL)

        mapping = []
        if arm is not None and ak:
            mapping.append(f"arm_or_model<-{ak}")
        if seed is not None and sk:
            mapping.append(f"seed<-{sk}")
        if steps is not None and stk:
            mapping.append(f"steps<-{stk}")
        if width is not None and wk:
            mapping.append(f"width<-{wk}")
        if lookback is not None and lbk:
            mapping.append(f"lookback_s<-{lbk}")
        if layers is not None and lyk:
            mapping.append(f"layers<-{lyk}")
        if lr is not None and lrk:
            mapping.append(f"learning_rate<-{lrk}")
        if gen_k:
            mapping.append(f"validation_log_loss<-{gen_k}")

        rows.append({
            "folder": folder,
            "file_name": fname,
            "run_path": run_path,
            "arm_or_model": fmt(arm),
            "seed": fmt(seed),
            "steps": fmt(steps),
            "width": fmt(width),
            "lookback_s": fmt(lookback),
            "layers": fmt(layers),
            "learning_rate": fmt(lr),
            "final_val_log_loss": fmt(final_v),
            "final_raw_val_log_loss": fmt(raw_v),
            "final_ema_val_log_loss": fmt(ema_v),
            "validation_log_loss": fmt(gen_v),
            "validation_log_loss_key": fmt(gen_k),
            "best_val_log_loss": fmt(best_v),
            "best_step": fmt(best_step),
            "test_rows_in_arrays": fmt(node.get("test_rows_in_arrays")),
            "test_rows_read": fmt(node.get("test_rows_read")),
            "test_rows_in_memory": fmt(node.get("test_rows_in_memory")),
            "test_rows_used": fmt(node.get("test_rows_used")),
            "undertrained_flag": fmt(under),
            "best_iteration": fmt(best_iter),
            "field_sources": "; ".join(mapping),
            "record_note": "; ".join(notes),
        })

    def walk(node, path, key_name):
        if isinstance(node, dict):
            sc = scalar_strict(node)
            dc = dict_strict(node)
            bc = broad_scalar(node)
            fallback = None if (key_name is None or re.fullmatch(r"\[\d+\]", key_name)) else key_name
            if sc:
                notes = []
                if dc:
                    notes.append("also has dict-valued validation key(s): " + ", ".join(dc))
                if bc:
                    notes.append("other broad-pattern keys (not extracted): " +
                                 ", ".join(f"{k}={node[k]!r}" for k in bc))
                make_row(node, path, fallback, notes)
            elif dc:
                # dict-valued validation metrics (e.g. per-arm losses): one row per inner entry
                for k in dc:
                    for inner_name, v in node[k].items():
                        if not is_num(v):
                            continue
                        arm_guess = fallback if fallback is not None else inner_name
                        make_row(node, path, arm_guess,
                                 [f"expanded from dict-valued key {k}; inner key {inner_name}"],
                                 validation_override=(v, f"{k}[{inner_name}]"),
                                 arm_forced=True)
            elif bc:
                # broad-pattern scalar keys with no standard key: one row per key
                for k in bc:
                    make_row(node, path, k, [f"validation key matched by broad pattern: {k}"],
                             validation_override=(node[k], k), arm_forced=True)
            elif history_has_val(node):
                make_row(node, path, fallback,
                         ["history-only: run has per-epoch validation values inside history but no "
                          "run-level validation key; not extracted"])
            for k, v in node.items():
                if k == "history":
                    continue
                walk(v, f"{path}.{k}" if path else k, k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]", f"[{i}]")

    walk(obj, "", None)
    return rows


def has_any_val(obj, depth=0):
    if depth > 10:
        return False
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ALL_VAL_KEYS or (BROAD.search(str(k)) and is_num(v)):
                return True
            if has_any_val(v, depth + 1):
                return True
    elif isinstance(obj, list):
        for v in obj:
            if has_any_val(v, depth + 1):
                return True
    return False


def test_flag(r):
    """True if any of the three named test-row fields is non-zero (boolean True counts as non-zero)."""
    for k in ["test_rows_in_arrays", "test_rows_read", "test_rows_in_memory"]:
        v = r[k]
        if v == "":
            continue
        if v == "true":
            return True
        if v == "false":
            continue
        try:
            if float(v) != 0:
                return True
        except ValueError:
            pass
    return False


def main():
    all_rows = []
    qualifying = []
    errors = []
    scanned = 0
    for root in ROOTS:
        for dp, dn, fn in os.walk(root):
            dn.sort()
            for f in sorted(fn):
                if not f.lower().endswith(".json"):
                    continue
                scanned += 1
                p = (dp + "/" + f).replace("\\", "/")
                try:
                    obj = load_json(p)
                except Exception as e:  # keep going on unreadable files
                    errors.append([p, str(e)[:160]])
                    continue
                if not has_any_val(obj):
                    continue
                qualifying.append(p)
                all_rows.extend(build_rows_for_file(p, obj))

    os.makedirs(os.path.dirname(WORK_CSV), exist_ok=True)
    with open(WORK_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    folders = {r["folder"] for r in all_rows}
    files = {(r["folder"], r["file_name"]) for r in all_rows}
    summary = {
        "work_csv": WORK_CSV,
        "rows": len(all_rows),
        "distinct_folders": len(folders),
        "distinct_files": len(files),
        "qualifying_files": len(qualifying),
        "json_scanned": scanned,
        "parse_errors": errors,
        "test_rows_read_nonzero_rows": sum(1 for r in all_rows if test_flag(r)),
        "test_rows_in_memory_true_rows": sum(1 for r in all_rows if r["test_rows_in_memory"] == "true"),
        "numeric_test_counter_nonzero_rows": sum(
            1 for r in all_rows
            if any(r[k] not in ("", "true", "false") and float(r[k]) != 0
                   for k in ["test_rows_in_arrays", "test_rows_read"])),
    }
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
