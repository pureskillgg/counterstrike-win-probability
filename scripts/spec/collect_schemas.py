"""Read-only collection for the 2024 vs 2026 csds schema comparison.

- Samples five complete match folders per day (seeded, reproducible).
- A folder is complete when its `csds` manifest is gzip JSON that parses and
  every channel the manifest declares has a parquet file of that name.
- Reads each manifest and each declared parquet file's SCHEMA via
  pyarrow.parquet.read_schema (footer only, never row data).
- Writes one JSON file to the scratchpad. Never writes under E:.
"""
import collections
import gzip
import json
import os
import random

import pyarrow.parquet as pq

E_ROOT = r"E:\csds_final\csds"
DAYS = [
    ("2024-02-02", os.path.join(E_ROOT, "csds_tail_recovered_20261011", "csds", "2024", "02", "02")),
    ("2024-02-12", os.path.join(E_ROOT, "csds_tail_recovered_20261011", "csds", "2024", "02", "12")),
] + [
    ("2026-10-%02d" % d, os.path.join(E_ROOT, "2026", "10", "%02d" % d)) for d in range(1, 9)
]
PICK = 5
SEED_PREFIX = "csds-schema-2024-vs-2026:"
OUT = (
    r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
    r"\7308a8e7-a47e-4034-b140-5c511acb0aa4\scratchpad\schema_sample.json"
)


def load_manifest(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw[:2] != b"\x1f\x8b":
        raise ValueError("manifest is not gzip")
    return json.loads(gzip.decompress(raw).decode("utf-8"))


def check_match(match_dir):
    """Return (doc, None) when complete, else (None, reason)."""
    mf = os.path.join(match_dir, "csds")
    if not os.path.isfile(mf):
        return None, "no csds manifest"
    try:
        doc = load_manifest(mf)
    except Exception:
        return None, "manifest parse error"
    chans = doc.get("channels") if isinstance(doc, dict) else None
    if not isinstance(chans, list):
        return None, "no channels list"
    for c in chans:
        name = c.get("channel")
        if not name or not os.path.isfile(os.path.join(match_dir, name)):
            return None, "missing parquet"
    return doc, None


def pick(day_dir):
    names = sorted(n for n in os.listdir(day_dir) if os.path.isdir(os.path.join(day_dir, n)))
    rng = random.Random(SEED_PREFIX + day_dir)
    rng.shuffle(names)
    chosen, rejects = [], collections.Counter()
    for n in names:
        md = os.path.join(day_dir, n)
        doc, why = check_match(md)
        if doc is None:
            rejects[why] += 1
            continue
        chosen.append((md, doc))
        if len(chosen) == PICK:
            break
    return chosen, rejects, len(names)


def read_schema(path):
    try:
        sch = pq.read_schema(path)
    except Exception as exc:  # report, do not abort
        msg = str(exc).splitlines()[0][:300] if str(exc) else ""
        return {"error": "%s: %s" % (type(exc).__name__, msg)}
    fields = []
    for i in range(len(sch)):
        f = sch.field(i)
        fields.append([f.name, str(f.type), bool(f.nullable)])
    return {"fields": fields}


def extract(match_dir, doc, day_label):
    chans = doc["channels"]
    names = [c.get("channel") for c in chans]
    rec = {
        "day": day_label,
        "match": os.path.basename(match_dir),
        "channel_entries": len(chans),
        "duplicate_channel_names": len(names) - len(set(names)),
        "channels": {},
    }
    for c in chans:
        ch = c.get("channel")
        entry = {
            "available": c["available"] if "available" in c else "<absent>",
            "manifest_columns": [
                [col.get("name"), col.get("type"), col.get("nullable"), col.get("origin")]
                for col in c.get("columns", [])
            ],
        }
        entry.update(read_schema(os.path.join(match_dir, ch)))
        rec["channels"][ch] = entry
    return rec


def main():
    out = {"days": [], "folders": {"2024": [], "2026": []}}
    for label, day_dir in DAYS:
        year = label[:4]
        chosen, rejects, total = pick(day_dir)
        out["days"].append({
            "day": label,
            "path": day_dir,
            "match_folders": total,
            "chosen": len(chosen),
            "rejected_before_pick": dict(rejects),
        })
        print("%s: %d match folders, chose %d, rejected before pick: %s"
              % (label, total, len(chosen), dict(rejects)))
        if len(chosen) < PICK:
            print("  WARNING: fewer than %d complete folders found" % PICK)
        for md, doc in chosen:
            rec = extract(md, doc, label)
            out["folders"][year].append(rec)
            errs = [ch for ch, e in rec["channels"].items() if "error" in e]
            print("  %s  channels=%d  schema_errors=%d %s"
                  % (rec["match"], rec["channel_entries"], len(errs), errs[:5]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
