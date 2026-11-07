"""Checks the 57,858-match training export (export_clean.tar) for personal data and split problems, and prints the numbers
used in docs/data/EXPORT_REVIEW.md.

Reads the archive without extracting it. Nothing is written except the printed report.
Inputs (paths can be overridden with the environment variables below):
  EXPORT_TAR     export_clean.tar, with the export_clean/ folder (match_keys.json, match_split.npy, README.txt, ...)
  MANIFEST_V2    split_manifest_v2.json, the split of record named in docs/final/MIRAGE_TEST_DEFINITION.md
  FROZEN_KEYS    frozen_test_keys.json (repo: data/benchmark/frozen_test_keys.json)
  HOLDOUT_B      holdout_22_25_test_keys.json (csds-cs2 data/combined)

Run with the pixi Python from the repo root:
    G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe scripts\\data\\review_export.py
"""

import collections
import io
import json
import os
import tarfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TAR = Path(os.environ.get("EXPORT_TAR", "C:/Users/13144/AppData/Local/Temp/export_clean.tar"))
MANIFEST = Path(os.environ.get("MANIFEST_V2", "C:/Users/13144/Documents/pureskill_internship/csds-cs2/data/combined/split_manifest_v2.json"))
FROZEN = Path(os.environ.get("FROZEN_KEYS", str(REPO / "data" / "benchmark" / "frozen_test_keys.json")))
HOLDOUT_B = Path(os.environ.get("HOLDOUT_B", "G:/pureskill/csds-cs2/data/combined/holdout_22_25_test_keys.json"))
LABEL = {0: "train", 1: "val", 2: "test"}  # the code mapping in docs/final/MIRAGE_TEST_DEFINITION.md


def member(tf, name):
    return tf.extractfile(f"export_clean/{name}").read()


def day(key):
    return "-".join(key.split("/")[1:4])


def main():
    with tarfile.open(TAR) as tf:
        readme = member(tf, "README.txt").decode("utf-8")
        verification = json.loads(member(tf, "verification.json"))
        loadout_names = json.loads(member(tf, "loadout_names.json"))
        keys = json.loads(member(tf, "match_keys.json"))
        codes = np.load(io.BytesIO(member(tf, "match_split.npy")))
        loadout_report = json.loads(member(tf, "loadout_report.json"))
        listing = [(m.name, m.size) for m in tf.getmembers() if m.isfile()]

    print("== Archive contents (name, bytes)")
    for name, size in listing:
        print(f"  {name}  {size:,}")
    print("== README stores per second:", readme.split("Stored per second")[1].split("\n")[0].strip(": ").strip())
    print("== Loadout columns (per round, aggregates only):", ", ".join(loadout_names))
    print("== verification.json:", json.dumps(verification))
    print("== loadout_report.json:", json.dumps(loadout_report))

    n = len(keys)
    print(f"== Match keys: {n:,} (unique {len(set(keys)):,}); codes: {codes.shape[0]:,}")
    shape_ok = all(k.startswith("csds/") and k.endswith("/csds") for k in keys)
    print("   every key has the csds/date/uuid/csds form:", shape_ok)
    dates = sorted(day(k) for k in keys)
    print("   date range:", dates[0], "to", dates[-1])
    months = collections.Counter(d[:7] for d in dates)
    print("   matches per month:", ", ".join(f"{m} {c:,}" for m, c in sorted(months.items())))

    name = [LABEL[int(c)] for c in codes]
    print("== Split labels in the export:", dict(collections.Counter(name)))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["assign"]
    agree = sum(1 for k, lab in zip(keys, name) if manifest.get(k) == lab)
    print(f"== Export split codes agree with split_manifest_v2 on {agree:,} of {n:,} keys")
    print("   manifest labels for the export keys:", dict(collections.Counter(manifest.get(k, "absent") for k in keys)))

    frozen = set(json.loads(FROZEN.read_text(encoding="utf-8")))
    in_export = [i for i, k in enumerate(keys) if k in frozen]
    print(f"== Frozen test keys: {len(frozen):,}; in the export: {len(in_export):,}")
    print("   their labels in the export:", dict(collections.Counter(name[i] for i in in_export)))

    holdout = set(json.loads(HOLDOUT_B.read_text(encoding="utf-8"))["match_keys"])
    print(f"== Holdout B keys: {len(holdout):,}; in the export: {len(holdout & set(keys)):,}")

    late = [i for i, k in enumerate(keys) if day(k) >= "2026-10-26"]
    print(f"== Matches dated 26 Oct 2026 or later: {len(late):,}")
    scorer = [i for i, k in enumerate(keys) if "2026-10-14" <= day(k) <= "2026-10-25"]
    print(f"== Matches dated 14 to 25 Oct 2026 (the scorer's test uploads): {len(scorer):,};",
          dict(collections.Counter(name[i] for i in scorer)))
    gdt = [i for i, k in enumerate(keys) if "2026-09-01" <= day(k) <= "2026-10-08"]
    print(f"== Matches dated 1 Sep to 8 Oct 2026 (the GDT window): {len(gdt):,};",
          dict(collections.Counter(name[i] for i in gdt)))


if __name__ == "__main__":
    main()
