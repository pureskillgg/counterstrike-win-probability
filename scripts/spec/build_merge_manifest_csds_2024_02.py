"""Build the proposed non-overwriting merge manifest and the C day-01 empty-folder list.

Reads only the inventory JSON (no reads under E:). Writes under notes\\spec.
Proposed plan (not executed):
  - base: copy A (B is name- and size-identical; kept as a cross-check)
  - m_bb0ad43b: add the 7 C-only files that A lacks
  - the 3 C-only folders: add all of their files
  - the 7 repaired files: A is byte-identical to C, so no action beyond the base copy
  - the 7 C files that duplicate A byte-for-byte are listed as skipped
  - the 219 empty C folders are not copied

Usage:
  pixi run python -I G:\\pureskill\\csds-cs2\\scripts\\spec\\build_merge_manifest_csds_2024_02.py <inv.json> <notes_spec_dir>
"""
import csv
import json
import os
import sys

REPAIRED = {
    "m_9805aa614020949e127f8783/bomb_action",
    "m_c40f829f09b8e2223e446fda/bomb_action",
    "m_3203b9941c8d8f5f23454468/other_death",
    "m_856031b476cfb6efe4bd3584/tick",
    "m_7dbe5dcf62ae89f8488e1934/player_footstep",
    "m_59e9835567700182adb3936c/weapon_fire",
    "m_1547a130b440d46bc3138806/player_hurt",
}


def main(inv_path, out_dir):
    inv = json.load(open(inv_path, encoding="utf-8"))
    fA, fB, fC = inv["A"]["files"], inv["B"]["files"], inv["C"]["files"]
    rows = []
    for k in sorted(fA):
        note = "base_copy_A"
        if k in REPAIRED:
            note = "base_copy_A (repaired; byte-identical to C)"
        rows.append([k, fA[k], "A", "copy_from_A", note])
    for k in sorted(fC):
        if k in fA:
            if k in REPAIRED and fA[k] == fC[k]:
                rows.append([k, fC[k], "C", "skip_duplicate", "identical to A (size and bytes)"])
            else:
                rows.append([k, fC[k], "C", "review", "name exists in A with different size"])
        else:
            rows.append([k, fC[k], "C", "add_from_C", "not in A or B"])

    manifest = os.path.join(out_dir, "merge_manifest_day01.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["relative_path", "size_bytes", "source_copy", "action", "note"])
        w.writerows(rows)

    empties = inv["C"]["empty_folders"]
    empty_path = os.path.join(out_dir, "C_day01_empty_match_folders.txt")
    with open(empty_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(empties)) + "\n")

    actions = {}
    bytes_by = {}
    for r in rows:
        actions[r[3]] = actions.get(r[3], 0) + 1
        bytes_by[r[3]] = bytes_by.get(r[3], 0) + r[1]
    print("manifest rows:", len(rows), "actions:", actions)
    print("bytes by action:", bytes_by)
    print("empty folders written:", len(empties))

    # Per-day C counts for days 02-25 (summary only).
    days = inv["C_days_02_25"]
    print("C day 02-25 per-day folders/files:")
    for d in sorted(days):
        print(d, days[d]["folders"], days[d]["files"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
