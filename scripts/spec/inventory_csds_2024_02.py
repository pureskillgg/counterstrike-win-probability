"""Read-only inventory of the 2024-02 CSDS copies (names and sizes only).

Writes one JSON file with, per copy:
  - match folders (day 01 immediate subdirectories)
  - every file as relative path -> size in bytes
  - folders that contain no files anywhere beneath them
  - for the tail copy C, file and folder counts for days 02..25 (summary only)

Nothing under E: is written. Usage:
  pixi run python -I G:\\pureskill\\csds-cs2\\scripts\\spec\\inventory_csds_2024_02.py <out_json>
"""
import json
import os
import sys

ROOTS = {
    "A": r"E:\csds_extracted\csds\2024\02\01",
    "B": r"E:\csds_recovered_20261011\csds\2024\02\01",
    "C": r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\01",
}
C_MONTH = r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02"


def inventory(root):
    match_folders = []
    files = {}
    empty_folders = []
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if entry.is_dir(follow_symlinks=False):
            match_folders.append(entry.name)
    for folder in match_folders:
        base = os.path.join(root, folder)
        folder_has_file = False
        all_dirs = []
        for dp, dn, fn in os.walk(base):
            all_dirs.append(dp)
            for name in fn:
                full = os.path.join(dp, name)
                rel = os.path.relpath(full, root).replace("\\", "/")
                files[rel] = os.path.getsize(full)
                folder_has_file = True
        if not folder_has_file:
            empty_folders.append(folder)
    return {
        "root": root,
        "match_folders": match_folders,
        "files": files,
        "empty_folders": empty_folders,
    }


def day_counts(day_root):
    if not os.path.isdir(day_root):
        return {"exists": False, "folders": 0, "files": 0}
    folders = 0
    files = 0
    for entry in os.scandir(day_root):
        if entry.is_dir(follow_symlinks=False):
            folders += 1
            for _dp, _dn, fn in os.walk(entry.path):
                files += len(fn)
    return {"exists": True, "folders": folders, "files": files}


def main(out_path):
    result = {}
    for key, root in ROOTS.items():
        result[key] = inventory(root)
        print(
            key,
            "match_folders=%d files=%d empty=%d"
            % (
                len(result[key]["match_folders"]),
                len(result[key]["files"]),
                len(result[key]["empty_folders"]),
            ),
            flush=True,
        )
    result["C_days_02_25"] = {}
    for d in range(2, 26):
        day = "%02d" % d
        result["C_days_02_25"][day] = day_counts(os.path.join(C_MONTH, day))
    tot_folders = sum(v["folders"] for v in result["C_days_02_25"].values())
    tot_files = sum(v["files"] for v in result["C_days_02_25"].values())
    print("C days 02-25 total folders=%d files=%d" % (tot_folders, tot_files), flush=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
