"""Read-only comparison of the 2024-02 CSDS day-01 copies, using the inventory JSON.

Phase 1 (no file reads): folder overlaps, A vs B identity, A vs C file name/size diff,
repaired-file checks, empty-folder listing.
Phase 2 (partial reads only): SHA-256 of first and last 1 MB for size-mismatched files.

Writes a JSON report and a CSV of per-file rows to the path given. Nothing under E: is written.
Usage:
  pixi run python -I G:\\pureskill\\csds-cs2\\scripts\\spec\\compare_csds_2024_02.py <inv.json> <out_dir>
"""
import csv
import hashlib
import json
import os
import sys

ONE_MB = 1024 * 1024

A_ROOT = r"E:\csds_extracted\csds\2024\02\01"
B_ROOT = r"E:\csds_recovered_20261011\csds\2024\02\01"
C_ROOT = r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\01"

REPAIRED = [
    "0109c126-e81b-404b-bc22-86629092c53e/bomb_action",
    "063e4cbf-811e-4a25-a72a-eeaa11829dcd/bomb_action",
    "0d681e4b-78fb-4c7a-a110-3e89b1de85c5/other_death",
    "1347edcf-9305-4aee-99ae-e6fb9b6c6ca6/tick",
    "aa756ae0-7c97-4910-875a-c48bfda28b59/player_footstep",
    "ac5e31e6-ef42-447a-9944-3495e1497ae3/weapon_fire",
    "eeed5f70-c429-4cba-a732-a85c0da980b1/player_hurt",
]


def folder_of(key):
    return key.split("/", 1)[0]


def edge_hashes(path, size):
    """SHA-256 of the first 1 MB and of the last 1 MB (or whole file if smaller)."""
    head_n = min(ONE_MB, size)
    with open(path, "rb") as fh:
        head = fh.read(head_n)
        if size > ONE_MB:
            tail_start = max(size - ONE_MB, head_n)
            fh.seek(tail_start)
            tail = fh.read(size - tail_start)
        else:
            tail = b""
    return {
        "head_sha256": hashlib.sha256(head).hexdigest(),
        "tail_sha256": hashlib.sha256(tail).hexdigest() if size > ONE_MB else None,
        "head_bytes": len(head),
        "tail_bytes": len(tail),
    }


def main(inv_path, out_dir):
    inv = json.load(open(inv_path, encoding="utf-8"))
    A, B, C = inv["A"], inv["B"], inv["C"]
    report = {}

    # ---- Task 1: match folder overlaps -----------------------------------
    sA, sB, sC = set(A["match_folders"]), set(B["match_folders"]), set(C["match_folders"])
    report["folders"] = {
        "A": len(sA), "B": len(sB), "C": len(sC),
        "A_and_B": len(sA & sB),
        "A_only": sorted(sA - sC),
        "C_only": sorted(sC - sA),
        "A_and_C": len(sA & sC),
        "B_only_vs_A": sorted(sB - sA),
        "A_only_vs_B": sorted(sA - sB),
        "A_and_B_identical_folder_sets": sA == sB,
        "C_only_count": len(sC - sA),
        "A_only_count": len(sA - sC),
        "A_and_B_only_in_both_count": len(sA & sB & sC),
    }

    # ---- Task 1 (cont.): A vs B file names and sizes ----------------------
    fA, fB, fC = A["files"], B["files"], C["files"]
    keysA, keysB = set(fA), set(fB)
    diff_AB_size = sorted(k for k in keysA & keysB if fA[k] != fB[k])
    report["A_vs_B"] = {
        "files_A": len(fA), "files_B": len(fB),
        "names_identical": keysA == keysB,
        "only_in_A": sorted(keysA - keysB),
        "only_in_B": sorted(keysB - keysA),
        "same_name_size_mismatch": diff_AB_size,
        "total_bytes_A": sum(fA.values()),
        "total_bytes_B": sum(fB.values()),
        "name_and_size_identical": keysA == keysB and not diff_AB_size,
    }

    # ---- Task 2: matches in both A and C ----------------------------------
    both_folders = sA & sC
    scope_A = {k: v for k, v in fA.items() if folder_of(k) in both_folders}
    scope_C = {k: v for k, v in fC.items() if folder_of(k) in both_folders}
    kA, kC = set(scope_A), set(scope_C)
    rows = []
    for k in sorted(kA | kC):
        a = scope_A.get(k)
        c = scope_C.get(k)
        if a is not None and c is not None:
            status = "same_size" if a == c else "size_mismatch"
        elif a is not None:
            status = "missing_in_C"
        else:
            status = "missing_in_A"
        rows.append({"path": k, "size_A": a, "size_C": c, "status": status})
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    report["task2_A_vs_C_in_shared_folders"] = {
        "shared_folders": len(both_folders),
        "files_in_A_shared_folders": len(scope_A),
        "files_in_C_shared_folders": len(scope_C),
        "status_counts": counts,
        "files_in_C_outside_shared_folders": sum(
            1 for k in fC if folder_of(k) not in both_folders
        ),
        "files_in_A_outside_shared_folders": sum(
            1 for k in fA if folder_of(k) not in both_folders
        ),
        "folders_with_C_files": sorted({folder_of(k) for k in fC}),
    }

    # ---- Task 3: the seven repaired files ---------------------------------
    rep_rows = []
    for rel in REPAIRED:
        rep_rows.append({
            "file": rel,
            "in_A": rel in fA, "size_A": fA.get(rel),
            "in_B": rel in fB, "size_B": fB.get(rel),
            "in_C": rel in fC, "size_C": fC.get(rel),
            "A_eq_C_size": (rel in fA and rel in fC and fA[rel] == fC[rel]),
            "folder_in_A": rel.split("/")[0] in sA,
            "folder_in_C": rel.split("/")[0] in sC,
        })
    report["task3_repaired"] = rep_rows

    # ---- Task 4: empty match folders in C ---------------------------------
    report["task4_C_empty_folders"] = {
        "count": len(C["empty_folders"]),
        "examples_first5": C["empty_folders"][:5],
        "empty_and_present_in_A": sum(1 for f in C["empty_folders"] if f in sA),
        "empty_and_A_has_files": sum(
            1 for f in C["empty_folders"]
            if any(folder_of(k) == f for k in fA)
        ),
        "nonempty_folders_in_C": sorted({folder_of(k) for k in fC}),
    }

    # ---- Task 5: partial-hash check on size mismatches (A vs C, shared folders)
    mismatched = [r["path"] for r in rows if r["status"] == "size_mismatch"]
    # Include the repaired files too, if they are present on both sides with different sizes.
    for rel in REPAIRED:
        if rel in fA and rel in fC and fA[rel] != fC[rel] and rel not in mismatched:
            mismatched.append(rel)
    hash_rows = []
    for rel in mismatched:
        pa = os.path.join(A_ROOT, rel.replace("/", os.sep))
        pc = os.path.join(C_ROOT, rel.replace("/", os.sep))
        row = {"file": rel, "size_A": fA[rel], "size_C": fC[rel]}
        row["A"] = edge_hashes(pa, fA[rel])
        row["C"] = edge_hashes(pc, fC[rel])
        row["head_match"] = row["A"]["head_sha256"] == row["C"]["head_sha256"]
        row["tail_match"] = row["A"]["tail_sha256"] == row["C"]["tail_sha256"]
        hash_rows.append(row)
    report["task5_edge_hashes"] = hash_rows

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "compare_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    with open(os.path.join(out_dir, "A_vs_C_file_rows.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "size_A", "size_C", "status"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(json.dumps({
        "folders": {k: v for k, v in report["folders"].items() if not isinstance(v, list)},
        "A_vs_B": {k: v for k, v in report["A_vs_B"].items() if not isinstance(v, list)},
        "task2": {k: v for k, v in report["task2_A_vs_C_in_shared_folders"].items() if not isinstance(v, list)},
        "task4": {k: v for k, v in report["task4_C_empty_folders"].items() if not isinstance(v, list)},
        "task5_count": len(hash_rows),
    }, indent=1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
