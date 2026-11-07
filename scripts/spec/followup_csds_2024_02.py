"""Read-only follow-up checks for the 2024-02 CSDS day-01 comparison.

1. Per-folder file-name completeness against the union of names seen in A.
2. Byte-for-byte equality of the seven repaired files across A, B and C (all under 2 MB).
3. First/last 1 MB SHA-256 sample for A vs B (names and sizes already match on all files).

Nothing under E: is written. Usage:
  pixi run python -I G:\\pureskill\\csds-cs2\\scripts\\spec\\followup_csds_2024_02.py <inv.json> <out_json>
"""
import filecmp
import hashlib
import json
import os
import random
import sys

ONE_MB = 1024 * 1024
ROOTS = {
    "A": r"E:\csds_extracted\csds\2024\02\01",
    "B": r"E:\csds_recovered_20261011\csds\2024\02\01",
    "C": r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\01",
}
REPAIRED = [
    "0109c126-e81b-404b-bc22-86629092c53e/bomb_action",
    "063e4cbf-811e-4a25-a72a-eeaa11829dcd/bomb_action",
    "0d681e4b-78fb-4c7a-a110-3e89b1de85c5/other_death",
    "1347edcf-9305-4aee-99ae-e6fb9b6c6ca6/tick",
    "aa756ae0-7c97-4910-875a-c48bfda28b59/player_footstep",
    "ac5e31e6-ef42-447a-9944-3495e1497ae3/weapon_fire",
    "eeed5f70-c429-4cba-a732-a85c0da980b1/player_hurt",
]


def local(rel, copy):
    return os.path.join(ROOTS[copy], rel.replace("/", os.sep))


def edge(path, size):
    h = hashlib.sha256()
    t = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(min(ONE_MB, size)))
        if size > ONE_MB:
            fh.seek(max(size - ONE_MB, ONE_MB))
            t.update(fh.read(size - max(size - ONE_MB, ONE_MB)))
    return h.hexdigest(), (t.hexdigest() if size > ONE_MB else None)


def main(inv_path, out_path):
    inv = json.load(open(inv_path, encoding="utf-8"))
    fA, fB, fC = inv["A"]["files"], inv["B"]["files"], inv["C"]["files"]
    out = {}

    # 1. Name completeness. Union of basenames over all A folders.
    per_folder = {}
    for k in fA:
        f, n = k.split("/", 1)
        per_folder.setdefault(f, set()).add(n)
    union_names = set().union(*per_folder.values())
    out["A_union_name_count"] = len(union_names)
    out["A_folder_name_count_histogram"] = {}
    for names in per_folder.values():
        out["A_folder_name_count_histogram"][str(len(names))] = (
            out["A_folder_name_count_histogram"].get(str(len(names)), 0) + 1
        )
    out["A_folders_missing_names"] = {
        f: sorted(union_names - names)
        for f, names in per_folder.items()
        if union_names - names
    }
    # Merged view of fd9f729f (A plus C-only names) vs the union.
    fd = "fd9f729f-b7a5-4021-bd71-a9aac40c79c8"
    c_names_fd = {k.split("/", 1)[1] for k in fC if k.startswith(fd + "/")}
    a_names_fd = per_folder.get(fd, set())
    out["fd9f729f_merged_names"] = len(a_names_fd | c_names_fd)
    out["fd9f729f_missing_vs_union_after_merge"] = sorted(union_names - (a_names_fd | c_names_fd))
    out["fd9f729f_extra_beyond_union_after_merge"] = sorted((a_names_fd | c_names_fd) - union_names)
    # Names of the three C-only folders against the A union.
    c_only = ["fde2b323-1bb1-4f41-b0bc-370d90e6e8dc", "feebe219-b1b3-42e7-9fcd-e3e9801d8ec0",
              "ffa6d31c-d72d-4bcf-b864-b03aa6743e77"]
    out["C_only_folder_name_check"] = {}
    for f in c_only:
        names = {k.split("/", 1)[1] for k in fC if k.startswith(f + "/")}
        out["C_only_folder_name_check"][f] = {
            "files": len(names),
            "missing_vs_A_union": sorted(union_names - names),
            "extra_vs_A_union": sorted(names - union_names),
        }

    # 2. Byte-level equality for the seven repaired files (small files, full compare).
    rep = []
    for rel in REPAIRED:
        row = {"file": rel, "size_A": fA.get(rel), "size_B": fB.get(rel), "size_C": fC.get(rel)}
        row["A_eq_B_bytes"] = filecmp.cmp(local(rel, "A"), local(rel, "B"), shallow=False)
        row["A_eq_C_bytes"] = filecmp.cmp(local(rel, "A"), local(rel, "C"), shallow=False)
        rep.append(row)
    out["repaired_byte_compare"] = rep

    # 3. Sample of A vs B: 7 repaired + 10 largest + 20 random (seeded), first/last 1 MB SHA-256.
    shared = sorted(set(fA) & set(fB))
    largest = sorted(shared, key=lambda k: fA[k], reverse=True)[:10]
    rng = random.Random(20241002)
    rand = rng.sample([k for k in shared if k not in largest and k not in REPAIRED], 20)
    sample = []
    for k in list(dict.fromkeys(REPAIRED + largest + rand)):
        sz = fA[k]
        ha, ta = edge(local(k, "A"), sz)
        hb, tb = edge(local(k, "B"), fB[k])
        sample.append({
            "file": k, "size": sz,
            "head_match": ha == hb, "tail_match": ta == tb,
        })
    out["A_vs_B_edge_sample"] = sample
    out["A_vs_B_edge_sample_total"] = len(sample)
    out["A_vs_B_edge_sample_all_match"] = all(r["head_match"] and r["tail_match"] for r in sample)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, (list, dict))}, indent=1))
    print("repaired A_eq_B:", [r["A_eq_B_bytes"] for r in rep])
    print("repaired A_eq_C:", [r["A_eq_C_bytes"] for r in rep])
    print("edge sample total:", len(sample), "all match:", out["A_vs_B_edge_sample_all_match"])
    print("fd9f729f missing after merge:", out["fd9f729f_missing_vs_union_after_merge"])
    print("A folders missing names:", len(out["A_folders_missing_names"]))
    print("histogram:", out["A_folder_name_count_histogram"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
