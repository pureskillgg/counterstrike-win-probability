import json
import os
import sys
import time
from collections import Counter

import numpy as np
import polars as pl

V2 = "/home/sagemaker-user/adam/builder/final/export_clean_v2/"
V3 = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout/"
V4 = "/home/sagemaker-user/adam/builder/final/export_clean_v4/"
FX = "/home/sagemaker-user/adam/builder/fixes/freeze_fix/"
OUT_DIR = "/home/sagemaker-user/adam/builder/fixes/readiness/"
OUT = OUT_DIR + "step_diff.json"
BLOCK = 4_000_000
NF = 19
FIELDS = ["hp", "armor_n", "armor_sum", "helmet_n", "kit_n", "defusing", "ak", "m4", "awp",
          "rifle", "sniper_other", "smg", "heavy", "no_primary", "flash", "smoke", "he", "fire", "decoy"]
T0 = time.time()


def prog(msg):
    print(f"{time.time() - T0:8.1f}s {msg}", file=sys.stderr, flush=True)


def main():
    keys = json.load(open(V2 + "match_keys.json"))
    keys3 = json.load(open(V3 + "match_keys.json"))
    keys4 = json.load(open(V4 + "match_keys.json"))
    NM = len(keys)
    kid = {k: i for i, k in enumerate(keys)}
    assert len(kid) == NM

    off = np.load(V4 + "offsets.npy").astype(np.int64)
    off2 = np.load(V2 + "offsets.npy").astype(np.int64)
    rm = np.load(V2 + "round_match.npy").astype(np.int64)
    rm4 = np.load(V4 + "round_match.npy").astype(np.int64)
    R = off.shape[0] - 1
    NS = int(off[-1])

    a16 = np.load(V3 + "loadout_step_i16.npy", mmap_mode="r")
    b16 = np.load(V4 + "loadout_step_i16.npy", mmap_mode="r")
    ae = np.load(V3 + "loadout_equip_i32.npy", mmap_mode="r")
    be = np.load(V4 + "loadout_equip_i32.npy", mmap_mode="r")

    checks = {
        "rounds": R,
        "matches": NM,
        "total_steps_from_offsets": NS,
        "match_keys_v2_eq_v3": keys == keys3,
        "match_keys_v2_eq_v4": keys == keys4,
        "offsets_v2_eq_v4": bool(np.array_equal(off, off2)),
        "round_match_v2_eq_v4": bool(np.array_equal(rm, rm4)),
        "offsets_nondecreasing": bool(np.all(np.diff(off) >= 0)),
        "v3_step_shape": list(a16.shape), "v3_step_dtype": str(a16.dtype),
        "v4_step_shape": list(b16.shape), "v4_step_dtype": str(b16.dtype),
        "v3_equip_shape": list(ae.shape), "v3_equip_dtype": str(ae.dtype),
        "v4_equip_shape": list(be.shape), "v4_equip_dtype": str(be.dtype),
        "int16_values_per_step": int(a16.shape[1] * a16.shape[2]),
        "int32_values_per_step": int(ae.shape[1]),
    }
    shape_ok = (a16.shape == b16.shape == (NS, 2, NF)
                and ae.shape == be.shape == (NS, 2)
                and str(a16.dtype) == "int16" and str(b16.dtype) == "int16"
                and str(ae.dtype) == "int32" and str(be.dtype) == "int32"
                and NS == 72567700 and len(rm) == R
                and checks["offsets_v2_eq_v4"] and checks["round_match_v2_eq_v4"]
                and checks["match_keys_v2_eq_v3"] and checks["match_keys_v2_eq_v4"])
    checks["shapes_and_alignment_ok"] = bool(shape_ok)
    if not shape_ok:
        raise SystemExit("shape or alignment check failed: " + json.dumps(checks))

    st3 = pl.read_csv(V3 + "match_loadout_status.csv", infer_schema_length=0)
    st4 = pl.read_csv(V4 + "match_loadout_status.csv", infer_schema_length=0)
    build3 = [""] * NM
    reason3 = [""] * NM
    split3 = [""] * NM
    date3 = [""] * NM
    scope4 = [""] * NM
    m_mismatch = 0
    unmapped3 = 0
    for m, mk, sp, dt, bs, rs in st3.select("m", "match_key", "split", "date", "build_status", "reason").rows():
        i = kid.get(mk)
        if i is None:
            unmapped3 += 1
            continue
        if int(m) != i:
            m_mismatch += 1
        build3[i] = bs or ""
        reason3[i] = rs or ""
        split3[i] = sp or ""
        date3[i] = dt or ""
    for mk, sc in st4.select("match_key", "v4_build_scope").rows():
        i = kid.get(mk)
        if i is not None:
            scope4[i] = sc or ""
    checks["status3_rows"] = st3.height
    checks["status4_rows"] = st4.height
    checks["status3_unmapped_keys"] = unmapped3
    checks["status3_m_index_mismatch"] = m_mismatch

    inv_keys = json.load(open(FX + "inv_matches.json"))
    is_inv_m = np.zeros(NM, dtype=bool)
    for k in inv_keys:
        is_inv_m[kid[k]] = True
    checks["inv_matches"] = int(is_inv_m.sum())

    acc_a = np.zeros(NM, np.int64)
    acc_b = np.zeros(NM, np.int64)
    acc_c = np.zeros(NM, np.int64)
    fb = np.zeros((2, NF), np.int64)
    eb = np.zeros(2, np.int64)
    n = Counter()

    vmin4 = np.full((2, NF), 32767, np.int64)
    vmax4 = np.full((2, NF), -32768, np.int64)
    vmin3 = np.full((2, NF), 32767, np.int64)
    vmax3 = np.full((2, NF), -32768, np.int64)
    neg4 = np.zeros((2, NF), np.int64)
    hp500_4 = np.zeros(2, np.int64)
    emin4 = np.full(2, 2**31 - 1, np.int64)
    emax4 = np.full(2, -(2**31), np.int64)
    emin3 = np.full(2, 2**31 - 1, np.int64)
    emax3 = np.full(2, -(2**31), np.int64)

    for s0 in range(0, NS, BLOCK):
        s1 = min(NS, s0 + BLOCK)
        x = np.asarray(a16[s0:s1])
        y = np.asarray(b16[s0:s1])
        xe = np.asarray(ae[s0:s1])
        ye = np.asarray(be[s0:s1])

        steps = np.arange(s0, s1, dtype=np.int64)
        r = np.searchsorted(off, steps, side="right") - 1
        m = rm[r]
        inv_s = is_inv_m[m]

        dz = x != y
        de = xe != ye
        ch_i16 = dz.any(axis=(1, 2))
        ch_e = de.any(axis=1)
        changed = ch_i16 | ch_e
        v3_i16 = (x != 0).any(axis=(1, 2))
        v3_e = (xe != 0).any(axis=1)
        v4_i16 = (y != 0).any(axis=(1, 2))
        v4_e = (ye != 0).any(axis=1)
        v3_nz = v3_i16 | v3_e
        v4_nz = v4_i16 | v4_e

        out = changed & ~inv_s
        a = out & ~v3_nz & v4_nz
        b = out & v3_nz & v4_nz
        c = out & v3_nz & ~v4_nz
        d = out & ~v3_nz & ~v4_nz
        diag = out & ~v3_i16 & v3_e

        n["steps_inv"] += int(inv_s.sum())
        n["changed_inv"] += int((changed & inv_s).sum())
        n["changed_out"] += int(out.sum())
        n["out_i16_changed"] += int((out & ch_i16).sum())
        n["out_equip_changed"] += int((out & ch_e).sum())
        n["out_equip_only"] += int((out & ~ch_i16 & ch_e).sum())
        n["case_a"] += int(a.sum())
        n["case_b"] += int(b.sum())
        n["case_c"] += int(c.sum())
        n["case_d"] += int(d.sum())
        n["diag_v3_i16_zero_equip_nonzero"] += int(diag.sum())
        n["out_steps_v3_empty"] += int((~inv_s & ~v3_nz).sum())
        n["out_steps_v4_empty"] += int((~inv_s & ~v4_nz).sum())

        acc_a += np.bincount(m[a], minlength=NM)
        acc_b += np.bincount(m[b], minlength=NM)
        acc_c += np.bincount(m[c], minlength=NM)

        if b.any():
            fb += dz[b].sum(axis=0, dtype=np.int64)
            eb += de[b].sum(axis=0, dtype=np.int64)

        vmin4 = np.minimum(vmin4, y.min(axis=0).astype(np.int64))
        vmax4 = np.maximum(vmax4, y.max(axis=0).astype(np.int64))
        vmin3 = np.minimum(vmin3, x.min(axis=0).astype(np.int64))
        vmax3 = np.maximum(vmax3, x.max(axis=0).astype(np.int64))
        neg4 += (y < 0).sum(axis=0, dtype=np.int64)
        hp500_4 += (y[:, :, 0] > 500).sum(axis=0, dtype=np.int64)
        emin4 = np.minimum(emin4, ye.min(axis=0).astype(np.int64))
        emax4 = np.maximum(emax4, ye.max(axis=0).astype(np.int64))
        emin3 = np.minimum(emin3, xe.min(axis=0).astype(np.int64))
        emax3 = np.maximum(emax3, xe.max(axis=0).astype(np.int64))

        prog(f"block {s0}-{s1} done")

    ch_out = acc_a + acc_b + acc_c
    changed_idx = np.nonzero(ch_out > 0)[0]
    total_by_group = Counter(f"{build3[i]} | {reason3[i]}" for i in range(NM))
    groups = {}
    for i in changed_idx:
        g = f"{build3[i]} | {reason3[i]}"
        d = groups.setdefault(g, {"matches_changed": 0, "matches_in_group_all": total_by_group[g],
                                  "changed_steps": 0, "case_a_steps": 0, "case_b_steps": 0,
                                  "case_c_steps": 0})
        d["matches_changed"] += 1
        d["changed_steps"] += int(ch_out[i])
        d["case_a_steps"] += int(acc_a[i])
        d["case_b_steps"] += int(acc_b[i])
        d["case_c_steps"] += int(acc_c[i])

    order = np.argsort(-ch_out, kind="stable")[:5]
    top5 = []
    for i in order:
        top5.append({
            "match_key": keys[i], "match_index": int(i), "date": date3[i], "split": split3[i],
            "v3_build_status": build3[i], "v3_reason": reason3[i],
            "changed_steps_outside_inv": int(ch_out[i]), "case_a": int(acc_a[i]),
            "case_b": int(acc_b[i]), "case_c": int(acc_c[i]),
        })

    field_b = {}
    field_b_total = {}
    for f in range(NF):
        field_b[FIELDS[f]] = {"CT": int(fb[0, f]), "T": int(fb[1, f])}
        field_b_total[FIELDS[f]] = int(fb[0, f] + fb[1, f])

    ranges = {}
    for f in range(NF):
        ranges[FIELDS[f]] = {
            "v3_min": int(min(vmin3[0, f], vmin3[1, f])), "v3_max": int(max(vmax3[0, f], vmax3[1, f])),
            "v4_min": int(min(vmin4[0, f], vmin4[1, f])), "v4_max": int(max(vmax4[0, f], vmax4[1, f])),
            "v4_negative_cells": int(neg4[0, f] + neg4[1, f]),
        }

    n_scope_new = sum(1 for i in changed_idx if scope4[i] != "")
    reconciles = (int(n["changed_out"]) == 10805134 and int(n["changed_inv"]) == 16485)

    res = {
        "task": "export_clean_v4 vs export_clean_v3_loadout per-second loadout arrays, read-only",
        "checks": checks,
        "steps": {
            "total": NS,
            "in_inv_matches": int(n["steps_inv"]),
            "outside_inv_matches": int(NS - n["steps_inv"]),
            "changed_in_inv_matches": int(n["changed_inv"]),
            "changed_outside_inv_matches": int(n["changed_out"]),
            "changed_outside_inv_i16_diff": int(n["out_i16_changed"]),
            "changed_outside_inv_equip_diff": int(n["out_equip_changed"]),
            "changed_outside_inv_equip_only": int(n["out_equip_only"]),
            "outside_inv_steps_v3_empty": int(n["out_steps_v3_empty"]),
            "outside_inv_steps_v4_empty": int(n["out_steps_v4_empty"]),
        },
        "case_a_v3_empty_or_missing_v4_has_values": int(n["case_a"]),
        "case_b_both_present_and_differ": int(n["case_b"]),
        "case_c_v3_present_v4_empty": int(n["case_c"]),
        "case_d_both_empty_but_changed": int(n["case_d"]),
        "diag_changed_outside_inv_v3_i16_all_zero_equip_nonzero": int(n["diag_v3_i16_zero_equip_nonzero"]),
        "case_b_cell_diffs_by_field_CT_T": field_b,
        "case_b_cell_diffs_by_field_total": field_b_total,
        "case_b_equip_diffs_CT_T": [int(eb[0]), int(eb[1])],
        "matches_total": NM,
        "matches_with_any_changed_step_outside_inv": int(len(changed_idx)),
        "matches_with_case_a": int((acc_a > 0).sum()),
        "matches_with_case_b": int((acc_b > 0).sum()),
        "matches_with_case_c": int((acc_c > 0).sum()),
        "changed_matches_that_are_new_in_v4_by_v4_build_scope": int(n_scope_new),
        "changed_matches_by_v3_status_group": groups,
        "top5_matches_by_changed_steps_outside_inv": top5,
        "reconciles_with_verify_v4_state": bool(reconciles),
        "verify_v4_state_reference": {"steps_changed_other_matches": 10805134,
                                      "steps_changed_inv_fixed_rounds": 16485,
                                      "steps_changed_inv_other_rounds": 0},
        "field_value_ranges_all_steps": ranges,
        "v4_hp_gt500_count_CT_T": [int(hp500_4[0]), int(hp500_4[1])],
        "v3_equip_min_CT_T": [int(emin3[0]), int(emin3[1])],
        "v3_equip_max_CT_T": [int(emax3[0]), int(emax3[1])],
        "v4_equip_min_CT_T": [int(emin4[0]), int(emin4[1])],
        "v4_equip_max_CT_T": [int(emax4[0]), int(emax4[1])],
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1), flush=True)
    prog("DONE")


if __name__ == "__main__":
    main()
