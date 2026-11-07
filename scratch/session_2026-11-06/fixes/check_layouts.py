"""Read-only: compare the two export layouts used by the position builders (export_clean for the bucket builder, export_clean_v2 for the assembler).
Only index arrays are read (keys, round ids, freeze ticks, step counts, split codes). No labels, no position data."""
import json
import os
import numpy as np

F = os.path.expanduser("~/adam/builder/final")
A, B = F + "/export_clean", F + "/export_clean_v2"
ka = json.load(open(A + "/match_keys.json")); kb = json.load(open(B + "/match_keys.json"))
print("matches: export_clean", len(ka), "| export_clean_v2", len(kb), "| same set", set(ka) == set(kb), "| same order", ka == kb, flush=True)
def rounds(D, keys):
    rm = np.load(D + "/round_match.npy").astype(np.int64); rn = np.load(D + "/round_num.npy").astype(np.int64)
    fr = np.load(D + "/round_freeze.npy").astype(np.int64); off = np.load(D + "/offsets.npy")
    n = np.diff(off)
    return {(keys[m], int(r)): (int(f), int(k)) for m, r, f, k in zip(rm, rn, fr, n)}, len(rm), int(off[-1])
ra, na, sa = rounds(A, ka); rb, nb, sb = rounds(B, kb)
print("rounds: export_clean", na, "steps", sa, "| export_clean_v2", nb, "steps", sb, flush=True)
both = set(ra) & set(rb)
print("(match, round) keys only in export_clean", len(set(ra) - set(rb)), "| only in v2", len(set(rb) - set(ra)), "| in both", len(both), flush=True)
fz = sum(1 for k in both if ra[k][0] != rb[k][0]); nd = sum(1 for k in both if ra[k][1] != rb[k][1])
print("freeze tick differs in", fz, "rounds | step count differs in", nd, "rounds", flush=True)
sp = np.load(B + "/match_split.npy")
idx_b = {k: i for i, k in enumerate(kb)}
ex = [k for k in ["csds/2023/11/23/m_c5fe35c5412787409568f8e7/csds", "csds/2023/11/24/m_dc8169e604c7aa9add7019b2/csds"]]
print("split code of the two probe-candidate matches (by v2 index):", [int(sp[idx_b[k]]) if k in idx_b else None for k in ex], flush=True)
