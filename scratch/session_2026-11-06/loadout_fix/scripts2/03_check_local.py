import sys, json
import numpy as np
S = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/loadout_fix"
sys.path.insert(0, S + "/scripts2")
from loadout_rule import FIELDS
off = np.load(S + "/offsets.npy").astype(np.int64)
NS = int(off[-1])
a16 = np.load(S + "/out/loadout_step_i16.npy", mmap_mode="r")
aeq = np.load(S + "/out/loadout_equip_i32.npy", mmap_mode="r")
print("shapes", a16.shape, a16.dtype, aeq.shape, aeq.dtype, "NS", NS)
F = FIELDS[:-1]
mx = np.full((2, len(F)), -10**9, dtype=np.int64)
mn = np.full((2, len(F)), 10**9, dtype=np.int64)
nz_steps = 0
eq_mx = np.full(2, -10**9, dtype=np.int64)
CH = 4_000_000
for i in range(0, NS, CH):
    blk = np.asarray(a16[i:i + CH]).astype(np.int64)
    mx = np.maximum(mx, blk.max(axis=0))
    mn = np.minimum(mn, blk.min(axis=0))
    eb = np.asarray(aeq[i:i + CH]).astype(np.int64)
    eq_mx = np.maximum(eq_mx, eb.max(axis=0))
    nz_steps += int(((blk != 0).any(axis=2)).any(axis=1).sum()) if False else int((np.abs(blk).sum(axis=(1, 2)) + np.abs(eb).sum(axis=1) > 0).sum())
print("steps with any nonzero value", nz_steps, "of", NS)
for j, f in enumerate(F):
    print(f"{f:14s} CT min {mn[0, j]:6d} max {mx[0, j]:6d} | T min {mn[1, j]:6d} max {mx[1, j]:6d}")
print("equip CT max", int(eq_mx[0]), "T max", int(eq_mx[1]))
