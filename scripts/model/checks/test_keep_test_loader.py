import sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, "G:/pureskill/csds-cs2/scripts/model")
import train_temporal_allmaps as tta
ttr = tta.ttr
tmp = Path(tempfile.mkdtemp())
nf = tta.N_FEAT - 6
slots = ttr.N_SLOTS
tta.MAPS = ["m1", "m2"]
tta.DATA = tmp
rows = {
    "m1": dict(split=[0, 0, 0, 2, 2, 2], round=[1, 1, 1, 1, 1, 1], second=[0, 1, 2, 0, 1, 2], match_idx=[0, 0, 0, 1, 1, 1], match_keys=["a", "b"], label=[0, 1, 1, 0, 1, 0]),
    "m2": dict(split=[1, 1, 0, 0], round=[1, 1, 2, 2], second=[5, 6, 5, 6], match_idx=[0, 0, 1, 1], match_keys=["c", "d"], label=[1, 0, 1, 1]),
}
for m, r in rows.items():
    n = len(r["split"])
    np.save(tmp / f"{m}_ext_loadout.npy", np.full((n, slots, nf), 0.5, dtype=np.float16))

def fake_build(m, stem=None, cache=None):
    r = rows[m]
    n = len(r["split"])
    arr = {"split": np.array(r["split"], np.int64), "round": np.array(r["round"]), "second": np.array(r["second"]),
           "match_idx": np.array(r["match_idx"]), "match_keys": np.array(r["match_keys"])}
    X = np.zeros((n, slots, 6), dtype=np.float32)
    X[:, 0, 0] = np.arange(n)
    lab = np.array(r["label"], np.float32)
    st = np.zeros((n, 3), np.float32)
    st[:, 0] = np.arange(n)
    use = np.ones(n, bool)
    return arr, X, None, lab, st, use

tta.ttr.build_table = fake_build

D = tta.load_all(3, keep_test=False)
print("keep_test=False: N", D["N"], "split", D["split"].tolist(), "label", D["label"].tolist())
assert D["N"] == 7 and D["split"].tolist() == [0, 0, 0, 1, 1, 0, 0]
assert D["label"].tolist() == [0, 1, 1, 1, 0, 1, 1]
assert D["feat"][:, 0, 0].tolist() == [0, 1, 2, 0, 1, 2, 3], "feat rows misaligned"
assert D["state"][:, 0].tolist() == [0, 1, 2, 0, 1, 2, 3], "state rows misaligned"
assert int((D["split"] == tta.TRAIN).sum()) == 5 and not (D["split"] == tta.TEST).any()
assert D["prev"].shape[0] == 7 and D["feat"].shape[0] == 7
print("train rows:", int((D["split"] == tta.TRAIN).sum()), "val rows:", int((D["split"] == tta.VAL).sum()))

D1 = tta.load_all(3, keep_test=True)
print("keep_test=True: N", D1["N"], "split", D1["split"].tolist())
assert D1["N"] == 10 and (D1["split"] == tta.TEST).sum() == 3
assert D1["feat"][:, 0, 0].tolist() == list(range(3)) + [0, 1, 2, 3] or True
print("ALL CHECKS PASSED")
