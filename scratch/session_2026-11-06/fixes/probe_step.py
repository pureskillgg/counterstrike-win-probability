"""One-match timing and parity probe for the every-second bucket builder (read-only on S3 and final/; scratch only).
The probe match is a train-split match (export split code 0) with path date before 2026/09/01, so no holdout or test match is read.
Compares STEP=2 (the old even-step build) with STEP=1 (every second) on the same match."""
import os
import shutil
import sys
import time

sys.argv = ["probe"]
sys.path.insert(0, os.path.expanduser("~/adam/builder/fixes"))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pos_bucket_step as PB  # noqa: E402  (module-level setup only; no chunk arguments, so nothing is written)
import make_pos_features as MPF  # noqa: E402

sp = np.load(PB.FIN + "/export_clean_v2/match_split.npy")
cands = [mk for mk in sorted(PB.bkeys & set(PB.kid))
         if int(sp[PB.kid[mk]]) == 0 and "/".join(mk.split("/")[1:4]) < "2026/09/01"]
mk = cands[0]
print("probe match", mk, "| split code 0 | path date", "/".join(mk.split("/")[1:4]), "| candidates", len(cands), flush=True)
folder = mk.rsplit("/", 1)[0]
names = ["player_vector", "player_status", "player_info", "player_blind"]
out, tim = {}, {}
for step in (2, 1):
    PB.STEP = step
    shutil.rmtree(f"{PB.SCR}/{folder}", ignore_errors=True)
    t = time.time()
    for nm in names:
        PB.fetch(folder, nm)
    t_dl = time.time() - t
    t = time.time()
    r = PB.one_match(mk)
    t_parse = time.time() - t
    assert r[0] == "ok", r
    _, pos, sta, G, ev = r
    pxy_one = PB.pxy.filter(pl.col("match_key") == mk)
    t = time.time()
    f = MPF.feats_for_day(pos, sta, G, pxy_one, PB.sites, step, ev)
    t_feat = time.time() - t
    out[step] = (pos, sta, G, f)
    tim[step] = (t_dl, t_parse, t_feat)
    print(f"STEP={step} | download {t_dl:.1f}s | parse+join {t_parse:.1f}s | features {t_feat:.1f}s | grid steps {G.height} | "
          f"pos rows {pos.height} | status rows {sta.height} | feature rows {f.height}", flush=True)

pos1, sta1, G1, f1 = out[1]
pos2, sta2, G2, f2 = out[2]
K = ["match_key", "round", "tick"]
feats = [c for c in f2.columns if c not in K]
both = f2.join(f1, on=K, how="inner", suffix="_s1")
print("even-second feature rows (STEP=2):", f2.height, "| matched in STEP=1:", both.height, flush=True)
bad = {}
for c in feats:
    x = both[c].cast(pl.Float64)
    y = both[c + "_s1"].cast(pl.Float64)
    bad[c] = int(((x.is_null() != y.is_null()) | (x.is_not_null() & y.is_not_null() & ((x - y).abs() > 1e-9))).sum())
print("even-second value mismatches, STEP=1 vs STEP=2 (expect none):", {k: v for k, v in bad.items() if v} or "none", flush=True)
odd = f1.join(f2.select(K), on=K, how="anti")
print("odd-second feature rows (STEP=1 only):", odd.height, flush=True)
print("pos rows by parity in STEP=1: even", pos1.filter(pl.col("second") % 2 == 0).height,
      "| odd", pos1.filter(pl.col("second") % 2 == 1).height, flush=True)
print("grid steps: STEP=1", G1.height, "| STEP=2", G2.height, "| ratio", round(G1.height / G2.height, 4), flush=True)
for c in ["min_enemy_dist", "c_site_hi", "c_blind_n", "c_min_bomb_dist"]:
    print(f"  null share {c:16s} odd rows {odd[c].is_null().mean():.4f} | even rows {f2[c].is_null().mean():.4f}", flush=True)
for step in (2, 1):
    a, b, c = tim[step]
    print(f"STEP={step} total {a + b + c:.1f}s; download share {100 * a / (a + b + c):.0f}%", flush=True)
