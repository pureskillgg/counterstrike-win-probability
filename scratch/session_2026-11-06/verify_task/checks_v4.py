import json
import collections
import numpy as np
import polars as pl

V2 = "/home/sagemaker-user/adam/builder/final/export_clean_v2/"
V3 = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout/"
V4 = "/home/sagemaker-user/adam/builder/final/export_clean_v4/"
FX = "/home/sagemaker-user/adam/builder/fixes/freeze_fix/"
FS = "/home/sagemaker-user/adam/builder/fixes/freeze_scope/"


def hdr(s):
    print("\n=== " + s, flush=True)


hdr("1a keys and status")
keys2 = json.load(open(V2 + "match_keys.json"))
keys4 = json.load(open(V4 + "built_match_keys.json"))
st4 = pl.read_csv(V4 + "match_loadout_status.csv", infer_schema_length=0)
st3 = pl.read_csv(V3 + "match_loadout_status.csv", infer_schema_length=0)
print("v2 match_keys:", len(keys2), "unique:", len(set(keys2)))
print("v4 built_match_keys:", len(keys4), "unique:", len(set(keys4)))
print("v4 built set == v2 set:", set(keys4) == set(keys2), "| same order:", keys4 == keys2)
print("v4 status rows:", st4.height, "| unique keys:", st4["match_key"].n_unique(),
      "| status keys == v2 set:", set(st4["match_key"].to_list()) == set(keys2))
print("v4 build_status:", st4.group_by("build_status").len().sort("build_status").rows())
print("v4 v4_batch:", st4.group_by("v4_batch").len().sort("v4_batch").rows())
print("v3 status rows:", st3.height, "| v3 build_status:", st3.group_by("build_status").len().sort("build_status").rows())
new_keys = st4.filter(pl.col("v4_batch").fill_null("") != "")["match_key"].to_list()
new_set = set(new_keys)
print("new keys (v4_batch non-empty):", len(new_keys), "| unique:", len(new_set))
v3_status = dict(zip(st3["match_key"].to_list(), st3["build_status"].to_list()))
print("v3 build_status of new keys:", dict(collections.Counter(v3_status.get(k) for k in new_keys)))
print("v3 build_status of other v4 keys:", dict(collections.Counter(v3_status.get(k) for k in keys4 if k not in new_set)))
common = [c for c in st3.columns if c in st4.columns]
m4 = {k: r for k, r in zip(st4["match_key"].to_list(), st4.select(common).rows())}
diffs = collections.Counter()
for k, r3 in zip(st3["match_key"].to_list(), st3.select(common).rows()):
    if k in new_set:
        continue
    for c, a, b in zip(common, r3, m4[k]):
        if a != b:
            diffs[c] += 1
print("status columns differing v3 vs v4 for v3-built keys:", dict(diffs))

hdr("1b array shapes, step count, per-match coverage")
off = np.load(V2 + "offsets.npy").astype(np.int64)
rm = np.load(V2 + "round_match.npy").astype(np.int64)
rn = np.load(V2 + "round_num.npy").astype(np.int64)
NS = int(off[-1])
dur = np.diff(off)
print("offsets size:", off.size, "| rounds (round_match):", rm.size, "| export steps offsets[-1]:", NS, "| offsets[0]:", int(off[0]))
print("steps_alive v4 shape:", np.load(V4 + "steps_alive.npy", mmap_mode="r").shape)
A4 = np.load(V4 + "loadout_step_i16.npy", mmap_mode="r")
E4 = np.load(V4 + "loadout_equip_i32.npy", mmap_mode="r")
A3 = np.load(V3 + "loadout_step_i16.npy", mmap_mode="r")
E3 = np.load(V3 + "loadout_equip_i32.npy", mmap_mode="r")
print("v4 step:", A4.shape, A4.dtype, "| v4 equip:", E4.shape, E4.dtype)
print("v3 step:", A3.shape, A3.dtype, "| v3 equip:", E3.shape, E3.dtype)
print("v4 rows == export steps:", A4.shape[0] == NS and E4.shape[0] == NS,
      "| v3 rows == export steps:", A3.shape[0] == NS and E3.shape[0] == NS)
mkid = {k: i for i, k in enumerate(keys2)}
per_match = np.zeros(len(keys2), np.int64)
np.add.at(per_match, rm, dur)
print("matches with zero export steps:", int((per_match == 0).sum()),
      "| sum per-match steps == export steps:", int(per_match.sum()) == NS)
print("round_match non-decreasing:", bool(np.all(np.diff(rm) >= 0)))
is_new_match = np.zeros(len(keys2), bool)
for k in new_keys:
    is_new_match[mkid[k]] = True
is_new_round = is_new_match[rm]
print("new matches:", int(is_new_match.sum()), "| new rounds:", int(is_new_round.sum()),
      "| export steps in new rounds:", int(dur[is_new_round].sum()))

ov = pl.read_csv(FX + "override_rounds.csv", infer_schema_length=0)
fixed_pairs = [(k, int(r)) for k, r, s in zip(ov["match_key"].to_list(), ov["round"].to_list(), ov["status"].to_list()) if s == "fixed"]
is_fixed_round = np.zeros(rm.size, bool)
for k, rd in fixed_pairs:
    idx = np.nonzero((rm == mkid[k]) & (rn == rd))[0]
    assert idx.size == 1, (k, rd, idx.size)
    is_fixed_round[idx[0]] = True
print("fixed rounds mapped:", int(is_fixed_round.sum()), "| fixed rounds in new matches:", int((is_fixed_round & is_new_round).sum()))

hdr("1c v3 vs v4 full comparison, every step")
BLOCK = 4_000_000
c = collections.Counter()
for s0 in range(0, NS, BLOCK):
    s1 = min(NS, s0 + BLOCK)
    x = np.asarray(A3[s0:s1])
    y = np.asarray(A4[s0:s1])
    xe = np.asarray(E3[s0:s1])
    ye = np.asarray(E4[s0:s1])
    ch = np.any(x != y, axis=(1, 2)) | np.any(xe != ye, axis=1)
    nz3 = np.any(x != 0, axis=(1, 2)) | np.any(xe != 0, axis=1)
    nz4 = np.any(y != 0, axis=(1, 2)) | np.any(ye != 0, axis=1)
    r = np.searchsorted(off, np.arange(s0, s1, dtype=np.int64), side="right") - 1
    nw = is_new_round[r]
    fx = is_fixed_round[r]
    ot = ~nw & ~fx
    c["steps_scanned"] += s1 - s0
    c["changed_total"] += int(ch.sum())
    c["changed_in_new_rounds"] += int((ch & nw).sum())
    c["changed_in_fixed_rounds"] += int((ch & fx).sum())
    c["changed_in_other_rounds"] += int((ch & ot).sum())
    c["steps_in_new_rounds"] += int(nw.sum())
    c["v3_nonzero_in_new_rounds"] += int((nz3 & nw).sum())
    c["v4_nonzero_in_new_rounds"] += int((nz4 & nw).sum())
    c["v4_zero_in_new_rounds"] += int((~nz4 & nw).sum())
    c["steps_in_fixed_rounds"] += int(fx.sum())
    c["steps_in_other_rounds"] += int(ot.sum())
    c["v3_nonzero_in_other_rounds"] += int((nz3 & ot).sum())
    c["v4_nonzero_in_other_rounds"] += int((nz4 & ot).sum())
print(dict(c), flush=True)

hdr("1d loadout_round.npy v2 vs v4")
LR2 = np.asarray(np.load(V2 + "loadout_round.npy", mmap_mode="r"))
LR4 = np.asarray(np.load(V4 + "loadout_round.npy", mmap_mode="r"))
ax = tuple(range(1, LR4.ndim))
print("loadout_round v2:", LR2.shape, LR2.dtype, "| v4:", LR4.shape, LR4.dtype, "| rows == rounds:", LR4.shape[0] == rm.size)
dr = np.nonzero(np.any(LR2 != LR4, axis=ax))[0]
print("rows changed v2->v4:", dr.size, "| all inside fixed rounds:", bool(np.all(is_fixed_round[dr])))
nzr = np.any(LR4 != 0, axis=ax)
print("v4 loadout_round nonzero rows: new rounds", int((nzr & is_new_round).sum()), "of", int(is_new_round.sum()),
      "| other rounds", int((nzr & ~is_new_round).sum()), "of", int((~is_new_round).sum()))

hdr("3 freeze-tick defect count after the fix")
sumj = json.load(open(FS + "scope_summary.json"))
print("scope_summary: affected_rounds", sumj["affected_rounds"], "| affected_matches", sumj["affected_matches"],
      "| rounds_with_dup_alive", sumj["rounds_with_dup_alive"], "| rounds_with_unrostered_alive", sumj["rounds_with_unrostered_alive"],
      "| rounds_alive_not_10", sumj["rounds_alive_not_10"])
sr = pl.read_parquet(FS + "scope_rounds.parquet").with_columns(pl.col("round").cast(pl.Int64))
print("scope_rounds columns:", sr.columns)
aff = sr.filter(pl.col("affected")).select("match_key", "round")
print("scope_rounds rows:", sr.height, "| affected rounds:", aff.height, "| affected matches:", aff["match_key"].n_unique())
ovp = pl.read_csv(FX + "override_rounds.csv", infer_schema_length=0).with_columns(pl.col("round").cast(pl.Int64))
print("override_rounds rows:", ovp.height, "| status:", ovp.group_by("status").len().sort("status").rows())
fx = ovp.filter(pl.col("status") == "fixed").select("match_key", "round")
print("fixed rounds:", fx.height, "| unique:", fx.unique().height,
      "| fixed inside affected set:", fx.join(aff, on=["match_key", "round"], how="inner").height)
rem = aff.join(fx, on=["match_key", "round"], how="anti")
print("affected rounds still unfixed (affected - fixed):", rem.height, "| matches with unfixed rounds:", rem["match_key"].n_unique())
print("unfixed reasons:", ovp.filter(pl.col("status") == "not_fixed").group_by("reason").len().sort("len", descending=True).rows())
orow = pl.read_parquet(FX + "override_rows.parquet")
print("override_rows:", orow.height, "| side values:", orow.group_by("side").len().sort("side").rows())
gg = orow.group_by("match_key", "round").agg(pl.len().alias("n"), (pl.col("side") == 3).sum().alias("ct"), (pl.col("side") == 2).sum().alias("t"))
print("fixed rounds gated:", gg.height, "| n != 10:", gg.filter(pl.col("n") != 10).height,
      "| ct != 5:", gg.filter(pl.col("ct") != 5).height, "| t != 5:", gg.filter(pl.col("t") != 5).height)
print("override rounds vs fixed table, set equal:", set(map(tuple, gg.select("match_key", "round").rows())) == set(map(tuple, fx.rows())))
print("DONE", flush=True)
