import json, collections
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

ROOT = "G:/pureskill/csds-cs2"
man = json.load(open(ROOT + "/deliverables/email_package_2026-10-19_review/results/split_manifest.json", encoding="utf-8"))
assign = man["assign"]
allowed = pa.array([k for k, v in assign.items() if v in ("train", "val")], type=pa.string())

dset = ds.dataset("G:/pureskill/csds-cs2/data/combined/deaths.parquet", format="parquet")
flt = ds.field("match_key").isin(allowed)
tbl = dset.to_table(columns=["match_key", "player_id"], filter=flt)
pid = tbl["player_id"]
n = tbl.num_rows
print("rows", n)
print("player_id null_count", pid.null_count)
small = pc.sum(pc.cast(pc.less_equal(pid, pa.scalar(63, pa.int64())), pa.int64())).as_py()
large = pc.sum(pc.cast(pc.greater(pid, pa.scalar(63, pa.int64())), pa.int64())).as_py()
print("rows player_id<=63:", small, " rows player_id>63:", large)
vals = pc.unique(pid).to_pylist()
big = sorted(v for v in vals if v is not None and v > 63)
print("distinct player_id total:", len(vals), " distinct >63:", len(big))
if big:
    print("large-value min/max:", big[0], big[-1])
    print("large-value count of exactly 65535:", sum(1 for v in big if v == 65535))
    print("large-value count within 65000..65535:", sum(1 for v in big if 65000 <= v <= 65535))
    print("large-value count below 65000:", sum(1 for v in big if v < 65000))
# how many matches carry the large values
mk = tbl["match_key"]
mask = pc.greater(pid, pa.scalar(63, pa.int64()))
mkl = pc.filter(mk, mask)
print("matches carrying large player_id values:", len(set(mkl.to_pylist())))
# per-match count of distinct player_id values among small ones (slot-like check)
g = tbl.group_by("match_key").aggregate([("player_id", "count_distinct")])
cd = g["player_id_count_distinct"].to_pylist()
print("matches:", len(cd), " max distinct player_id per match:", max(cd), " median:", sorted(cd)[len(cd)//2])
