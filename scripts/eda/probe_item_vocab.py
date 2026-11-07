import glob
import pyarrow.compute as pc
import pyarrow.parquet as pq
base = "G:/pureskill/csds-cs2/tomes/2026-10-01_to_08/tome/csds/"
sets = {}
for ch in ["item_equip", "item_pickup", "item_dropped", "item_refund"]:
    vals = set()
    for p in sorted(glob.glob(base + f"basic_{ch}.2026-09-18,2026-10-08/dataframe_*")):
        t = pq.read_table(p, columns=["item"]).column("item")
        vals |= set(pc.unique(t).to_pylist())
    sets[ch] = vals
    print(ch, len(vals), sorted(vals)[:12])
allv = set().union(*sets.values())
for ch, v in sets.items():
    print(ch, "distinct item names:", len(v), "| lowercase-internal-looking:", sum(1 for x in v if x == x.lower() and " " not in x))
print("names in item_equip not in item_dropped:", sorted(sets["item_equip"] - sets["item_dropped"])[:20])
print("names in item_dropped not in item_equip:", sorted(sets["item_dropped"] - sets["item_equip"])[:20])
print("refund vs dropped overlap:", len(sets["item_refund"] & sets["item_dropped"]), "of", len(sets["item_refund"]))
