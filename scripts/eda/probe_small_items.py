import glob
import pandas as pd
import pyarrow.parquet as pq
base = "G:/pureskill/csds-cs2/tomes/2026-10-01_to_08/tome/csds/"
r = pq.read_table(glob.glob(base + "basic_item_refund.2026-09-18,2026-10-08/dataframe_*")[0]).to_pandas()
print("item_refund rows", len(r), "matches", r.match_key.nunique())
print(r.item.value_counts().head(10).to_dict())
print("refund def_index counts:", r.def_index.value_counts().head(5).to_dict())
print("refund round range", r["round"].min(), r["round"].max(), "tick min", r.tick.min())
print("refund rows per match (top 3):", r.match_key.value_counts().head(3).to_dict())
d = pq.read_table(glob.glob(base + "basic_item_dropped.2026-09-18,2026-10-08/dataframe_*")[0], columns=["reason", "item", "def_index", "entity_id", "x_pos", "y_pos", "z_pos", "match_key", "round", "tick"]).to_pandas()
print("dropped reason:", d.reason.value_counts(dropna=False).to_dict())
print("dropped entity_id min/max:", d.entity_id.min(), d.entity_id.max(), "nunique", d.entity_id.nunique())
print("dropped x_pos range:", d.x_pos.min(), d.x_pos.max(), "nulls", d.x_pos.isna().sum())
print("dropped def_index nulls:", d.def_index.isna().sum(), "distinct", d.def_index.nunique())
print("dropped item top:", d.item.value_counts().head(8).to_dict())
