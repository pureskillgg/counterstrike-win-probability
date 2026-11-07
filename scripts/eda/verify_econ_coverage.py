"""Coverage by match_type and timing of economy/weapon events relative to each round.

Read-only. Uses header match_type, round_state round_freeze_end (live start) and
round_end (round over) from the 2026-10-01_to_08 csds tome.

Run:  cd G:\\pureskill\\csds-cs2 ; pixi run python -I scripts\\eda\\verify_econ_coverage.py
"""
import glob
import os

import pandas as pd
import pyarrow.parquet as pq

ROOT = "G:/pureskill/csds-cs2/tomes/2026-10-01_to_08/tome/csds"


def folder(ch):
    c = glob.glob(os.path.join(ROOT, "basic_%s.*" % ch))
    assert len(c) == 1, (ch, c)
    return c[0]


def read_ch(ch, cols):
    frames = []
    for p in sorted(glob.glob(os.path.join(folder(ch), "dataframe_*"))):
        have = pq.ParquetFile(p).schema_arrow.names
        use = [c for c in cols if c in have]
        frames.append(pq.read_table(p, columns=use).to_pandas())
    return pd.concat(frames, ignore_index=True)


hdr = pq.read_table(os.path.join(ROOT, "header", "dataframe_00000"),
                    columns=["key", "match_type", "platform", "match_date"]).to_pandas()
hdr["batch"] = ["10-08" if str(d)[:10] == "2026-10-08" else
                ("10-01..07" if "2026-10-01" <= str(d)[:10] <= "2026-10-07" else "other")
                for d in hdr.match_date]
print("match_type counts:", hdr.match_type.value_counts().to_dict(), flush=True)
print("match_type by batch:", hdr.groupby(["batch", "match_type"]).size().to_dict(), flush=True)

CH = ["rank_update", "bullet_damage", "item_equip", "item_pickup", "item_dropped", "item_refund",
      "weapon_action", "grenade_state", "grenade_bounce", "molotov_state", "molotov_fire", "weapon_fire"]
present = {}
for ch in CH:
    keys = set(read_ch(ch, ["match_key"]).match_key.unique())
    present[ch] = keys
    row = hdr.assign(p=hdr.key.isin(keys))
    by_type = row.groupby("match_type").p.agg(["sum", "count"])
    by_batch = row.groupby("batch").p.agg(["sum", "count"])
    print("%-14s present %4d | type: %s | batch: %s" % (
        ch, len(keys),
        "; ".join("%s %d/%d" % (t, r["sum"], r["count"]) for t, r in by_type.iterrows()),
        "; ".join("%s %d/%d" % (t, r["sum"], r["count"]) for t, r in by_batch.iterrows())), flush=True)

comp = set(hdr.loc[hdr.match_type == "competitive", "key"])
print("rank_update == bullet_damage match sets:", present["rank_update"] == present["bullet_damage"], flush=True)
print("rank_update == competitive match set:", present["rank_update"] == comp,
      "| |rank & comp| =", len(present["rank_update"] & comp), "| comp-only:", len(comp - present["rank_update"]),
      "| rank-only:", len(present["rank_update"] - comp), flush=True)

# Timing relative to each round: pre-live (before round_freeze_end), live, post-round (after round_end)
rs = read_ch("round_state", ["match_key", "round", "event_type", "tick"])
freeze = (rs[rs.event_type == "round_freeze_end"].groupby(["match_key", "round"]).tick.min()
          .rename("freeze").reset_index())
end = read_ch("round_end", ["match_key", "round", "tick"]).groupby(["match_key", "round"]).tick.max() \
    .rename("end").reset_index()
print("rounds with freeze_end:", len(freeze), "| round_end rows:", len(end), flush=True)

for ch in ["item_equip", "item_pickup", "item_dropped", "item_refund", "molotov_state", "molotov_fire",
           "grenade_state", "weapon_fire"]:
    df = read_ch(ch, ["match_key", "round", "tick"])
    m = df.merge(freeze, on=["match_key", "round"], how="left").merge(end, on=["match_key", "round"], how="left")
    n = len(m)
    pre = int((m.tick < m.freeze).sum())
    post = int((m.tick > m.end).sum())
    nofz = int(m.freeze.isna().sum())
    live = n - pre - post - nofz
    print("%-14s rows %9d | pre-live %.3f | live %.3f | after round_end %.3f | no freeze_end row %d" % (
        ch, n, pre / n, live / n, post / n, nofz), flush=True)
