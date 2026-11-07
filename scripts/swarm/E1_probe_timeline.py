"""E1 probe: header population, round timeline, and item-event placement vs round_state ticks.

G: tomes only. Run: cd G:\\pureskill\\csds-cs2 ; pixi run python -I -X utf8 scripts\\swarm\\E1_probe_timeline.py
"""

import glob
import os

import pandas as pd
import pyarrow.parquet as pq

ROOT = r"G:\pureskill\csds-cs2"
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")


def pages(channel):
    d = [x for x in glob.glob(os.path.join(TOME, "basic_%s.*" % channel)) if os.path.isdir(x)]
    assert len(d) == 1, (channel, d)
    return sorted(glob.glob(os.path.join(d[0], "dataframe_*")))


def load(channel, cols=None):
    tbls = [pq.read_table(p, columns=cols) for p in pages(channel)]
    import pyarrow as pa
    return pa.concat_tables(tbls).to_pandas()


def main():
    hdr = pq.read_table(os.path.join(TOME, "header", "dataframe_00000")).to_pandas()
    print("header rows", len(hdr), "cols", list(hdr.columns)[:40])
    print("is_wingman", hdr["is_wingman"].value_counts(dropna=False).to_dict())
    print("match_type", hdr["match_type"].value_counts(dropna=False).head(10).to_dict())
    print("unique keys", hdr["key"].nunique())
    print("max_rounds", hdr["max_rounds"].value_counts(dropna=False).to_dict() if "max_rounds" in hdr else "no max_rounds col")
    print("tick_rate", hdr["tick_rate"].value_counts(dropna=False).to_dict())
    print("day counts", hdr["key"].str.extract(r"csds/2026/10/(d{2})/")[0].value_counts().sort_index().to_dict())

    rs = load("round_state", ["round", "tick", "event_type", "phase", "match_key"])
    print("round_state event types:", rs["event_type"].value_counts().to_dict())
    print("n matches in round_state", rs["match_key"].nunique())

    # timeline for first match, first 3 rounds
    mk = sorted(rs["match_key"].unique())[0]
    sub = rs[(rs.match_key == mk) & (rs["round"] <= 3)].sort_values(["round", "tick"])
    print("\nTIMELINE", mk)
    print(sub[["round", "tick", "event_type", "phase"]].to_string(max_rows=80))

    re_ = load("round_end", ["round", "tick", "winner_team_code", "match_key"])
    mkr = re_[re_.match_key == mk].sort_values("round")
    print(mkr.head(4).to_string())

    ie = load("item_equip", ["round", "tick", "player_id", "item", "weapon_type_code", "match_key"])
    ie = ie[(ie.match_key == mk) & (ie["round"] <= 3)].sort_values(["round", "tick"])
    print("\nITEM_EQUIP first 3 rounds, n=", len(ie))
    print(ie.head(40).to_string())
    print("item names", ie["item"].value_counts().head(30).to_dict())

    ip = load("item_pickup", ["round", "tick", "player_id", "item", "def_index", "match_key"])
    ip = ip[(ip.match_key == mk) & (ip["round"] <= 3)].sort_values(["round", "tick"])
    print("\nITEM_PICKUP first 3 rounds n=", len(ip))
    print(ip.head(25).to_string())

    rf = load("item_refund", ["round", "tick", "player_id", "item", "def_index", "match_key"])
    rf = rf[(rf.match_key == mk) & (rf["round"] <= 4)].sort_values(["round", "tick"])
    print("\nITEM_REFUND first 4 rounds n=", len(rf))
    print(rf.to_string())


if __name__ == "__main__":
    main()
