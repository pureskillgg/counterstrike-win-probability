"""E1 step 1: cache the G: tome columns the economy proxy needs (2026 canonical tome, days 01 to 08).

Writes parquet caches under logs\\swarm\\E1\\cache (not under E:, not report files).
Columns are projected per channel, so only the fields used by E1 are read.
Run: cd G:\\pureskill\\csds-cs2 ; pixi run python -I -X utf8 scripts\\swarm\\E1_cache.py
"""

import glob
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = r"G:\pureskill\csds-cs2"
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
CACHE = os.path.join(ROOT, "logs", "swarm", "E1", "cache")

SPEC = {
    "header": ["key", "is_wingman", "max_rounds", "match_type", "platform", "match_date", "tick_rate"],
    "round_end": ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"],
    "round_start": ["match_key", "round", "tick"],
    "round_state": ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"],
    "player_info": ["match_key", "round", "player_id", "team_code", "wins", "rank"],
    "item_equip": ["match_key", "round", "tick", "player_id", "item", "weapon_type_code", "player_team_code"],
    "item_pickup": ["match_key", "round", "tick", "player_id", "item", "def_index", "player_team_code"],
    "item_dropped": ["match_key", "round", "tick", "player_id", "item", "def_index", "reason", "player_team_code"],
    "item_refund": ["match_key", "round", "tick", "player_id", "item", "def_index", "player_team_code"],
}


def pages(name):
    if name == "header":
        return [os.path.join(TOME, "header", "dataframe_00000")]
    d = [x for x in glob.glob(os.path.join(TOME, "basic_%s.*" % name)) if os.path.isdir(x)]
    assert len(d) == 1, (name, d)
    return sorted(glob.glob(os.path.join(d[0], "dataframe_*")))


def main():
    os.makedirs(CACHE, exist_ok=True)
    for name, cols in SPEC.items():
        t0 = time.perf_counter()
        tbls = []
        for p in pages(name):
            tb = pq.read_table(p, columns=cols)
            tbls.append(tb)
        tbl = pa.concat_tables(tbls)
        out = os.path.join(CACHE, "%s.parquet" % name)
        pq.write_table(tbl, out, compression="zstd")
        print("%-12s rows=%9d cols=%d  %.1fs -> %s" % (name, tbl.num_rows, tbl.num_columns,
                                                       time.perf_counter() - t0, out), flush=True)


if __name__ == "__main__":
    main()
