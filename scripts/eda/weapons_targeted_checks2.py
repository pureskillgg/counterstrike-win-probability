"""Second round of targeted weapons checks (pyarrow, read-only).

  A. weapon_action: what the rows-per-round outlier matches contain.
  B. entity_id == 999 and grenade_state tick_throw == -3: spread over matches and event types.
"""
import glob
import os
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"


def pages(ch):
    folder = glob.glob(os.path.join(TOME, f"basic_{ch}.*"))[0]
    return sorted(p for p in glob.glob(os.path.join(folder, "dataframe_*")) if os.path.isfile(p))


def read(ch, cols):
    return pa.concat_tables([pq.read_table(p, columns=cols) for p in pages(ch)])


def check_a():
    print("== A. weapon_action: rows per distinct round, by match")
    t = read("weapon_action", ["match_key", "round", "tick", "player_id", "event_type"])
    mk = t.column("match_key").to_pylist()
    rd = t.column("round").to_pylist()
    ev = t.column("event_type").to_pylist()
    pid = t.column("player_id").to_pylist()
    tick = t.column("tick").to_pylist()
    rows = Counter(mk)
    rset = {}
    for k, r in zip(mk, rd):
        rset.setdefault(k, set()).add(r)
    ratio = {k: rows[k] / len(rset[k]) for k in rows}
    vals = np.array(list(ratio.values()))
    print(f"  matches={len(rows)} rows/round median={np.median(vals):.1f} p90={np.percentile(vals, 90):.1f} "
          f"max={vals.max():.1f}")
    order = sorted(ratio, key=lambda k: -ratio[k])
    show = order[:3] + [order[len(order) // 2]]
    for k in show:
        idx = [i for i, x in enumerate(mk) if x == k]
        evc = Counter(ev[i] for i in idx).most_common(4)
        players = len({pid[i] for i in idx})
        ticks = len({tick[i] for i in idx})
        per_rp = Counter((rd[i], pid[i]) for i in idx)
        print(f"  {k.split('/')[-1][:8]} rows={rows[k]} rounds={len(rset[k])} rows/round={ratio[k]:.1f} "
              f"events={evc} players={players} distinct_ticks={ticks} "
              f"rows_per_round_player_median={np.median(list(per_rp.values())):.0f} max={max(per_rp.values())}")
    print("  (last line is the median-ratio match)")


def check_b():
    print("== B. entity_id == 999 and tick_throw == -3")
    for ch in ["grenade_state", "grenade_bounce", "molotov_state", "molotov_fire"]:
        cols = ["match_key", "entity_id"] + (["event_type"] if ch != "grenade_bounce" else [])
        t = read(ch, cols)
        sub = t.filter(pc.equal(t.column("entity_id"), 999))
        n_mk = len(set(sub.column("match_key").to_pylist()))
        ev = Counter(sub.column("event_type").to_pylist()).most_common(3) if "event_type" in cols else "n/a"
        print(f"  {ch}: entity_id==999 rows={sub.num_rows} matches={n_mk} events={ev}")
    t = read("grenade_state", ["event_type", "tick_throw", "match_key"])
    sub = t.filter(pc.equal(t.column("tick_throw"), -3))
    print(f"  grenade_state tick_throw==-3 rows={sub.num_rows} "
          f"events={Counter(sub.column('event_type').to_pylist()).most_common(3)} "
          f"matches={len(set(sub.column('match_key').to_pylist()))}")


if __name__ == "__main__":
    check_a()
    check_b()
