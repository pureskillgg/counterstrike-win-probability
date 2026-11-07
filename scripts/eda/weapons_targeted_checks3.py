"""Third round of targeted weapons checks (pyarrow, read-only).

  A. weapon_action reload spam: gap between consecutive reload ticks per (match, round, player),
     outlier matches vs all matches.
  B. entity_id == 999: entity_id_fixed values on those rows (is raw 999 a real index or a placeholder?).
  C. grenade_state tick_throw == -3 and molotov_fire entity_id_fixed == -1: event context.
"""
import glob
import os
from collections import Counter, defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
OUTLIERS = None  # filled from the second run's ranking below


def pages(ch):
    folder = glob.glob(os.path.join(TOME, f"basic_{ch}.*"))[0]
    return sorted(p for p in glob.glob(os.path.join(folder, "dataframe_*")) if os.path.isfile(p))


def read(ch, cols):
    return pa.concat_tables([pq.read_table(p, columns=cols) for p in pages(ch)])


def check_a():
    print("== A. weapon_action reload gaps (ticks between consecutive reloads of the same player in a round)")
    t = read("weapon_action", ["match_key", "round", "tick", "player_id", "event_type"])
    t = t.filter(pc.equal(t.column("event_type"), "reload"))
    mk = t.column("match_key").to_pylist()
    rd = t.column("round").to_pylist()
    pid = t.column("player_id").to_pylist()
    tk = t.column("tick").to_pylist()
    grp = defaultdict(list)
    for a, b, c, d in zip(mk, rd, pid, tk):
        grp[(a, b, c)].append(d)
    gaps_by_match = defaultdict(list)
    rows_by_match = Counter(mk)
    for (a, b, c), ticks in grp.items():
        ticks.sort()
        gaps_by_match[a].extend(np.diff(ticks).tolist())
    med_rows = np.median(list(rows_by_match.values()))
    worst = sorted(rows_by_match, key=lambda k: -rows_by_match[k])[:3]
    allg = np.concatenate([np.array(g) for g in gaps_by_match.values() if g]) if gaps_by_match else np.array([])
    print(f"  all matches: reload gaps n={allg.size} median={np.median(allg):.0f} p10={np.percentile(allg, 10):.0f} "
          f"share<=2 ticks={np.mean(allg <= 2):.3f}")
    for k in worst:
        g = np.array(gaps_by_match[k])
        print(f"  {k.split('/')[-1][:8]} reloads={rows_by_match[k]} gaps median={np.median(g):.0f} "
              f"share<=2 ticks={np.mean(g <= 2):.3f}")
    print(f"  median reload rows per match={med_rows:.0f}")


def check_b():
    print("== B. entity_id == 999: entity_id_fixed on those rows")
    for ch in ["grenade_state", "grenade_bounce", "molotov_state", "molotov_fire"]:
        cols = ["entity_id", "entity_id_fixed"] if ch in ("grenade_state", "molotov_state", "molotov_fire") else \
               ["entity_id"]
        if ch == "grenade_bounce":
            cols = ["entity_id"]
        t = read(ch, cols)
        m = pc.equal(t.column("entity_id"), 999)
        if "entity_id_fixed" in t.column_names:
            fx = t.filter(m).column("entity_id_fixed").to_pylist()
            if fx:
                print(f"  {ch}: n={len(fx)} entity_id_fixed min={min(fx)} max={max(fx)} distinct={len(set(fx))} "
                      f"top={Counter(fx).most_common(3)}")
        else:
            print(f"  {ch}: n={t.filter(m).num_rows} (no entity_id_fixed column)")


def check_c():
    print("== C. grenade_state tick_throw == -3 and molotov_fire entity_id_fixed == -1")
    t = read("grenade_state", ["event_type", "tick_throw", "match_key"])
    s = t.filter(pc.equal(t.column("tick_throw"), -3))
    print(f"  grenade_state tick_throw==-3: n={s.num_rows} events={Counter(s.column('event_type').to_pylist()).most_common(3)} "
          f"matches={len(set(s.column('match_key').to_pylist()))}")
    t = read("molotov_fire", ["entity_id_fixed", "match_key"])
    s = t.filter(pc.equal(t.column("entity_id_fixed"), -1))
    print(f"  molotov_fire entity_id_fixed==-1: n={s.num_rows} matches={len(set(s.column('match_key').to_pylist()))}")


if __name__ == "__main__":
    check_a()
    check_b()
    check_c()
