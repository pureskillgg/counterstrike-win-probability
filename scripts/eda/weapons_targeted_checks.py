"""Targeted follow-up checks for the weapons EDA (pyarrow, read-only).

Verifies the items the profiler flagged before they are ranked as goofy:
  1. Tick values 65535 / 32767 / 9999 and second=999 (tick 63936): spike or real timestamp?
  2. molotov_state: -1 encoding vs null encoding (player_id, tick_throw, player positions).
  3. weapon_action outlier match: rows per distinct round vs the group.
  4. weapon_fire phi/theta ranges (from notes/eda/weapons.json).
  5. weapon_fire weapon_name vs player_weapon_code: share of rows that disagree with the code's main name.
  6. weapon_fire player_id_fixed nulls (26 rows): examples.
"""
import glob
import json
import os
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
JSON_PATH = r"G:\pureskill\csds-cs2\notes\eda\weapons.json"
CHANNELS = ["weapon_fire", "weapon_action", "grenade_state", "grenade_bounce", "molotov_state", "molotov_fire"]


def pages(ch):
    folder = glob.glob(os.path.join(TOME, f"basic_{ch}.*"))[0]
    return sorted(p for p in glob.glob(os.path.join(folder, "dataframe_*")) if os.path.isfile(p))


def tick_counts(ch):
    parts_v, parts_c = [], []
    for p in pages(ch):
        vc = pc.value_counts(pq.read_table(p, columns=["tick"]).column(0).combine_chunks())
        parts_v.append(vc.field("values"))
        parts_c.append(vc.field("counts"))
    agg = (pa.table({"v": pa.chunked_array(parts_v), "c": pa.chunked_array(parts_c)})
           .group_by("v").aggregate([("c", "sum")]))
    return {int(v): int(c) for v, c in zip(agg.column("v").to_pylist(), agg.column("c_sum").to_pylist())}


def check_tick_spikes():
    print("== 1. tick sentinel-like values: count, neighbours (v-1, v+1), matches containing it")
    targets = [65535, 32767, 9999, 63936]
    for ch in CHANNELS:
        tc = tick_counts(ch)
        line = []
        for v in targets:
            n = tc.get(v, 0)
            nb = (tc.get(v - 1, 0), tc.get(v + 1, 0))
            line.append(f"{v}: n={n} nb={nb}")
        print(f"  {ch}: " + " | ".join(line))
    # matches holding tick==65535 in weapon_fire
    mk_hits = Counter()
    for p in pages("weapon_fire"):
        t = pq.read_table(p, columns=["match_key", "tick"])
        m = pc.equal(t.column("tick"), pa.scalar(65535, pa.int64()))
        sub = t.column("match_key").filter(m)
        for k in sub.to_pylist():
            mk_hits[k] += 1
    print(f"  weapon_fire tick==65535: matches={len(mk_hits)} top_match_rows={mk_hits.most_common(2)}")


def check_molotov_state():
    print("== 2. molotov_state -1 vs null encoding")
    p = pages("molotov_state")[0]
    t = pq.read_table(p, columns=["event_type", "player_id", "tick_throw", "player_x_pos", "burn_duration", "extinguisher_id"])
    et = t.column("event_type").to_pylist()
    pid = t.column("player_id").to_numpy()
    tt = t.column("tick_throw").to_numpy()
    xnull = np.asarray(t.column("player_x_pos").is_null().to_numpy(zero_copy_only=False))
    neg = pid == -1
    print(f"  player_id==-1: rows={int(neg.sum())}; event types={Counter([et[i] for i in np.flatnonzero(neg)]).most_common(4)}")
    print(f"  tick_throw==-1 rows={int((tt == -1).sum())} (same rows as player_id==-1: {bool(np.array_equal(neg, tt == -1))})")
    print(f"  player_x_pos null rows={int(xnull.sum())}; event types among them={Counter([et[i] for i in np.flatnonzero(xnull)]).most_common(4)}")
    print(f"  player_x_pos null AND player_id==-1: {int((xnull & neg).sum())}")
    bd = t.column("burn_duration").to_numpy(zero_copy_only=False)
    print(f"  burn_duration==-1 rows={int((bd == -1).sum())}; event types={Counter([et[i] for i in np.flatnonzero(bd == -1)]).most_common(3)}")


def check_weapon_action_outlier():
    print("== 3. weapon_action: rows per distinct round by match (top 3 by rows/round, and median)")
    rows, rounds = Counter(), {}
    for p in pages("weapon_action"):
        t = pq.read_table(p, columns=["match_key", "round"])
        mk = t.column("match_key").to_pylist()
        rd = t.column("round").to_pylist()
        for a, b in zip(mk, rd):
            rows[a] += 1
            rounds.setdefault(a, set()).add(b)
    ratio = {k: rows[k] / max(len(rounds[k]), 1) for k in rows}
    med = float(np.median(list(ratio.values())))
    top = sorted(ratio.items(), key=lambda kv: -kv[1])[:3]
    print(f"  median rows/round={med:.1f}")
    for k, r in top:
        print(f"    {k.split('/')[-1][:8]} rows={rows[k]} rounds={len(rounds[k])} rows/round={r:.1f} "
              f"max_round={max(rounds[k])}")


def check_angles_and_codes():
    d = json.load(open(JSON_PATH, encoding="utf-8"))
    wf = d["channels"]["weapon_fire"]["columns"]
    print("== 4. weapon_fire angle ranges (from JSON)")
    for nm in ["player_phi_ang", "player_theta_ang"]:
        c = wf[nm]
        print(f"  {nm}: min={c['min']:.3f} max={c['max']:.3f} median={c['median']:.3f} nulls={c['nulls']}")
    print("== 5. weapon_fire name vs code: rows disagreeing with the code's main name (from JSON pairs)")
    wc = d["channels"]["weapon_fire"]["weapon_code_name_pairs"]
    # Full pair counts are only stored for codes/names with more than one value; rebuild totals from them.
    code_tot, code_top = {}, {}
    for code, names in wc["codes_with_more_than_one_name"].items():
        total = sum(names.values())
        code_tot[code] = total
        code_top[code] = max(names.values())
    mismatch = sum(code_tot[c] - code_top[c] for c in code_tot)
    total_rows = d["channels"]["weapon_fire"]["rows"]
    print(f"  codes with >1 name: {len(code_tot)}; rows outside the main name of those codes: {mismatch} "
          f"({mismatch / total_rows:.5%} of {total_rows} rows)")


def check_fixed_nulls():
    print("== 6. weapon_fire player_id_fixed nulls: examples")
    shown = 0
    for p in pages("weapon_fire"):
        t = pq.read_table(p, columns=["match_key", "round", "tick", "player_id", "player_id_fixed", "weapon_name"])
        m = t.column("player_id_fixed").is_null().to_numpy(zero_copy_only=False)
        for i in np.flatnonzero(m)[:3]:
            row = {c: t.column(c)[int(i)].as_py() for c in t.column_names}
            row["match_key"] = row["match_key"].split("/")[-1][:8]
            print(f"  {row}")
            shown += 1
        if shown >= 3:
            break


if __name__ == "__main__":
    check_tick_spikes()
    check_molotov_state()
    check_weapon_action_outlier()
    check_angles_and_codes()
    check_fixed_nulls()
