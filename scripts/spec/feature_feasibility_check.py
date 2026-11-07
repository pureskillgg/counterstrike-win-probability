"""Feature feasibility checks (read-only) for notes/spec/feature_feasibility.md.

1. Compares every tome channel's parquet columns with notes/spec/csds_dictionary.csv.
2. Checks each (table, column) pair the feasibility table relies on: in the dictionary, and in the tome.
3. Tests two conflicts in the notes: molotov_fire in the tome, and bomb site_code encoding.
4. Header map and round-time fields; round length on Target_Saved rounds (time-out, no plant).
5. player_info rank/wins behaviour within a match; rank_update timing.
6. Lists one E: match folder (directory listing only; nothing is read or written under E:).
"""
import collections
import csv
import glob
import os
import statistics

import pyarrow.parquet as pq

ROOT = r"G:\pureskill\csds-cs2"
DICT = os.path.join(ROOT, "notes", "spec", "csds_dictionary.csv")
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
E_MATCH = r"E:\csds_final\csds\2026\10\01\m_a661aadb0b4b87e35cdec94c"


def load_dictionary():
    cols = collections.defaultdict(set)
    with open(DICT, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cols[row["TableName"]].add(row["FieldName"])
    return cols


def tome_tables():
    out = {}
    for name in sorted(os.listdir(TOME)):
        path = os.path.join(TOME, name)
        if not os.path.isdir(path):
            continue
        base = name.split(".")[0]
        table = base[len("basic_"):] if base.startswith("basic_") else base
        out[table] = path
    return out


def pages(path):
    return sorted(glob.glob(os.path.join(path, "dataframe_*")))


TOMES = tome_tables()


def column_values(table, col):
    out = []
    for p in pages(TOMES[table]):
        out.extend(pq.read_table(p, columns=[col]).column(col).to_pylist())
    return out


CLAIMS = {
    "player_status": ["money", "current_equipment_cost", "round_start_equipment_cost",
                      "freezetime_end_equipment_cost", "equipment_value_calc", "health", "armor",
                      "place_name", "inv_primary", "inv_secondary", "is_in_buy_zone", "has_helmet",
                      "has_defuser", "player_controller_id"],
    "player_vector": ["x_pos", "weapon_code", "phi_ang", "team_code"],
    "tick": ["previous_phase"],
    "grenade_vector": ["x_pos", "grenade_type"],
    "world_item_vector": ["x_pos", "item"],
    "player_inputs": ["button_attack"],
    "player_hurt": ["health", "armor", "health_removed", "armor_removed", "effective_health_removed",
                    "attacker_id_fixed", "hit_box_code", "weapon_name", "attacker_team_code"],
    "bullet_damage": ["distance", "num_penetrations", "is_attacker_in_air"],
    "grenade_state": ["event_type", "entity_id_fixed", "tick_throw", "player_id_fixed", "x_pos"],
    "molotov_fire": ["fire_index", "is_burning", "x_pos", "entity_id", "entity_id_fixed"],
    "molotov_state": ["entity_id_fixed", "tick_throw", "burn_duration", "player_id_fixed",
                      "was_extinguished_by_smoke", "was_thrown_into_smoke"],
    "player_blind": ["blind_duration", "attacker_id_fixed", "player_id_fixed", "attacker_team_code",
                     "player_x_pos"],
    "item_equip": ["item", "weapon_type_code", "def_index"],
    "item_pickup": ["item", "def_index"],
    "item_dropped": ["reason", "item"],
    "item_refund": ["item"],
    "weapon_fire": ["weapon_name", "player_weapon_code"],
    "weapon_action": ["event_type"],
    "player_death": ["player_x_pos", "attacker_x_pos", "assister_x_pos", "attacker_id", "assister_id",
                     "is_flash_assist", "is_through_smoke", "is_attacker_blind", "is_noscope",
                     "weapon_name", "is_headshot", "player_id_fixed", "attacker_id_fixed",
                     "assister_id_fixed"],
    "player_footstep": ["player_x_pos", "player_id_fixed"],
    "bomb_state": ["event_type", "site_code", "player_x_pos", "player_id_fixed"],
    "bomb_action": ["event_type", "site_code"],
    "bomb_defuse": ["has_kit"],
    "round_state": ["phase", "t_score", "ct_score", "t_score_raw", "event_type"],
    "header": ["freeze_time", "round_time_defuse", "bomb_time", "max_rounds", "map_name",
               "is_wingman", "tick_rate"],
    "player_info": ["rank", "rank_type", "wins", "rank_raw", "rank_platform", "team_code",
                    "elo_platform", "radar_color_code"],
    "player_personal": ["steam_id", "name", "is_bot", "player_id_pawn"],
    "rank_update": ["rank_change", "rank_new", "win_count"],
    "round_end": ["winner_team_code", "win_reason_code", "player_count", "round"],
    "score_update": ["old_score", "new_score"],
    "team_change": ["old_team_code", "new_team_code", "silent"],
    "round_mvp": ["mvp_count"],
    "round_start": ["tick"],
}


def main():
    dic = load_dictionary()
    print("== tome channels vs dictionary ==")
    print("dictionary tables:", len(dic), "tome channel folders:", len(TOMES))
    print("in dictionary, not in tome:", sorted(set(dic) - set(TOMES)))
    print("in tome, not in dictionary:", sorted(set(TOMES) - set(dic)))
    schema = {}
    for table in sorted(TOMES):
        ps = pages(TOMES[table])
        names = pq.read_schema(ps[0]).names
        rows = sum(pq.ParquetFile(p).metadata.num_rows for p in ps)
        schema[table] = names
        extra = sorted(set(names) - dic.get(table, set()) - {"match_key"})
        missing = sorted(dic.get(table, set()) - set(names))
        print(f"{table}: pages={len(ps)} rows={rows} cols={len(names)} "
              f"not_in_dict={extra} dict_not_in_tome={missing}")

    print("\n== claimed (table.column): dict / tome ==")
    for table, cols in CLAIMS.items():
        parts = []
        for col in cols:
            in_dict = "y" if col in dic.get(table, set()) else "N"
            if table in schema:
                in_tome = "y" if col in schema[table] else "N"
            else:
                in_tome = "nochannel"
            parts.append(f"{col}[{in_dict}/{in_tome}]")
        print(f"{table}: " + " ".join(parts))

    print("\n== bomb site_code encodings in tome ==")
    for table in ("bomb_state", "bomb_action"):
        vals = collections.Counter(v for v in column_values(table, "site_code") if v is not None)
        print(table, "distinct", len(vals), "has 281:", 281 in vals, "has 282:", 282 in vals,
              "top5", vals.most_common(5))

    print("\n== header: maps and round/time fields ==")
    maps = collections.Counter(column_values("header", "map_name"))
    print("matches", sum(maps.values()), "maps", maps.most_common())
    for col in ("freeze_time", "round_time_defuse", "bomb_time", "max_rounds", "is_wingman"):
        print(col, collections.Counter(column_values("header", col)).most_common(6))

    print("\n== round length on Target_Saved (code 12) rounds, from live start ==")
    live = {}
    for mk, rnd, ev, sec in zip(column_values("round_state", "match_key"),
                                column_values("round_state", "round"),
                                column_values("round_state", "event_type"),
                                column_values("round_state", "second")):
        if ev == "round_freeze_end":
            live[(mk, rnd)] = sec
    lengths = []
    for mk, rnd, sec, code in zip(column_values("round_end", "match_key"),
                                  column_values("round_end", "round"),
                                  column_values("round_end", "second"),
                                  column_values("round_end", "win_reason_code")):
        if code == 12 and (mk, rnd) in live:
            lengths.append(sec - live[(mk, rnd)])
    if lengths:
        lengths.sort()
        q = statistics.quantiles(lengths, n=20)
        print("n", len(lengths), "min", round(lengths[0], 2), "5%", round(q[0], 2),
              "median", round(statistics.median(lengths), 2), "95%", round(q[-1], 2),
              "max", round(lengths[-1], 2))
        print("most common rounded lengths", collections.Counter(round(x) for x in lengths).most_common(5))

    print("\n== player_info: rank and wins within a match ==")
    mk = column_values("player_info", "match_key")
    pid = column_values("player_info", "player_id")
    rk = column_values("player_info", "rank")
    wins = column_values("player_info", "wins")
    rt = column_values("player_info", "rank_type")
    rank_seen = collections.defaultdict(set)
    wins_seen = collections.defaultdict(set)
    for a, b, c, w in zip(mk, pid, rk, wins):
        rank_seen[(a, b)].add(c)
        wins_seen[(a, b)].add(w)
    n = len(rank_seen)
    print("match-player pairs", n,
          "constant rank share", round(sum(1 for v in rank_seen.values() if len(v) == 1) / n, 4),
          "constant wins share", round(sum(1 for v in wins_seen.values() if len(v) == 1) / n, 4))
    print("rank==0 share", round(sum(1 for c in rk if c == 0) / len(rk), 4),
          "rank==9999 rows", sum(1 for c in rk if c == 9999),
          "rank_type==-1 share", round(sum(1 for x in rt if x == -1) / len(rt), 4))

    print("\n== rank_update timing ==")
    rmk = column_values("rank_update", "match_key")
    rrd = column_values("rank_update", "round")
    last_round = {}
    for a, b in zip(column_values("round_end", "match_key"), column_values("round_end", "round")):
        last_round[a] = max(last_round.get(a, 0), b)
    at_last = sum(1 for a, b in zip(rmk, rrd) if b == last_round.get(a))
    print("rank_update rows", len(rmk), "matches", len(set(rmk)),
          "rows in the match's last round", at_last,
          "round counter top3", collections.Counter(rrd).most_common(3))

    print("\n== E: match folder (listing only) ==")
    try:
        names = sorted(os.listdir(E_MATCH))
        print("entries:", names)
    except OSError as exc:
        print("not readable:", exc)


if __name__ == "__main__":
    main()
