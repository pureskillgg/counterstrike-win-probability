# -*- coding: utf-8 -*-
"""
verify_A1_2_round8.py
Resolves the round-8 disagreement between basic_player_info.team_code and engine spawn/death player_team_code
in the claimed matches. For each disagreeing player-round it reports the event's tick relative to the
round's round_start tick and round_end tick (ticks are engine ticks), and the spawn-only side (spawn is
recorded at round start, so it is the least ambiguous side source).
Also: side of planters in round 8 (bomb_begin_plant events), for the claimed matches.
"""
import os, glob, collections
import polars as pl

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
CLAIMED = [
    "csds/2026/10/01/7368f3ae-7353-4f68-8d04-7c97f22ea2b8/csds",
    "csds/2026/10/01/ec8a424b-067f-4209-a42c-36d5a65cd97f/csds",
    "csds/2026/10/02/5429f41b-7721-4232-9784-bd9a3e2045a5/csds",
    "csds/2026/10/02/5f014ad2-63f8-415d-9952-5984b2924721/csds",
    "csds/2026/10/02/83d3ede2-5226-4035-8f32-87a32f083665/csds",
    "csds/2026/10/02/b2e50666-9b0b-4dcd-b8cc-59703e6a3064/csds",
    "csds/2026/10/02/d69b442a-8c83-4700-8a25-a035082e0ba8/csds",
    "csds/2026/10/02/e60396ec-5ff6-4a18-a2c9-b7187b650485/csds",
    "csds/2026/10/03/2093742d-edc0-4041-961c-a261f4f26876/csds",
    "csds/2026/10/03/cbb071c2-83a9-4167-b4c1-6fed1e145da0/csds",
]
claimed = set(CLAIMED)


def chan(name):
    folder = [d for d in os.listdir(T) if d.startswith(name + ".")][0]
    return sorted(glob.glob(os.path.join(T, folder, "dataframe_*")))


def scan(name, cols):
    return pl.scan_parquet(chan(name)).select(cols).collect()


pi = scan("basic_player_info", ["match_key", "round", "player_id", "team_code"]).filter(pl.col("match_key").is_in(list(claimed)))
pp = scan("basic_player_personal", ["match_key", "round", "player_id", "steam_id"]).filter(pl.col("match_key").is_in(list(claimed)))
sp = scan("basic_player_spawn", ["match_key", "round", "tick", "player_id", "player_team_code"]).filter(pl.col("match_key").is_in(list(claimed)))
dt = scan("basic_player_death", ["match_key", "round", "tick", "player_id", "player_team_code"]).filter(pl.col("match_key").is_in(list(claimed)))
rs_ = scan("basic_round_start", ["match_key", "round", "tick"]).filter(pl.col("match_key").is_in(list(claimed)))
re_ = scan("basic_round_end", ["match_key", "round", "tick", "winner_team_code"]).filter(pl.col("match_key").is_in(list(claimed)))
ba = scan("basic_bomb_action", ["match_key", "round", "tick", "event_type", "player_id", "player_team_code"]).filter(pl.col("match_key").is_in(list(claimed)))

pmap = {(r[0], r[1]): r[2] for r in pp.group_by(["match_key", "player_id"]).agg(pl.col("steam_id").first().alias("sid")).iter_rows()}
pi_side = {}
for mk, r, pid, tc in pi.iter_rows():
    if (mk, pid) in pmap and tc in (2, 3):
        pi_side[(mk, r, pmap[(mk, pid)])] = tc

rstart = {(mk, r): t for mk, r, t in rs_.iter_rows()}
rend = {(mk, r): (t, w) for mk, r, t, w in re_.iter_rows()}

print("round_start rows:", rs_.height, "| round_end rows:", re_.height)
print("\n== per claimed match: round 8 and 9 ticks, and disagreeing events ==")
for k in CLAIMED:
    print("\n", k.split("/")[3], k.split("/")[4][:8], "round_start r8/r9 ticks:", rstart.get((k, 8)), rstart.get((k, 9)),
          "round_end r8 tick/winner:", rend.get((k, 8)))
    # player_info sides in round 8 and 9
    for r in (8, 9):
        sides = sorted([(sid, tc) for (mk, rr, sid), tc in pi_side.items() if mk == k and rr == r])
        print("   pinfo r%d:" % r, sides)
    # events in round 8 for the claimed match: spawn and death side vs pinfo
    for name, df in (("spawn", sp), ("death", dt)):
        sub = df.filter((pl.col("match_key") == k) & (pl.col("round") == 8))
        for pid, tick, tc in sub.select(["player_id", "tick", "player_team_code"]).iter_rows():
            sid = pmap.get((k, pid))
            pinf = pi_side.get((k, 8, sid))
            if pinf is not None and tc in (2, 3) and tc != pinf:
                rel = "after_round_end" if rend.get((k, 8)) and tick > rend[(k, 8)][0] else "before_round_end"
                print("   DISAGREE %s r8 player=%s tick=%s event_side=%s pinfo_side=%s (%s)" % (name, sid, tick, tc, pinf, rel))
    sp8 = sp.filter((pl.col("match_key") == k) & (pl.col("round") == 8))
    spawn_dis = 0
    spawn_ok = 0
    for pid, tc in sp8.select(["player_id", "player_team_code"]).iter_rows():
        sid = pmap.get((k, pid))
        pinf = pi_side.get((k, 8, sid))
        if pinf is None or tc not in (2, 3):
            continue
        if tc == pinf:
            spawn_ok += 1
        else:
            spawn_dis += 1
    print("   round 8 spawn-only agree with pinfo:", spawn_ok, "| disagree:", spawn_dis)
    # planters in round 8
    pl8 = ba.filter((pl.col("match_key") == k) & (pl.col("round") == 8) & (pl.col("event_type") == "bomb_begin_plant"))
    print("   round 8 plant events (engine side):", pl8.select(["tick", "player_team_code"]).rows())
    pl9 = ba.filter((pl.col("match_key") == k) & (pl.col("round") == 9) & (pl.col("event_type") == "bomb_begin_plant"))
    print("   round 9 plant events (engine side):", pl9.select(["tick", "player_team_code"]).rows())
