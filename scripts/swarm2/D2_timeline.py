"""D2 timeline: plain-text round timeline for example 2024 matches (affected and clean).

For rounds 10-15 of each example match prints: side of team X, recorded winner, reason code, round_start tick,
freeze_end tick, round_end tick, bomb_planted ticks (with planter side), bomb outcome events (type, tick),
and player_death events (tick range, deaths by side). Side of a death uses team membership from the first valid round.

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\D2_timeline.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
label = "2024"


def read(ch, cols):
    fd = sorted(glob.glob(str(BASE / f"basic_{ch}.*")))[0]
    return pl.scan_parquet(sorted(glob.glob(fd + "/dataframe_*"))).select(cols).collect()


rounds = pl.read_parquet(ROOT / "logs" / "swarm2" / f"D2_rounds_{label}.parquet")
pi = read("player_info", ["match_key", "round", "player_id", "team_code"]).unique(["match_key", "round", "player_id"], keep="first")
rs_start = read("round_start", ["match_key", "round", "tick"]).sort("tick").unique(["match_key", "round"], keep="first")
re_ = read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).sort("tick").unique(["match_key", "round"], keep="first")
rstate = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
fe = rstate.filter(pl.col("event_type") == "round_freeze_end").sort("tick").unique(["match_key", "round"], keep="first")
bs = read("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
dth = read("player_death", ["match_key", "round", "tick", "player_id"])

# membership
first_valid = rounds.filter(pl.col("sx").is_not_null()).group_by("match_key").agg(pl.col("round").min().alias("F"))
pim = pi.join(first_valid, on="match_key").filter(pl.col("round") == pl.col("F"))
memX = set(tuple(x) for x in pim.filter(pl.col("team_code") == 2).select(["match_key", "player_id"]).rows())


def trace(mk, lo=10, hi=15):
    print(f"\n==== {mk}")
    r = rounds.filter((pl.col("match_key") == mk) & (pl.col("round") >= lo) & (pl.col("round") <= hi)).sort("round")
    for row in r.iter_rows(named=True):
        rd = row["round"]
        sx = row["sx"]
        sxp = row["sx_prev"]
        print(f"-- round {rd}: sx={sx} sx_prev={sxp} recorded_winner={row['winner']} reason={row['win_reason_code']} s_reason={row['s_reason']} s_score={row['s_score']}")
        s = rs_start.filter((pl.col("match_key") == mk) & (pl.col("round") == rd))
        f = fe.filter((pl.col("match_key") == mk) & (pl.col("round") == rd))
        e = re_.filter((pl.col("match_key") == mk) & (pl.col("round") == rd))
        print("   round_start tick:", s["tick"].to_list(), " freeze_end tick:", f["tick"].to_list(),
              " scores t/ct at freeze:", list(zip(f["t_score"].to_list(), f["ct_score"].to_list())),
              " round_end tick:", e["tick"].to_list())
        b = bs.filter((pl.col("match_key") == mk) & (pl.col("round") == rd)).sort("tick")
        print("   bomb events:", [(x["event_type"], x["tick"], x["player_team_code"]) for x in b.iter_rows(named=True)])
        d = dth.filter((pl.col("match_key") == mk) & (pl.col("round") == rd))
        if d.height:
            sides = []
            for x in d.iter_rows(named=True):
                inX = (mk, x["player_id"]) in memX
                if sx is None:
                    sides.append(None)
                else:
                    sides.append(sx if inX else 5 - sx)
            n2 = sum(1 for v in sides if v == 2)
            n3 = sum(1 for v in sides if v == 3)
            print(f"   deaths: n={d.height} tick range {d['tick'].min()}..{d['tick'].max()} victim side counts T={n2} CT={n3}")
        else:
            print("   deaths: none")


# affected example: first round-13 bomb mismatch in days 12+
aff = rounds.filter((pl.col("round") == 13) & pl.col("sx").is_not_null() & pl.col("day").is_in(["12", "13"]))
aff_bomb = []
for mk in aff["match_key"].unique().to_list()[:200]:
    b = bs.filter((pl.col("match_key") == mk) & (pl.col("round") == 13) & pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
    if b.height:
        ev = b["event_type"][0]
        side_bomb = 2 if ev == "bomb_exploded" else 3
        w = aff.filter(pl.col("match_key") == mk)["winner"][0]
        if side_bomb != w:
            aff_bomb.append(mk)
    if len(aff_bomb) >= 2:
        break
clean = rounds.filter((pl.col("round") == 13) & pl.col("sx").is_not_null() & pl.col("day").is_in(["01"]))
clean_bomb = []
for mk in clean["match_key"].unique().to_list()[:50]:
    b = bs.filter((pl.col("match_key") == mk) & (pl.col("round") == 13) & pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
    if b.height:
        clean_bomb.append(mk)
    if len(clean_bomb) >= 1:
        break
for mk in aff_bomb[:2]:
    trace(mk)
for mk in clean_bomb[:1]:
    trace(mk)
