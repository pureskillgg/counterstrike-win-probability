"""verify_D3_4_e_format_coverage.py: format and coverage checks for the 8-round group, and map / day split of the flip.

Checks:
  E1 players per match (distinct weapon_fire player ids) for (8,) vs (12,) complete matches (2v2 would show ~4).
  E2 final starter score pairs for complete (8,) matches (MR8 first-to-9 would give a 9 winner; 16-round cap).
  E3 coverage: for (8,) matches without a player-side flip at 8|9, is there enough player data at the boundary?
  E4 first-post-swap disagreement (score delta side vs round_end winner) at round s+1 by map and by folder day,
     for the (12,) group (s = 12) and the (8,) group (s = 8).
ASSUMPTIONS (marked):
  A17 complete match = header starter total equals the number of round_end rows for the match.
  A18 player-side flip boundary as in verify_D3_4_d_player_sides.py (>= 4 common players, >= 60% changed).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D3_4_e_format_coverage.py
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(200)

TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
SIDE_BY_REASON = {1: 2, 9: 2, 13: 2, 17: 2, 7: 3, 8: 3, 11: 3, 12: 3, 18: 3}


def scan(ch, cols):
    folder = TOME / "header" if ch == "header" else TOME / f"basic_{ch}{SUF}"
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.scan_parquet(files).select(cols)


def day_of(m):
    return int(m.split("/")[3])


def clean(dt, dc):
    return (dc == 1 and dt == 0) or (dt == 1 and dc == 0)


def dside(dt, dc):
    if dt == 1 and dc == 0:
        return 2
    if dc == 1 and dt == 0:
        return 3
    return None


fe = (scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).collect()
      .filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"], maintain_order=True)
      .agg(pl.col("t_score").first(), pl.col("ct_score").first()))
live = {(m, r): (t, c) for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
by_m = collections.defaultdict(list)
for (m, r) in live:
    by_m[m].append(r)


def swaps_of(m):
    out = []
    for r in sorted(by_m[m]):
        if (m, r + 1) not in live:
            continue
        t1, c1 = live[(m, r)]
        t2, c2 = live[(m, r + 1)]
        dt, dc = t2 - t1, c2 - c1
        if clean(dt, dc) or (dt, dc) == (0, 0):
            continue
        if dt * dc <= 0:
            out.append(r)
    return tuple(out)


RE = scan("round_end", ["match_key", "round", "winner_team_code", "win_reason_code"]).collect().unique(
    ["match_key", "round"], keep="first", maintain_order=True)
win = {(m, r): (w, rc) for m, r, w, rc in RE.select(["match_key", "round", "winner_team_code", "win_reason_code"]).iter_rows()}
n_end = collections.Counter(m for (m, _) in win)
H = scan("header", ["key", "map_name", "t_starters_score_final", "ct_starters_score_final"]).collect()
hdr = {k: (mp, tf, cf) for k, mp, tf, cf in H.iter_rows()}

matches = [m for m in by_m if 12 <= day_of(m) <= 25]
sig = {m: swaps_of(m) for m in matches}
groups = collections.defaultdict(list)
for m in matches:
    groups[str(sig[m])].append(m)

# ---------- E1: distinct players per match ----------
wf = (scan("weapon_fire", ["match_key", "round", "player_id_fixed", "player_team_code"])
      .filter(pl.col("player_team_code").is_in([2, 3]))
      .group_by(["match_key", "player_id_fixed"]).agg(pl.len().alias("rows"))
      .collect())
players_of = collections.Counter(wf["match_key"].to_list())
print("== E1 distinct players (weapon_fire) per match ==")
for g in ["(12,)", "(8,)"]:
    cnt = collections.Counter()
    for m in groups[g]:
        if hdr[m][1] + hdr[m][2] == n_end[m]:
            cnt[players_of.get(m, 0)] += 1
    print(f"  {g} complete: {sorted(cnt.items())[:12]}")

# ---------- E2: final score pairs for complete (8,) matches ----------
pairs = collections.Counter()
for m in groups["(8,)"]:
    tf, cf = hdr[m][1], hdr[m][2]
    if tf + cf == n_end[m]:
        pairs[(max(tf, cf), min(tf, cf))] += 1
print("\n== E2 final score pairs (winner, loser), complete (8,) matches ==")
print("  ", sorted(pairs.items(), key=lambda x: -x[1])[:15])

# ---------- E3: coverage at boundary 8|9 for (8,) matches with and without a player flip ----------
JOIN = (scan("weapon_fire", ["match_key", "round", "player_id_fixed", "player_team_code"])
        .filter(pl.col("player_team_code").is_in([2, 3]))
        .group_by(["match_key", "round", "player_id_fixed"])
        .agg(pl.col("player_team_code").mode().first().alias("side"))
        .collect())
nxt = JOIN.with_columns((pl.col("round") - 1).alias("round")).rename({"side": "side_next"})
jj = JOIN.join(nxt, on=["match_key", "round", "player_id_fixed"], how="inner")
ag = jj.group_by(["match_key", "round"]).agg(pl.len().alias("n"), (pl.col("side") != pl.col("side_next")).sum().alias("f"))
bd = collections.defaultdict(dict)
for m, r, n, f in ag.select(["match_key", "round", "n", "f"]).iter_rows():
    bd[m][r] = (n, f)
print("\n== E3 coverage at boundary 8|9 (common players), (8,) matches ==")
cov = collections.Counter()
for m in groups["(8,)"]:
    n8 = bd.get(m, {}).get(8, (0, 0))
    fl = n8[0] >= 4 and n8[1] / n8[0] >= 0.6
    cov["flip" if fl else ("no_flip_enough_players" if n8[0] >= 4 else "insufficient_players")] += 1
print("  ", dict(cov))
print("  flip@8 fraction among matches with >=4 common players:",
      round(cov["flip"] / max(1, cov["flip"] + cov["no_flip_enough_players"]), 3))

# ---------- E4: first-post-swap disagreement by map and by day ----------
def k1_table(s, label):
    bymap = collections.defaultdict(lambda: [0, 0])
    byday = collections.defaultdict(lambda: [0, 0])
    for m in groups[str((s,))]:
        if (m, s + 1) not in win or (m, s + 1) not in live or (m, s + 2) not in live:
            continue
        t1, c1 = live[(m, s + 1)]
        t2, c2 = live[(m, s + 2)]
        dt, dc = t2 - t1, c2 - c1
        if not clean(dt, dc):
            continue
        ds = dside(dt, dc)
        dis = int(ds != win[(m, s + 1)][0])
        bymap[hdr[m][0]][0] += 1
        bymap[hdr[m][0]][1] += dis
        byday[day_of(m)][0] += 1
        byday[day_of(m)][1] += dis
    print(f"\n== E4 {label}: disagree at round {s+1} (score delta side vs round_end winner) ==")
    print("  by map:", {k: f"{v[1]}/{v[0]} ({100*v[1]/v[0]:.0f}%)" for k, v in sorted(bymap.items(), key=lambda x: -x[1][0])})
    print("  by folder day:", {k: f"{v[1]}/{v[0]}" for k, v in sorted(byday.items())})


k1_table(12, "(12,) group")
k1_table(8, "(8,) group")
