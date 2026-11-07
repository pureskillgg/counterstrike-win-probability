"""verify_D3_4_d_player_sides.py: does a real side swap happen at round 8 (the format question)?

Independent swap evidence: each player's side (player_team_code on basic_weapon_fire) per round.
A halftime side swap shows as most common players changing side across the boundary r | r+1.
Compares the player-side swap boundary with the score-based signature from D3_swap_signature (re-derived here)
and with the map name and round count.

ASSUMPTIONS (marked):
  A14 player side = mode of player_team_code over that player's weapon_fire rows in the round (codes 2/3 only).
  A15 a boundary is a player-side swap when >= 4 players appear in both rounds and >= 60% of them change side.
  A16 score-based signature rule identical to the claim (see verify_D3_4_a_scores.py).

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D3_4_d_player_sides.py
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


def scan(ch, cols):
    folder = TOME / "header" if ch == "header" else TOME / f"basic_{ch}{SUF}"
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.scan_parquet(files).select(cols)


def day_of(m):
    return int(m.split("/")[3])


def clean(dt, dc):
    return (dc == 1 and dt == 0) or (dt == 1 and dc == 0)


# ---------- score-based signature (A16) ----------
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


H = scan("header", ["key", "map_name", "t_starters_score_final", "ct_starters_score_final"]).collect()
hdr = {k: (mp, tf, cf) for k, mp, tf, cf in H.iter_rows()}
RE = scan("round_end", ["match_key", "round"]).collect().unique(["match_key", "round"], keep="first")
n_end = collections.Counter(RE["match_key"].to_list())
sig = {m: swaps_of(m) for m in by_m if 12 <= day_of(m) <= 25}
print(f"score-based matches day 12..25: {len(sig)}")

# ---------- player sides from weapon_fire ----------
wf = (scan("weapon_fire", ["match_key", "round", "player_id_fixed", "player_team_code"])
      .filter(pl.col("player_team_code").is_in([2, 3]))
      .group_by(["match_key", "round", "player_id_fixed"])
      .agg(pl.col("player_team_code").mode().first().alias("side"))
      .collect())
print(f"weapon_fire (match, round, player) side rows: {wf.height}")
nxt = wf.with_columns((pl.col("round") - 1).alias("round")).rename({"side": "side_next"})
j = wf.join(nxt, on=["match_key", "round", "player_id_fixed"], how="inner")
agg = (j.group_by(["match_key", "round"])
       .agg(pl.len().alias("n"), (pl.col("side") != pl.col("side_next")).sum().alias("flips")))
bnd = collections.defaultdict(dict)
for m, r, n, f in agg.select(["match_key", "round", "n", "flips"]).iter_rows():
    if n >= 4:
        bnd[m][r] = (n, f)

flip_bounds = {}
for m in sig:
    fb = [r for r, (n, f) in bnd.get(m, {}).items() if f / n >= 0.6]
    flip_bounds[m] = tuple(sorted(fb))

# ---------- cross-tab ----------
print("\n== player-side swap boundaries vs score signature (day 12..25) ==")
xt = collections.Counter()
for m in sig:
    xt[(str(sig[m]), str(flip_bounds[m]))] += 1
for (sg, fbs), v in sorted(xt.items(), key=lambda x: (-x[1])):
    if v >= 3:
        print(f"  score sig {sg:<12} player-flip boundaries {fbs:<22} n={v}")

print("\n== per score-signature group: share of matches with a player-side flip at each boundary ==")
groups = collections.defaultdict(list)
for m in sig:
    groups[str(sig[m])].append(m)
for g in sorted(groups, key=lambda x: -len(groups[x])):
    ms = groups[g]
    cnt = collections.Counter()
    for m in ms:
        for b in flip_bounds[m]:
            cnt[b] += 1
    have_any = sum(1 for m in ms if flip_bounds[m])
    top = ", ".join(f"{b}|{b+1}:{cnt[b]}" for b in sorted(cnt, key=lambda b: -cnt[b])[:5])
    print(f"  {g:<12} n={len(ms):<5} with any player flip={have_any:<5} top boundaries: {top}")

print("\n== (8,) matches: map, header totals, player flip at 8|9 and 12|13 ==")
mp8 = collections.Counter()
for m in groups.get("(8,)", []):
    mp8[(hdr[m][0], 8 in flip_bounds[m], 12 in flip_bounds[m])] += 1
for k, v in sorted(mp8.items(), key=lambda x: -x[1]):
    print(f"  map={k[0]:<12} flip@8={k[1]!s:<6} flip@12={k[2]!s:<6} n={v}")

print("\n== (12,) matches: flip at 12|13 share by map ==")
mp12 = collections.defaultdict(lambda: [0, 0])
for m in groups.get("(12,)", []):
    mp12[hdr[m][0]][0] += 1
    mp12[hdr[m][0]][1] += int(12 in flip_bounds[m])
print({k: f"{v[1]}/{v[0]}" for k, v in sorted(mp12.items(), key=lambda x: -x[1][0])})

print("\n== () matches: flip at 12|13 share (hidden-swap check) ==")
cnt = collections.Counter()
for m in groups.get("()", []):
    cnt["n"] += 1
    cnt["flip@12"] += int(12 in flip_bounds[m])
    cnt["flip@8"] += int(8 in flip_bounds[m])
    cnt["any"] += int(bool(flip_bounds[m]))
print(dict(cnt))

print("\n== (8,) completeness: header total vs round_end rows ==")
comp = collections.Counter()
for m in groups.get("(8,)", []):
    tf, cf = hdr[m][1], hdr[m][2]
    comp["complete" if tf + cf == n_end[m] else "incomplete"] += 1
print(dict(comp))
