"""verify_D3_4_f_day_window.py: is the first-post-swap flip specific to folder days 12..25?

Strongest alternative: the flip is a data-window artifact (e.g., the 2024 tail copy) and not a swap effect.
Test: run the same score-based swap signature and the same first-post-swap comparison on folder days 01..11
(the tome holds them too). If the 5v5 12-swap exists in days 01..11 but the flip is absent there, the flip is
window-specific. If it is present there as well, the flip is not confined to days 12..25.

ASSUMPTIONS (marked):
  A19 folder day = key day (csds/2024/02/DD/...), as in the claim.
  A20 same swap rule and same first-post-swap comparison as verify_D3_4_a_scores.py (A4, A5).
  A21 the player count per match (distinct weapon_fire players) separates 5v5 (10) from 2v2 (4).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D3_4_f_day_window.py
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
RE = scan("round_end", ["match_key", "round", "winner_team_code"]).collect().unique(
    ["match_key", "round"], keep="first", maintain_order=True)
win = {(m, r): w for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows()}
players = collections.Counter(
    scan("weapon_fire", ["match_key", "player_id_fixed", "player_team_code"])
    .filter(pl.col("player_team_code").is_in([2, 3]))
    .group_by(["match_key", "player_id_fixed"]).agg(pl.len().alias("n")).collect()["match_key"].to_list())


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


H = scan("header", ["key", "map_name"]).collect()
mp_of = {k: mp for k, mp in H.iter_rows()}

rows = collections.defaultdict(collections.Counter)
for m in by_m:
    d = day_of(m)
    band = "d01-07" if d <= 7 else ("d08-11" if d <= 11 else "d12-25")
    ss = swaps_of(m)
    lab = str(ss) if ss else "()"
    rows[band]["matches"] += 1
    rows[band][f"sig{lab}"] += 1
    rows[band][f"players{players.get(m, 0)}"] += 1
    if ss and ss[0] in (8, 12):
        s = ss[0]
        if (m, s + 1) in win and (m, s + 1) in live and (m, s + 2) in live:
            t1, c1 = live[(m, s + 1)]
            t2, c2 = live[(m, s + 2)]
            dt, dc = t2 - t1, c2 - c1
            if clean(dt, dc):
                rows[band][f"k1_at{s}_n"] += 1
                rows[band][f"k1_at{s}_disagree"] += int(dside(dt, dc) != win[(m, s + 1)])

for band in ["d01-07", "d08-11", "d12-25"]:
    print(f"== {band} ==")
    keys = sorted(rows[band].items(), key=lambda x: -x[1])
    print("  " + "  ".join(f"{k}={v}" for k, v in keys))
print("\n(1) day-window test: does a 12-swap exist in days 01-11, and does the flip follow it there?")
print("(2) players per match: 10 = 5v5, 4 = 2v2")

# ---- independent player-side check of the 12-swaps in each window (A18 of verify_D3_4_d_player_sides.py) ----
wf = (scan("weapon_fire", ["match_key", "round", "player_id_fixed", "player_team_code"])
      .filter(pl.col("player_team_code").is_in([2, 3]))
      .group_by(["match_key", "round", "player_id_fixed"])
      .agg(pl.col("player_team_code").mode().first().alias("side"))
      .collect())
nxt = wf.with_columns((pl.col("round") - 1).alias("round")).rename({"side": "side_next"})
jj = wf.join(nxt, on=["match_key", "round", "player_id_fixed"], how="inner")
ag = jj.group_by(["match_key", "round"]).agg(pl.len().alias("n"), (pl.col("side") != pl.col("side_next")).sum().alias("f"))
pflip = collections.defaultdict(dict)
for m, r, n, f in ag.select(["match_key", "round", "n", "f"]).iter_rows():
    if n >= 4:
        pflip[m][r] = f / n >= 0.6
print("\n(3) player-side flip at 12|13 among score-based (12,) matches, by window:")
for band_lo, band_hi, name in [(1, 7, "d01-07"), (12, 25, "d12-25")]:
    tot = yes = 0
    for m in by_m:
        if band_lo <= day_of(m) <= band_hi and swaps_of(m) == (12,):
            tot += 1
            yes += int(pflip.get(m, {}).get(12, False))
    print(f"  {name}: {yes}/{tot} player-confirmed 12|13 flips")
