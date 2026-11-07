"""A5 probe: are post-round_end deaths (and post-end bomb explosions) repeat deaths of victims who
already died in the live window (duplicate victims), or first deaths of players still alive?
Also: outcome of rounds whose bomb explodes after round_end. Read-only; output to stdout.
"""

import collections
import importlib.util
from pathlib import Path

import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
_spec = importlib.util.spec_from_file_location("bakeoff", ROOT / "scripts" / "model" / "bakeoff.py")
bk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bk)
fv = bk.fv
T, CT, TICK = bk.T, bk.CT, bk.TICK_RATE
WIN = fv.WIN_REASON

RE = fv.read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"])
RS = fv.read("round_state", ["match_key", "round", "tick", "event_type"])
DE = fv.read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code", "weapon_name"])
BS = fv.read("bomb_state", ["match_key", "round", "tick", "event_type"])

frz = {}
for m, r, t in RS.filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "tick"]).iter_rows():
    frz.setdefault((m, int(r)), int(t))
end = {}
rw = {}
for m, r, t, w, rs in RE.select(["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).iter_rows():
    end.setdefault((m, int(r)), int(t))
    rw.setdefault((m, int(r)), (w, rs))

# first in-window death per (round, victim)
first_in = {}
deaths_by = collections.defaultdict(list)
for m, r, t, pid, team, w in DE.select(["match_key", "round", "tick", "player_id", "player_team_code",
                                        "weapon_name"]).sort("tick").iter_rows():
    k = (m, int(r))
    deaths_by[(k, pid)].append((t, team, w))
    if k in frz and k in end and frz[k] <= t <= end[k]:
        first_in.setdefault((k, pid), t)

post = collections.Counter()
post_w = collections.Counter()
post_gap_rep = []
for (k, pid), lst in deaths_by.items():
    if k not in frz or k not in end:
        continue
    for t, team, w in lst:
        if t <= end[k] or t < frz[k]:
            continue
        has_in = (k, pid) in first_in
        key = "victim already died in window" if has_in else "victim alive at round_end (first death)"
        post[key] += 1
        post_w[(key, w)] += 1
        if has_in:
            post_gap_rep.append((t - end[k]) / TICK)
print(f"post-round_end deaths (all matches): {dict(post)}")
print("by weapon (top 12):")
for (key, w), n in post_w.most_common(12):
    print(f"   {n:>6}  {key:<42} {w}")
gs = sorted(post_gap_rep)
if gs:
    n = len(gs)
    print(f"gap after round_end, repeat deaths: quantiles "
          f"{[round(gs[min(n-1, int(q*(n-1)))], 2) for q in (0, 0.5, 0.95, 1)]}")

print("\nwithin-window duplicate victims (all matches), i.e. the same pid dying twice in the window:")
dupin = collections.Counter()
for (k, pid), lst in deaths_by.items():
    if k not in frz or k not in end:
        continue
    inwin = [t for t, _, _ in lst if frz[k] <= t <= end[k]]
    if len(inwin) > 1:
        dupin["groups"] += 1
        dupin["extra rows"] += len(inwin) - 1
print(dict(dupin))

print("\nbomb_exploded rows after round_end: round outcome and CT/T deaths after end in that round:")
ex = collections.Counter()
exp_after = []
for m, r, t in BS.filter(pl.col("event_type") == "bomb_exploded").select(["match_key", "round", "tick"]).iter_rows():
    k = (m, int(r))
    if k in end and t > end[k]:
        w, rs = rw.get(k, (None, None))
        ex[(WIN.get(rs, rs), "winner " + str(w))] += 1
        exp_after.append((t - end[k]) / TICK)
print(dict(ex))
es = sorted(exp_after)
if es:
    n = len(es)
    print(f"gap quantiles (s): {[round(es[min(n-1, int(q*(n-1)))], 2) for q in (0, 0.25, 0.5, 0.75, 1)]}")

print("\nbomb_exploded rows in window, by round outcome:")
exw = collections.Counter()
for m, r, t in BS.filter(pl.col("event_type") == "bomb_exploded").select(["match_key", "round", "tick"]).iter_rows():
    k = (m, int(r))
    if k in end and frz.get(k, 10**12) <= t <= end[k]:
        exw[WIN.get(rw.get(k, (None, None))[1], "?")] += 1
print(dict(exw))
print("done")
