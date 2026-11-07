"""A5 probe: in-window 'world' deaths and in-window team_change rows, for all matches and for the
bakeoff kept set (rounds with a 5v5 start, no disconnect or takeover, live start, 10 players).
Read-only; output to stdout.
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
T, CT, START, TICK = bk.T, bk.CT, bk.START, bk.TICK_RATE

RE = fv.read("round_end", ["match_key", "round", "tick", "player_count"])
RS = fv.read("round_state", ["match_key", "round", "tick", "event_type"])
DE = fv.read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code", "weapon_name"])
TC = fv.read("team_change", ["match_key", "round", "tick", "player_id", "old_team_code", "new_team_code", "is_bot"])
PI = fv.read("player_info", ["match_key", "round", "player_id", "team_code"])
DC = fv.read("player_disconnect", ["match_key", "round", "tick"])
BT = fv.read("bot_takeover", ["match_key", "round", "tick"])

frz = {}
for m, r, t in RS.filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "tick"]).iter_rows():
    frz.setdefault((m, int(r)), int(t))
end = {}
pc = {}
for m, r, t, n in RE.select(["match_key", "round", "tick", "player_count"]).iter_rows():
    end.setdefault((m, int(r)), int(t))
    pc.setdefault((m, int(r)), n)

# kept-set approximation: 10 players with 5 T and 5 CT at start, player_count 10, no disconnect or takeover
pl_by = collections.defaultdict(dict)
for m, r, pid, tc in PI.select(["match_key", "round", "player_id", "team_code"]).iter_rows():
    pl_by[(m, int(r))][pid] = tc
bad_rounds = set()
for m, r in list(DC.select(["match_key", "round"]).iter_rows()) + list(BT.select(["match_key", "round"]).iter_rows()):
    bad_rounds.add((m, int(r)))
kept = set()
for k, d in pl_by.items():
    if pc.get(k) != 10 or k in bad_rounds or k not in frz or k not in end:
        continue
    if len(d) == 10 and sum(1 for v in d.values() if v == T) == START and sum(1 for v in d.values() if v == CT) == START:
        kept.add(k)
print(f"approximate kept rounds: {len(kept)}")

print("\n== world deaths ==")
for label, rounds in (("all matches", None), ("approx kept", kept)):
    inw = collections.Counter()
    gaps = []
    for m, r, t, pid, team, w in DE.select(["match_key", "round", "tick", "player_id", "player_team_code",
                                            "weapon_name"]).iter_rows():
        k = (m, int(r))
        if rounds is not None and k not in rounds:
            continue
        if k not in frz or k not in end:
            continue
        if frz[k] <= t <= end[k]:
            inw["in window"] += 1
            if w == "world":
                inw["in window world"] += 1
                gaps.append((t - frz[k]) / TICK)
    print(f"{label}: {dict(inw)}")
    if gaps:
        gs = sorted(gaps)
        n = len(gs)
        print(f"   in-window world deaths, seconds after freeze end: quantiles "
              f"{[round(gs[min(n - 1, int(q * (n - 1)))], 2) for q in (0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)]}; "
              f"under 2 s: {sum(1 for g in gs if g < 2)}")

print("\n== team_change rows ==")
tcw = collections.Counter()
for m, r, t, pid, old, new, bot in TC.select(["match_key", "round", "tick", "player_id", "old_team_code",
                                              "new_team_code", "is_bot"]).iter_rows():
    k = (m, int(r))
    if k not in frz or k not in end:
        tcw["no freeze/end"] += 1
        continue
    if frz[k] <= t <= end[k]:
        tcw["in window"] += 1
        if k in kept:
            tcw["in window, kept round"] += 1
    elif t < frz[k]:
        tcw["before freeze"] += 1
    else:
        tcw["after end"] += 1
print(dict(tcw))
print("done")
