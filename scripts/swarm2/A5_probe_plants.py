"""A5 probe: bomb events. Compares bomb_state (used by bakeoff.build_states) with the bomb events
that also appear in round_state, and checks rounds whose outcome needs a plant (Target_Bombed,
Bomb_Defused) but have no plant row in the same round. Read-only; output to stdout.
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
BS = fv.read("bomb_state", ["match_key", "round", "tick", "event_type"])
BOMB = ["bomb_planted", "bomb_exploded", "bomb_defused"]

print("== A. event counts by channel ==")
rs_b = RS.filter(pl.col("event_type").is_in(BOMB))
bs_b = BS.filter(pl.col("event_type").is_in(BOMB))
print("round_state bomb rows:", dict(collections.Counter(rs_b["event_type"].to_list())))
print("bomb_state bomb rows: ", dict(collections.Counter(bs_b["event_type"].to_list())))

frz = {}
for m, r, t in RS.filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "tick"]).iter_rows():
    frz.setdefault((m, int(r)), int(t))
end = {}
rwin = {}
for m, r, t, w, rs in RE.select(["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).iter_rows():
    end.setdefault((m, int(r)), int(t))
    rwin.setdefault((m, int(r)), (w, rs))


def win_of(k, t):
    if k not in frz or k not in end:
        return "no window"
    if t < frz[k]:
        return "before freeze"
    if t > end[k]:
        return "after end"
    return "in window"


print("\n== B. per (match, round): does bomb_state agree with round_state for each bomb event type ==")
bs_set = collections.defaultdict(set)
for m, r, t, ev in bs_b.select(["match_key", "round", "tick", "event_type"]).iter_rows():
    bs_set[(m, int(r))].add((ev, int(t)))
rs_set = collections.defaultdict(set)
for m, r, t, ev in rs_b.select(["match_key", "round", "tick", "event_type"]).iter_rows():
    rs_set[(m, int(r))].add((ev, int(t)))
agree = collections.Counter()
only_rs = collections.Counter()
only_bs = collections.Counter()
ex_rs_only = []
ex_bs_only = []
for k in set(bs_set) | set(rs_set):
    for ev in BOMB:
        a = {t for e, t in bs_set.get(k, set()) if e == ev}
        b = {t for e, t in rs_set.get(k, set()) if e == ev}
        if a == b:
            agree[(ev, "same ticks" if a else "absent in both")] += 1
        elif b and not a:
            only_rs[ev] += 1
            if len(ex_rs_only) < 5:
                ex_rs_only.append((k[0][:30], k[1], ev, sorted(b)[:3], rwin.get(k)))
        elif a and not b:
            only_bs[ev] += 1
            if len(ex_bs_only) < 5:
                ex_bs_only.append((k[0][:30], k[1], ev, sorted(a)[:3]))
        else:
            agree[(ev, "different ticks")] += 1
print("agreement:", dict(agree))
print("present only in round_state:", dict(only_rs), "examples", ex_rs_only)
print("present only in bomb_state:", dict(only_bs), "examples", ex_bs_only)

print("\n== C. outcome vs plant rows (all matches) ==")
out = collections.Counter()
for k, (w, rs) in rwin.items():
    if rs not in (1, 7, 9, 12, 8):
        continue
    has_bs = any(e == "bomb_planted" for e, _ in bs_set.get(k, set()))
    has_rs = any(e == "bomb_planted" for e, _ in rs_set.get(k, set()))
    out[(WIN.get(rs, rs), "BS plant" if has_bs else "no BS plant", "RS plant" if has_rs else "no RS plant")] += 1
for key, n in sorted(out.items(), key=lambda kv: -kv[1]):
    print(f"   {key}: {n}")

print("\n== D. Target_Bombed / Bomb_Defused rounds with no BS plant: where are their bomb events? ==")
miss = [k for k, (w, rs) in rwin.items() if rs in (1, 7) and not any(e == "bomb_planted" for e, _ in bs_set.get(k, set()))]
print(f"rounds: {len(miss)}; matches: {len({k[0] for k in miss})}")
per_match = collections.Counter(k[0] for k in miss)
print(f"rounds per affected match: {dict(sorted(collections.Counter(per_match.values()).items()))}")
explode_in = collections.Counter()
nb = collections.Counter()
for k in miss:
    ev_list = sorted(bs_set.get(k, set()))
    kinds = tuple(sorted({e for e, _ in ev_list}))
    nb[kinds] += 1
    for e, t in ev_list:
        explode_in[(e, win_of(k, t))] += 1
print("bomb_state event kinds present in those rounds:", dict(nb))
print("their event timing (event, window):", dict(explode_in))
# neighbouring-round plants
nbr = collections.Counter()
ex_nb = []
for k in miss:
    m, r = k
    for dr in (-1, 1):
        kk = (m, r + dr)
        if kk in bs_set:
            pl_ticks = [t for e, t in bs_set[kk] if e == "bomb_planted"]
            if pl_ticks:
                nbr[("round %+d has plant" % dr, win_of(kk, pl_ticks[0]))] += 1
                if len(ex_nb) < 4:
                    ex_nb.append((m[:30], r, dr, pl_ticks, frz.get(kk), end.get(kk)))
print("plants in neighbouring rounds of missing-plant rounds:", dict(nbr))
print("examples:", ex_nb)

print("\n== E. tick of bomb events vs the round's window, rounds that have a BS plant ==")
tw = collections.Counter()
for k in rwin:
    for e, t in bs_set.get(k, set()):
        tw[(e, win_of(k, t))] += 1
print(dict(tw))

print("\n== F. time from freeze end to plant (rounds with in-window plant), seconds ==")
gaps = []
for k in rwin:
    pts = [t for e, t in bs_set.get(k, set()) if e == "bomb_planted" and k in frz and frz[k] <= t <= end[k]]
    if pts and k in frz:
        gaps.append((min(pts) - frz[k]) / TICK)
gaps.sort()
if gaps:
    n = len(gaps)
    print({q: round(gaps[min(n - 1, int(q * (n - 1)))], 2) for q in (0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)})
print("done")
