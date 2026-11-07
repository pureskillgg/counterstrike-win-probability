"""verify D7-6 (alt5): is the own ledger needed for the post-onset label fix? Field-based label test.

F(r) = winner side from the score-field increment between round_freeze_end r and r+1 (T field +1 -> 2, CT field +1 -> 3).
Valid when no side change lies between r and r+1. 'side change follows r' = DP swap flag at r OR player-level
side change at r->r+1 (overlap of T-side spawn ids < 0.5, both sides >= 1 player).
Reads only. Independent of winner_team_code (used only as the outcome being compared).
"""
import sys
import collections

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import verify_D7_6_common as C  # noqa: E402

D = C.load()
rows = C.rows_for(D)
post = [x for x in rows if x["period"] == "post"]
pre = [x for x in rows if x["period"] == "pre"]

spawn = C.read("basic_player_spawn", ["match_key", "round", "player_id", "player_team_code"])
SP = collections.defaultdict(lambda: collections.defaultdict(set))
for s in spawn:
    if s["player_team_code"] in (2, 3):
        SP[(s["match_key"], s["round"])][s["player_team_code"]].add(s["player_id"])


def pswap(k, r):
    a = SP.get((k, r))
    b = SP.get((k, r + 1))
    if not a or not b or not a.get(2) or not b.get(2) or not a.get(3) or not b.get(3):
        return None
    t0, t1 = a[2], b[2]
    return len(t0 & t1) / max(1, min(len(t0), len(t1))) < 0.5


def field_label(k, r):
    a = D["FE"].get((k, r))
    b = D["FE"].get((k, r + 1))
    if a is None or b is None:
        return None
    dt_, dc = b[0] - a[0], b[1] - a[1]
    if (dt_, dc) == (1, 0):
        return 2
    if (dt_, dc) == (0, 1):
        return 3
    return None


def side_change_follows(k, r):
    lg = D["LEDGER"].get((k, r))
    dp = bool(lg and lg[1])
    ps = pswap(k, r)
    return dp or bool(ps)


for per, sub in [("post", post), ("pre", pre)]:
    print(f"== {per}: field-label F(r) vs ledger L and winner_code w ==")
    cls = collections.Counter()
    for x in sub:
        F = field_label(x["k"], x["r"])
        scf = side_change_follows(x["k"], x["r"])
        flip = x["w"] != x["L"]
        cls[(
            "side change follows" if scf else "no side change follows",
            "winner!=L" if flip else "winner==L",
            "F None" if F is None else ("F==L" if F == x["L"] else "F!=L"),
            "F==w" if F is not None and F == x["w"] else "F!=w",
        )] += 1
    for key in sorted(cls):
        print("  ", key, cls[key])

print("\n== post flips (winner!=L) at no-side-change rounds: F agrees with L? by round ==")
byr = collections.Counter()
for x in post:
    if x["w"] == x["L"] or side_change_follows(x["k"], x["r"]):
        continue
    F = field_label(x["k"], x["r"])
    byr[(x["r"], "F None" if F is None else ("F==L" if F == x["L"] else "F!=L"))] += 1
for key in sorted(byr, key=lambda z: (z[0], z[1])):
    print("  ", key, byr[key])

print("\n  post flips (winner!=L) where a side change FOLLOWS the round, by round:")
byr2 = collections.Counter()
for x in post:
    if x["w"] == x["L"] or not side_change_follows(x["k"], x["r"]):
        continue
    F = field_label(x["k"], x["r"])
    byr2[(x["r"], "F None" if F is None else ("F==L" if F == x["L"] else "F!=L"))] += 1
for key in sorted(byr2, key=lambda z: (z[0], z[1])):
    print("  ", key, byr2[key])

print("\n  (rows with a side change following the round are where the DP or player codes are needed)")
