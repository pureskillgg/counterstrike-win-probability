"""verify D7-6 (alt3): the DP swap-at-8 matches (post) that the player-level check could not verify, plus the
round-27 ledger-error set, and the field-based label property of the ledger.

Independent of winner_team_code. Uses verify_D7_6_common (own ledger from round_freeze_end) and
basic_player_spawn / basic_player_death team codes.
ASSUMPTION: player-level side change at boundary r->r+1 = overlap of round-r T-side ids with round-(r+1) T-side ids
< 0.5 of the smaller set; here the minimum set size is relaxed to >= 1 (reported separately).
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
DPS = collections.defaultdict(list)
for (k, r), v in D["LEDGER"].items():
    if v[1]:
        DPS[k].append(r)
max_round = collections.defaultdict(int)
for (k, r) in D["RE"]:
    max_round[k] = max(max_round[k], r)
MR_OF = {}


def side_sizes(k, r):
    v = SP.get((k, r))
    if not v:
        return (0, 0)
    return (len(v.get(2, ())), len(v.get(3, ())))


def pswap_lenient(k, r):
    a = SP.get((k, r))
    b = SP.get((k, r + 1))
    if not a or not b or not a.get(2) or not b.get(2):
        return None
    t0, t1 = a[2], b[2]
    ov = len(t0 & t1) / max(1, min(len(t0), len(t1)))
    return ov < 0.5


# ---------------------------------------------------------------------------
# A. DP swap at 8 (post): player-level evidence, spawn sizes, match length
# ---------------------------------------------------------------------------
s8 = [k for k, rs in DPS.items() if 8 in rs and D["MDT"][k] >= C.ONSET]
print(f"== A. post matches with DP swap after round 8: {len(s8)}")
print("  spawn sizes (T,CT) at rounds 7, 8, 9 (most common):")
for rr in (7, 8, 9):
    print(f"    r{rr}:", collections.Counter(side_sizes(k, rr) for k in s8).most_common(5))
print("  lenient player swap at 8->9:", collections.Counter(pswap_lenient(k, 8) for k in s8))
print("  max round recorded in these matches (distribution):", sorted(collections.Counter(max_round[k] for k in s8).items()))
print("  freeze-end sum (t+c) at round 9 (should be 8 in any format):",
      collections.Counter(sum(D["FE"][(k, 9)]) for k in s8 if (k, 9) in D["FE"]).most_common(3))
print("  score pattern at 8 -> 9 (t8,c8)->(t9,c9) examples:",
      [(D["FE"].get((k, 8)), D["FE"].get((k, 9))) for k in s8[:6]])
# MR12 contrast: DP swap at 12, max round
s12 = [k for k, rs in DPS.items() if 12 in rs and D["MDT"][k] >= C.ONSET]
print(f"  contrast: post DP swap after 12: {len(s12)} matches; max round distribution (top 8):",
      sorted(collections.Counter(max_round[k] for k in s12).items())[:8])

# round-9 flips without DP swap at 8
r9f_noswap = [x for x in post if x["r"] == 9 and not x["fas"] and x["w"] != x["L"]]
print(f"\n  post r9 flips without DP swap at 8: {len(r9f_noswap)}; lenient player swap 8->9 (flip matches):",
      collections.Counter(pswap_lenient(x["k"], 8) for x in r9f_noswap))
print("    their DP swap rounds:", collections.Counter(tuple(sorted(DPS.get(x["k"], []))) for x in r9f_noswap).most_common(5))

# ---------------------------------------------------------------------------
# B. round-27 set: DP-missed swap at 27 vs ledger-error classification
# ---------------------------------------------------------------------------
print("\n== B. post round-27 flips vs player-level swap at 27->28 ==")
r27 = [x for x in post if x["r"] == 27 and x["w"] != x["L"]]
print("  n r27 flips:", len(r27), "| DP swap after 27 among them:", sum(1 for x in r27 if x["k"] in DPS and 27 in DPS[x["k"]]))
print("  lenient player swap 27->28 among them:", collections.Counter(pswap_lenient(x["k"], 27) for x in r27))
r27_agree = [x for x in post if x["r"] == 27 and x["w"] == x["L"]]
print("  r27 agreeing rows:", len(r27_agree), "| lenient player swap 27->28:",
      collections.Counter(pswap_lenient(x["k"], 27) for x in r27_agree))

# ---------------------------------------------------------------------------
# C. pre-period flips: which rounds (baseline floor)
# ---------------------------------------------------------------------------
print("\n== C. pre-period winner!=ledger rounds and player-level swap ==")
pf = [x for x in pre if x["w"] != x["L"]]
print("  pre flips by round:", sorted(collections.Counter(x["r"] for x in pf).items()))
print("  pre flips: lenient player swap at r->r+1:", collections.Counter(pswap_lenient(x["k"], x["r"]) for x in pf))
print("  pre flips with DP swap after the round:", sum(1 for x in pf if x["k"] in DPS and x["r"] in DPS[x["k"]]))

# ---------------------------------------------------------------------------
# D. share of post flips by the evidence class (headline partition)
# ---------------------------------------------------------------------------
print("\n== D. partition of the 2,642 post flips by independent evidence ==")
part = collections.Counter()
for x in post:
    if x["w"] == x["L"]:
        continue
    if x["bs"] is not None:
        part["bomb: ledger matches bomb (winner wrong)" if x["bs"] == x["L"] else "bomb: winner matches bomb (ledger wrong)"] += 1
    else:
        e = None
        # elimination (recomputed here, same rule as verify_D7_6_alt.py)
        DE = collections.defaultdict(set)
        if not hasattr(sys.modules[__name__], "_DE"):
            death = C.read("basic_player_death", ["match_key", "round", "player_id"])
            for s in death:
                DE[(s["match_key"], s["round"])].add(s["player_id"])
            globals()["_DE"] = DE
        DE = globals()["_DE"]
        v = SP.get((x["k"], x["r"]))
        if v and v.get(2) and v.get(3):
            d = DE.get((x["k"], x["r"]), set())
            a2, a3 = v[2] - d, v[3] - d
            if not a2 and a3:
                e = 3
            elif not a3 and a2:
                e = 2
        if e is None:
            part["non-bomb: no elimination call"] += 1
        elif e == x["L"]:
            part["non-bomb: elimination matches ledger (winner wrong)"] += 1
        elif e == x["w"]:
            part["non-bomb: elimination matches winner (ledger wrong)"] += 1
for key, c in sorted(part.items()):
    print(f"  {key}: {c}")
print("  total post flips:", sum(part.values()))
