"""verify D7-6 (alternatives): is the post-onset flip a winner_team_code defect, or a ledger artifact?

Independent round-outcome sources used (none of them is the score ledger or winner_team_code):
  (1) bomb outcome events (basic_bomb_state): exploded -> T, defused -> CT. Rounds with a bomb outcome only.
  (2) elimination: a non-bomb round in which one side has no living players (from basic_player_spawn
      team codes at the round and basic_player_death victims). Rounds with a bomb plant are excluded.
      ASSUMPTION: a side with zero survivors lost the round by elimination; both sides alive -> no call.
Diagnostic only: pre-period rows, where the claim says winner_team_code is clean.

Reads only. Imports verify_D7_6_common (same definitions as verify_D7_6_recompute.py).
"""
import sys
import collections
import datetime as dt

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import verify_D7_6_common as C  # noqa: E402

D = C.load()
rows = C.rows_for(D)
post = [x for x in rows if x["period"] == "post"]
pre = [x for x in rows if x["period"] == "pre"]
print(f"ledger-known rows: {len(rows)} | post {len(post)} | pre {len(pre)}")

# ---------------------------------------------------------------------------
# 1. per-match flip signatures (post): which rounds disagree, per match
# ---------------------------------------------------------------------------
print("\n== 1. post per-match flip signatures (rounds with winner_team_code != ledger) ==")
per_match = collections.defaultdict(list)
for x in post:
    per_match[x["k"]].append(x)
sig = collections.Counter()
sig_fas9 = collections.Counter()
for k, xs in per_match.items():
    fl = tuple(sorted(x["r"] for x in xs if x["w"] != x["L"]))
    sig[fl] += 1
    if any(x["r"] == 9 and x["fas"] for x in xs):
        sig_fas9[fl] += 1
print("  distinct signatures:", len(sig), "| matches:", len(per_match))
print("  top 15 signatures (flip rounds: matches):", sig.most_common(15))
print("  matches with a fas-9 row:", sum(1 for k, xs in per_match.items() if any(x['r'] == 9 and x['fas'] for x in xs)))
print("  signatures among fas-9 matches (top 10):", sig_fas9.most_common(10))

# ---------------------------------------------------------------------------
# 2. worked examples
# ---------------------------------------------------------------------------
print("\n== 2. worked examples ==")


def show(k, r0=1, r1=26):
    print("  match", k[:52], "| map", D["MAP"][k], "| mdt", D["MDT"][k])
    for r in range(r0, r1 + 1):
        fe = D["FE"].get((k, r))
        lg = D["LEDGER"].get((k, r))
        rw = D["RE"].get((k, r))
        if fe is None and rw is None:
            continue
        print(f"    r{r:>2} freeze(t,c)={str(fe):<9} ledger={str(lg and lg[0]):<4} swap_after={str(lg and lg[1]):<5} "
              f"winner_code={str(rw and rw['winner_team_code']):<4} msg={str(rw and rw['win_reason_message'])[18:]:<20} "
              f"bomb={sorted(D['BOMB'].get((k, r), []))}")


ex9 = next((k for k, xs in per_match.items() if any(x["r"] == 9 and x["fas"] and x["w"] != x["L"] for x in xs)
            and any(x["r"] == 13 and x["fas"] for x in xs)), None)
if ex9:
    show(ex9)
ex13 = next((k for k, xs in per_match.items() if sorted(x["r"] for x in xs if x["w"] != x["L"]) == [13]), None)
if ex13:
    print("  (single flip at round 13)")
    show(ex13)

# ---------------------------------------------------------------------------
# 3. bomb-outcome 2x2 by round group (rounds with a bomb outcome)
# ---------------------------------------------------------------------------
print("\n== 3. bomb outcome vs ledger vs winner_team_code, by round group ==")
print("  cols: A = ledger==bomb & winner!=bomb (winner is the outlier) | B = winner==bomb & ledger!=bomb "
      "(ledger is the outlier) | C = both agree | D = neither")
GROUPS = {
    "r9 fas": lambda x: x["r"] == 9 and x["fas"],
    "r9 non-fas": lambda x: x["r"] == 9 and not x["fas"],
    "r13 fas": lambda x: x["r"] == 13 and x["fas"],
    "r13 non-fas": lambda x: x["r"] == 13 and not x["fas"],
    "other fas (not 9/13)": lambda x: x["fas"] and x["r"] not in (9, 13),
    "other non-fas (not 9/13)": lambda x: (not x["fas"]) and x["r"] not in (9, 13),
}
for per, sub_all in [("pre", pre), ("post", post)]:
    for g, cond in GROUPS.items():
        sub = [x for x in sub_all if x["bs"] is not None and cond(x)]
        cA = sum(1 for x in sub if x["L"] == x["bs"] and x["w"] != x["bs"])
        cB = sum(1 for x in sub if x["w"] == x["bs"] and x["L"] != x["bs"])
        cC = sum(1 for x in sub if x["L"] == x["bs"] and x["w"] == x["bs"])
        cD = sum(1 for x in sub if x["L"] != x["bs"] and x["w"] != x["bs"])
        print(f"  {per:4} {g:<26} n={len(sub):<6} A={cA:<5} B={cB:<4} C={cC:<6} D={cD}")

# ---------------------------------------------------------------------------
# 4. elimination check on non-bomb rounds (independent of ledger and winner_team_code)
# ---------------------------------------------------------------------------
print("\n== 4. elimination check on non-bomb rounds ==")
spawn = C.read("basic_player_spawn", ["match_key", "round", "player_id", "player_team_code"])
death = C.read("basic_player_death", ["match_key", "round", "player_id", "player_team_code"])
print("  spawn rows:", len(spawn), "| team codes:", collections.Counter(s["player_team_code"] for s in spawn).most_common(6))
print("  death rows:", len(death), "| victim team codes:", collections.Counter(s["player_team_code"] for s in death).most_common(6))
SP = collections.defaultdict(lambda: collections.defaultdict(set))
for s in spawn:
    if s["player_team_code"] in (2, 3):
        SP[(s["match_key"], s["round"])][s["player_team_code"]].add(s["player_id"])
DE = collections.defaultdict(set)
for s in death:
    DE[(s["match_key"], s["round"])].add(s["player_id"])
sizes = collections.Counter((len(v.get(2, ())), len(v.get(3, ()))) for v in SP.values())
print("  per-round (T spawns, CT spawns) most common:", sizes.most_common(6))


def elim(k, r):
    v = SP.get((k, r))
    if not v or not v.get(2) or not v.get(3):
        return None
    d = DE.get((k, r), set())
    alive2 = v[2] - d
    alive3 = v[3] - d
    if not alive2 and alive3:
        return 3
    if not alive3 and alive2:
        return 2
    return None


for per, sub_all in [("pre", pre), ("post", post)]:
    nb = [x for x in sub_all if x["bs"] is None]
    res = collections.Counter()
    for x in nb:
        e = elim(x["k"], x["r"])
        flip = x["w"] != x["L"]
        res[("flip" if flip else "agree", "E=None" if e is None else ("E=L" if e == x["L"] else ("E=w" if e == x["w"] else "E=other")))] += 1
    print(f"  {per} non-bomb rows: n={len(nb)}")
    for key in sorted(res):
        print("     ", key, res[key])

print("\n  post non-bomb flips by round group (E=L means elimination agrees with ledger, E=w with winner_team_code):")
for g, cond in GROUPS.items():
    sub = [x for x in post if x["bs"] is None and x["w"] != x["L"] and cond(x)]
    c = collections.Counter()
    for x in sub:
        e = elim(x["k"], x["r"])
        c["E=None" if e is None else ("E=L" if e == x["L"] else ("E=w" if e == x["w"] else "E=other"))] += 1
    print(f"   {g:<26} flips(non-bomb)={len(sub):<5} {dict(c)}")

# ---------------------------------------------------------------------------
# 5. per-round-number elimination evidence for the claimed flip rounds (post, non-bomb)
# ---------------------------------------------------------------------------
print("\n== 5. post non-bomb rounds 9 and 13 by fas, all rows (ledger/winner/elimination) ==")
for rr in (9, 13):
    for fl in (True, False):
        sub = [x for x in post if x["bs"] is None and x["r"] == rr and x["fas"] == fl]
        c = collections.Counter()
        for x in sub:
            e = elim(x["k"], x["r"])
            c[("winner!=L" if x["w"] != x["L"] else "winner==L",
               "E=None" if e is None else ("E=L" if e == x["L"] else ("E=w" if e == x["w"] else "E=other")))] += 1
        print(f"   r{rr} fas={fl}: n={len(sub)} {dict(c)}")

# ---------------------------------------------------------------------------
# 6. single-round regime check: is the DP swap at round 8 real? (swap-8 matches, post)
# ---------------------------------------------------------------------------
print("\n== 6. post matches with a DP swap at round 8 (fas-9): rounds 8-13 detail for 2 examples ==")
s8 = [k for k, xs in per_match.items() if any(x["r"] == 9 and x["fas"] for x in xs)]
print("  matches with fas-9 rows (post):", len(s8))
d8 = collections.Counter(D["MDT"][k].strftime("%m-%d") for k in s8)
print("  fas-9 matches by day:", sorted(d8.items()))
for k in s8[:1]:
    show(k, 6, 15)
