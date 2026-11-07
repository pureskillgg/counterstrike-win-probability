"""verify D7-6 (alt2): does the ledger's swap placement agree with player-level side changes, which rounds carry
the outlier cases, and does build_num (not map) track the flip regime?

Uses verify_D7_6_common (own ledger from round_freeze_end scores; never winner_team_code) plus
basic_player_spawn team codes (player-level side per round, independent of score and winner_team_code).
ASSUMPTION: a player-level swap between round r and r+1 = fewer than half of round-r T-side players are on T in r+1
(overlap ratio < 0.5), using only rounds with >= 3 spawned players per side on both rounds.
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

spawn = C.read("basic_player_spawn", ["match_key", "round", "player_id", "player_team_code"])
SP = collections.defaultdict(lambda: collections.defaultdict(set))
for s in spawn:
    if s["player_team_code"] in (2, 3):
        SP[(s["match_key"], s["round"])][s["player_team_code"]].add(s["player_id"])

# ---------------------------------------------------------------------------
# (a) per-match structure: DP swap rounds (ledger) and flip rounds, post and pre
# ---------------------------------------------------------------------------
print("== (a) per-match structure: (DP swap rounds, flip rounds) ==")
dp_swaps = collections.defaultdict(list)
for (k, r), v in D["LEDGER"].items():
    if v[1]:
        dp_swaps[k].append(r)
for per, sub in [("post", post), ("pre", pre)]:
    per_m = collections.defaultdict(list)
    for x in sub:
        per_m[x["k"]].append(x)
    cnt = collections.Counter()
    for k, xs in per_m.items():
        fl = tuple(sorted(x["r"] for x in xs if x["w"] != x["L"]))
        sw = tuple(sorted(dp_swaps.get(k, [])))
        cnt[(sw, fl)] += 1
    print(f"  {per}: matches={len(per_m)} distinct (swap,flip) structures={len(cnt)}")
    for key, c in cnt.most_common(12):
        print("     swaps(after r)=", key[0], " flips=", key[1], " matches=", c)

# ---------------------------------------------------------------------------
# (b) DP swap vs player-level swap (spawn team codes), by round boundary
# ---------------------------------------------------------------------------
print("\n== (b) DP swap vs player-level swap at each round boundary r -> r+1 ==")


def player_swap(k, r):
    a = SP.get((k, r))
    b = SP.get((k, r + 1))
    if not a or not b:
        return None
    if min(len(a.get(2, ())), len(a.get(3, ()))) < 3 or min(len(b.get(2, ())), len(b.get(3, ()))) < 3:
        return None
    t0, t1 = a[2], b[2]
    overlap = len(t0 & t1) / max(1, min(len(t0), len(t1)))
    return overlap < 0.5


def dp_swap(k, r):
    v = D["LEDGER"].get((k, r))
    return None if v is None else v[1]


tab = collections.Counter()
for (k, r), v in D["LEDGER"].items():
    per = "pre" if D["MDT"][k] < C.ONSET else "post"
    ps = player_swap(k, r)
    if ps is None:
        continue
    grp = r if r in (8, 12, 24, 27) else "other"
    tab[(per, grp, bool(v[1]), ps)] += 1
for key in sorted(tab, key=lambda z: (z[0], str(z[1]), z[2], z[3])):
    print("  ", key, tab[key])
print("  (columns: period, boundary round r (r->r+1), DP swap, player-level swap, count)")

# ---------------------------------------------------------------------------
# (c) outlier cases: which rounds carry 'winner != ledger' with bomb or elimination evidence
# ---------------------------------------------------------------------------
print("\n== (c) flips by round number, post, with flip sub-class ==")
ELIM_CACHE = {}
death = C.read("basic_player_death", ["match_key", "round", "player_id"])
DE = collections.defaultdict(set)
for s in death:
    DE[(s["match_key"], s["round"])].add(s["player_id"])


def elim(k, r):
    v = SP.get((k, r))
    if not v or not v.get(2) or not v.get(3):
        return None
    d = DE.get((k, r), set())
    a2 = v[2] - d
    a3 = v[3] - d
    if not a2 and a3:
        return 3
    if not a3 and a2:
        return 2
    return None


byr = collections.defaultdict(collections.Counter)
for x in post:
    if x["w"] == x["L"]:
        continue
    if x["bs"] is not None:
        cls = "bomb:bomb==L" if x["bs"] == x["L"] else "bomb:bomb==w"
    else:
        e = elim(x["k"], x["r"])
        cls = "elim:none" if e is None else ("elim:E=L" if e == x["L"] else "elim:E=w")
    byr[x["r"]][cls] += 1
for rr in sorted(byr):
    print(f"  r{rr:>2}: {dict(byr[rr])}")

print("\n  post ledger-vs-bomb in AGREEING rounds (winner==ledger), bomb rounds only, by round number:")
bb = collections.defaultdict(collections.Counter)
for x in post:
    if x["bs"] is None or x["w"] != x["L"]:
        continue
    bb[x["r"]]["bomb==L" if x["bs"] == x["L"] else "bomb!=L"] += 1
for rr in sorted(bb):
    if bb[rr].get("bomb!=L", 0):
        print(f"  r{rr:>2}: {dict(bb[rr])}")
print("  (listed only where bomb!=L while winner==L; i.e., both winner and ledger disagree with bomb)")

print("\n  ledger-vs-bomb disagreements in post, all rounds (winner side irrelevant), by round:")
lb = collections.Counter()
for x in post:
    if x["bs"] is not None and x["bs"] != x["L"]:
        lb[x["r"]] += 1
print("  ", sorted(lb.items()))

# ---------------------------------------------------------------------------
# (d) build_num vs flip regime (post), and date range per build
# ---------------------------------------------------------------------------
print("\n== (d) build_num vs flip regime ==")
bmatch = collections.defaultdict(lambda: {"n": 0, "flip": 0, "dates": []})
per_m = collections.defaultdict(list)
for x in rows:
    per_m[x["k"]].append(x)
for k, xs in per_m.items():
    b = D["BLD"][k]
    per = "pre" if D["MDT"][k] < C.ONSET else "post"
    key = (per, b)
    bmatch[key]["n"] += 1
    bmatch[key]["flip"] += int(any(x["w"] != x["L"] for x in xs))
    bmatch[key]["dates"].append(D["MDT"][k])
for key in sorted(bmatch):
    v = bmatch[key]
    print(f"  {key}: matches={v['n']} with_flip={v['flip']} "
          f"first={min(v['dates'])} last={max(v['dates'])}")
# post matches on build 9842 (the 'no flip' build) and their dates
odd = sorted((D["MDT"][k], k) for k in per_m if D["BLD"][k] == 9842 and D["MDT"][k] >= C.ONSET)
print("  post matches on build 9842:", len(odd), "| first 5:", [(str(a), b[-12:]) for a, b in odd[:5]])
print("  pre matches on build 9961:", sum(1 for k in per_m if D["BLD"][k] == 9961 and D["MDT"][k] < C.ONSET))
print("  build 9961 flip rate among post matches:",
      round(sum(v['flip'] for kk, v in bmatch.items() if kk == ('post', 9961)) /
            max(1, sum(v['n'] for kk, v in bmatch.items() if kk == ('post', 9961))), 4))

# ---------------------------------------------------------------------------
# (e) examples of round-13 flips with no DP swap at 12 (non-fas r13 flips)
# ---------------------------------------------------------------------------
print("\n== (e) examples: post round-13 flip without DP swap at 12 ==")
ex = [x for x in post if x["r"] == 13 and not x["fas"] and x["w"] != x["L"]][:2]
for x in ex:
    k = x["k"]
    print("  match", k[-40:], "| map", x["map"], "| mdt", x["mdt"], "| DP swaps", sorted(dp_swaps.get(k, [])),
          "| bomb13", sorted(D["BOMB"].get((k, 13), [])))
    for r in range(7, 17):
        fe = D["FE"].get((k, r))
        lg = D["LEDGER"].get((k, r))
        rw = D["RE"].get((k, r))
        ps = player_swap(k, r - 1) if r > 7 else None
        print(f"    r{r:>2} freeze={str(fe):<9} ledger={str(lg and lg[0]):<4} swap_after={str(lg and lg[1]):<5} "
              f"winner_code={str(rw and rw['winner_team_code']):<4} player_swap(r-1->r)={ps}")
