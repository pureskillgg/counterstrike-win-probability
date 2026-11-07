"""verify D7-6 (alt4): (i) round-25 flips vs player-level side change at 24->25; (ii) is the 16-round (short) format
present before onset; (iii) does the post winner defect depend on format (short vs 12-round halves) or on build.

Independent of winner_team_code for the side evidence (player spawn team codes). Uses verify_D7_6_common.
ASSUMPTION: short-format match = max recorded round <= 16 (first-to-9 cap); 12-round format otherwise is only
used as a contrast group (max round >= 13 is not required, only the swap pattern).
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
MAXR = collections.defaultdict(int)
for (k, r) in D["RE"]:
    MAXR[k] = max(MAXR[k], r)


def pswap(k, r):
    a = SP.get((k, r))
    b = SP.get((k, r + 1))
    if not a or not b or not a.get(2) or not b.get(2):
        return None
    t0, t1 = a[2], b[2]
    return len(t0 & t1) / max(1, min(len(t0), len(t1))) < 0.5


# (i) round 25 flips vs player swap at 24->25
print("== (i) post round-25 rows: player-level side change at 24->25 ==")
r25 = [x for x in post if x["r"] == 25]
c = collections.Counter()
for x in r25:
    c[("flip" if x["w"] != x["L"] else "agree", pswap(x["k"], 24))] += 1
print("  (winner!=ledger or ==, player swap 24->25):", dict(c))

# (ii) short-format presence before and after onset (by max round and DP/player swap at 8)
print("\n== (ii) short-format (max round <= 16) matches by period ==")
for per, lo, hi in [("pre", None, C.ONSET), ("post", C.ONSET, None)]:
    ks = [k for k in D["MDT"] if (lo is None or D["MDT"][k] >= lo) and (hi is None or D["MDT"][k] < hi)]
    short = [k for k in ks if MAXR[k] and MAXR[k] <= 16 and any(r >= 8 for r in range(8, 9) if (k, r) in D["FE"])]
    dp8 = sum(1 for k in short if 8 in DPS.get(k, []))
    ps8 = collections.Counter(pswap(k, 8) for k in short)
    print(f"  {per}: matches with rounds recorded={len(ks)} short(max<=16)={len(short)} "
          f"DP swap after 8={dp8} player swap 8->9={dict(ps8)}")

# (iii) winner defect by format (short vs long) and by period
print("\n== (iii) winner!=ledger rate by format and period (ledger-known rows) ==")
for per, sub in [("pre", pre), ("post", post)]:
    for fmt in ["short(max<=16)", "long(max>16)"]:
        s = [x for x in sub if ((MAXR[x["k"]] <= 16) == (fmt.startswith("short")))]
        f = sum(1 for x in s if x["w"] != x["L"])
        print(f"  {per:4} {fmt:<16} rows={len(s):<6} flips={f:<5} rate={100.0 * f / max(1, len(s)):.2f}%")

print("\n  post flips by format and round:")
pf = collections.Counter()
for x in post:
    if x["w"] != x["L"]:
        pf[("short" if MAXR[x["k"]] <= 16 else "long", x["r"])] += 1
for key in sorted(pf):
    print("   ", key, pf[key])
