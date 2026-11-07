# verify_A1_1_clinch.py
# Follow-up for A1-1: is "a team reached the win threshold" a valid game-over test in this tome?
# Uses the per-match table written by verify_A1_1_recompute.py (scratchpad CSV) plus round_end winners.
# Reads only the 2026 tome. ASSUMPTION: threshold = max_rounds//2 + 1, regulation rounds are r <= max_rounds.
import os, glob, collections, csv
import polars as pl

print("== verify_A1_1_clinch ==")
ROOT = r"G:\pureskill\csds-cs2"
T = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
SCRATCH = r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"
rows = list(csv.DictReader(open(os.path.join(SCRATCH, "verify_A1_1_permatch.csv"), encoding="utf-8")))
for r in rows:
    for c in ["mr", "thr", "S", "hdr_t", "hdr_c", "n", "rnd_t", "rnd_c", "first_reach", "day"]:
        if c != "day":
            r[c] = int(r[c])
    r["ot"] = r["n"] > r["mr"]

rp = sorted(glob.glob(os.path.join(T, "basic_round_end.2026-09-18,2026-10-08", "dataframe_*")))
re_ = pl.concat([pl.read_parquet(p, columns=["match_key", "round", "winner_team_code"]) for p in rp], how="diagonal_relaxed")
rounds = collections.defaultdict(dict)
for k, r, w in re_.select(["match_key", "round", "winner_team_code"]).iter_rows():
    rounds[k][int(r)] = w

def swap_after(r, mr):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)

# Regulation-only clinch: first round r <= mr where a team's round-derived total reaches thr.
def reg_clinch(k, mr, thr):
    rm = rounds[k]
    cum_t = 0; cum_c = 0; swaps = 0
    for r in sorted(rm):
        if r > mr:
            break
        w = rm[r]
        if (w == 2) == (swaps % 2 == 0):
            cum_t += 1
        else:
            cum_c += 1
        if max(cum_t, cum_c) >= thr:
            return r
        if swap_after(r, mr):
            swaps += 1
    return -1

for r in rows:
    r["reg_clinch"] = reg_clinch(r["key"], r["mr"], r["thr"])

def cat_of(r):
    return r["cat"]

print("\n== 1. Overtime matches (n > max_rounds) by category ==")
print("OT matches by cat:", dict(collections.Counter(r["cat"] for r in rows if r["ot"])))
print("OT matches by day:", dict(collections.Counter(r["day"] for r in rows if r["ot"])))
print("max n by cat:", {c: max(r["n"] for r in rows if r["cat"] == c) for c in ["A_agree", "B_sum_eq_split_diff", "C_sum_short"]})
print("n distribution for OT matches:", sorted(collections.Counter(r["n"] for r in rows if r["ot"]).items()))
print("OT matches: header S and n:", [(r["day"], r["n"], r["S"], r["hdr_t"], r["hdr_c"]) for r in rows if r["ot"]])

print("\n== 2. Regulation-only clinch in agreeing matches (A), n <= max_rounds ==")
A_reg = [r for r in rows if r["cat"] == "A_agree" and not r["ot"]]
clinch_at_end = [r for r in A_reg if r["reg_clinch"] == r["n"]]
clinch_before = [r for r in A_reg if 0 < r["reg_clinch"] < r["n"]]
no_clinch = [r for r in A_reg if r["reg_clinch"] == -1]
print("A regulation matches:", len(A_reg), "| clinch exactly at last round:", len(clinch_at_end),
      "| clinch BEFORE last round (game went on after a team had thr):", len(clinch_before),
      "| no clinch in regulation:", len(no_clinch))
print("clinch-before-last: n vs max_rounds:", collections.Counter((r["n"] == r["mr"], r["mr"]) for r in clinch_before))
print("clinch-before-last: by day:", dict(collections.Counter(r["day"] for r in clinch_before)))
print("clinch-before-last: examples (key tail, mr, thr, reg_clinch, n, S):")
for r in clinch_before[:8]:
    print("   ", r["key"][-40:], r["mr"], r["thr"], r["reg_clinch"], r["n"], r["S"], (r["rnd_t"], r["rnd_c"]))
print("no-clinch regulation: n vs max_rounds:", collections.Counter((r["n"] == r["mr"], r["mr"]) for r in no_clinch))
print("no-clinch with n < mr (end before a clinch and before max rounds):", sum(1 for r in no_clinch if r["n"] < r["mr"]),
      "| n == mr:", sum(1 for r in no_clinch if r["n"] == r["mr"]))

print("\n== 3. Shortfall set C, regulation only (n <= max_rounds) ==")
C_reg = [r for r in rows if r["cat"] == "C_sum_short" and not r["ot"]]
C_ot = [r for r in rows if r["cat"] == "C_sum_short" and r["ot"]]
print("C regulation count:", len(C_reg), "| C overtime count:", len(C_ot))
print("C regulation: clinch exactly at last recorded round:", sum(1 for r in C_reg if r["reg_clinch"] == r["n"]),
      "| clinch before last:", sum(1 for r in C_reg if 0 < r["reg_clinch"] < r["n"]),
      "| no clinch:", sum(1 for r in C_reg if r["reg_clinch"] == -1))
print("C regulation no-clinch rows:", [(r["key"][-40:], r["mr"], r["n"], r["S"], (r["hdr_t"], r["hdr_c"]), (r["rnd_t"], r["rnd_c"])) for r in C_reg if r["reg_clinch"] == -1])
print("C regulation clinch-at-last rows, count by day:", dict(collections.Counter(r["day"] for r in C_reg if r["reg_clinch"] == r["n"])))
print("C regulation clinch-at-last: header winner team agrees with round-derived winner team:",
      sum(1 for r in C_reg if r["reg_clinch"] == r["n"] and ((r["hdr_t"] > r["hdr_c"]) == (r["rnd_t"] > r["rnd_c"]))),
      "of", sum(1 for r in C_reg if r["reg_clinch"] == r["n"]))
print("C regulation: rounds after clinch that header adds (S - n), by type:")
ext = collections.Counter()
for r in C_reg:
    if r["reg_clinch"] == r["n"]:
        ext["clinch_at_last"] += 1
print("   ", dict(ext))

print("\n== 4. Which team carries the header excess (C matches) ==")
def excess_team(r):
    dt = r["hdr_t"] - r["rnd_t"]; dc = r["hdr_c"] - r["rnd_c"]
    if dt == 0 and dc == 0:
        return "none"
    if dt > 0 and dc == 0:
        return "T_starters_only"
    if dc > 0 and dt == 0:
        return "CT_starters_only"
    return "both_or_negative"
print("C all:", dict(collections.Counter(excess_team(r) for r in rows if r["cat"] == "C_sum_short")))
# For each C match: is the excess on the team that did NOT reach threshold by round-derived totals?
ex_loser = 0; ex_winner = 0; ex_none = 0; ex_both = 0
for r in rows:
    if r["cat"] != "C_sum_short":
        continue
    dt = r["hdr_t"] - r["rnd_t"]; dc = r["hdr_c"] - r["rnd_c"]
    rnd_winner_is_t = r["rnd_t"] > r["rnd_c"]
    if dt > 0 and dc > 0:
        ex_both += 1
    elif dt == 0 and dc == 0:
        ex_none += 1
    else:
        excess_on_t = dt > 0
        if excess_on_t == (not rnd_winner_is_t):
            ex_loser += 1
        else:
            ex_winner += 1
print("C excess on the round-derived LOSER side:", ex_loser, "| on the round-derived WINNER side:", ex_winner,
      "| on both:", ex_both, "| none:", ex_none)

print("\n== 5. Engine cross-check on the same matches (score_update last value per team_code) ==")
pages = sorted(glob.glob(os.path.join(T, "basic_score_update.2026-09-18,2026-10-08", "dataframe_*")))
su = pl.concat([pl.read_parquet(p, columns=["match_key", "round", "tick", "team_code", "new_score"]) for p in pages], how="diagonal_relaxed")
su = su.sort(["match_key", "round", "tick"])
latest = {}
for mk, tc, ns in su.select(["match_key", "team_code", "new_score"]).iter_rows():
    latest[(mk, tc)] = ns
print("score_update rows", su.height, "| team_code values", sorted(su.get_column('team_code').unique().to_list()))
bad_maxeng = 0; agree_maxeng = 0
for r in rows:
    if r["cat"] != "C_sum_short":
        continue
    a = latest.get((r["key"], 2)); b = latest.get((r["key"], 3))
    if a is None or b is None:
        continue
    eng_max = max(a, b)
    if eng_max == max(r["rnd_t"], r["rnd_c"]):
        agree_maxeng += 1
    else:
        bad_maxeng += 1
print("C matches where engine max team score == round-derived max team score:", agree_maxeng, "| differ:", bad_maxeng)
eng_pairs = collections.Counter()
for r in rows:
    if r["cat"] != "C_sum_short":
        continue
    a = latest.get((r["key"], 2)); b = latest.get((r["key"], 3))
    eng_pairs[(min(a, b), max(a, b), min(r["rnd_t"], r["rnd_c"]), max(r["rnd_t"], r["rnd_c"]))] += 1
print("engine (lo,hi) vs round-derived (lo,hi) sample:", list(eng_pairs.items())[:10])
print("C matches where engine sum == round rows:", sum(1 for r in rows if r["cat"] == "C_sum_short" and (latest.get((r["key"],2),0)+latest.get((r["key"],3),0)) == r["n"]), "of 80")
print("C matches where engine sum == header S:", sum(1 for r in rows if r["cat"] == "C_sum_short" and (latest.get((r["key"],2),0)+latest.get((r["key"],3),0)) == r["S"]), "of 80")
