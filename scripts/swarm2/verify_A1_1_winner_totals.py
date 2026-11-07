# verify_A1_1_winner_totals.py
# For the 80 shortfall matches: does the header excess sit only on the round-derived loser?
# Also: header max-team total vs round max-team total, and the regulation clinch subset.
import csv, os, collections
SCRATCH = r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"
rows = list(csv.DictReader(open(os.path.join(SCRATCH, "verify_A1_1_permatch.csv"), encoding="utf-8")))
for r in rows:
    for c in ["mr", "thr", "S", "hdr_t", "hdr_c", "n", "rnd_t", "rnd_c", "first_reach"]:
        r[c] = int(r[c])
    r["ot"] = r["n"] > r["mr"]
print("== verify_A1_1_winner_totals ==")
C = [r for r in rows if r["cat"] == "C_sum_short"]
loser_side = 0; winner_side = 0; both = 0; none = 0
max_eq = 0; max_eq_reg = 0; reg = 0
for r in C:
    dt = r["hdr_t"] - r["rnd_t"]; dc = r["hdr_c"] - r["rnd_c"]
    rnd_win_t = r["rnd_t"] > r["rnd_c"]
    pos_t = dt > 0; pos_c = dc > 0
    if pos_t and pos_c:
        both += 1
    elif not pos_t and not pos_c:
        none += 1
    else:
        excess_is_t = pos_t
        loser_is_t = not rnd_win_t
        if excess_is_t == loser_is_t:
            loser_side += 1
        else:
            winner_side += 1
    hmax = max(r["hdr_t"], r["hdr_c"]); rmax = max(r["rnd_t"], r["rnd_c"])
    if hmax == rmax:
        max_eq += 1
    if not r["ot"]:
        reg += 1
        if hmax == rmax:
            max_eq_reg += 1
print("C 80: excess on round-derived LOSER side (exactly one side positive):", loser_side,
      "| on WINNER side:", winner_side, "| both positive:", both, "| none positive:", none)
print("C 80: header max-team total equals round max-team total:", max_eq, "of 80")
print("C regulation 73: header max-team total equals round max-team total:", max_eq_reg, "of", reg)
# Regulation clinch subset: winner (max team) equals 13 in both sources?
rc = [r for r in C if (not r["ot"]) and r["first_reach"] == r["n"]]
print("C regulation clinch-at-last 71: header max == thr:", sum(1 for r in rc if max(r['hdr_t'], r['hdr_c']) == r['thr']),
      "| round max == thr:", sum(1 for r in rc if max(r['rnd_t'], r['rnd_c']) == r['thr']),
      "| header max-team total = round max-team total:", sum(1 for r in rc if max(r['hdr_t'], r['hdr_c']) == max(r['rnd_t'], r['rnd_c'])))
print("C regulation clinch-at-last 71: loser's header excess (S - n) range:",
      min(r['S'] - r['n'] for r in rc), "to", max(r['S'] - r['n'] for r in rc))
# Proposed-fix completion rule check: 'round count equals engine final sum and a team reached threshold'
# Here: a team reached threshold within the recorded rounds (regulation) or overtime; count agreeing matches it would flag.
A = [r for r in rows if r["cat"] == "A_agree"]
flag_rule = [r for r in A if max(r["rnd_t"], r["rnd_c"]) < r["thr"]]
print("\nProposed completion rule ('a team reached threshold'): agreeing matches it would mark incomplete:", len(flag_rule),
      "| of which n < max_rounds:", sum(1 for r in flag_rule if r["n"] < r["mr"]),
      "| of which n == max_rounds (full-length, no team at threshold):", sum(1 for r in flag_rule if r["n"] == r["mr"]))
flag_C = [r for r in C if max(r["rnd_t"], r["rnd_c"]) < r["thr"]]
print("   flagged matches the same rule would mark incomplete:", len(flag_C), [ (r['key'][-40:], r['n'], r['mr']) for r in flag_C])
