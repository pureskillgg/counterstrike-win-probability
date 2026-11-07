# VERIFIER for A1-1, part 5: test the swap rule the claim relies on, against engine end-of-match side totals.
# Round-derived team totals use round_end winner_team_code (2 = T side, 3 = CT side) and the swap rule
#   swaps_before(r) = [r > max_rounds//2] + #{x in {max+3, max+9, ...}: x < r}   (ASSUMPTION, same as A1_header_vs_rounds.py)
#   starters_T wins if winner side == T and swaps_before is even, etc.
# Engine end state: fin_t, fin_ct (t_score, ct_score on the cs_win_panel_match row).
# For a regulation match, starters_T total should equal fin_ct and starters_CT total should equal fin_t.
# Categories per match: rule_ok_engine (swap rule agrees with engine end), hdr_ok_engine (header t == fin_ct, ct == fin_t),
# claim_split_ok (round-derived starters == header t/ct, the claim's split check).
import os, glob, collections
import polars as pl

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
def pages(ch):
    return sorted(glob.glob(os.path.join(T, ch, "dataframe_*")))
def chdir(prefix):
    return [x for x in os.listdir(T) if x.startswith(prefix + ".")][0]

h = pl.concat([pl.scan_parquet(p).select(["key", "max_rounds", "t_starters_score_final", "ct_starters_score_final"])
               for p in pages("header")], how="vertical").collect()
rs = pl.concat([pl.scan_parquet(p).select(["match_key", "tick", "event_type", "t_score", "ct_score"])
                for p in pages(chdir("basic_round_state"))], how="diagonal_relaxed").collect()
wp = (rs.filter(pl.col("event_type") == "cs_win_panel_match").sort(["match_key", "tick"]).group_by("match_key").last()
        .select(["match_key", "t_score", "ct_score"]))
fin = {k: (t, c) for k, t, c in wp.iter_rows()}
re_ = pl.concat([pl.scan_parquet(p).select(["match_key", "round", "winner_team_code"]) for p in pages(chdir("basic_round_end"))],
                how="diagonal_relaxed").collect()
rows = collections.defaultdict(list)
for k, r, w in re_.iter_rows():
    rows[k].append((r, w))

def swaps_before(r, mr):
    s = 1 if r > mr // 2 else 0
    x = mr + 3
    while x < r:
        s += 1; x += 6
    return s

cnt = collections.Counter()
examples = collections.defaultdict(list)
for k, mr, tS, cS in h.select(["key", "max_rounds", "t_starters_score_final", "ct_starters_score_final"]).iter_rows():
    st = 0; sc = 0
    for r, w in rows[k]:
        if w not in (2, 3):
            continue
        sw = swaps_before(r, mr)
        starters_t_on_T = (sw % 2 == 0)
        if (w == 2) == starters_t_on_T:
            st += 1
        else:
            sc += 1
    ft, fc = fin[k]
    day = k.split("/")[3]
    n = len(rows[k])
    rule_ok_engine = (st == fc and sc == ft)
    hdr_ok_engine = (tS == fc and cS == ft)
    claim_split_ok = (st == tS and sc == cS)
    sum_ok = (n == tS + cS)
    cnt[("sum_ok", sum_ok)] += 1
    cnt[("rule_ok_engine", rule_ok_engine)] += 1
    cnt[("hdr_ok_engine", hdr_ok_engine)] += 1
    cnt[("claim_split_ok", claim_split_ok)] += 1
    cat = (sum_ok, claim_split_ok, rule_ok_engine, hdr_ok_engine)
    cnt[("cat", cat)] += 1
    if sum_ok and not claim_split_ok:
        examples["B_sum_ok_split_bad"].append((k, mr, n, tS, cS, st, sc, ft, fc, rule_ok_engine, hdr_ok_engine))
    if not sum_ok:
        examples["C_sum_bad"].append((k, mr, n, tS + cS, tS, cS, st, sc, ft, fc, rule_ok_engine, hdr_ok_engine))

print("matches:", len(list(h.iter_rows())))
for key in ["sum_ok", "claim_split_ok", "rule_ok_engine", "hdr_ok_engine"]:
    print(key, {str(k[1]): v for k, v in cnt.items() if k[0] == key})
print("category (sum_ok, claim_split_ok, rule_ok_engine, hdr_ok_engine) -> count:")
for k, v in sorted(((k[1], v) for k, v in cnt.items() if k[0] == "cat"), key=lambda z: -z[1]):
    print("   ", k, v)
print("B (sum agrees, claim split disagrees) count:", len(examples["B_sum_ok_split_bad"]))
for e in examples["B_sum_ok_split_bad"]:
    print("   B", e)
print("C (sum disagrees) sample of first 10:")
for e in examples["C_sum_bad"][:10]:
    print("   C", e)
print("C summary: rule_ok_engine:", sum(1 for e in examples["C_sum_bad"] if e[10]),
      "hdr_ok_engine:", sum(1 for e in examples["C_sum_bad"] if e[11]), "of", len(examples["C_sum_bad"]))
print("done")

# --- breakdown of the orientation-disagreement groups by day and overtime (rows > max_rounds) ---
brk = collections.Counter()
for k, mr, tS, cS in h.select(["key", "max_rounds", "t_starters_score_final", "ct_starters_score_final"]).iter_rows():
    st = 0; sc = 0
    for r, w in rows[k]:
        if w not in (2, 3):
            continue
        sw = swaps_before(r, mr)
        if (w == 2) == (sw % 2 == 0):
            st += 1
        else:
            sc += 1
    ft, fc = fin[k]
    n = len(rows[k])
    grp = (tS + cS == n, (st == tS and sc == cS), (st == fc and sc == ft))
    brk[(k.split("/")[3], "OT" if n > mr else "reg", grp)] += 1
print("breakdown (day, OT or regulation, (sum_ok, claim_split_ok, rule_ok_engine)) -> count")
for kk, v in sorted(brk.items()):
    print("   ", kk, v)
print("done2")
