"""verify D5-9, step 6: tie ambiguity and the 'no halftime' control contamination.

  (a) regime-A 'none' matches (no clean swap at 8 or 12) that still have round-13 bomb rows: what are
      fe(12), fe(13) and the boundary-12 class? Are they tie-ambiguous MR12 halftimes (last-round-half label 11)?
  (b) how many boundary-9 classes are 'swap' (the claim's '9->10 swap, 51 matches' caveat) in the 2v2 set
  (c) the 2v2 set: the round-9 truth requires fe(9) and fe(10) to be untied; count ties at fe(9)
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_9_06_ties.py
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_9")
R = pl.read_parquet(SCR / "rounds.parquet")
M = pl.read_parquet(SCR / "matches.parquet")
meta = {r["match_key"]: r for r in M.to_dicts()}

fe = defaultdict(dict)
rend = defaultdict(dict)
for r in R.iter_rows(named=True):
    if r["fe_t"] is not None:
        fe[r["match_key"]][r["round"]] = (r["fe_t"], r["fe_ct"])
    if r["end_winner"] is not None:
        rend[r["match_key"]][r["round"]] = r


def bound(mk):
    out = {}
    F = fe[mk]
    for r in sorted(F):
        if r + 1 not in F:
            continue
        t0, c0 = F[r]
        t1, c1 = F[r + 1]
        dt, dc = t1 - t0, c1 - c0
        s = (dt + dc) == 1
        ns = s and dt in (0, 1) and dc in (0, 1)
        sw = s and (t1 - c0) in (0, 1) and (c1 - t0) in (0, 1)
        if ns and sw:
            out[r] = ("ambig", None)
        elif ns:
            out[r] = ("noswap", 2 if dt == 1 else 3)
        elif sw:
            out[r] = ("swap", 2 if (c1 - t0) == 1 else 3)
        else:
            out[r] = ("bad", None)
    return out


BND = {mk: bound(mk) for mk in meta}
swap8 = {mk for mk in BND if BND[mk].get(8, ("",))[0] == "swap"}
swap12 = {mk for mk in BND if BND[mk].get(12, ("",))[0] == "swap"}
ann7 = {mk for mk in meta if meta[mk].get("round_announce_last_round_half") == [7]}
ann11 = {mk for mk in meta if (meta[mk].get("round_announce_last_round_half") or [None])[0] == 11}

print("=" * 100)
print("STEP 6(a): regime-A 'none' matches (no clean swap at 8 or 12) with round-13 bomb rows")
none_A = {mk for mk in meta if meta[mk]["regime"] == "A" and mk not in swap8 and mk not in swap12}
c = Counter()
examples = []
for mk in none_A:
    row = R.filter((pl.col("match_key") == mk) & (pl.col("round") == 13))
    if row.height == 0:
        continue
    row = row.row(0, named=True)
    if row["n_defused"] + row["n_exploded"] == 0:
        continue
    b12 = BND[mk].get(12, ("none",))[0]
    b13 = BND[mk].get(13, ("none",))[0]
    tie13 = (fe[mk].get(13, (None, None))[0] == fe[mk].get(13, (None, None))[1])
    tie12 = (fe[mk].get(12, (None, None))[0] == fe[mk].get(12, (None, None))[1])
    c[("bomb13_row", "b12=" + b12, "fe13_tied=" + str(tie13), "ann11=" + str(mk in ann11))] += 1
    if len(examples) < 3:
        examples.append((mk, fe[mk].get(12), fe[mk].get(13), fe[mk].get(14), b12, row["end_winner"],
                         row["n_defused"], row["n_exploded"]))
for k, v in sorted(c.items()):
    print("  ", k, v)
print("  examples (key, fe12, fe13, fe14, boundary12 class, round_end13 winner, defused, exploded):")
for e in examples:
    print("   ", e)
print("  regime-A 'none' with last-round-half label 11:", sum(1 for mk in none_A if mk in ann11),
      "of", len(none_A))

print("\n" + "=" * 100)
print("STEP 6(b): boundary-9 classes in the 2v2 halftime set (claim: a '9->10 swap' in 51 matches)")
set2v2 = (swap8 | ann7) - swap12
cc = Counter()
for mk in set2v2:
    cc[(meta[mk]["regime"], BND[mk].get(9, ("no_pair",))[0])] += 1
for k in sorted(cc):
    print("  ", k, cc[k])
sw9 = [mk for mk in set2v2 if BND[mk].get(9, ("",))[0] == "swap"]
print("  matches with boundary-9 class 'swap':", len(sw9))
for mk in sw9[:5]:
    print("   ", mk, fe[mk].get(9), fe[mk].get(10), fe[mk].get(11), rend[mk].get(9, {}).get("end_winner") if rend[mk].get(9) else None)

print("\n" + "=" * 100)
print("STEP 6(c): ties at freeze-end 9 and 10 among the 2v2 halftime set (round-9 truth needs untied fe(9)->fe(10))")
tie9 = Counter()
for mk in set2v2:
    f9 = fe[mk].get(9)
    f10 = fe[mk].get(10)
    if f9 is None:
        tie9[(meta[mk]["regime"], "no fe9")] += 1
    elif f9[0] == f9[1]:
        tie9[(meta[mk]["regime"], "fe9 tied")] += 1
    elif f10 is None:
        tie9[(meta[mk]["regime"], "no fe10")] += 1
    else:
        tie9[(meta[mk]["regime"], "untied")] += 1
for k in sorted(tie9):
    print("  ", k, tie9[k])

print("\nDONE step 6")
