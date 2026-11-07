"""verify D5-9, step 5: independent anchors for the 'first post-halftime round' mechanism.

  (a) 2v2 halftime matches including the 94 tie-ambiguous-at-8 matches (label-7 announcement):
      round-9 inversion using the 9->10 boundary (which does not depend on the 8->9 swap)
  (b) 5v5 MR12 halftime (swap at 12->13): round-13 bomb truth vs round_end vs score truth, and last-victim truth
  (c) control groups for the same anchors (regime A no halftime; regime B halftime)
  (d) single-label signature: inversion rate at labels 8..11 and 13..16 around the swap, per group
  (e) regime vs calendar day for the 2v2 halftime set
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_9_05_anchor.py
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_9")
R = pl.read_parquet(SCR / "rounds.parquet")
M = pl.read_parquet(SCR / "matches.parquet")
meta = {r["match_key"]: r for r in M.to_dicts()}
LATE = {f"{d:02d}" for d in range(12, 26)}

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
# 2v2 halftime set incl. tie-ambiguous at 8: label-7 last-round-half announcement and not an MR12 swap
ann7 = {mk for mk in meta if meta[mk].get("round_announce_last_round_half") == [7]}
set2v2 = (swap8 | ann7) - swap12
tie8 = set2v2 - swap8
print("=" * 100)
print("STEP 5(a): 2v2 halftime set (swap8 421 + tie-ambiguous-at-8 label-7 matches):", len(set2v2),
      " tie-only extras:", len(tie8))
print("  by regime:", dict(Counter(meta[mk]["regime"] for mk in set2v2)),
      " extras by regime:", dict(Counter(meta[mk]["regime"] for mk in tie8)))


def inv_rate(mks, rnd, label):
    usable = inv = 0
    for mk in mks:
        b = BND[mk].get(rnd)
        if b is None or b[0] not in ("noswap", "swap") or rnd not in rend[mk]:
            continue
        usable += 1
        if rend[mk][rnd]["end_winner"] != b[1]:
            inv += 1
    print(f"  {label:<60} usable={usable:5d} inverted={inv:5d} " +
          (f"rate={100*inv/usable:6.2f}%" if usable else "rate=   n/a"))


print("\nSTEP 5(a): round-9 inversion (truth from boundary 9->10 which is independent of the 8->9 swap)")
inv_rate({m for m in set2v2 if meta[m]["regime"] == "A"}, 9, "2v2 halftime, regime A (swap8 + tie extras)")
inv_rate({m for m in tie8 if meta[m]["regime"] == "A"}, 9, "   of which tie-only extras, regime A")
inv_rate({m for m in set2v2 if meta[m]["regime"] == "B"}, 9, "2v2 halftime, regime B (swap8 + tie extras)")
inv_rate({m for m in tie8 if meta[m]["regime"] == "B"}, 9, "   of which tie-only extras, regime B")

print("\nSTEP 5(b): 5v5 MR12 halftime, round 13: bomb, victim, score truth")
s12A = {m for m in swap12 if meta[m]["regime"] == "A"}
s12B = {m for m in swap12 if meta[m]["regime"] == "B"}
noA = {m for m in meta if meta[m]["regime"] == "A" and m not in swap8 and m not in swap12 and m not in ann7}
noB = {m for m in meta if meta[m]["regime"] == "B" and m not in swap8 and m not in swap12 and m not in ann7}


def bomb_check(mks, rnd, label):
    c = Counter()
    for mk in mks:
        row = R.filter((pl.col("match_key") == mk) & (pl.col("round") == rnd))
        if row.height == 0:
            c["no_row"] += 1
            continue
        row = row.row(0, named=True)
        if row["end_winner"] is None:
            c["no_round_end"] += 1
            continue
        if row["n_defused"] > 0 and row["n_exploded"] == 0:
            truth = 3
        elif row["n_exploded"] > 0 and row["n_defused"] == 0:
            truth = 2
        else:
            truth = None
        b = BND[mk].get(rnd)
        score = b[1] if b and b[0] in ("noswap", "swap") else None
        if truth is None:
            c["no_bomb_outcome"] += 1
            continue
        c["bomb_rows"] += 1
        if row["end_winner"] != truth:
            c["round_end_vs_bomb_mismatch"] += 1
        if score is not None:
            c["score_usable_with_bomb"] += 1
            if score != truth:
                c["score_vs_bomb_mismatch"] += 1
            if score != row["end_winner"]:
                c["score_vs_round_end_mismatch"] += 1
    print(f"  {label:<40}", dict(sorted(c.items())))


bomb_check(s12A, 13, "regime A, MR12 halftime (swap12), label 13")
bomb_check(s12B, 13, "regime B, MR12 halftime (swap12), label 13")
bomb_check(noA, 13, "regime A, no halftime (control), label 13")
bomb_check(noB, 13, "regime B, no halftime (control), label 13")
bomb_check({m for m in set2v2 if meta[m]["regime"] == "A"}, 9, "regime A, 2v2 halftime, label 9")
bomb_check({m for m in set2v2 if meta[m]["regime"] == "B"}, 9, "regime B, 2v2 halftime, label 9")


def victim_check(mks, rnd, label):
    c = Counter()
    for mk in mks:
        row = R.filter((pl.col("match_key") == mk) & (pl.col("round") == rnd))
        if row.height == 0:
            continue
        row = row.row(0, named=True)
        if row["end_winner"] is None or row["n_defused"] + row["n_exploded"] > 0:
            continue
        lv = row["last_victim_team"]
        if lv not in (2, 3):
            c["no_victim"] += 1
            continue
        truth = 3 if lv == 2 else 2
        c["n"] += 1
        if row["end_winner"] != truth:
            c["round_end_vs_victim_mismatch"] += 1
        b = BND[mk].get(rnd)
        if b and b[0] in ("noswap", "swap"):
            c["score_usable"] += 1
            if b[1] != truth:
                c["score_vs_victim_mismatch"] += 1
    print(f"  {label:<40}", dict(sorted(c.items())))


print("\nSTEP 5(b): last-victim check, label 13 (no-bomb rounds)")
victim_check(s12A, 13, "regime A, MR12 halftime, label 13")
victim_check(s12B, 13, "regime B, MR12 halftime, label 13")
victim_check(noA, 13, "regime A, no halftime, label 13")
victim_check(noB, 13, "regime B, no halftime, label 13")

print("\n" + "=" * 100)
print("STEP 5(d): inversion rate around the first post-halftime label (regime A, 5v5 MR12 vs 2v2 MR8)")
for lab in [8, 9, 10, 11, 12, 13, 14, 15]:
    for name, mks in [("A 2v2 halftime", {m for m in set2v2 if meta[m]["regime"] == "A"}),
                      ("A 5v5 MR12 halftime", s12A)]:
        usable = inv = 0
        for mk in mks:
            b = BND[mk].get(lab)
            if b is None or b[0] not in ("noswap", "swap") or lab not in rend[mk]:
                continue
            usable += 1
            if rend[mk][lab]["end_winner"] != b[1]:
                inv += 1
        print(f"  label {lab:>2}  {name:<22} usable={usable:5d} inverted={inv:5d}")

print("\n" + "=" * 100)
print("STEP 5(e): calendar day spread of 2v2 halftime set by regime")
for rg in ["A", "B"]:
    days = Counter(meta[m]["day"] for m in set2v2 if meta[m]["regime"] == rg)
    print("  regime", rg, "days:", dict(sorted(days.items())))
print("  regime A 2v2 halftime count:", sum(1 for m in set2v2 if meta[m]["regime"] == "A"),
      " regime B:", sum(1 for m in set2v2 if meta[m]["regime"] == "B"))

print("\nDONE step 5")
