"""verify D5-9, step 2: recompute the swap-8 set, the round-9 inversion rate, and the controls.

Method (independent of the D5 scripts):
  * For each match and each boundary r -> r+1 use only freeze-end scores (side-attached t/ct fields):
      noswap fit: t(r+1)-t(r) in {0,1} and ct(r+1)-ct(r) in {0,1} (sum must rise by exactly 1)
      swap fit:   t(r+1)-ct(r) in {0,1} and ct(r+1)-t(r) in {0,1}   (halftime-style physical side swap)
    Classified noswap / swap / ambig (both fit) / bad (neither fit). No round_end winner is used.
  * Winner implied by the score step (team truth from fields):
      noswap: T (2) if t field +1, else CT (3); swap: T (2) if ct field rose by 1 (old T team), else CT (3).
  * Bomb truth: bomb_defused -> CT (3), bomb_exploded -> T (2). Not using bomb player_team_code.
  * Last-victim truth (no-bomb rounds only): winner = opposite of the team code of the last death.
Compares round_end winner_team_code against these truths. Scratchpad inputs only; no writes to G: or E:.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_9_02_check.py
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)

SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_9")
R = pl.read_parquet(SCR / "rounds.parquet")
M = pl.read_parquet(SCR / "matches.parquet")

LATE = {f"{d:02d}" for d in range(12, 26)}
EARLY = {f"{d:02d}" for d in range(1, 8)}

meta = {r["match_key"]: r for r in M.to_dicts()}

# ---------- per-match freeze-end and round_end dictionaries ----------
fe = defaultdict(dict)      # mk -> label -> (t, ct, t_raw, ct_raw)
rend = defaultdict(dict)    # mk -> label -> dict
for r in R.iter_rows(named=True):
    mk, lab = r["match_key"], r["round"]
    if r["fe_t"] is not None:
        fe[mk][lab] = (r["fe_t"], r["fe_ct"], r["fe_t_raw"], r["fe_ct_raw"])
    if r["end_winner"] is not None:
        rend[mk][lab] = r


def classify(mk):
    """Return dict boundary r -> (cls, truth_winner) using freeze-end fields only."""
    out = {}
    F = fe[mk]
    for r in sorted(F):
        if r + 1 not in F:
            continue
        t0, c0, _, _ = F[r]
        t1, c1, _, _ = F[r + 1]
        dt, dc = t1 - t0, c1 - c0
        sum_ok = (dt + dc) == 1
        ns = dt in (0, 1) and dc in (0, 1) and sum_ok
        sw = (t1 - c0) in (0, 1) and (c1 - t0) in (0, 1) and sum_ok
        if not sum_ok:
            cls, truth = "bad", None
        elif ns and sw:
            cls, truth = "ambig", None
        elif ns:
            cls, truth = "noswap", (2 if dt == 1 else 3)
        elif sw:
            cls, truth = "swap", (2 if (c1 - t0) == 1 else 3)
        else:
            cls, truth = "bad", None
        out[r] = (cls, truth)
    return out


BND = {mk: classify(mk) for mk in meta}

print("=" * 100)
print("STEP 2 CHECK: boundary classes (all matches, all boundaries, freeze-end fields only)")
cc = Counter(c for mk in BND for (c, _) in BND[mk].values())
print("boundary class counts:", dict(cc))
bad_matches = sum(1 for mk in BND if any(c == "bad" for (c, _) in BND[mk].values()))
print("matches with >=1 bad boundary:", bad_matches, "of", len(BND))

# ---------- regime and swap-8 / swap-12 sets ----------
def regime(mk):
    return meta[mk]["regime"]


def grp(mk):
    d = meta[mk]["day"]
    if d in LATE:
        return "late"
    if d in EARLY:
        return "early"
    return "other"


swap8 = {mk for mk in BND if BND[mk].get(8, (None,))[0] == "swap"}
swap12 = {mk for mk in BND if BND[mk].get(12, (None,))[0] == "swap"}
print("\nswap at 8->9 (freeze-end only): total", len(swap8), "| by regime:",
      dict(Counter(regime(m) for m in swap8)))
print("swap at 12->13 (freeze-end only): total", len(swap12), "| by regime:",
      dict(Counter(regime(m) for m in swap12)))
s8n12 = {m for m in swap8 if m not in swap12}
print("swap8 and no swap12:", len(s8n12), "| by regime:", dict(Counter(regime(m) for m in s8n12)))
print("D5 claim: 421 (A 253, B 168)")
print("swap8 and swap12 both:", len(swap8 & swap12))

# ---------- round-9 inversion (score-step truth vs round_end winner) ----------
def inv_stats(mks, label, rnd, boundary_truth=True):
    n_total = len(mks)
    usable = inv = 0
    noclass = 0
    for mk in mks:
        b = BND[mk].get(rnd)
        if b is None or b[0] not in ("noswap", "swap"):
            noclass += 1
            continue
        if rnd not in rend[mk]:
            noclass += 1
            continue
        usable += 1
        if rend[mk][rnd]["end_winner"] != b[1]:
            inv += 1
    rate = (100.0 * inv / usable) if usable else float("nan")
    print(f"  {label:<55} matches={n_total:5d} usable={usable:5d} inverted={inv:5d} rate={rate:6.2f}%  (unusable {noclass})")
    return usable, inv


print("\n" + "=" * 100)
print("STEP 2 CHECK: round-9 inversion, score-step truth (boundary 9->10), by group")
A_s8 = {m for m in s8n12 if regime(m) == "A"}
B_s8 = {m for m in s8n12 if regime(m) == "B"}
print("  (claim: A 240 of 253 inverted (95%); B 0 of 168)")
inv_stats(A_s8, "regime A, swap8 (no swap12)", 9)
inv_stats(B_s8, "regime B, swap8 (no swap12)", 9)
# control: regime A matches with no halftime swap at 8 and none at 12
A_nos = {m for m in BND if regime(m) == "A" and m not in swap8 and m not in swap12}
B_nos = {m for m in BND if regime(m) == "B" and m not in swap8 and m not in swap12}
inv_stats(A_nos, "regime A, no halftime swap at 8 or 12 (control)", 9)
inv_stats(B_nos, "regime B, no halftime swap at 8 or 12 (control)", 9)
# control: regime A swap12 (MR12 halftime) round 13
A_s12 = {m for m in swap12 if regime(m) == "A"}
B_s12 = {m for m in swap12 if regime(m) == "B"}
print("  round-13 controls (boundary 13->14, MR12 halftime at 12):")
inv_stats(A_s12, "regime A, swap12 -> round 13", 13)
inv_stats(B_s12, "regime B, swap12 -> round 13", 13)

# ---------- round_end vs score truth, every label, regime A vs B, swap8 vs not ----------
print("\n" + "=" * 100)
print("STEP 2 CHECK: round_end winner vs score-step truth at each label (all matches), inverted counts")
tab = defaultdict(lambda: [0, 0])  # (regime, label) -> [usable, inverted]
for mk in BND:
    rg = regime(mk)
    for rnd, (cls, truth) in BND[mk].items():
        if cls not in ("noswap", "swap"):
            continue
        if rnd not in rend[mk]:
            continue
        tab[(rg, rnd)][0] += 1
        if rend[mk][rnd]["end_winner"] != truth:
            tab[(rg, rnd)][1] += 1
labels = sorted({k[1] for k in tab})
print("label  |  regime A usable/inverted (rate)     |  regime B usable/inverted (rate)")
for lab in labels:
    if lab > 30:
        continue
    a = tab.get(("A", lab), [0, 0])
    b = tab.get(("B", lab), [0, 0])
    ra = f"{100*a[1]/a[0]:6.2f}%" if a[0] else "   n/a"
    rb = f"{100*b[1]/b[0]:6.2f}%" if b[0] else "   n/a"
    print(f"{lab:>5}  |  {a[0]:6d} / {a[1]:5d} ({ra})       |  {b[0]:6d} / {b[1]:5d} ({rb})")

# ---------- bomb truth at round 9 (independent of score fields and of player_team_code) ----------
print("\n" + "=" * 100)
print("STEP 2 CHECK: bomb truth at round 9 (bomb_defused -> CT, bomb_exploded -> T)")
rows9 = R.filter(pl.col("round") == 9).to_dicts()
bomb9 = defaultdict(lambda: Counter())
for r in rows9:
    mk = r["match_key"]
    if r["end_winner"] is None:
        continue
    bomb_truth = None
    if r["n_defused"] > 0 and r["n_exploded"] == 0:
        bomb_truth = 3
    elif r["n_exploded"] > 0 and r["n_defused"] == 0:
        bomb_truth = 2
    grpk = "swap8" if mk in s8n12 else "other"
    key = (regime(mk), grpk)
    if bomb_truth is None:
        bomb9[key]["no_bomb_outcome"] += 1
        continue
    bomb9[key]["with_bomb"] += 1
    if r["end_winner"] != bomb_truth:
        bomb9[key]["round_end_disagrees_bomb"] += 1
    b = BND[mk].get(9)
    if b and b[0] in ("noswap", "swap"):
        bomb9[key]["score_step_usable"] += 1
        if b[1] != bomb_truth:
            bomb9[key]["score_step_disagrees_bomb"] += 1
        if b[1] != r["end_winner"]:
            bomb9[key]["score_step_disagrees_round_end"] += 1
for key in sorted(bomb9):
    print(" ", key, dict(bomb9[key]))

# ---------- last-victim truth at round 9 (no-bomb rounds) ----------
print("\n" + "=" * 100)
print("STEP 2 CHECK: last-victim truth at round 9 (no-bomb rounds): opposite of last victim team")
vic = defaultdict(lambda: Counter())
for r in rows9:
    mk = r["match_key"]
    if r["end_winner"] is None or r["n_defused"] + r["n_exploded"] > 0:
        continue
    lv = r["last_victim_team"]
    if lv not in (2, 3):
        vic[(regime(mk), "swap8" if mk in s8n12 else "other")]["no_victim"] += 1
        continue
    truth = 3 if lv == 2 else 2
    key = (regime(mk), "swap8" if mk in s8n12 else "other")
    vic[key]["n"] += 1
    if r["end_winner"] != truth:
        vic[key]["round_end_disagrees_victim"] += 1
    b = BND[mk].get(9)
    if b and b[0] in ("noswap", "swap"):
        vic[key]["score_usable"] += 1
        if b[1] != truth:
            vic[key]["score_disagrees_victim"] += 1
for key in sorted(vic):
    print(" ", key, dict(vic[key]))

# ---------- player_team_code of last victim vs fields, is it side or identity? (sanity) ----------
print("\n" + "=" * 100)
print("STEP 2 CHECK: raw vs side-attached score fields (t_score vs t_score_raw)")
diff = Counter()
for mk in meta:
    for lab, (t, c, tr, cr) in fe[mk].items():
        diff[("t_eq_raw" if t == tr else "t_ne_raw", "c_eq_raw" if c == cr else "c_ne_raw")] += 1
print(dict(diff))
ex = []
for mk in list(s8n12)[:400]:
    for lab in sorted(fe[mk]):
        t, c, tr, cr = fe[mk][lab]
        if t != tr or c != cr:
            ex.append((regime(mk), lab, t, c, tr, cr))
            break
print("first differing (regime, label, t, ct, t_raw, ct_raw) among swap8 matches (up to 15):")
for e in ex[:15]:
    print(" ", e)

# ---------- final scores and max team score per match (first-to-N check) ----------
print("\n" + "=" * 100)
print("STEP 2 CHECK: maximum team score reached before the last round, by group")
mx = defaultdict(lambda: Counter())
for mk in meta:
    if not fe[mk]:
        continue
    labs = sorted(fe[mk])
    maxscore_before_last = max(max(fe[mk][l][0], fe[mk][l][1]) for l in labs)
    g = "swap8" if mk in s8n12 else ("swap12" if mk in swap12 else "none")
    key = (regime(mk), g)
    mx[key][f"max_team_score_fe_ge_{min(maxscore_before_last, 16)}"] += 1
    mx[key]["matches"] += 1
    mx[key]["_sum_max_fe"] += maxscore_before_last
for key in sorted(mx):
    d = mx[key]
    n = d["matches"]
    print(" ", key, "matches", n, "mean max fe team score", round(d["_sum_max_fe"] / n, 2))
    dd = {k: v for k, v in sorted(d.items()) if k.startswith("max_team")}
    print("     ", dd)

print("\nDONE step 2")
