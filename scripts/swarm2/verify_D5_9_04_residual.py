"""verify D5-9, step 4: residual checks.

  (a) matches with a last-round-half announcement at label 7 that are NOT in the 421 swap-8 set
      (the claim says label 7 is only seen in the 421; "other matches: label 11")
  (b) 4-player (2v2) matches by group, regime, and max round label
  (c) bomb tick alignment for the regime-A swap-8 bomb rounds at label 9 (does the bomb sit inside the
      label-9 freeze-end..round_end window, i.e. is the bomb really round 9's outcome?)
  (d) first-to-N test restated: a team with >= 9 wins at any freeze-end (would contradict a 9-win threshold)
  (e) claim joint table (late regime A, labels 9 and 13): alternative usability definition (no ambig exclusion)
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_9_04_residual.py
"""
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_9")
R = pl.read_parquet(SCR / "rounds.parquet")
M = pl.read_parquet(SCR / "matches.parquet")
meta = {r["match_key"]: r for r in M.to_dicts()}
LATE = {f"{d:02d}" for d in range(12, 26)}

fe = defaultdict(dict)
rend = defaultdict(dict)
for r in R.iter_rows(named=True):
    if r["fe_t"] is not None:
        fe[r["match_key"]][r["round"]] = (r["fe_t"], r["fe_ct"], r["fe_tick"])
    if r["end_winner"] is not None:
        rend[r["match_key"]][r["round"]] = r


def bound(mk):
    out = {}
    F = fe[mk]
    for r in sorted(F):
        if r + 1 not in F:
            continue
        t0, c0, _ = F[r]
        t1, c1, _ = F[r + 1]
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
set421 = {mk for mk in swap8 if mk not in swap12}
n_players = {}
PI = pl.read_parquet(sorted(glob.glob(str(BASE / f"basic_player_info{SUF}" / "dataframe_*")))[0],
                     columns=["match_key", "player_id"])
for r in PI.group_by("match_key").agg(pl.col("player_id").n_unique().alias("n")).iter_rows(named=True):
    n_players[r["match_key"]] = r["n"]

print("=" * 100)
print("STEP 4(a): label-7 last-round-half matches outside the 421")
ann7 = [mk for mk in meta if meta[mk].get("round_announce_last_round_half") == [7]]
print("  matches with last-round-half label exactly [7]:", len(ann7))
out7 = [mk for mk in ann7 if mk not in set421]
print("  of which outside the 421 swap-8 set:", len(out7))
c = Counter()
for mk in out7:
    c[(meta[mk]["regime"], n_players.get(mk), meta[mk]["max_end_label"] if meta[mk]["max_end_label"] is None else
       ("<=8" if meta[mk]["max_end_label"] <= 8 else ">=9"))] += 1
print("  (regime, n_players, max label bucket):", dict(c))
print("  boundary 8 class for those matches:", Counter(BND[mk].get(8, ("no_fe_pair",))[0] for mk in out7))
print("  max_end_label distribution:", dict(sorted(Counter(meta[mk]["max_end_label"] for mk in out7).items())))
print("  mean max_end_label:", round(sum(meta[mk]["max_end_label"] for mk in out7) / len(out7), 2) if out7 else None)
print("  example keys:", out7[:3])

print("\n" + "=" * 100)
print("STEP 4(b): 4-player matches by group and regime")
pl4 = Counter()
for mk in meta:
    if n_players.get(mk) == 4:
        g = "swap8" if mk in set421 else ("swap12" if mk in swap12 else "none")
        pl4[(meta[mk]["regime"], g)] += 1
print("  4-player matches (regime, group):", dict(sorted(pl4.items())))
for rg in ["A", "B"]:
    for g in ["none", "swap8"]:
        mks = [mk for mk in meta if n_players.get(mk) == 4 and meta[mk]["regime"] == rg and
               (("swap8" if mk in set421 else "none") == g)]
        print(f"  regime {rg} 4p {g}: max_end_label dist {dict(sorted(Counter(meta[mk]['max_end_label'] for mk in mks).items()))}")
n5 = Counter()
for mk in meta:
    if n_players.get(mk) == 10:
        n5[(meta[mk]["regime"], "swap8" if mk in set421 else ("swap12" if mk in swap12 else "none"))] += 1
print("  10-player matches (regime, group):", dict(sorted(n5.items())))

print("\n" + "=" * 100)
print("STEP 4(c): bomb tick alignment (label 9 and label 13) against freeze-end window")
rows = R.filter(pl.col("round").is_in([9, 13])).to_dicts()
al = Counter()
for r in rows:
    mk = r["match_key"]
    if r["n_defused"] + r["n_exploded"] == 0:
        continue
    g = "swap8" if mk in set421 else ("swap12" if mk in swap12 else "none")
    key = (r["round"], meta[mk]["regime"], g)
    lo, hi = r["fe_tick"], r["end_tick"]
    if lo is None or hi is None:
        al[key + ("no_window",)] += 1
        continue
    al[key + ("has_bomb_row",)] += 1
for k in sorted(al):
    if k[-1] == "has_bomb_row" or k[-1] == "no_window":
        print("  ", k, al[k])
# explicit bomb-tick check via bomb_state rows
BSf = sorted(glob.glob(str(BASE / f"basic_bomb_state{SUF}" / "dataframe_*")))[0]
BS = pl.read_parquet(BSf, columns=["match_key", "round", "tick", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
BSr = BS.join(R.select(["match_key", "round", "fe_tick", "end_tick"]), on=["match_key", "round"], how="left")
BSr = BSr.with_columns(inside=(pl.col("tick") >= pl.col("fe_tick")) & (pl.col("tick") <= pl.col("end_tick")))
BSr = BSr.filter(pl.col("round").is_in([9, 13]))
BSr = BSr.with_columns(g=pl.col("match_key").map_elements(
    lambda mk: "swap8" if mk in set421 else ("swap12" if mk in swap12 else "none"), return_dtype=pl.String),
                       reg=pl.col("match_key").map_elements(lambda mk: meta[mk]["regime"], return_dtype=pl.String))
print("  defuse/explode ticks inside [freeze_end, round_end] window of the same label:")
print(BSr.group_by(["round", "reg", "g", "inside"]).len().sort(["round", "reg", "g", "inside"]))

print("\n" + "=" * 100)
print("STEP 4(d): does any freeze-end show a team at >= 9 wins (9-win threshold check)")
mx9 = Counter()
for mk in meta:
    if not fe[mk]:
        continue
    g = "swap8" if mk in set421 else ("swap12" if mk in swap12 else "none")
    m = max(max(fe[mk][l][0], fe[mk][l][1]) for l in fe[mk])
    mx9[(meta[mk]["regime"], g, m if m >= 8 else "<8")] += 1
for k in sorted(mx9, key=lambda x: (x[0], x[1], str(x[2]))):
    if k[1] == "swap8" or k[2] in (8, 9, 10, 11, 12, 13, 14, 15, "<8") and k[1] != "swap12":
        print("  ", k, mx9[k])
print("  count of freeze-ends with max team score >= 9 in 2v2 (swap8) matches:",
      sum(1 for mk in set421 if max(max(fe[mk][l][0], fe[mk][l][1]) for l in fe[mk]) >= 9))

print("\n" + "=" * 100)
print("STEP 4(e): claim joint table, late regime A, alternative usability (any clean fit, ambig allowed)")
joint = Counter()
for mk in meta:
    if meta[mk]["regime"] != "A" or meta[mk]["day"] not in LATE:
        continue
    b9 = BND[mk].get(9)
    b13 = BND[mk].get(13)
    u9 = b9 is not None and b9[0] in ("noswap", "swap", "ambig") and 9 in rend[mk]
    u13 = b13 is not None and b13[0] in ("noswap", "swap", "ambig") and 13 in rend[mk]
    if not (u9 and u13):
        joint["not both"] += 1
        continue
    # truth where unambiguous; ambiguous -> skip
    t9 = b9[1]
    t13 = b13[1]
    if t9 is None or t13 is None:
        joint["ambig at 9 or 13"] += 1
        continue
    i9 = rend[mk][9]["end_winner"] != t9
    i13 = rend[mk][13]["end_winner"] != t13
    joint[(i9, i13, "swap8" if mk in set421 else "other")] += 1
for k in sorted(joint, key=str):
    print("  ", k, joint[k])
print("  claim: 13-inverted-only 1,621; 9-inverted-only 161; none both")

print("\nDONE step 4")
