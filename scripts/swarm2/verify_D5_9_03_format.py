"""verify D5-9, step 3: format and mode checks for the swap-8 matches.

Tests the claim's format identification (first-to-8, "MR15", halftime after round 8) against:
  * player_count on round_end, number of distinct players per match (player_info), map, match_type
  * round_announce_last_round_half and round_announce_match_point labels
  * halftime phase rows in round_state (engine's own phase field) at the boundary
  * final score implied by freeze-end scores plus the last round winner
  * joint late-regime-A table for labels 9 and 13 (claim: 1,621 / 161 / none)
Reads scratchpad parquet from verify_D5_9_01_build.py and the 2024 tome player_info channel (projected).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_9_03_format.py
"""
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_9")
R = pl.read_parquet(SCR / "rounds.parquet")
M = pl.read_parquet(SCR / "matches.parquet")
meta = {r["match_key"]: r for r in M.to_dicts()}

# ---------- freeze-end-only classification (same rule as step 2) ----------
fe = defaultdict(dict)
for r in R.iter_rows(named=True):
    if r["fe_t"] is not None:
        fe[r["match_key"]][r["round"]] = (r["fe_t"], r["fe_ct"])


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
grp = {}
for mk in meta:
    if mk in swap8 and mk not in swap12:
        grp[mk] = "swap8"
    elif mk in swap12:
        grp[mk] = "swap12"
    else:
        grp[mk] = "none"

# ---------- per-match attributes ----------
pc_mode = {mk: meta[mk]["mode_player_count"] for mk in meta}
print("=" * 100)
print("STEP 3: player_count (round_end, mode per match) by group and regime")
t = Counter()
for mk in meta:
    t[(meta[mk]["regime"], grp[mk], pc_mode[mk])] += 1
for k in sorted(t, key=lambda x: (x[0], x[1], -1 if x[2] is None else x[2])):
    print("  ", k, t[k])

# distinct players per match from player_info (round, player_id, team_code)
pif = sorted(glob.glob(str(BASE / f"basic_player_info{SUF}" / "dataframe_*")))
PI = pl.read_parquet(pif[0], columns=["match_key", "round", "player_id", "team_code"])
npl = PI.group_by("match_key").agg(pl.col("player_id").n_unique().alias("n_players"),
                                   pl.col("round").n_unique().alias("n_info_rounds"))
npl_map = {r["match_key"]: r for r in npl.to_dicts()}
print("\nSTEP 3: distinct players per match (player_info), by group and regime")
c = defaultdict(list)
for mk in meta:
    if mk in npl_map:
        c[(meta[mk]["regime"], grp[mk])].append(npl_map[mk]["n_players"])
    else:
        c[(meta[mk]["regime"], grp[mk])].append(None)
for k in sorted(c):
    vals = [v for v in c[k] if v is not None]
    cnt = Counter(vals)
    print("  ", k, "matches", len(c[k]), "n_players distribution:", dict(sorted(cnt.items())))

print("\nSTEP 3: map and match_type by group")
mp = Counter((grp[mk], meta[mk]["map_name"]) for mk in meta)
for k in sorted(mp):
    print("  ", k, mp[k])
print("  match_type:", Counter((grp[mk], meta[mk]["match_type"]) for mk in meta))

# ---------- announcements ----------
print("\n" + "=" * 100)
print("STEP 3: round_announce_last_round_half labels and round_announce_match_point labels by group")


def lab_str(v):
    if v is None:
        return "none"
    return ",".join(str(x) for x in v)


a = Counter()
b = Counter()
for mk in meta:
    a[(grp[mk], lab_str(meta[mk].get("round_announce_last_round_half")))] += 1
    b[(grp[mk], lab_str(meta[mk].get("round_announce_match_point")))] += 1
print(" last_round_half label(s) by group (top):")
for k, v in sorted(a.items(), key=lambda kv: (kv[0][0], -kv[1]))[:40]:
    print("  ", k, v)
print(" match_point label(s) by group (top):")
for k, v in sorted(b.items(), key=lambda kv: (kv[0][0], -kv[1]))[:40]:
    print("  ", k, v)

# ---------- halftime phase rows ----------
print("\n" + "=" * 100)
print("STEP 3: engine phase == 'halftime' rows by round label (count of matches with halftime rows at that label)")
ht = defaultdict(Counter)
for r in R.filter(pl.col("n_halftime_phase_rows") > 0).iter_rows(named=True):
    ht[grp[r["match_key"]]][r["round"]] += 1
for g in ["swap8", "swap12", "none"]:
    print("  ", g, dict(sorted(ht[g].items())))

# ---------- max label, sums, final scores ----------
print("\n" + "=" * 100)
print("STEP 3: max round_end label and final freeze-end sums by group and regime")
for rg in ["A", "B"]:
    for g in ["swap8", "swap12", "none"]:
        mks = [mk for mk in meta if meta[mk]["regime"] == rg and grp[mk] == g]
        mx = Counter(meta[mk]["max_end_label"] for mk in mks)
        print(f"  regime {rg} {g}: matches {len(mks)}; max_end_label dist: {dict(sorted(mx.items()))}")

print("\nSTEP 3: final freeze-end sum (last fe label) for swap8, by regime")
for rg in ["A", "B"]:
    mks = [mk for mk in meta if meta[mk]["regime"] == rg and grp[mk] == "swap8"]
    sums = Counter()
    for mk in mks:
        L = max(fe[mk])
        sums[(fe[mk][L][0] + fe[mk][L][1])] += 1
    print("  regime", rg, "fe sum at last label:", dict(sorted(sums.items())))

print("\nSTEP 3: maximum single-team freeze-end score (over all labels) for swap8 by regime")
for rg in ["A", "B"]:
    mks = [mk for mk in meta if meta[mk]["regime"] == rg and grp[mk] == "swap8"]
    mxs = Counter(max(max(fe[mk][l]) for l in fe[mk]) for mk in mks)
    print("  regime", rg, dict(sorted(mxs.items())))

# first-to-8 test: a team with 8 wins at a freeze-end that is NOT the last freeze-end would
# mean the match continued after 8 wins (impossible in a first-to-8 format). Count those.
print("\nSTEP 3: swap8 matches where a team already has >=8 wins at a freeze-end that is followed by another round")
cnt8 = Counter()
for rg in ["A", "B"]:
    mks = [mk for mk in meta if meta[mk]["regime"] == rg and grp[mk] == "swap8"]
    for mk in mks:
        labs = sorted(fe[mk])
        hit_nonlast = any(max(fe[mk][l]) >= 8 for l in labs[:-1])
        hit_last = max(fe[mk][labs[-1]]) >= 8
        cnt8[(rg, "reached8_before_last_fe" if hit_nonlast else "not_before_last_fe",
              "last_fe_has_8" if hit_last else "last_fe_below_8")] += 1
for k in sorted(cnt8):
    print("  ", k, cnt8[k])
print("  (first-to-8 predicts: no team reaches 8 at a freeze-end followed by another round)")

# ---------- regime A joint table for labels 9 and 13 (late regime A, both usable) ----------
print("\n" + "=" * 100)
print("STEP 3: joint table, late regime A: round 9 and round 13 usability and inversion (score-step truth)")
rend = defaultdict(dict)
for r in R.iter_rows(named=True):
    if r["end_winner"] is not None:
        rend[r["match_key"]][r["round"]] = r["end_winner"]
joint = Counter()
for mk in meta:
    if meta[mk]["regime"] != "A" or meta[mk]["day"] not in {f"{d:02d}" for d in range(12, 26)}:
        continue
    b9 = BND[mk].get(9)
    b13 = BND[mk].get(13)
    u9 = b9 is not None and b9[0] in ("noswap", "swap") and 9 in rend[mk]
    u13 = b13 is not None and b13[0] in ("noswap", "swap") and 13 in rend[mk]
    if not (u9 and u13):
        joint[("not both usable", grp[mk])] += 1
        continue
    i9 = rend[mk][9] != b9[1]
    i13 = rend[mk][13] != b13[1]
    joint[(f"both usable: 9 inv={i9} 13 inv={i13}", grp[mk])] += 1
for k in sorted(joint):
    print("  ", k, joint[k])

# ---------- also: late regime A, usable label 9 only, by group ----------
print("\nSTEP 3: late regime A, label-9 usable and inverted, by group")
s9 = Counter()
for mk in meta:
    if meta[mk]["regime"] != "A" or meta[mk]["day"] not in {f"{d:02d}" for d in range(12, 26)}:
        continue
    b9 = BND[mk].get(9)
    if b9 is None or b9[0] not in ("noswap", "swap") or 9 not in rend[mk]:
        s9[(grp[mk], "unusable")] += 1
        continue
    s9[(grp[mk], "usable_inverted" if rend[mk][9] != b9[1] else "usable_ok")] += 1
for k in sorted(s9):
    print("  ", k, s9[k])

print("\nDONE step 3")
