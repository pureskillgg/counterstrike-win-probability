"""verify D5-9 labels: per-label localisation of the round_end inversion, ordering/lag tests,
format identity from announcements, and the joint 9/13 table.

Uses the scratchpad tables from verify_D5_9_build.py. Writes nothing to the project or to E:.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_9_labels.py
"""
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\VD59")

H = pl.read_parquet(SCR / "header.parquet")
FE = pl.read_parquet(SCR / "fe.parquet")
MK = pl.read_parquet(SCR / "marks.parquet")
REr = pl.read_parquet(SCR / "end_raw.parquet")
BS = pl.read_parquet(SCR / "bomb.parquet")
DE = pl.read_parquet(SCR / "deaths.parquet")

regA = set(MK.filter(pl.col("event_type") == "round_announce_match_start")["match_key"].to_list())
RE = (REr.sort(["match_key", "round", "tick"])
      .group_by(["match_key", "round"], maintain_order=True).agg(
          pl.col("winner_team_code").first().alias("w"),
          pl.col("tick").first().alias("end_tick"),
          pl.col("win_reason_code").first().alias("reason"),
          pl.col("player_count").first().alias("pc")))
BO = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
BO = BO.with_columns(
    bo_side=pl.when(pl.col("evs").list.contains("bomb_exploded") & ~pl.col("evs").list.contains("bomb_defused")).then(2)
    .when(pl.col("evs").list.contains("bomb_defused") & ~pl.col("evs").list.contains("bomb_exploded")).then(3)
    .otherwise(None)).select(["match_key", "round", "bo_side"])
PL = BS.filter(pl.col("event_type") == "bomb_planted").group_by(["match_key", "round"]).agg(pl.len().alias("n_pl"))
DEs = DE.sort(["match_key", "round", "tick"])
LVt = DEs.group_by(["match_key", "round"]).agg(pl.col("tick").max().alias("lv_tick"))
LV = (DEs.join(LVt, on=["match_key", "round"]).filter(pl.col("tick") == pl.col("lv_tick"))
      .group_by(["match_key", "round"]).agg(pl.col("player_team_code").n_unique().alias("lv_nteam"),
                                            pl.col("player_team_code").max().alias("lv_victim")))
R = (FE.select(["match_key", "round", "fe_t", "fe_ct", "fe_tick"])
     .join(RE, on=["match_key", "round"], how="full", coalesce=True)
     .join(BO, on=["match_key", "round"], how="left")
     .join(PL, on=["match_key", "round"], how="left")
     .join(LV, on=["match_key", "round"], how="left"))
rounds = defaultdict(dict)
for row in R.iter_rows(named=True):
    rounds[row["match_key"]][row["round"]] = row
tick_rate = {r["match_key"]: r["tick_rate"] for r in H.iter_rows(named=True)}
mapname = {r["match_key"]: r["map_name"] for r in H.iter_rows(named=True)}


def g(mk, r):
    return rounds.get(mk, {}).get(r)


def classify(t, ct, t2, ct2, w):
    if None in (t, ct, t2, ct2, w) or w not in (2, 3):
        return "na"
    a = 1 if w == 2 else 0
    b = 1 if w == 3 else 0
    ns = (t2 == t + a) and (ct2 == ct + b)
    sw = (t2 == ct + b) and (ct2 == t + a)
    if ns and sw:
        return "ambig"
    if ns:
        return "ns"
    if sw:
        return "sw"
    return "other"


def boundary(mk, r):
    a_, b_ = g(mk, r), g(mk, r + 1)
    if not a_ or not b_:
        return "absent"
    return classify(a_.get("fe_t"), a_.get("fe_ct"), b_.get("fe_t"), b_.get("fe_ct"), a_.get("w"))


cls8 = {mk: boundary(mk, 8) for mk in rounds}
cls12 = {mk: boundary(mk, 12) for mk in rounds}
sw8 = {mk for mk, v in cls8.items() if v == "sw"}
sw12 = {mk for mk, v in cls12.items() if v == "sw"}
claimA = {mk for mk in sw8 if mk not in sw12 and mk in regA}
claimB = {mk for mk in sw8 if mk not in sw12 and mk not in regA}
ctrl12A = {mk for mk in sw12 if mk not in sw8 and mk in regA}
ctrl12B = {mk for mk in sw12 if mk not in sw8 and mk not in regA}
allA = regA
allB = set(rounds) - regA
otherA = allA - claimA - ctrl12A
otherB = allB - claimB - ctrl12B
print("groups: claimA", len(claimA), "claimB", len(claimB), "ctrl12A", len(ctrl12A), "ctrl12B", len(ctrl12B),
      "otherA", len(otherA), "otherB", len(otherB))


def pct(a, b):
    return f"{100.0 * a / b:5.1f}%" if b else "   -  "


# ---------- A. per-label agreement of round_end winner with bomb and last-victim truth ----------
print("\n=== A. per-label agreement: round_end winner vs bomb truth and vs last-victim truth")
print("     (disagree % shown; bomb n and last-victim n shown; LV only on no-bomb rounds with one victim team)")


def per_label(mks, label):
    print(f"\n  [{label}] matches={len(mks)}")
    print("   r   bomb_n  bomb_dis%   lv_n   lv_dis%")
    for r in range(1, 31):
        bn = bd = ln = ld = 0
        for mk in mks:
            row = g(mk, r)
            if not row or row.get("w") not in (2, 3):
                continue
            if row.get("bo_side") is not None:
                bn += 1
                bd += int(row["bo_side"] != row["w"])
            elif row.get("n_pl") is None and row.get("lv_nteam") == 1 and row.get("lv_victim") in (2, 3):
                ln += 1
                ld += int((5 - row["lv_victim"]) != row["w"])
        if bn or ln:
            print(f"  {r:3d}  {bn:6d}  {pct(bd, bn):>8s}  {ln:6d}  {pct(ld, ln):>8s}")


per_label(claimA, "claim group regime A (sw 8->9, no sw 12->13): MR8-type halftime at 8")
per_label(claimB, "claim group regime B (sw 8->9): MR8-type halftime at 8")
per_label(ctrl12A, "MR12 control regime A (sw 12->13, no sw 8->9)")
per_label(ctrl12B, "MR12 control regime B (sw 12->13, no sw 8->9)")
per_label(otherA, "other regime A (no sw at 8 or 12 classified)")
per_label(otherB, "other regime B (no sw at 8 or 12 classified)")

# ---------- B. ordering and window tests ----------
print("\n=== B. tick ordering fe(r) < end(r) < fe(r+1), and bomb tick inside [fe(r), end(r)]")


def ordering(mks, label):
    viol = ok = 0
    for mk in mks:
        for r in range(1, 30):
            a_, b_ = g(mk, r), g(mk, r + 1)
            if not a_ or not b_:
                continue
            if None in (a_.get("fe_tick"), a_.get("end_tick"), b_.get("fe_tick")):
                continue
            if a_["fe_tick"] < a_["end_tick"] < b_["fe_tick"]:
                ok += 1
            else:
                viol += 1
    print(f"  {label}: pairs ok {ok}, violations {viol}")


ordering(claimA, "claim A")
ordering(claimB, "claim B")
ordering(ctrl12A, "MR12 control A")
ordering(allB, "all regime B")

# bomb window per label for claim A (single pass)
inside = defaultdict(lambda: [0, 0])
bo_rows = BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
bo_by = {(r["match_key"], r["round"]): r["tick"] for r in bo_rows.sort("tick").group_by(["match_key", "round"]).agg(
    pl.col("tick").first()).iter_rows(named=True)}
for (mk, r), tk in bo_by.items():
    if mk not in claimA and mk not in claimB:
        continue
    row = g(mk, r)
    if not row or row.get("fe_tick") is None or row.get("end_tick") is None:
        continue
    grp = "A" if mk in claimA else "B"
    inside[(grp, r)][0 if row["fe_tick"] <= tk <= row["end_tick"] else 1] += 1
print("  bomb tick inside its labelled window [fe(r), end(r)] (inside, outside):")
for key in sorted(inside):
    if key[1] in (7, 8, 9, 10, 11):
        print("   ", key, inside[key])

# ---------- C. lag test: does round_end(9) equal the sw-mapped step at 8->9 (i.e. round 8's result)? ----------
print("\n=== C. lag/shift test on claim groups: compare w_9 with the 8->9 step under side-swap mapping")
for label, mks in [("claim A", claimA), ("claim B", claimB)]:
    c = Counter()
    for mk in mks:
        r8, r9 = g(mk, 8), g(mk, 9)
        if not r8 or not r9 or None in (r8.get("fe_t"), r8.get("fe_ct"), r9.get("fe_t"), r9.get("fe_ct")):
            continue
        dt = r9["fe_t"] - r8["fe_ct"]   # swapped mapping: T field now holds CT's old score
        dc = r9["fe_ct"] - r8["fe_t"]
        if (dt, dc) == (1, 0):
            sw_truth = 3
        elif (dt, dc) == (0, 1):
            sw_truth = 2
        else:
            c["sw_step_unusable"] += 1
            continue
        c["n"] += 1
        c["w9 == sw-step(8->9)"] += int(r9["w"] == sw_truth)
        c["w8 == sw-step(8->9)"] += int(r8["w"] == sw_truth)
    print(f"  {label}: {dict(c)}")

# ---------- D. joint table, regime A, usable steps at 9 and 13 (step-based and LV-based) ----------
print("\n=== D. joint table: round_end inversion at 9 and at 13 (regime A, step truth where usable)")


def step_ns(r_row, n_row):
    dt, dc = n_row["fe_t"] - r_row["fe_t"], n_row["fe_ct"] - r_row["fe_ct"]
    if (dt, dc) == (1, 0):
        return 2
    if (dt, dc) == (0, 1):
        return 3
    return None


joint = Counter()
for mk in allA:
    r9, r10 = g(mk, 9), g(mk, 10)
    r13, r14 = g(mk, 13), g(mk, 14)
    if not (r9 and r10 and r9.get("w") in (2, 3)) or None in (r9.get("fe_t"), r10.get("fe_t")):
        j9 = None
    else:
        s9 = step_ns(r9, r10)
        j9 = None if s9 is None else (s9 != r9["w"])
    if not (r13 and r14 and r13.get("w") in (2, 3)) or None in (r13.get("fe_t"), r14.get("fe_t")):
        j13 = None
    else:
        s13 = step_ns(r13, r14)
        j13 = None if s13 is None else (s13 != r13["w"])
    if j9 is None or j13 is None:
        joint["unusable"] += 1
    else:
        joint[(f"inv9={j9}", f"inv13={j13}")] += 1
print("  ", dict(joint))

# ---------- E. format identity from announcements, all matches ----------
print("\n=== E. halftime-format identity from last-round-half announcement (label 7 = 8-round half, 11 = 12-round half)")
ann = MK.filter(pl.col("event_type") == "round_announce_last_round_half")
lab_by_mk = {r["match_key"]: r["min_round"] for r in ann.iter_rows(named=True)}
ann_multi = ann.filter(pl.col("n") > 1).height
print("  matches with a last-round-half announcement:", len(lab_by_mk), " (rows with n>1:", ann_multi, ")")
print("  announcement label by group:")
for name, mks in [("claimA", claimA), ("claimB", claimB), ("ctrl12A", ctrl12A), ("ctrl12B", ctrl12B),
                  ("otherA", otherA), ("otherB", otherB)]:
    print(f"    {name:8s}", dict(Counter(lab_by_mk.get(mk) for mk in mks).most_common()))
lab7 = {mk for mk, v in lab_by_mk.items() if v == 7}
print("  label-7 matches:", len(lab7), " in claim A/B:", len(lab7 & (claimA | claimB)),
      " regime A:", len(lab7 & regA), " regime B:", len(lab7 - regA))
extra7 = lab7 - claimA - claimB
print("  label-7 matches NOT in claim group:", len(extra7))
print("    their cls8:", dict(Counter(cls8.get(mk) for mk in extra7)),
      " cls12:", dict(Counter(cls12.get(mk) for mk in extra7)))
print("    regime:", dict(Counter("A" if mk in regA else "B" for mk in extra7)))
print("    maps:", Counter(mapname[mk] for mk in extra7).most_common(6))

# ---------- F. final score structure for label-7 matches (any group) ----------
print("\n=== F. final score structure, label-7 matches")


def last_label(mk):
    rs = [r for r, v in rounds[mk].items() if v.get("w") in (2, 3) or v.get("fe_t") is not None]
    return max(rs) if rs else None


def final(mk):
    N = last_label(mk)
    row = g(mk, N)
    if not row or row.get("fe_t") is None or row.get("w") not in (2, 3):
        return None
    return N, row["fe_t"] + (row["w"] == 2), row["fe_ct"] + (row["w"] == 3)


fin = Counter()
ex = []
for mk in lab7:
    f = final(mk)
    if f is None:
        fin["no_final"] += 1
        continue
    N, t, ct = f
    grp = ("claimA" if mk in claimA else "claimB" if mk in claimB else "other")
    grp += "/" + ("A" if mk in regA else "B")
    fin[(grp, "max=" + str(max(t, ct)))] += 1
    fin[(grp, "sum==N" if t + ct == N else "sum!=N")] += 1
    if max(t, ct) < 9:
        ex.append((mk, N, t, ct, grp))
for k, v in sorted(fin.items(), key=lambda kv: str(kv[0])):
    print("  ", k, v)
print("  label-7 matches with max final < 9 (key, last label, t, ct, group):")
for e in ex[:20]:
    print("   ", e)
print("  total with max < 9:", len(ex))

# ---------- G. gaps: round_end(r) -> freeze_end(r+1) for halftime and ordinary boundaries ----------
print("\n=== G. timing: seconds from round_end(r) to freeze_end(r+1)")


def gap_stats(mks, r):
    out = []
    for mk in mks:
        a_, b_ = g(mk, r), g(mk, r + 1)
        if a_ and b_ and a_.get("end_tick") is not None and b_.get("fe_tick") is not None:
            out.append((b_["fe_tick"] - a_["end_tick"]) / (tick_rate.get(mk) or 64))
    if not out:
        return "n=0"
    out.sort()
    return f"n={len(out)} median={st.median(out):.1f}s min={out[0]:.1f}s max={out[-1]:.1f}s"


ordinary_B = {mk for mk in allB if cls8.get(mk) in ("ns",) and cls12.get(mk) in ("ns", "absent")}
print("  ordinary regime-B boundary 5->6:", gap_stats(allB, 5))
print("  ordinary regime-B boundary 10->11:", gap_stats(allB, 10))
print("  claim A boundary 8->9:", gap_stats(claimA, 8))
print("  claim B boundary 8->9:", gap_stats(claimB, 8))
print("  claim A boundary 9->10:", gap_stats(claimA, 9))
print("  claim B boundary 9->10:", gap_stats(claimB, 9))
print("  MR12 control A boundary 12->13:", gap_stats(ctrl12A, 12))
print("  MR12 control B boundary 12->13:", gap_stats(ctrl12B, 12))

# ---------- H. player_count (round_end) by group ----------
print("\n=== H. round_end player_count by group (rows, all labels)")
for name, mks in [("claimA", claimA), ("claimB", claimB), ("ctrl12A", ctrl12A), ("otherA", otherA)]:
    c = Counter()
    for mk in mks:
        for r, row in rounds[mk].items():
            if row.get("pc") is not None:
                c[row["pc"]] += 1
    print(f"  {name}: {c.most_common(5)}")
