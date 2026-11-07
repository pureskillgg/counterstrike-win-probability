"""verify D5-9 final: (1) ordinary-round timing in regime A vs B (is the 8->9 gap short for a halftime
reason or just regime-wide?); (2) the affected set (round-13 bomb mismatch, player_count==20) and its roster
size; (3) the 1,621 / 161 joint counts restricted to the non-affected late regime-A matches.

Uses scratchpad tables from verify_D5_9_build.py. Writes nothing to the project or to E:.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_9_final.py
"""
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\VD59")
H = pl.read_parquet(SCR / "header.parquet")
FE = pl.read_parquet(SCR / "fe.parquet")
MK = pl.read_parquet(SCR / "marks.parquet")
REr = pl.read_parquet(SCR / "end_raw.parquet")
BS = pl.read_parquet(SCR / "bomb.parquet")
ROST = pl.read_parquet(SCR / "roster.parquet")

regA = set(MK.filter(pl.col("event_type") == "round_announce_match_start")["match_key"].to_list())
lh = {r["match_key"]: r["min_round"] for r in
      MK.filter(pl.col("event_type") == "round_announce_last_round_half").iter_rows(named=True)}
tick_rate = {r["match_key"]: r["tick_rate"] for r in H.iter_rows(named=True)}
roster = {r["match_key"]: r["n_players"] for r in ROST.iter_rows(named=True)}
RE = (REr.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"], maintain_order=True).agg(
    pl.col("winner_team_code").first().alias("w"), pl.col("tick").first().alias("end_tick"),
    pl.col("player_count").first().alias("pc")))
BO = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).group_by(["match_key", "round"])
      .agg(pl.col("event_type").unique().alias("evs")))
BO = BO.with_columns(
    bo_side=pl.when(pl.col("evs").list.contains("bomb_exploded") & ~pl.col("evs").list.contains("bomb_defused")).then(2)
    .when(pl.col("evs").list.contains("bomb_defused") & ~pl.col("evs").list.contains("bomb_exploded")).then(3)
    .otherwise(None)).select(["match_key", "round", "bo_side"])
R = (FE.select(["match_key", "round", "fe_t", "fe_ct", "fe_tick"])
     .join(RE, on=["match_key", "round"], how="full", coalesce=True)
     .join(BO, on=["match_key", "round"], how="left"))
rounds = defaultdict(dict)
for row in R.iter_rows(named=True):
    rounds[row["match_key"]][row["round"]] = row


def g(mk, r):
    return rounds.get(mk, {}).get(r)


# ---------- 1. ordinary timing ----------
print("=== 1. timing: seconds from round_end(r) to freeze_end(r+1), by regime and label (all matches)")


def gap(mks, r):
    out = []
    for mk in mks:
        a_, b_ = g(mk, r), g(mk, r + 1)
        if a_ and b_ and a_.get("end_tick") is not None and b_.get("fe_tick") is not None:
            out.append((b_["fe_tick"] - a_["end_tick"]) / (tick_rate.get(mk) or 64))
    if not out:
        return "n=0"
    out.sort()
    return f"n={len(out)} median={st.median(out):.1f}s p10={out[int(.1*len(out))]:.1f}s p90={out[int(.9*len(out))]:.1f}s"


A_mk = [m for m in rounds if m in regA]
B_mk = [m for m in rounds if m not in regA]
A5 = [m for m in A_mk if lh.get(m) == 11]
B5 = [m for m in B_mk if lh.get(m) == 11]
A2 = [m for m in A_mk if lh.get(m) == 7]
B2 = [m for m in B_mk if lh.get(m) == 7]
for r in [3, 6, 10, 11]:
    print(f"  5v5 (ann 11) regime A, boundary {r}->{r+1}: {gap(A5, r)}")
    print(f"  5v5 (ann 11) regime B, boundary {r}->{r+1}: {gap(B5, r)}")
print(f"  2v2 (ann 7) regime A, boundary 8->9 (halftime): {gap(A2, 8)}")
print(f"  2v2 (ann 7) regime B, boundary 8->9 (halftime): {gap(B2, 8)}")
print(f"  2v2 (ann 7) regime A, boundary 9->10: {gap(A2, 9)}")
print(f"  2v2 (ann 7) regime B, boundary 9->10: {gap(B2, 9)}")
print(f"  5v5 (ann 11) regime A, boundary 12->13 (halftime): {gap(A5, 12)}")
print(f"  5v5 (ann 11) regime B, boundary 12->13 (halftime): {gap(B5, 12)}")

# ---------- 2. affected set (e2 definition) and roster ----------
print("\n=== 2. affected set: round-13 bomb mismatch with player_count==20 (e2 definition), and roster size")
aff = []
for mk, rr in rounds.items():
    row = rr.get(13)
    if row and row.get("bo_side") is not None and row.get("pc") == 20 and row.get("w") in (2, 3):
        if row["bo_side"] != row["w"]:
            aff.append(mk)
aff = sorted(set(aff))
print("  affected matches (e2 definition, recomputed):", len(aff))
print("  roster of affected matches (distinct players in kills):", Counter(roster.get(m) for m in aff).most_common(5))
print("  affected matches in regime A:", sum(1 for m in aff if m in regA), " ann labels:", Counter(lh.get(m) for m in aff))

# ---------- 3. joint 9 / 13 table for non-affected late regime-A matches ----------
print("\n=== 3. joint 9/13 table, regime A, excluding the affected set (usable steps at 9 and 13)")


def step_ns(a_, b_):
    dt, dc = b_["fe_t"] - a_["fe_t"], b_["fe_ct"] - a_["fe_ct"]
    if (dt, dc) == (1, 0):
        return 2
    if (dt, dc) == (0, 1):
        return 3
    return None


def joint(mks):
    c = Counter()
    for mk in mks:
        r9, r10, r13, r14 = g(mk, 9), g(mk, 10), g(mk, 13), g(mk, 14)
        if not (r9 and r10 and r13 and r14) or None in (r9.get("fe_t"), r10.get("fe_t"), r13.get("fe_t"), r14.get("fe_t")):
            c["unusable"] += 1
            continue
        if r9.get("w") not in (2, 3) or r13.get("w") not in (2, 3):
            c["unusable"] += 1
            continue
        s9, s13 = step_ns(r9, r10), step_ns(r13, r14)
        if s9 is None or s13 is None:
            c["unusable"] += 1
            continue
        c[(f"inv9={s9 != r9['w']}", f"inv13={s13 != r13['w']}")] += 1
    return dict(c)


affset = set(aff)
print("  all regime A:", joint(A_mk))
print("  regime A excluding affected set:", joint([m for m in A_mk if m not in affset]))
print("  affected set only:", joint([m for m in A_mk if m in affset]))
