"""Verifier D3-5: is the 2024 round-13 defect in round_end labels (winner-derived lead wrong) or in
round_state (score-lead feature wrong)? Independent recomputation on the 2024 G: tome, all 4,435 matches.

Read only. Memory: column-projected small tables (round_end, round_state, bomb_state, player_death, header).
Output goes to stdout; the run wrapper writes logs/swarm2/verify_D3_5.log.

Sources compared per (match, round):
  A  round_end.winner_team_code (side code: 2 = T, 3 = CT)                       [claimed defective]
  B  round_end.win_reason_code mapped to the winning side (CS2 reason semantics)  [same row as A]
  C  round_state score transition r -> r+1 at round_freeze_end (side-attached fields, +1 to the winner)
     resolved with the MR12 halftime swap schedule only where the pattern is ambiguous (ASSUMPTION)
  D  bomb outcome: bomb_exploded -> T, bomb_defused -> CT (rounds with an outcome row only)
  E  elimination: all five distinct players of one side died and no bomb was planted -> other side

Assumptions (ASSUMPTION): 5 players per side; MR12 regulation (swap after the 12th completed round);
round_state fields are side-attached (t_score = score of the team currently on T); score increments are +1.
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(220)
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3
SWAP_AFTER = 12  # ASSUMPTION: MR12 regulation, halftime swap after the 12th completed round
# CS2 round_end win reasons -> side of the winner. Surrender codes: the losing side surrendered.
REASON_SIDE = {1: T, 9: T, 13: T, 18: T, 7: CT, 8: CT, 11: CT, 12: CT, 17: CT}


def g_read(ch, cols=None):
    files = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def day_of(m):
    return m.split("/")[3]


def grp_of(m):
    return "day12_25" if int(day_of(m)) >= 12 else "day01_07"


print("== load ==")
HD = pl.concat([pl.read_parquet(f, columns=["key", "t_starters_score_final", "ct_starters_score_final"])
                for f in sorted(glob.glob(str(TOME / "header" / "dataframe_*")))])
RE = g_read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
RS = g_read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
BS = g_read("bomb_state", ["match_key", "round", "tick", "event_type"])
DE = g_read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code"])
print("tables:", {"round_end": RE.height, "round_state": RS.height, "bomb_state": BS.height,
                  "player_death": DE.height, "header": HD.height})

# ---------- per-(match, round) maps ----------
A = {}  # (m, r) -> (winner, reason, tick)
for m, r, tk, w, rc in RE.sort("tick").select(["match_key", "round", "tick", "winner_team_code",
                                                "win_reason_code"]).iter_rows():
    A.setdefault((m, int(r)), (int(w), int(rc), int(tk)))
FE = {}  # (m, r) -> (tick, t, c), first round_freeze_end by tick
fe_rows = RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
fe_count = collections.Counter(zip(fe_rows["match_key"].to_list(), fe_rows["round"].to_list()))
for m, r, tk, t, c in fe_rows.select(["match_key", "round", "tick", "t_score", "ct_score"]).iter_rows():
    FE.setdefault((m, int(r)), (int(tk), t, c))
BOMB = collections.defaultdict(set)
for m, r, ev in BS.filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"])) \
        .select(["match_key", "round", "event_type"]).iter_rows():
    BOMB[(m, int(r))].add(ev)
DEATHS = collections.defaultdict(lambda: collections.defaultdict(set))  # (m,r) -> team -> {pid}
for m, r, pid, team in DE.select(["match_key", "round", "player_id", "player_team_code"]).iter_rows():
    DEATHS[(m, int(r))][team].add(pid)

matches = sorted(set(HD["key"].to_list()))
FE_BY_MATCH = collections.defaultdict(dict)
for (m, r), v in FE.items():
    FE_BY_MATCH[m][r] = v
print("matches in header:", len(matches), "| round_end matches:", len({k[0] for k in A}),
      "| freeze_end matches:", len({k[0] for k in FE}))
print("freeze_end duplicate (match,round) keys:", sum(1 for v in fe_count.values() if v > 1))


def classify(t, c, t2, c2):
    """Return the set of (winner_side, swap_flag) explaining the freeze r -> r+1 score change."""
    out = set()
    if (t2, c2) == (t + 1, c):
        out.add((T, 0))
    if (t2, c2) == (t, c + 1):
        out.add((CT, 0))
    if (t2, c2) == (c, t + 1):
        out.add((T, 1))
    if (t2, c2) == (c + 1, t):
        out.add((CT, 1))
    return out


# ---------- C: score-implied winner per transition ----------
C = {}  # (m, r) -> resolved winner side from freeze r -> r+1
C_status = collections.Counter()
swap_obs = collections.Counter()  # (k_r) where a swap flag was observed unambiguously
swap_sched_ok = collections.Counter()
swap_sched_bad = collections.Counter()
for m in matches:
    fes = FE_BY_MATCH.get(m, {})
    rs = sorted(fes)
    for r in rs:
        if (r + 1) not in fes:
            continue
        _, t, c = fes[r]
        _, t2, c2 = fes[r + 1]
        if None in (t, c, t2, c2):
            C_status["null_score"] += 1
            continue
        cands = classify(t, c, t2, c2)
        k_r = t + c
        sched_swap = int(k_r < SWAP_AFTER <= k_r + 1)  # swap if this round completes the 12th round
        if not cands:
            C_status["no_pattern"] += 1
            continue
        winners = {w for w, _ in cands}
        if len(cands) == 1:
            w, s = next(iter(cands))
            C[(m, r)] = w
            C_status["unambiguous"] += 1
            swap_obs[(k_r, s)] += 1
            if s == sched_swap:
                swap_sched_ok[k_r] += 1
            else:
                swap_sched_bad[k_r] += 1
        else:
            # ambiguous: resolve by the schedule's swap flag if it picks one candidate
            pick = [w for w, s in cands if s == sched_swap]
            if len(pick) == 1:
                C[(m, r)] = pick[0]
                C_status["ambiguous_resolved_by_schedule"] += 1
            else:
                C_status["ambiguous_unresolved"] += 1
print("\n== C: score transitions ==")
for k, v in sorted(C_status.items()):
    print(f"  {k}: {v}")
print("swap flag observed by completed count k_r (k_r, swap): count:", sorted(swap_obs.items()))
print("schedule agrees with unambiguous swap flags, by k_r:", sorted(swap_sched_ok.items()))
print("schedule disagrees, by k_r:", sorted(swap_sched_bad.items()))

# ---------- main comparison table ----------
rows = []
for (m, r), (w, rc, tk) in A.items():
    grp = grp_of(m)
    rs_side = REASON_SIDE.get(rc, -1)
    bomb = BOMB.get((m, r), set())
    bside = None
    if "bomb_exploded" in bomb:
        bside = T
    elif "bomb_defused" in bomb:
        bside = CT
    dd = DEATHS.get((m, r), {})
    nT = len(dd.get(T, set()))
    nCT = len(dd.get(CT, set()))
    eside = None
    if "bomb_planted" not in bomb and "bomb_exploded" not in bomb and "bomb_defused" not in bomb:
        if nCT >= 5 and nT < 5:
            eside = T
        elif nT >= 5 and nCT < 5:
            eside = CT
    rows.append({"m": m, "r": r, "grp": grp, "day": int(day_of(m)), "A": w, "B": rs_side,
                 "C": C.get((m, r), -1), "D": -1 if bside is None else bside,
                 "E": -1 if eside is None else eside, "has_C": (m, r) in C,
                 "has_D": bside is not None, "has_E": eside is not None})
df = pl.DataFrame(rows)
print("\nrows (round_end rows):", df.height, "| with C:", int(df["has_C"].sum()),
      "| with D:", int(df["has_D"].sum()), "| with E:", int(df["has_E"].sum()))

cmp = df.filter(pl.col("has_C"))
print("\n== A vs C (round_end winner vs score transition), by group and round ==")
t1 = cmp.group_by(["grp", "r"]).agg(pl.len().alias("n"), (pl.col("A") != pl.col("C")).sum().alias("A_ne_C"))
t1 = t1.with_columns((pl.col("A_ne_C") / pl.col("n")).round(4).alias("rate")).sort(["grp", "r"])
print(t1.filter(pl.col("r") <= 26).rows())
print("total A!=C by group:", cmp.group_by("grp").agg(pl.len().alias("n"), (pl.col("A") != pl.col("C")).sum().alias("ne")).rows())

print("\n== B vs C (reason side vs score transition) ==")
bc = cmp.filter(pl.col("B") >= 0)
print(bc.group_by("grp").agg(pl.len().alias("n"), (pl.col("B") != pl.col("C")).sum().alias("B_ne_C")).rows())
print("== B vs A (reason side vs round_end winner) ==")
print(df.filter(pl.col("B") >= 0).group_by("grp").agg(pl.len().alias("n"), (pl.col("B") != pl.col("A")).sum().alias("B_ne_A")).rows())

print("\n== D (bomb outcome) vs A and vs C ==")
d = df.filter(pl.col("has_D"))
print("bomb-outcome rows:", d.height)
print(d.group_by(["grp", "r"]).agg(pl.len().alias("n"), (pl.col("D") != pl.col("A")).sum().alias("D_ne_A"),
                                  ((pl.col("D") != pl.col("C")) & (pl.col("C") >= 0)).sum().alias("D_ne_C"),
                                  (pl.col("C") >= 0).sum().alias("withC"))
      .filter(pl.col("r").is_in([12, 13, 14, 25])).sort(["grp", "r"]).rows())
d_all = d.with_columns((pl.col("D") != pl.col("A")).alias("ne"))
print("bomb rows D!=A, by group x round<=12 vs 13 vs >13:",
      d_all.with_columns(pl.when(pl.col("r") < 13).then(pl.lit("<13")).when(pl.col("r") == 13).then(pl.lit("13"))
                         .otherwise(pl.lit(">13")).alias("rb"))
      .group_by(["grp", "rb"]).agg(pl.len().alias("n"), pl.col("ne").sum().alias("D_ne_A")).sort(["grp", "rb"]).rows())
dc = d.filter(pl.col("C") >= 0)
print("bomb rows with C: D==C:", int((dc["D"] == dc["C"]).sum()), "D!=C:", int((dc["D"] != dc["C"]).sum()))

print("\n== E (elimination) vs A and vs C ==")
e = df.filter(pl.col("has_E"))
print("elimination-determined rows:", e.height)
print(e.group_by(["grp", "r"]).agg(pl.len().alias("n"), (pl.col("E") != pl.col("A")).sum().alias("E_ne_A"),
                                  ((pl.col("E") != pl.col("C")) & (pl.col("C") >= 0)).sum().alias("E_ne_C"))
      .filter(pl.col("r").is_in([12, 13, 14])).sort(["grp", "r"]).rows())
ec = e.filter(pl.col("C") >= 0)
print("elimination rows with C: E==C:", int((ec["E"] == ec["C"]).sum()), "E!=C:", int((ec["E"] != ec["C"]).sum()))
print("elimination rows E!=A total, by group:", e.group_by("grp").agg((pl.col("E") != pl.col("A")).sum().alias("E_ne_A"), pl.len().alias("n")).rows())

# save the summary numbers used by the verdict
summ = {
    "A_ne_C_by_grp": cmp.group_by("grp").agg(pl.len().alias("n"), (pl.col("A") != pl.col("C")).sum().alias("ne")).to_dicts(),
}
print("\nsummary:", summ)
