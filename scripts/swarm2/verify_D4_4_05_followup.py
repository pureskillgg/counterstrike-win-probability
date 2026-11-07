r"""Verifier D4-4 follow-up.

(a) 2026 control reproduced with per-match halftime = header max_rounds // 2 (as the claimed control's
    log describes), and a check that score-step halftime groups agree with header max_rounds.
(b) 2024: round-13 winner from the no-swap rule on boundary 13->14 (no halftime is possible after round 13
    in MR12; the swap test is never positive at 13->14 in the data, see verify_D4_4_03 output), so
    ties at halftime no longer remove the round-13 check. Cross-tab against label, bomb and S12 header fit.
(c) Composition of the claimed 825 round-13 header flips by score-step halftime group.

Usage: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D4_4_05_followup.py
"""
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2\tomes")
T, CT = 2, 3


def base(era):
    return ROOT / ("2024-02-g" if era == "2024" else "2026-10-01_to_08") / "tome" / "csds"


def files(era, ch):
    b = base(era)
    fd = b / "header" if ch == "header" else sorted(b.glob(f"basic_{ch}.*"))[0]
    return sorted(glob.glob(str(fd / "dataframe_*")))


def classify(a, b):
    t0, c0 = a
    t1, c1 = b
    ns = (t1 - t0, c1 - c0)
    sw = (t1 - c0, c1 - t0)
    ns_ok = ns in ((1, 0), (0, 1))
    sw_ok = sw in ((1, 0), (0, 1))
    if ns_ok and sw_ok:
        return "ambig", None
    if ns_ok:
        return "noswap", (T if ns == (1, 0) else CT)
    if sw_ok:
        return "swap", (CT if sw == (1, 0) else T)
    return "bad", None


def load(era):
    hdr_cols = ["key", "t_starters_score_final", "ct_starters_score_final"]
    H = pl.scan_parquet(files(era, "header")).collect()
    if "max_rounds" in H.columns:
        hdr_cols.append("max_rounds")
    HD = {r["key"]: r for r in H.select(hdr_cols).iter_rows(named=True)}
    RE = pl.scan_parquet(files(era, "round_end")).select(["match_key", "round", "tick", "winner_team_code"]).collect()
    RE1 = RE.sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first", maintain_order=True)
    BS = (pl.scan_parquet(files(era, "bomb_state")).select(["match_key", "round", "tick", "event_type"])
          .filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"])).collect())
    BO = BS.group_by(["match_key", "round"]).agg(
        (pl.col("event_type") == "bomb_exploded").any().alias("exp"),
        (pl.col("event_type") == "bomb_defused").any().alias("dfs"))
    FE = (pl.scan_parquet(files(era, "round_state")).select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
          .filter(pl.col("event_type") == "round_freeze_end").collect())
    FE = FE.sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first", maintain_order=True)
    rows = defaultdict(dict)
    for r in RE1.iter_rows(named=True):
        rows[r["match_key"]].setdefault(r["round"], {})["L"] = r["winner_team_code"]
    for r in BO.iter_rows(named=True):
        d = rows[r["match_key"]].setdefault(r["round"], {})
        d["B"] = T if (r["exp"] and not r["dfs"]) else (CT if (r["dfs"] and not r["exp"]) else None)
        d["hasB"] = True
    for r in FE.iter_rows(named=True):
        rows[r["match_key"]].setdefault(r["round"], {})["fe"] = (r["t_score"], r["ct_score"])
    return HD, rows


def fit(wa, n, tA, tB):
    if (wa, n - wa) == (tA, tB):
        return "ok"
    if (wa, n - wa) == (tB, tA):
        return "swapped"
    return "bad"


def wins_A(rr, n, sched):
    wa = 0; swaps = 0
    for r in range(1, n + 1):
        side_A = T if swaps % 2 == 0 else CT
        wa += (rr[r]["L"] == side_A)
        if r in sched:
            swaps += 1
    return wa


def complete_matches(HD, rows, maxn=24):
    out = []
    for m in rows:
        if m not in HD:
            continue
        rr = rows[m]
        Lr = sorted(r for r in rr if "L" in rr[r])
        n = len(Lr)
        tA = int(HD[m]["t_starters_score_final"]); tB = int(HD[m]["ct_starters_score_final"])
        if Lr == list(range(1, n + 1)) and n == tA + tB and n <= maxn:
            out.append((m, n, tA, tB))
    return out


def score_halftime(rr, n):
    swaps = []
    for r in range(1, n):
        a = rr.get(r, {}).get("fe"); b = rr.get(r + 1, {}).get("fe")
        if a is None or b is None:
            continue
        cls, _ = classify(a, b)
        if cls == "swap":
            swaps.append(r)
    return tuple(swaps)


# ---------------- (a) 2026 control with max_rounds // 2 halftime
HD26, rows26 = load("2026")
C26 = complete_matches(HD26, rows26)
print(f"(a) 2026 complete regulation n<=24: {len(C26)}")
mr_dist = Counter(); c_mr = Counter(); c_mr_fit = Counter()
for m, n, tA, tB in C26:
    mr = int(HD26[m]["max_rounds"]) if HD26[m].get("max_rounds") is not None else None
    mr_dist[mr] += 1
    sched = {mr // 2} if mr else {12}
    c_mr[fit(wins_A(rows26[m], n, sched), n, tA, tB)] += 1
    st = score_halftime(rows26[m], n)
    c_mr_fit[(mr, st)] += 1
print(f"    max_rounds distribution (complete regulation): {dict(sorted(mr_dist.items(), key=lambda kv: (kv[0] is None, kv[0])))}")
print(f"    header fit with halftime = max_rounds//2: ok={c_mr['ok']} swapped={c_mr['swapped']} bad={c_mr['bad']}")
print("    score-step halftime position by header max_rounds (max_rounds, swap_after -> count):")
for k, v in sorted(c_mr_fit.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
    print(f"      max_rounds={k[0]} swap_after={list(k[1])}: {v}")

# ---------------- (b) 2024 round-13 with no-swap rule at 13->14
HD24, rows24 = load("2024")
C24 = complete_matches(HD24, rows24)
print(f"\n(b) 2024 complete regulation n<=24: {len(C24)}")
grp = {}
for m, n, tA, tB in C24:
    grp[m] = score_halftime(rows24[m], n)
gname = {(12,): "swap@12", (8,): "swap@8", (): "no-swap-found"}
cnt_grp = Counter(gname.get(grp[m], str(grp[m])) for m, *_ in C24)
print("    score-step halftime groups:", dict(cnt_grp))

rows_out = Counter()
header_vs_score = Counter()
flip_comp = Counter()
short24 = 0
for m, n, tA, tB in C24:
    rr = rows24[m]
    if 13 not in rr or "L" not in rr[13]:
        short24 += 1
        continue
    g = gname.get(grp[m], str(grp[m]))
    fe13 = rr.get(13, {}).get("fe"); fe14 = rr.get(14, {}).get("fe")
    w13 = None
    if fe13 is not None and fe14 is not None:
        dt, dc = fe14[0] - fe13[0], fe14[1] - fe13[1]
        if (dt, dc) == (1, 0):
            w13 = T
        elif (dt, dc) == (0, 1):
            w13 = CT
    L13 = rr[13]["L"]
    B13 = rr[13].get("B")
    hasB = bool(rr[13].get("hasB"))
    if w13 is None:
        sc = "no-score"
    else:
        sc = "score=label" if w13 == L13 else "score!=label"
    bstat = "bomb-none" if not hasB else ("bomb=score" if (B13 is not None and w13 is not None and B13 == w13) else ("bomb!=score" if (w13 is not None and B13 is not None) else "bomb-undetermined"))
    rows_out[(g, sc, bstat)] += 1
    # header fit under S12 with the round-13 label as recorded, and with the round-13 label replaced by score-step
    s12 = fit(wins_A(rr, n, {12}), n, tA, tB)
    header_vs_score[(g, s12, sc)] += 1
    # composition of the claimed single-flip set: S12-bad, round-13 flip fixes, no bomb row at 13
    if s12 != "ok" and not hasB:
        wa = wins_A(rr, n, {12})
        side_A13 = CT  # S12: team A on CT in round 13
        isA = (L13 == side_A13)
        new_wa = wa - (1 if isA else 0) + (0 if isA else 1)
        if (new_wa, n - new_wa) == (tA, tB):
            flip_comp[g] += 1

print("\n    round-13 score-step (no-swap rule at 13->14) vs label vs bomb, by halftime group:")
for k, v in sorted(rows_out.items()):
    print(f"      {k[0]:<14} {k[1]:<13} {k[2]:<20} n={v}")

print("\n    header S12 fit vs round-13 score-step status, by halftime group:")
for k, v in sorted(header_vs_score.items()):
    print(f"      {k[0]:<14} S12={k[1]:<8} r13:{k[2]:<13} n={v}")

print("\n    claimed 825-type set (S12 misfit, no bomb row at 13, round-13 flip reconciles), by halftime group:")
for k, v in sorted(flip_comp.items()):
    print(f"      {k:<14} n={v}")
print(f"      total = {sum(flip_comp.values())}")
