"""A7: score feature coverage. Excluded matches, winner-derived lead, and team-attached patterns.

Question
  bakeoff.py feeds round_state ct_score - t_score (live start) as ct_score_minus_t_score. The score
  lead check (scripts/model/sufficiency/score_lead_fix.py) found zero disagreement with the
  winner-derived lead on the 2026 kept set. This script covers every match in both tomes:
    - 2026 tome: 1,752 matches (kept set from scripts/eda/extract_state_1hz.bakeoff_kept(), read only)
    - 2024 tome: 4,435 matches (all outside the bake-off kept set)

Per match, three independent tests
  A. Winner-derived lead vs round_state lead. Side-coded winners (2 = T, 3 = CT) and the swap rule
     give L_w(r) = score(CT side) - score(T side) before round r. Compared with
     L_s(r) = ct_score - t_score at round_freeze_end. Only rounds whose earlier winners are all known
     are compared. DIS = count of disagreeing rounds.
  B. Score-only attachment test (no winner codes used). For each boundary r -> r+1 with both
     freeze rows, the total score rises by exactly one. Side-attached scores cross over at a side
     swap (t' = c + [CT won], c' = t + [T won]) and do not cross otherwise. Team-attached scores never
     cross. A boundary is classified noswap-only, swap-only, both (ambiguous) or neither. Compared
     with the schedule (swap_expected). ATT = scheduled swaps with a noswap-only pattern, plus
     unscheduled swap-only patterns. ATT > 0 is a team-attached-like signature.
  C. Recorded winner vs score-inferred winner on unambiguous, schedule-consistent boundaries.
     WINMM = count of such mismatches.

Assumptions (marked in the report)
  ASSUMPTION MR2026: max_rounds from the 2026 header (a real column).
  ASSUMPTION MR2024: the 2024 header has no max_rounds. For each 2024 match the candidate 24 (MR12)
    and 30 (MR15) are scored by ATT-type inconsistencies; the smaller count is used. Reported.
  ASSUMPTION Round 1 freeze is 0-0 is checked, not assumed.

Run from the project root:
  "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\A7_score_coverage.py
"""

import collections
import csv
import glob
import importlib.util
import io
import re
import sys
import time
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)

ROOT = Path(r"G:\pureskill\csds-cs2")
T, CT = 2, 3
TOMES = {
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
}
DETAIL_OUT = ROOT / "logs" / "swarm2" / "A7_match_detail.log"
# Candidate max_rounds for 2024 (no header column). 2026 headers show 16 and 24, so 16 must be a candidate.
MR24_CANDIDATES = tuple(range(8, 33))
ASSUMED_MR_2024_DEFAULT = 24
RULE_2024 = {}
KEY_RE = re.compile(r"/(\d{4})/(\d{2})/(\d{2})/")

T0 = time.time()


def load_bakeoff():
    spec = importlib.util.spec_from_file_location(
        "extract_state_1hz", ROOT / "scripts" / "eda" / "extract_state_1hz.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scan(base, suf, ch):
    return pl.scan_parquet(str(base / f"basic_{ch}{suf}" / "dataframe_*"))


def swap_expected(r, mr):
    """Same rule as bakeoff.swap_expected (halftime round, then after the 3rd round of each 6-round OT)."""
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


def boundary_class(t, c, t2, c2):
    """Score-only classification of the boundary r -> r+1 from freeze-end scores."""
    dT, dC = t2 - t, c2 - c
    xT, xC = t2 - c, c2 - t
    nos = dT in (0, 1) and dC in (0, 1) and dT + dC == 1
    swp = xT in (0, 1) and xC in (0, 1) and xT + xC == 1
    if nos and swp:
        return "ambig"
    if nos:
        return "noswap"
    if swp:
        return "swap"
    return "neither"


def load_tome(label, base, suf):
    hdr_cols = ["key", "match_date", "t_starters_score_final", "ct_starters_score_final"]
    hdr_all = pl.scan_parquet(str(base / "header" / "dataframe_*")).collect_schema().names()
    if "max_rounds" in hdr_all:
        hdr_cols.append("max_rounds")
    H_df = pl.scan_parquet(str(base / "header" / "dataframe_*")).select(hdr_cols).collect()
    H = {r["key"]: r for r in H_df.iter_rows(named=True)}

    re_df = scan(base, suf, "round_end").select(
        ["match_key", "round", "tick", "winner_team_code"]).collect()
    fe_df = scan(base, suf, "round_state").filter(pl.col("event_type") == "round_freeze_end").select(
        ["match_key", "round", "tick", "t_score", "ct_score"]).collect()

    dup_re = re_df.height - re_df.select(["match_key", "round"]).unique().height
    dup_fe = fe_df.height - fe_df.select(["match_key", "round"]).unique().height
    dup_re_rows = (re_df.filter(pl.struct(["match_key", "round"]).is_duplicated()).sort(
        ["match_key", "round", "tick"]))
    dup_re_diff = 0
    if dup_re_rows.height:
        g = dup_re_rows.group_by(["match_key", "round"]).agg(pl.col("winner_team_code").n_unique().alias("nw"))
        dup_re_diff = int((g["nw"] > 1).sum())
    dup_fe_rows = fe_df.filter(pl.struct(["match_key", "round"]).is_duplicated())
    dup_fe_diff = 0
    if dup_fe_rows.height:
        g = dup_fe_rows.group_by(["match_key", "round"]).agg(
            pl.col("t_score").n_unique().alias("nt"), pl.col("ct_score").n_unique().alias("nc"))
        dup_fe_diff = int(((g["nt"] > 1) | (g["nc"] > 1)).sum())

    W, F = collections.defaultdict(dict), collections.defaultdict(dict)
    for x in re_df.sort(["match_key", "round", "tick"]).iter_rows(named=True):
        W[x["match_key"]].setdefault(int(x["round"]), x["winner_team_code"])
    for x in fe_df.sort(["match_key", "round", "tick"]).iter_rows(named=True):
        F[x["match_key"]].setdefault(int(x["round"]), (x["t_score"], x["ct_score"]))

    info = {"header_matches": len(H), "round_end_rows": re_df.height, "round_end_dup_keys": dup_re,
            "round_end_dup_keys_conflicting_winner": dup_re_diff,
            "freeze_rows": fe_df.height, "freeze_dup_keys": dup_fe,
            "freeze_dup_keys_conflicting_score": dup_fe_diff,
            "matches_with_round_end": len(W), "matches_with_freeze": len(F),
            "header_has_max_rounds": "max_rounds" in hdr_cols}
    return H, W, F, info


def analyze_match(m, mr, W_m, F_m):
    """Returns the per-match flags and the per-round detail rows."""
    out = collections.Counter()
    details = []
    rounds_all = sorted(set(W_m) | set(F_m))
    max_r = rounds_all[-1] if rounds_all else 0
    out["rounds_end"] = len(W_m)
    out["rounds_freeze"] = len(F_m)
    out["gap_rounds_no_winner"] = sum(1 for r in range(1, max_r + 1) if r not in W_m)
    if 1 in F_m and F_m[1] != (0, 0):
        out["round1_freeze_not_zero"] = 1

    # --- A. winner-derived lead vs round_state lead
    # The chain starts at 0-0 before round 1. If a winner is missing, the chain stops; it restarts
    # (re-anchors) at the next freeze row, whose side scores are absolute (side-attached). A
    # re-anchor round is not compared (the comparison would be trivial).
    cur = {T: 0, CT: 0}
    valid = True
    dis_rounds, first_dis = [], None
    compared = 0
    anchors = 0
    for r in range(1, max_r + 1):
        w = W_m.get(r)
        f = F_m.get(r)
        anchored_here = False
        if not valid and f is not None:
            cur = {T: f[0], CT: f[1]}
            valid = True
            anchored_here = True
            anchors += 1
        if valid and f is not None and not anchored_here:
            Lw = cur[CT] - cur[T]
            Ls = f[1] - f[0]
            compared += 1
            if Lw != Ls:
                dis_rounds.append(r)
                if first_dis is None:
                    first_dis = r
                details.append(("DIS", r, Lw, Ls, w))
        if w is None:
            valid = False
        elif w in (T, CT):
            cur[w] += 1
        else:
            valid = False
            out["winner_code_not_2_or_3"] += 1
        if valid and swap_expected(r, mr):
            cur[T], cur[CT] = cur[CT], cur[T]
    out["compared_rounds"] = compared
    out["anchors"] = anchors
    out["dis_rounds"] = len(dis_rounds)
    out["first_dis_round"] = first_dis if first_dis is not None else 0

    # --- B and C. score-only boundaries and winner comparison
    rs_sorted = sorted(F_m)
    winmm = []
    for r in rs_sorted:
        if r + 1 not in F_m:
            continue
        t, c = F_m[r]
        t2, c2 = F_m[r + 1]
        cls = boundary_class(t, c, t2, c2)
        sched = swap_expected(r, mr)
        out["boundaries"] += 1
        if cls == "ambig":
            out["bd_ambig"] += 1
            continue
        if cls == "neither":
            out["bd_neither"] += 1
            details.append(("NEITHER", r, t, c, t2, c2))
            continue
        expect = "swap" if sched else "noswap"
        if cls == expect:
            out["bd_ok"] += 1
            if cls == "noswap":
                w_s = T if (t2 - t) == 1 else CT
            else:
                w_s = CT if (t2 - c) == 1 else T
            w_rec = W_m.get(r)
            if w_rec is not None and w_rec in (T, CT) and w_rec != w_s:
                winmm.append(r)
                details.append(("WINMM", r, w_rec, w_s, t, c, t2, c2))
        else:
            if sched:
                out["att_sched_noswap"] += 1
                details.append(("ATT_SCHED", r, t, c, t2, c2))
            else:
                out["att_unsched_swap"] += 1
                details.append(("ATT_UNSCHED", r, t, c, t2, c2))
    out["winmm_rounds"] = len(winmm)
    out["first_winmm_round"] = winmm[0] if winmm else 0
    out["att"] = out["att_sched_noswap"] + out["att_unsched_swap"]
    out["winmm_list"] = ",".join(str(x) for x in winmm)
    out["dis_list"] = ",".join(str(x) for x in dis_rounds)
    return out, details


def mr_fit_2024(F_m, candidates):
    """Number of unambiguous boundaries inconsistent with the schedule, per candidate max_rounds."""
    res = {}
    for mr in candidates:
        bad = 0
        for r in F_m:
            if r + 1 not in F_m:
                continue
            t, c = F_m[r]
            t2, c2 = F_m[r + 1]
            cls = boundary_class(t, c, t2, c2)
            if cls in ("ambig", "neither"):
                continue
            expect = "swap" if swap_expected(r, mr) else "noswap"
            bad += int(cls != expect)
        res[mr] = bad
    return res


def day_of(m):
    g = KEY_RE.search(m)
    return g.group(3) if g else "?"


def mr_fit_full(F_m, candidates):
    """Per candidate: (bad, ok). bad = unambiguous boundaries inconsistent with the schedule;
    ok = unambiguous boundaries consistent with it."""
    res = {}
    for mr in candidates:
        bad = ok = 0
        for r in F_m:
            if r + 1 not in F_m:
                continue
            t, c = F_m[r]
            t2, c2 = F_m[r + 1]
            cls = boundary_class(t, c, t2, c2)
            if cls in ("ambig", "neither"):
                continue
            expect = "swap" if swap_expected(r, mr) else "noswap"
            if cls == expect:
                ok += 1
            else:
                bad += 1
        res[mr] = (bad, ok)
    return res


PREFERRED_MR = (24, 16)


def mr_best(F_m, candidates, prefer=None):
    """Score-only fit. Lowest bad count; among those, the most ok boundaries (longest consistent
    record). Remaining ties go to `prefer` (the halftime-announcement rule value, when given),
    then the standard values 24 then 16, else the smallest.
    Returns (best mr, its bad count, number of candidates tied on (bad, ok), {mr: bad})."""
    full = mr_fit_full(F_m, candidates)
    best_bad = min(b for b, _ in full.values())
    pool = [k for k in candidates if full[k][0] == best_bad]
    best_ok = max(full[k][1] for k in pool)
    pool2 = [k for k in pool if full[k][1] == best_ok]
    if prefer is not None and prefer in pool2:
        best = prefer
    else:
        best = next((p for p in PREFERRED_MR if p in pool2), min(pool2))
    return best, best_bad, len(pool2), {k: full[k][0] for k in candidates}


def half_rule(base, suf):
    """cross_period's 2024 rule: 2 x (first round with round_announce_last_round_half) + 2; else 24.
    Returns {match_key: rule mr} for every header match."""
    hdr = pl.scan_parquet(str(base / "header" / "dataframe_*")).select(["key"]).collect()["key"].to_list()
    ev = (pl.scan_parquet(str(base / f"basic_round_state{suf}" / "dataframe_*"))
          .filter(pl.col("event_type") == "round_announce_last_round_half")
          .group_by("match_key").agg(pl.col("round").min().alias("r")).collect())
    half = {m: int(r) for m, r in ev.iter_rows()}
    return {m: (half[m] * 2 + 2 if m in half else 24) for m in hdr}


def main():
    print("== A7 score feature coverage ==", flush=True)
    bk_mod = load_bakeoff()
    print("\n== kept set (bakeoff port, read only) ==", flush=True)
    kept = bk_mod.bakeoff_kept()
    kept_final = set(kept["kept_final"])
    rd_matches = {m for (m, _r) in kept["rd"].keys()}
    kept_round_keys = set(kept["rounds_ok"])
    kept_rounds_in_kept_matches = set(kept["rd"].keys())
    print(f"kept matches {len(kept_final)}; kept rounds {len(kept_round_keys)}; "
          f"channel-kept matches with rounds {len(rd_matches)}", flush=True)

    global RULE_2024
    RULE_2024 = half_rule(*TOMES["2024"])
    all_rows = []  # per-match rows for detail output
    summary = {}
    for label, (base, suf) in TOMES.items():
        print(f"\n== tome {label} ==", flush=True)
        H, W, F, info = load_tome(label, base, suf)
        print(f"  headers {info['header_matches']}; round_end rows {info['round_end_rows']} "
              f"(dup keys {info['round_end_dup_keys']}, conflicting {info['round_end_dup_keys_conflicting_winner']}); "
              f"freeze rows {info['freeze_rows']} (dup keys {info['freeze_dup_keys']}, "
              f"conflicting {info['freeze_dup_keys_conflicting_score']}); "
              f"header max_rounds column: {info['header_has_max_rounds']}", flush=True)
        n_zero_mr = 0
        rows = []
        for m in sorted(H):
            Fm = F.get(m, {})
            best_mr, best_bad, ties, fit = mr_best(Fm, MR24_CANDIDATES, prefer=RULE_2024.get(m, 24))
            if label == "2026":
                hdr_mr = H[m].get("max_rounds")
                if hdr_mr is not None:
                    mr = int(hdr_mr)
                    mr_src = "header"
                    mr_fit_ok = int(fit.get(mr, 10 ** 9) == best_bad)
                else:
                    mr = ASSUMED_MR_2024_DEFAULT
                    mr_src = "assumed_24"
                    mr_fit_ok = -1
            else:
                mr = best_mr
                mr_src = "score_fit"
                mr_fit_ok = -1
                rule_mr = RULE_2024.get(m, 24)
                mr = mr_best(Fm, MR24_CANDIDATES, prefer=rule_mr)[0]
                mr_src = "score_fit_tiebreak_rule"
            res, details = analyze_match(m, mr, W.get(m, {}), Fm)
            kind = "kept" if (label == "2026" and m in kept_final) else (
                "2026_excluded" if label == "2026" else "2024")
            in_rd = (label == "2026" and m in rd_matches)
            rows.append({"tome": label, "match": m, "day": day_of(m), "group": kind,
                         "in_channel_kept_rd": int(in_rd), "mr": mr, "mr_src": mr_src,
                         "mr_fit_best_bad": best_bad, "mr_fit_ties": ties, "mr_fit_ok": mr_fit_ok,
                         **{k: res.get(k, 0) for k in (
                             "rounds_end", "rounds_freeze", "compared_rounds", "anchors", "dis_rounds",
                             "first_dis_round", "bd_ok", "bd_ambig", "bd_neither",
                             "att_sched_noswap", "att_unsched_swap", "att", "winmm_rounds",
                             "first_winmm_round", "gap_rounds_no_winner", "round1_freeze_not_zero",
                             "winner_code_not_2_or_3")},
                         "dis_list": res.get("dis_list", ""), "winmm_list": res.get("winmm_list", ""),
                         "details": details})
        summary[label] = {"info": info, "rows": rows}
    print(f"\nrun time so far {time.time() - T0:.1f}s", flush=True)

    # ---------------- report tables
    def is_flag(r, key):
        return r[key] > 0

    def grp_stats(rows, name):
        n = len(rows)
        if n == 0:
            print(f"  {name}: no matches")
            return
        dis = [r for r in rows if r["dis_rounds"] > 0]
        att = [r for r in rows if r["att"] > 0]
        wmm = [r for r in rows if r["winmm_rounds"] > 0]
        bad = [r for r in rows if r["bd_neither"] > 0]
        dis_only = [r for r in dis if r["att"] == 0]
        att_only = [r for r in att if r["dis_rounds"] == 0]
        both = [r for r in dis if r["att"] > 0]
        print(f"  {name}: matches {n}; with DIS>0 {len(dis)}; with ATT>0 {len(att)}; "
              f"with WINMM>0 {len(wmm)}; with NEITHER>0 {len(bad)}")
        print(f"    DIS>0 and ATT=0 {len(dis_only)}; ATT>0 and DIS=0 {len(att_only)}; DIS>0 and ATT>0 {len(both)}")
        print(f"    rounds compared {sum(r['compared_rounds'] for r in rows)}; DIS rounds "
              f"{sum(r['dis_rounds'] for r in rows)}; boundaries ok {sum(r['bd_ok'] for r in rows)}, "
              f"ATT {sum(r['att'] for r in rows)} (sched {sum(r['att_sched_noswap'] for r in rows)}, "
              f"unsched {sum(r['att_unsched_swap'] for r in rows)}), WINMM rounds {sum(r['winmm_rounds'] for r in rows)}")

    r26 = summary["2026"]["rows"]
    r24 = summary["2024"]["rows"]
    print("\n== 1. Inside vs outside the bake-off kept set (2026 tome) ==")
    grp_stats([r for r in r26 if r["group"] == "kept"], "2026 kept set (inside)")
    grp_stats([r for r in r26 if r["group"] == "2026_excluded"], "2026 excluded (outside)")
    grp_stats(r26, "2026 all matches")
    for lab in ("2026", "2024"):
        rr = summary[lab]["rows"]
        fr = sum(r["rounds_freeze"] for r in rr)
        cmp_ = sum(r["compared_rounds"] for r in rr)
        n_anch = sum(r["anchors"] for r in rr)
        n_m_anch = sum(1 for r in rr if r["anchors"] > 0)
        print(f"  coverage {lab}: freeze rows {fr}; compared by the winner-derived chain {cmp_} "
              f"({100 * cmp_ / max(fr, 1):.1f}%); re-anchors {n_anch} in {n_m_anch} matches")
    print("  2026 excluded matches that are inside channel-kept rounds but not kept: "
          f"{sum(1 for r in r26 if r['group'] == '2026_excluded' and r['in_channel_kept_rd'])}; "
          "not in channel-kept rounds: "
          f"{sum(1 for r in r26 if r['group'] == '2026_excluded' and not r['in_channel_kept_rd'])}")
    kept_rounds_check = 0
    print(f"  kept-round check: kept-set rounds in DIS list (from kept matches): "
          f"{sum(1 for r in r26 if r['group'] == 'kept' for x in r['dis_list'].split(',') if x)}")

    print("\n== 2. 2024 tome (all outside the bake-off kept set) ==")
    grp_stats(r24, "2024 all matches")
    print(f"  2024 max_rounds chosen by score-only fit (candidates {MR24_CANDIDATES[0]}..{MR24_CANDIDATES[-1]}): "
          f"{dict(sorted(collections.Counter(r['mr'] for r in r24).items()))}")
    print(f"  2024 fit best bad count distribution: "
          f"{dict(sorted(collections.Counter(r['mr_fit_best_bad'] for r in r24).items()))}; "
          f"matches with a unique best candidate: {sum(1 for r in r24 if r['mr_fit_ties'] == 1)}; "
          f"ties: {sum(1 for r in r24 if r['mr_fit_ties'] > 1)}")
    print(f"  2024 matches with best bad count 0: {sum(1 for r in r24 if r['mr_fit_best_bad'] == 0)}; "
          f"with best bad count > 0: {sum(1 for r in r24 if r['mr_fit_best_bad'] > 0)}")

    print("\n== 3. 2026 team-attached claim by day (HANDOFF: 15-20% on days 01-02) ==")
    for d in sorted({r["day"] for r in r26}):
        rs = [r for r in r26 if r["day"] == d]
        n_att = sum(1 for r in rs if r["att"] > 0)
        n_dis = sum(1 for r in rs if r["dis_rounds"] > 0)
        print(f"  day 2026-10-{d}: matches {len(rs)}; ATT>0 {n_att} ({100 * n_att / len(rs):.1f}%); "
              f"DIS>0 {n_dis} ({100 * n_dis / len(rs):.1f}%); kept {sum(1 for r in rs if r['group'] == 'kept')}")
    print(f"  2026 header max_rounds agrees with the score-only fit (header mr has the best bad count): "
          f"{sum(1 for r in r26 if r['mr_fit_ok'] == 1)} of {sum(1 for r in r26 if r['mr_fit_ok'] >= 0)}; "
          f"header max_rounds values: {dict(sorted(collections.Counter(r['mr'] for r in r26).items()))}")
    d1_7 = [r for r in r26 if r["day"] in ("01", "02", "04", "07")]
    print(f"  days 01,02,04,07 combined: matches {len(d1_7)}; ATT>0 {sum(1 for r in d1_7 if r['att'] > 0)}; "
          f"DIS>0 {sum(1 for r in d1_7 if r['dis_rounds'] > 0)}")

    print("\n== 4. 2024 by day ==")
    for d in sorted({r["day"] for r in r24}):
        rs = [r for r in r24 if r["day"] == d]
        n_dis = sum(1 for r in rs if r["dis_rounds"] > 0)
        n_att = sum(1 for r in rs if r["att"] > 0)
        n_wmm = sum(1 for r in rs if r["winmm_rounds"] > 0)
        print(f"  2024-02-{d}: matches {len(rs)}; DIS>0 {n_dis} ({100 * n_dis / len(rs):.1f}%); "
              f"WINMM>0 {n_wmm}; ATT>0 {n_att}")

    print("\n== 5. 2024 first disagreement round and winner mismatch rounds ==")
    first_dis = collections.Counter(r["first_dis_round"] for r in r24 if r["dis_rounds"] > 0)
    print(f"  first DIS round (matches): {sorted(first_dis.items())[:30]}")
    wm_rounds = collections.Counter()
    for r in r24:
        for x in r["winmm_list"].split(","):
            if x:
                wm_rounds[int(x)] += 1
    print(f"  WINMM by round number (top 30): {sorted(wm_rounds.items())[:30]}")
    n13 = sum(1 for r in r24 if 13 in [int(x) for x in r["winmm_list"].split(",") if x])
    print(f"  matches with a WINMM at round 13: {n13}")
    dis_r = collections.Counter()
    for r in r24:
        for x in r["dis_list"].split(","):
            if x:
                dis_r[int(x)] += 1
    print(f"  DIS by round number (top 25): {sorted(dis_r.items())[:25]}")
    print(f"  total DIS rounds 2024: {sum(dis_r.values())}; matches with DIS: "
          f"{sum(1 for r in r24 if r['dis_rounds'] > 0)}")

    print("\n== 6. Data checks ==")
    for label in ("2026", "2024"):
        i = summary[label]["info"]
        rr = summary[label]["rows"]
        print(f"  {label}: round1 freeze not 0-0: {sum(r['round1_freeze_not_zero'] for r in rr)}; "
              f"winner codes outside {{2,3}} in matches: {sum(1 for r in rr if r['winner_code_not_2_or_3'] > 0)}; "
              f"matches with gap in winner sequence: {sum(1 for r in rr if r['gap_rounds_no_winner'] > 0)}; "
              f"matches with NEITHER boundary: {sum(1 for r in rr if r['bd_neither'] > 0)}")
    print(f"  2024 matches with ATT>0 after the fit: {sum(1 for r in r24 if r['att'] > 0)}")

    # detail output
    DETAIL_OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["tome", "match", "day", "group", "in_channel_kept_rd", "mr", "mr_src", "mr_fit_best_bad",
            "mr_fit_ties", "mr_fit_ok", "rounds_end",
            "rounds_freeze", "compared_rounds", "anchors", "dis_rounds", "first_dis_round", "bd_ok", "bd_ambig",
            "bd_neither", "att_sched_noswap", "att_unsched_swap", "att", "winmm_rounds",
            "first_winmm_round", "gap_rounds_no_winner", "round1_freeze_not_zero",
            "winner_code_not_2_or_3", "dis_list", "winmm_list"]
    with open(DETAIL_OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for label in ("2026", "2024"):
            for r in summary[label]["rows"]:
                w.writerow([r[c] for c in cols])
    print(f"\nwrote {DETAIL_OUT}; total run {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
