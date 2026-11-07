"""D6: detect the actual halftime side-swap round per match from player team_code changes.

Hypothesis D6: affected 2024 matches (round-13 bomb-side mismatch) do not swap sides after round 12.
Method
  1. Source: basic_player_info (round, player_id, team_code), one row per player per round.
  2. For each match, reference round = first round with >= 8 players. Orientation of round r =
     share of players (present in both r and the reference round) whose team_code differs from the
     reference round. orientation 1 = sides flipped relative to the reference round.
     A swap after completed round s = orientation change between s and s+1.
  3. Calibration: the same detector on the 2026 tome, where header max_rounds exists. The detected
     swap rounds are compared with swap_expected(r, max_rounds) from scripts/model/bakeoff.py.
  4. 2024: detected swap rounds per match, cross-tabbed with the round-13 bomb-side mismatch
     (same definition as scripts/swarm/e2_side_check_r13.py: round 13, player_count == 20,
     bomb outcome present, bomb side (3 if defused else 2) != winner_team_code).
Run: "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/D6_swap_detect.py
"""

import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASES = {
    "2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
    "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds",
}
T, CT = 2, 3
OUTCOMES = ["bomb_defused", "bomb_exploded"]


def files(base, ch):
    if ch == "header":
        return sorted(glob.glob(str(base / "header" / "dataframe_*")))
    fd = sorted(base.glob(f"basic_{ch}.*"))[0]
    return sorted(glob.glob(str(fd / "dataframe_*")))


def read(base, ch, cols):
    fs = files(base, ch)
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def swap_expected(r, max_rounds):
    """Copied from scripts/model/bakeoff.py (swap_expected): swap after completed round r."""
    return r == max_rounds // 2 or (r > max_rounds and (r - max_rounds - 3) % 6 == 0)


def detect_swaps(rounds_map):
    """rounds_map: {round: {player_id: team_code}}. Returns swap boundaries and QC fields."""
    rounds = sorted(rounds_map)
    ref = next((r for r in rounds if len(rounds_map[r]) >= 8), rounds[0])
    t0 = rounds_map[ref]
    seq = []
    for r in rounds:
        cur = rounds_map[r]
        common = [p for p in cur if p in t0]
        if len(common) < 4:
            continue
        frac = sum(1 for p in common if cur[p] != t0[p]) / len(common)
        seq.append((r, 1 if frac > 0.5 else 0, max(frac, 1.0 - frac), len(common)))
    bounds, gap_bounds = [], []
    for a, b in zip(seq, seq[1:]):
        if a[1] != b[1]:
            (bounds if b[0] == a[0] + 1 else gap_bounds).append(a[0])
    return {
        "ref": ref,
        "seq": seq,
        "bounds": bounds,
        "gap_bounds": gap_bounds,
        "min_purity": min((s[2] for s in seq), default=1.0),
        "n_low_purity": sum(1 for s in seq if s[2] < 0.8),
        "rounds": rounds,
    }


def load_player_info(base):
    pi = read(base, "player_info", ["match_key", "round", "player_id", "team_code"])
    pi = pi.unique(["match_key", "round", "player_id"], keep="first")
    grouped = collections.defaultdict(lambda: collections.defaultdict(dict))
    for m, r, p, t in pi.iter_rows():
        grouped[m][r][p] = t
    return pi, grouped


def header_frame(base, era):
    cols = ["key", "match_date", "t_starters_score_final", "ct_starters_score_final"]
    if era == "2026":
        cols.append("max_rounds")
    return read(base, "header", cols)


# ---------------------------------------------------------------- 1. calibration on 2026
print("== 1. Calibration on 2026 tome (header max_rounds known) ==")
base26 = BASES["2026"]
pi26, g26 = load_player_info(base26)
H26 = {r["key"]: r for r in header_frame(base26, "2026").iter_rows(named=True)}
print(f"2026 player_info rows: {pi26.height}; matches with player_info: {len(g26)}; header rows: {len(H26)}")
print(f"2026 max_rounds distribution: {dict(collections.Counter(r['max_rounds'] for r in H26.values()))}")

cal = collections.Counter()
cal_bounds = collections.Counter()
cal_bad = []
cal_by_mr = collections.defaultdict(collections.Counter)
for m, rm in g26.items():
    h = H26.get(m)
    if h is None:
        cal["no_header"] += 1
        continue
    mr = int(h["max_rounds"])
    d = detect_swaps(rm)
    rounds = set(d["rounds"])
    pred = [r for r in d["rounds"] if (r + 1) in rounds and swap_expected(r, mr)]
    ok = pred == d["bounds"] and d["n_low_purity"] == 0 and not d["gap_bounds"]
    cal["consistent" if ok else "inconsistent"] += 1
    cal_by_mr[mr]["consistent" if ok else "inconsistent"] += 1
    cal_bounds[tuple(d["bounds"])] += 1
    if not ok and len(cal_bad) < 12:
        cal_bad.append((m[:44], mr, d["ref"], d["bounds"], pred, d["gap_bounds"], d["n_low_purity"], round(d["min_purity"], 3)))
print(f"2026 matches checked: {sum(cal.values()) - cal['no_header']}; outcome: {dict(cal)}")
print(f"2026 consistency by max_rounds: { {k: dict(v) for k, v in sorted(cal_by_mr.items())} }")
print(f"2026 detected swap-boundary patterns (top 10): {cal_bounds.most_common(10)}")
print("2026 inconsistent examples (key, max_rounds, ref, detected, predicted, gap, low_purity_rounds, min_purity):")
for row in cal_bad:
    print("  ", row)

# ---------------------------------------------------------------- 2. detection on 2024
print("\n== 2. 2024 tome: detected swap boundaries ==")
base24 = BASES["2024"]
pi24, g24 = load_player_info(base24)
H24 = {r["key"]: r for r in header_frame(base24, "2024").iter_rows(named=True)}
print(f"2024 player_info rows: {pi24.height}; matches with player_info: {len(g24)}; header rows: {len(H24)}")

det = {}
for m, rm in g24.items():
    det[m] = detect_swaps(rm)
first_b = collections.Counter(d["bounds"][0] if d["bounds"] else None for d in det.values())
print(f"first detected swap round (completed round) distribution: {dict(sorted(first_b.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)))}")
pat = collections.Counter(tuple(d["bounds"]) for d in det.values())
print(f"boundary patterns (top 15): {pat.most_common(15)}")
print(f"matches with gap-straddling boundary: {sum(1 for d in det.values() if d['gap_bounds'])}; "
      f"matches with any low-purity round: {sum(1 for d in det.values() if d['n_low_purity'] > 0)}")
print(f"ref round distribution: {dict(sorted(collections.Counter(d['ref'] for d in det.values()).items()))}")

# ---------------------------------------------------------------- 3. mismatch flag (e2 definition)
print("\n== 3. Round-13 bomb-side mismatch (e2 definition, player_count == 20) ==")
RE = read(base24, "round_end", ["match_key", "round", "winner_team_code", "player_count"])
RE = RE.unique(["match_key", "round"], keep="first")
BS = read(base24, "bomb_state", ["match_key", "round", "event_type", "player_id", "player_team_code"])
outcome = (BS.filter(pl.col("event_type").is_in(OUTCOMES))
           .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
planter = (BS.filter(pl.col("event_type") == "bomb_planted")
           .group_by(["match_key", "round"]).agg(pl.col("player_team_code").first().alias("planter_team")))
J = (RE.join(outcome, on=["match_key", "round"], how="inner")
     .join(planter, on=["match_key", "round"], how="left")
     .with_columns(
         pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bomb_side"))
     .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch")))
J20 = J.filter(pl.col("player_count") == 20)
r13 = J20.filter(pl.col("round") == 13)
print(f"round-13 rows with bomb outcome (player_count==20): {r13.height}; mismatches: {int(r13['mismatch'].sum())}")
r13_mm = set(r13.filter(pl.col("mismatch"))["match_key"].to_list())
r13_ok = set(r13.filter(~pl.col("mismatch"))["match_key"].to_list())
other_mm = J20.filter((pl.col("round") != 13) & pl.col("mismatch"))
print(f"other-round mismatches (player_count==20): {other_mm.height}")
other_mm_matches = set(other_mm["match_key"].to_list())

# ---------------------------------------------------------------- 4. cross-tab
print("\n== 4. Cross-tab: first detected swap vs round-13 mismatch ==")
ct = collections.Counter()
for m, d in det.items():
    grp = "r13_mismatch" if m in r13_mm else ("r13_match" if m in r13_ok else "no_r13_bomb")
    fb = d["bounds"][0] if d["bounds"] else None
    ct[(grp, fb)] += 1
for k in sorted(ct, key=lambda x: (x[0], -1 if x[1] is None else x[1])):
    print(f"  {k[0]:<13} first_swap_after={str(k[1]):<5} matches={ct[k]}")

aff_fb = collections.Counter(det[m]["bounds"][0] if det[m]["bounds"] else None for m in r13_mm if m in det)
print(f"affected (r13 mismatch) matches with player_info: {sum(aff_fb.values())}; first swap distribution: "
      f"{dict(sorted(aff_fb.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)))}")

# affected matches: pattern list
aff_pat = collections.Counter(tuple(det[m]["bounds"]) for m in r13_mm if m in det)
print(f"affected boundary patterns (top 12): {aff_pat.most_common(12)}")

# ---------------------------------------------------------------- 5. planter team consistency
print("\n== 5. Planter team on bomb_state vs player_info team in round 13 (affected and not) ==")
pinfo_team = {}
for m, r, p, t in pi24.filter(pl.col("round") == 13).select(["match_key", "round", "player_id", "team_code"]).iter_rows():
    pinfo_team[(m, p)] = t
pl13 = BS.filter((pl.col("round") == 13) & (pl.col("event_type") == "bomb_planted")).select(
    ["match_key", "player_team_code", "player_id"]).iter_rows()
agree = collections.Counter()
for m, pt, pid in pl13:
    grp = "r13_mismatch" if m in r13_mm else "r13_match_or_other"
    pi_t = pinfo_team.get((m, pid))
    agree[(grp, "no_pinfo" if pi_t is None else ("agree" if pi_t == pt else "disagree"))] += 1
print(f"  planter team check (group, result): {dict(agree)}")

# ---------------------------------------------------------------- 6. dump per match for follow-up
out = ROOT / "logs" / "swarm2" / "D6_per_match.log"
out.parent.mkdir(parents=True, exist_ok=True)
hdr = ["match_key", "match_date", "ref", "bounds", "gap_bounds", "n_low_purity", "min_purity",
       "r13_group", "rounds_pinfo", "final_t", "final_ct"]
with open(out, "w", encoding="utf-8") as f:
    f.write("\t".join(hdr) + "\n")
    for m in sorted(det):
        d = det[m]
        h = H24.get(m, {})
        grp = "r13_mismatch" if m in r13_mm else ("r13_match" if m in r13_ok else "no_r13_bomb")
        f.write("\t".join(str(x) for x in [
            m, h.get("match_date", ""), d["ref"], ",".join(map(str, d["bounds"])),
            ",".join(map(str, d["gap_bounds"])), d["n_low_purity"], f"{d['min_purity']:.3f}", grp,
            len(d["rounds"]), h.get("t_starters_score_final", ""), h.get("ct_starters_score_final", "")]) + "\n")
print(f"\nper-match table written to {out}")

# ---------------------------------------------------------------- 7. second source: player_spawn team codes
print("\n== 7. Independent second source: player_spawn team codes (2024) vs player_info ==")
sp = read(base24, "player_spawn", ["match_key", "round", "player_id", "player_team_code"])
sp = sp.unique(["match_key", "round", "player_id"], keep="first")
sp_map = collections.defaultdict(lambda: collections.defaultdict(dict))
for m, r, p, t in sp.iter_rows():
    if t is None:
        continue
    sp_map[m][r][p] = t
sp_cmp = collections.Counter()
sp_diff = []
for m, rm in sp_map.items():
    if m not in det:
        sp_cmp["not_in_player_info"] += 1
        continue
    d2 = detect_swaps(rm)
    same = d2["bounds"] == det[m]["bounds"]
    sp_cmp["same_bounds" if same else "diff_bounds"] += 1
    if not same and len(sp_diff) < 10:
        sp_diff.append((m[:44], det[m]["bounds"], d2["bounds"], det[m]["ref"], d2["ref"]))
print(f"spawn-based vs player_info-based swap boundaries: {dict(sp_cmp)}; matches with spawn data: {len(sp_map)}")
for row in sp_diff:
    print("   diff:", row)

# ---------------------------------------------------------------- 8. mismatch position relative to swap
print("\n== 8. Round-13 mismatch position relative to the detected swap (all rounds with outcome, player_count==20) ==")
tot, mmc = collections.Counter(), collections.Counter()
for m, r, mm in J20.select(["match_key", "round", "mismatch"]).iter_rows():
    d = det.get(m)
    if d is None or not d["bounds"]:
        k = ("no_swap", None)
    else:
        s1 = d["bounds"][0]
        rel = r - (s1 + 1)
        k = (f"s1={s1}", max(-3, min(3, rel)))
    tot[k] += 1
    mmc[k] += int(mm)
for k in sorted(tot, key=lambda x: (x[0], -99 if x[1] is None else x[1])):
    print(f"  {k[0]:<8} rel_to_first_post_swap_round={str(k[1]):<5} n={tot[k]:>6} mismatch={mmc[k]:>4} "
          f"rate={mmc[k] / tot[k]:.3f}")

# ---------------------------------------------------------------- 9. header totals vs detected format
print("\n== 9. 2024 header final scores vs detected format ==")
agg = collections.defaultdict(collections.Counter)
for m, d in det.items():
    h = H24.get(m)
    if h is None:
        continue
    s1 = d["bounds"][0] if d["bounds"] else None
    ft, fc = int(h["t_starters_score_final"]), int(h["ct_starters_score_final"])
    n = len(d["rounds"])
    g = str(s1)
    agg[g]["matches"] += 1
    agg[g]["n_rounds_eq_header_sum"] += int(n == ft + fc)
    agg[g]["max_final"] += 0
    if s1 == 8:
        agg[g]["max_final_ge_9"] += int(max(ft, fc) >= 9)
    elif s1 == 12:
        agg[g]["max_final_ge_13"] += int(max(ft, fc) >= 13)
    agg[g]["n_rounds_le_16"] += int(n <= 16)
    agg[g]["n_rounds_gt_24"] += int(n > 24)
for g in sorted(agg, key=lambda x: (x == "None", x)):
    print(f"  first_swap_after={g:<5} {dict(agg[g])}")
none_n = collections.Counter(len(d["rounds"]) for m, d in det.items() if not d["bounds"])
print(f"  matches with no detected swap, distinct rounds distribution: {dict(sorted(none_n.items()))}")

# ---------------------------------------------------------------- 10. 2026 misses
print("\n== 10. 2026 calibration misses (predicted swap at 8, none detected) ==")
shown = 0
for m, rm in g26.items():
    h = H26.get(m)
    if h is None or int(h["max_rounds"]) != 16:
        continue
    d = detect_swaps(rm)
    if d["bounds"]:
        continue
    if shown < 2:
        print(f"  {m[:60]} rounds={d['rounds']} ref={d['ref']}")
        print(f"    seq (round, orient, purity, n_common): {[(s[0], s[1], round(s[2], 2), s[3]) for s in d['seq']]}")
    shown += 1
print(f"  total 16-round 2026 matches with no detected swap: {shown}")

# ---------------------------------------------------------------- 11. score-field swap detection (round_state freeze_end)
print("\n== 11. Score-field check (round_state round_freeze_end t_score/ct_score), independent of player_info ==")
RS = read(base24, "round_state", ["match_key", "round", "event_type", "t_score", "ct_score"])
RS = (RS.filter(pl.col("event_type") == "round_freeze_end")
      .unique(["match_key", "round"], keep="first"))
sc = collections.defaultdict(dict)
for m, r, t, c in RS.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    sc[m][r] = (t, c)
print(f"matches with freeze_end scores: {len(sc)}; round rows: {sum(len(v) for v in sc.values())}")


def label_pair(scm, r):
    """Label transition r -> r+1: 'no' (sides kept), 'swap' (sides swapped), 'both', 'none'."""
    if r + 1 not in scm or r not in scm:
        return None
    t0, c0 = scm[r]
    t1, c1 = scm[r + 1]
    no = (t1 - t0, c1 - c0) in [(1, 0), (0, 1)]
    sw = (t1 - c0, c1 - t0) in [(1, 0), (0, 1)]
    if no and sw:
        return "both"
    return "no" if no else ("swap" if sw else "none")


def score_winner(scm, r):
    """Winner code (2 or 3) of round r from score fields, using the transition label at r."""
    lab = label_pair(scm, r)
    if lab not in ("no", "swap"):
        return None
    t0, c0 = scm[r]
    t1, c1 = scm[r + 1]
    if lab == "no":
        return T if (t1 - t0, c1 - c0) == (1, 0) else CT
    return T if (t1 - c0, c1 - t0) == (0, 1) else CT


lab_count = collections.Counter()
score_swaps = {}
for m, scm in sc.items():
    labs = {r: label_pair(scm, r) for r in scm if (r + 1) in scm}
    for lab in labs.values():
        lab_count[lab] += 1
    score_swaps[m] = [r for r, lab in sorted(labs.items()) if lab == "swap"]
print(f"transition labels (all matches): {dict(lab_count)}")
cmp = collections.Counter()
for m, sw in score_swaps.items():
    if m not in det:
        continue
    cmp["same_swap_set" if sw == det[m]["bounds"] else "diff_swap_set"] += 1
print(f"score-field swap set vs player_info swap set: {dict(cmp)}")
diff_ex = [(m[:44], score_swaps[m][:4], det[m]["bounds"][:4]) for m in score_swaps
           if m in det and score_swaps[m] != det[m]["bounds"]][:8]
for row in diff_ex:
    print("   diff:", row)

# score winner vs round_end winner, all rounds with both scores
REW = RE.select(["match_key", "round", "winner_team_code"])
rew = {(m, r): w for m, r, w in REW.iter_rows()}
agree = collections.Counter()
for m, scm in sc.items():
    for r in scm:
        w_sc = score_winner(scm, r)
        w_re = rew.get((m, r))
        if w_sc is None or w_re is None:
            agree["unscorable"] += 1
            continue
        agree["agree" if w_sc == w_re else "disagree"] += 1
print(f"score-derived round winner vs round_end winner_team_code (all rounds): {dict(agree)}")

# ---------------------------------------------------------------- 12. round-13 winner: score fields vs round_end vs bomb
print("\n== 12. Round-13 winner from score fields vs round_end winner vs bomb outcome ==")
bomb_side = {}
for row in J.select(["match_key", "round", "bomb_side", "player_count"]).iter_rows():
    m, r, bsd, pc = row
    if r == 13:
        bomb_side[m] = bsd
tab = collections.Counter()
for m in r13_mm | r13_ok:
    scm = sc.get(m, {})
    w_sc = score_winner(scm, 13) if scm else None
    w_re = rew.get((m, 13))
    grp = "affected" if m in r13_mm else "control"
    sw_at_12 = 12 in score_swaps.get(m, [])
    tab[(grp, "score_swap_at_12" if sw_at_12 else "no_score_swap_at_12",
         "score=" + ("none" if w_sc is None else str(w_sc)),
         "round_end=" + str(w_re), "bomb=" + str(bomb_side.get(m)))] += 1
for k in sorted(tab, key=lambda x: (x[0], x[1], x[2], x[3], x[4])):
    print(f"  {k[0]:<9} {k[1]:<22} {k[2]:<11} {k[3]:<14} {k[4]:<11} n={tab[k]}")

# ---------------------------------------------------------------- 13. spawn team codes, non-null rows only
print("\n== 13. player_spawn team codes (non-null) vs player_info team codes, same match/round/player ==")
pi_team = {}
for m, r, p, t in pi24.select(["match_key", "round", "player_id", "team_code"]).iter_rows():
    pi_team[(m, r, p)] = t
sp_ok = collections.Counter()
sp_round1 = collections.Counter()
for m, r, p, t in sp.iter_rows():
    if t is None:
        sp_ok["spawn_null"] += 1
        continue
    pt = pi_team.get((m, r, p))
    if pt is None:
        sp_ok["no_player_info_row"] += 1
        continue
    sp_ok["agree" if pt == t else "disagree"] += 1
    if r == 1:
        sp_round1["agree" if pt == t else "disagree"] += 1
print(f"spawn rows: {dict(sp_ok)}; round 1 non-null: {dict(sp_round1)}")

# ---------------------------------------------------------------- 14. unambiguous score transitions vs player_info swaps
print("\n== 14. Score transitions (unambiguous only) vs player_info swap rounds ==")
unamb = collections.Counter()
for m, scm in sc.items():
    if m not in det:
        continue
    pib = set(det[m]["bounds"])
    for r in scm:
        lab = label_pair(scm, r)
        if lab in ("no", "swap"):
            unamb[(lab, "pi_swap" if r in pib else "pi_no_swap")] += 1
print(f"  (score label, player_info swap at this round): {dict(sorted(unamb.items()))}")
print("  reading: 'swap' with pi_no_swap or 'no' with pi_swap would be contradictions")

# ---------------------------------------------------------------- 15. score vs round_end disagreement by swap position
print("\n== 15. Score-derived winner vs round_end winner: disagreements by swap group and position ==")
dis_tab = collections.Counter()
dis_round = collections.Counter()
for m, scm in sc.items():
    d = det.get(m)
    s1 = d["bounds"][0] if (d and d["bounds"]) else None
    for r in scm:
        w_sc = score_winner(scm, r)
        w_re = rew.get((m, r))
        if w_sc is None or w_re is None:
            continue
        rel = None if s1 is None else max(-2, min(3, r - (s1 + 1)))
        key = (f"s1={s1}", rel)
        dis_tab[(key, "agree" if w_sc == w_re else "disagree")] += 1
        if w_sc != w_re:
            dis_round[r] += 1
keys = sorted({k for k, _ in dis_tab}, key=lambda x: (str(x[0]), -99 if x[1] is None else x[1]))
for k in keys:
    a, dg = dis_tab[(k, "agree")], dis_tab[(k, "disagree")]
    print(f"  {k[0]:<8} rel={str(k[1]):<5} n={a + dg:>6} disagree={dg:>5} rate={dg / (a + dg):.4f}")
print(f"  disagreements by absolute round (top 15): {dis_round.most_common(15)}")
print(f"  total disagreements: {sum(dis_round.values())}")

# ---------------------------------------------------------------- 16. affected matches: score-vs-round_end at round 13 and the 2 max-round groups
print("\n== 16. Round-13 score-vs-round_end disagreement, by detected format ==")
r13_t = collections.Counter()
for m, scm in sc.items():
    if 13 not in scm:
        continue
    w_sc = score_winner(scm, 13)
    w_re = rew.get((m, 13))
    if w_sc is None or w_re is None:
        continue
    fmt = str(det[m]["bounds"][0]) if (m in det and det[m]["bounds"]) else "none"
    grp = "affected" if m in r13_mm else ("control" if m in r13_ok else "no_bomb")
    r13_t[(fmt, grp, "agree" if w_sc == w_re else "disagree")] += 1
for k in sorted(r13_t):
    print(f"  first_swap={k[0]:<5} {k[1]:<9} {k[2]:<9} n={r13_t[k]}")

# ---------------------------------------------------------------- 17. first-post-swap disagreements by match day (2024)
print("\n== 17. First-post-swap-round disagreement by match day (2024) ==")
day_tab = collections.Counter()
for m, scm in sc.items():
    d = det.get(m)
    if not d or not d["bounds"]:
        continue
    day = m.split("/")[3]
    s1 = d["bounds"][0]
    r_post = s1 + 1
    w_sc = score_winner(scm, r_post)
    w_re = rew.get((m, r_post))
    if w_sc is None or w_re is None:
        day_tab[(day, "unscorable")] += 1
        continue
    day_tab[(day, "agree" if w_sc == w_re else "disagree")] += 1
days = sorted({k[0] for k in day_tab})
print("  day  agree  disagree  unscorable  rate_disagree")
for dd in days:
    a, dg, u = day_tab[(dd, "agree")], day_tab[(dd, "disagree")], day_tab[(dd, "unscorable")]
    rate = dg / (a + dg) if (a + dg) else float("nan")
    print(f"  {dd}  {a:>6}  {dg:>8}  {u:>10}  {rate:.3f}")

# ---------------------------------------------------------------- 18. contradictions and spawn subset check
print("\n== 18. Contradictions and spawn subset check ==")
contra = []
for m, scm in sc.items():
    if m not in det:
        continue
    pib = set(det[m]["bounds"])
    for r in scm:
        if label_pair(scm, r) == "swap" and r not in pib:
            contra.append((m[:44], r, det[m]["bounds"], det[m]["ref"], len(det[m]["rounds"])))
print(f"  'swap' score labels at non-player_info swap rounds: {len(contra)}")
for row in contra[:6]:
    print("   ", row)
sub = collections.Counter()
for m, rm in sp_map.items():
    if m not in det:
        continue
    d2 = detect_swaps(rm)
    pib = set(det[m]["bounds"])
    sb = set(d2["bounds"])
    if sb == pib:
        sub["same"] += 1
    elif sb <= pib:
        sub["spawn_subset_of_pi"] += 1
    else:
        sub["spawn_has_extra_bound"] += 1
print(f"  spawn (non-null) vs player_info bounds: {dict(sub)}")

# ---------------------------------------------------------------- 19. 2026 control: score-field vs round_end
print("\n== 19. 2026 control: score-field winner vs round_end winner (should be clean) ==")
RS26 = read(base26, "round_state", ["match_key", "round", "event_type", "t_score", "ct_score"])
RS26 = RS26.filter(pl.col("event_type") == "round_freeze_end").unique(["match_key", "round"], keep="first")
sc26 = collections.defaultdict(dict)
for m, r, t, c in RS26.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    sc26[m][r] = (t, c)
RE26 = read(base26, "round_end", ["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first")
rew26 = {(m, r): w for m, r, w in RE26.iter_rows()}
c26 = collections.Counter()
c26_round = collections.Counter()
for m, scm in sc26.items():
    for r in scm:
        w_sc = score_winner(scm, r)
        w_re = rew26.get((m, r))
        if w_sc is None or w_re is None:
            c26["unscorable"] += 1
            continue
        if w_sc == w_re:
            c26["agree"] += 1
        else:
            c26["disagree"] += 1
            c26_round[r] += 1
print(f"  2026 score vs round_end (all rounds): {dict(c26)}; disagreements by round (top 10): {c26_round.most_common(10)}")

# ---------------------------------------------------------------- 20. non-round-13 bomb mismatches vs score-derived winner
print("\n== 20. Bomb-vs-round_end mismatches outside round 13: which source the score fields support ==")
oth = (J20.filter(pl.col("mismatch") & (pl.col("round") != 13))
       .select(["match_key", "round", "bomb_side", "winner_team_code"]).iter_rows())
c20 = collections.Counter()
c20_round = collections.Counter()
for m, r, bsd, wre in oth:
    scm = sc.get(m, {})
    w_sc = score_winner(scm, r) if scm else None
    c20_round[r] += 1
    if w_sc is None:
        c20["score_unscorable"] += 1
    elif w_sc == bsd:
        c20["score agrees with bomb (round_end odd)"] += 1
    elif w_sc == wre:
        c20["score agrees with round_end (bomb odd)"] += 1
    else:
        c20["other"] += 1
print(f"  non-13 mismatch rounds: {sum(c20_round.values())}; rounds: {dict(c20_round.most_common(8))}")
print(f"  {dict(c20)}")

# ---------------------------------------------------------------- 21. undetermined matches: score-field swap rounds
print("\n== 21. Matches with no player_info swap: score-field unambiguous swap rounds ==")
und = [m for m, d in det.items() if not d["bounds"]]
c21 = collections.Counter()
for m in und:
    scm = sc.get(m, {})
    sw = [r for r in sorted(scm) if label_pair(scm, r) == "swap"]
    c21[tuple(sw[:2])] += 1
print(f"  undetermined matches: {len(und)}; score-field swap patterns: {dict(c21.most_common(10))}")

# ---------------------------------------------------------------- 22. map name vs detected format
print("\n== 22. Map name vs detected format ==")
H24m = read(base24, "header", ["key", "map_name"])
map24 = {r["key"]: r["map_name"] for r in H24m.iter_rows(named=True)}
mc = collections.Counter()
for m, d in det.items():
    s = str(d["bounds"][0]) if d["bounds"] else "none"
    mc[(map24.get(m, "?"), s)] += 1
print("  2024 (first swap: 8 = 16-round, 12 = 24-round, none = undetermined):")
for mp in sorted({k[0] for k in mc}, key=lambda x: -sum(v for k, v in mc.items() if k[0] == x)):
    row = {s: mc[(mp, s)] for s in ["8", "12", "none"]}
    print(f"    {mp:<16} n={sum(row.values()):>5}  s8={row['8']:>4}  s12={row['12']:>5}  none={row['none']:>4}")
H26m = read(base26, "header", ["key", "map_name", "max_rounds"])
mc26 = collections.Counter((r["map_name"], r["max_rounds"]) for r in H26m.iter_rows(named=True))
print("  2026 (header max_rounds):")
for mp in sorted({k[0] for k in mc26}, key=lambda x: -sum(v for k, v in mc26.items() if k[0] == x)):
    print(f"    {mp:<16} mr24={mc26.get((mp, 24), 0):>5}  mr16={mc26.get((mp, 16), 0):>5}")

# ---------------------------------------------------------------- 23. per-round agree/disagree for first post-swap rounds (2024, score vs round_end)
print("\n== 23. Score-derived vs round_end winner, selected absolute rounds (2024, all matches) ==")
ab = collections.Counter()
for m, scm in sc.items():
    for r in (9, 12, 13, 14, 25, 26, 27, 28):
        if r not in scm:
            continue
        w_sc = score_winner(scm, r)
        w_re = rew.get((m, r))
        if w_sc is None or w_re is None:
            ab[(r, "unscorable")] += 1
        else:
            ab[(r, "agree" if w_sc == w_re else "disagree")] += 1
for r in (9, 12, 13, 14, 25, 26, 27, 28):
    a, dg, u = ab[(r, "agree")], ab[(r, "disagree")], ab[(r, "unscorable")]
    rate = dg / (a + dg) if (a + dg) else float("nan")
    print(f"  round {r:>2}: agree={a:>5} disagree={dg:>5} unscorable={u:>5} rate_disagree={rate:.3f}")
# 2024 days 12-25 first-post-swap summary
tot_days = sum(v for k, v in day_tab.items() if k[1] in ("agree", "disagree") and k[0] >= "12")
dis_days = sum(v for k, v in day_tab.items() if k[1] == "disagree" and k[0] >= "12")
print(f"  first-post-swap rounds with a scorable check, days 12-25: {tot_days}; disagree: {dis_days}")
print(f"  affected (r13 bomb mismatch) matches: score transition label at 12 -> swap: "
      f"{sum(1 for m in r13_mm if 12 in score_swaps.get(m, []))}, no-swap: 0 (labels: 'both' or 'swap' only; see section 14)")
