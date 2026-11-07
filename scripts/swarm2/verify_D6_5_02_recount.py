"""D6-5 independent verifier.

Recomputes the claimed defect scope from the tome parquet files (not from the earlier logs) and tests
which source is wrong at the disagreeing rounds, using references that do not use round_end's
winner_team_code:
  (a) bomb outcome events (bomb_defused -> CT wins, bomb_exploded -> T wins; physical, team-code free)
  (b) player-death wipe rule, only on rounds with no bomb planted and a clean 5v5 roster:
        one side fully dead -> the other side won; no side wiped and no bomb -> time-out -> CT won.
Score-derived winner is re-implemented here from round_state round_freeze_end t_score/ct_score
(transition r -> r+1: 'no' = sides kept, 'swap' = sides swapped at halftime / OT boundary).

Run:
  "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_5_02_recount.py 2024
  "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_5_02_recount.py 2026
Memory: column-projected lazy scans per channel; largest in-memory frame is player_info (~0.8M rows).
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
T, CT = 2, 3
BASES = {
    "2024": Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"),
    "2026": Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"),
}
ERA = sys.argv[1] if len(sys.argv) > 1 else "2024"
BASE = BASES[ERA]


def lazy(ch):
    dirs = sorted(BASE.glob(f"basic_{ch}.*"))
    assert len(dirs) == 1, (ch, dirs)
    fs = sorted(glob.glob(str(dirs[0] / "dataframe_*")))
    return pl.concat([pl.scan_parquet(f) for f in fs], how="vertical_relaxed")


def trans(scm, r):
    """Transition label r -> r+1 from freeze_end scores."""
    if r not in scm or (r + 1) not in scm:
        return None
    t0, c0 = scm[r]
    t1, c1 = scm[r + 1]
    no = (t1 - t0, c1 - c0) in ((1, 0), (0, 1))
    sw = (t1 - c0, c1 - t0) in ((1, 0), (0, 1))
    if no and sw:
        return "both"
    if no:
        return "no"
    if sw:
        return "swap"
    return "none"


def win_from_score(scm, r):
    lab = trans(scm, r)
    if lab not in ("no", "swap"):
        return None
    t0, c0 = scm[r]
    t1, c1 = scm[r + 1]
    if lab == "no":
        return T if (t1 - t0) == 1 else CT
    # swap: the team that had the T score (t0) and gained is now CT, so winner side in round r is T
    return T if (c1 - t0) == 1 else CT


def pct(a, b):
    return f"{100.0 * a / b:.2f}%" if b else "n/a"


print(f"=== ERA {ERA} tome: {BASE}")

# ------------------------------------------------------------------ load
fe = (lazy("round_state").filter(pl.col("event_type") == "round_freeze_end")
      .select(["match_key", "round", "t_score", "ct_score"]).collect())
fe_rows = fe.height
fe = fe.unique(["match_key", "round"], keep="first")
print(f"round_freeze_end rows {fe_rows}, unique (match,round) {fe.height}")

re_raw = lazy("round_end").select(["match_key", "round", "winner_team_code", "win_reason_message",
                                   "player_count", "tick"]).collect()
re_keys = re_raw.select(["match_key", "round"]).unique().height
re_ = re_raw.unique(["match_key", "round"], keep="first")
print(f"round_end rows {re_raw.height}, unique (match,round) {re_keys}; null winner rows: "
      f"{re_.filter(pl.col('winner_team_code').is_null()).height}")

bs = (lazy("bomb_state").filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"]))
      .select(["match_key", "round", "event_type"]).collect())
pinfo = (lazy("player_info").select(["match_key", "round", "player_id", "team_code"]).collect()
         .unique(["match_key", "round", "player_id"], keep="first"))
death = lazy("player_death").select(["match_key", "round", "player_id"]).collect()
print(f"bomb rows {bs.height}; player_info unique rows {pinfo.height}; player_death rows {death.height}")

scm_all = collections.defaultdict(dict)
for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    scm_all[m][r] = (t, c)
rew = {}
for m, r, w, rs, pc, tk in re_.select(["match_key", "round", "winner_team_code", "win_reason_message",
                                       "player_count", "tick"]).iter_rows():
    rew[(m, r)] = {"w": w, "reason": rs, "pc": pc, "tick": tk}
print(f"matches with freeze_end: {len(scm_all)}")

# ------------------------------------------------------------------ 1. headline recount (claim definition)
print("\n== 1. Headline recount: score-derived winner vs round_end winner (all freeze_end rounds) ==")
cnt = collections.Counter()
dis = []
for m, scm in scm_all.items():
    for r in scm:
        lab = trans(scm, r)
        cnt[f"transition_{lab}"] += 1
        w_sc = win_from_score(scm, r)
        rr = rew.get((m, r))
        w_re = rr["w"] if rr else None
        if rr is None:
            cnt["no_round_end_row"] += 1
        if w_sc is None or w_re is None:
            cnt["unscorable"] += 1
        elif w_sc == w_re:
            cnt["agree"] += 1
        else:
            cnt["disagree"] += 1
            dis.append((m, r))
total_rows = sum(len(v) for v in scm_all.values())
scorable = cnt["agree"] + cnt["disagree"]
print(f"freeze_end round rows: {total_rows}")
print(f"transition labels: { {k: v for k, v in cnt.items() if k.startswith('transition_')} }")
print(f"agree={cnt['agree']} disagree={cnt['disagree']} unscorable={cnt['unscorable']} "
      f"(no round_end row among freeze_end rounds: {cnt['no_round_end_row']})")
print(f"scorable (score and round_end both usable) = {scorable}; disagree / scorable = {pct(cnt['disagree'], scorable)}; "
      f"disagree / freeze_end rows = {pct(cnt['disagree'], total_rows)}")

# ------------------------------------------------------------------ 2. where the disagreements sit (score-based swaps only)
print("\n== 2. Disagreement positions, using score-derived swap labels only (no player_info) ==")
by_r = collections.Counter(r for _, r in dis)
print(f"disagreements by absolute round: {sorted(by_r.items())}")
prev_lab = collections.Counter()
for m, r in dis:
    scm = scm_all[m]
    prev_lab[trans(scm, r - 1) if (r - 1) in scm else "no_prev_round"] += 1
print(f"label at (round-1 -> round) for disagreeing rounds: {dict(prev_lab)}")
swap_rounds_hist = collections.Counter()
for m, scm in scm_all.items():
    for r in scm:
        if trans(scm, r) == "swap":
            swap_rounds_hist[r] += 1
print(f"score-derived swap boundaries (completed round -> count): {sorted(swap_rounds_hist.items())}")
# every disagreement should be the first round after a score-derived swap boundary
post_swap_count = sum(1 for m, r in dis if (r - 1) in scm_all[m] and trans(scm_all[m], r - 1) == "swap")
print(f"disagreements that are first post-swap rounds (score-derived): {post_swap_count} of {len(dis)}")
dis_sets = collections.Counter(r for m, r in dis)
print(f"disagreeing rounds by day (2024 'match_day' = path segment 4): ",
      dict(sorted(collections.Counter(m.split('/')[3] for m, r in dis).items())) if ERA == "2024" else "n/a")

# ------------------------------------------------------------------ 3. bomb-team-code-free evidence
print("\n== 3. Bomb outcome (team-code free) at the disagreeing rounds ==")
bomb = collections.defaultdict(set)
for m, r, e in bs.iter_rows():
    bomb[(m, r)].add(e)
conflict = sum(1 for v in bomb.values() if {"bomb_defused", "bomb_exploded"} <= v)
multi_plant = sum(1 for v in bomb.values() if "bomb_planted" in v)
print(f"rounds with a bomb row: {len(bomb)}; with both defused and exploded: {conflict}; with bomb_planted: {multi_plant}")


def bomb_side(m, r):
    ev = bomb.get((m, r), set())
    if "bomb_defused" in ev:
        return CT
    if "bomb_exploded" in ev:
        return T
    return None


b_tab = collections.Counter()
for m, r in dis:
    bs_side = bomb_side(m, r)
    scm = scm_all[m]
    w_sc = win_from_score(scm, r)
    w_re = rew[(m, r)]["w"]
    if bs_side is None:
        b_tab[(r, "no_outcome")] += 1
    else:
        b_tab[(r, f"bomb==score:{bs_side == w_sc}", f"bomb==round_end:{bs_side == w_re}")] += 1
for k in sorted(b_tab, key=str):
    print(f"  round {k[0]:>2} {k[1:]}: n={b_tab[k]}")

# population check: bomb vs round_end over every round with an outcome (team-code free), by round
pop = collections.Counter()
for (m, r), ev in bomb.items():
    bsd = bomb_side(m, r)
    rr = rew.get((m, r))
    if bsd is None or rr is None:
        continue
    pop[("mismatch" if bsd != rr["w"] else "match", r if r in (9, 12, 13, 14, 27, 28) else "other")] += 1
print(f"bomb outcome vs round_end, all rounds with outcome: {dict(sorted(pop.items(), key=str))}")

# ------------------------------------------------------------------ 4. wipe rule (team-code free except roster side)
print("\n== 4. Player-death wipe rule (no bomb planted, clean 5v5 roster) ==")
roster = collections.defaultdict(dict)
for m, r, p, t in pinfo.iter_rows():
    if t in (T, CT):
        roster[(m, r)][p] = t
roster_sizes = collections.Counter()
for k, ro in roster.items():
    nT = sum(1 for s in ro.values() if s == T)
    nC = len(ro) - nT
    roster_sizes[(nT, nC)] += 1
print(f"roster (T,CT) size pairs, top 8: {roster_sizes.most_common(8)}")
dead = collections.defaultdict(set)
for m, r, p in death.iter_rows():
    dead[(m, r)].add(p)
planted = {k for k, v in bomb.items() if "bomb_planted" in v}


def wipe_implied(m, r):
    if (m, r) in planted:
        return "planted", None
    ro = roster.get((m, r), {})
    nT = sum(1 for s in ro.values() if s == T)
    nC = len(ro) - nT
    if nT != 5 or nC != 5:
        return "not_5v5", None
    dset = dead.get((m, r), set())
    wT = all(p in dset for p, s in ro.items() if s == T)
    wC = all(p in dset for p, s in ro.items() if s == CT)
    if wT and wC:
        return "both_wiped", None
    if wT:
        return "wipe", CT
    if wC:
        return "wipe", T
    return "timeout", CT


# calibration: rounds where round_end and score agree (assumed correct) vs disagreements
disagree_set = set(dis)
cal = collections.Counter()
for m, scm in scm_all.items():
    for r in scm:
        rr = rew.get((m, r))
        w_sc = win_from_score(scm, r)
        if rr is None or w_sc is None:
            continue
        kind, imp = wipe_implied(m, r)
        if kind in ("planted", "not_5v5", "both_wiped"):
            cal[("skipped", kind)] += 1
            continue
        grp = "disagree" if (m, r) in disagree_set else "agree"
        cal[(grp, kind, f"rule==round_end:{imp == rr['w']}", f"rule==score:{imp == w_sc}")] += 1
print("  group, rule kind, rule vs round_end, rule vs score -> n")
for k in sorted(cal, key=str):
    print(f"  {str(k):<120} n={cal[k]}")

# ------------------------------------------------------------------ 5. unscorable first-post-swap rounds
print("\n== 5. First post-swap rounds (score-derived) that are unscorable ==")
fps = collections.Counter()
fps_detail = collections.Counter()
for m, scm in scm_all.items():
    for r in scm:
        if (r - 1) not in scm or trans(scm, r - 1) != "swap":
            continue
        day = m.split("/")[3] if ERA == "2024" else "all"
        w_sc = win_from_score(scm, r)
        rr = rew.get((m, r))
        w_re = rr["w"] if rr else None
        lab_now = trans(scm, r)
        has_bomb = (m, r) in bomb and bomb_side(m, r) is not None
        if w_sc is None or w_re is None:
            fps[(day, "unscorable")] += 1
            fps_detail[("unscorable", f"round_end_missing={w_re is None}", f"label_at_r={lab_now}", f"bomb_outcome={has_bomb}")] += 1
        else:
            fps[(day, "agree" if w_sc == w_re else "disagree")] += 1
days = sorted({k[0] for k in fps})
print("  day  agree  disagree  unscorable")
for dd in days:
    print(f"  {dd}  {fps[(dd, 'agree')]:>6}  {fps[(dd, 'disagree')]:>8}  {fps[(dd, 'unscorable')]:>10}")
for k in sorted(fps_detail, key=str):
    print(f"  {str(k):<100} n={fps_detail[k]}")
unsc_total = sum(v for k, v in fps.items() if k[1] == "unscorable")
if ERA == "2024":
    unsc_12_25 = sum(v for k, v in fps.items() if k[1] == "unscorable" and "12" <= k[0] <= "25")
    print(f"  first-post-swap unscorable, days 12-25: {unsc_12_25}; all days: {unsc_total}")

# ------------------------------------------------------------------ 6. 2024 rounds with disagreement: examples
print("\n== 6. Examples (first 4 disagreements): freeze_end scores r-1, r, r+1, r+2 and round_end winners ==")
for m, r in dis[:4]:
    scm = scm_all[m]
    seq = [scm.get(x) for x in (r - 1, r, r + 1, r + 2)]
    ws = [rew.get((m, x), {}).get("w") for x in (r - 1, r, r + 1, r + 2)]
    print(f"  {m[-44:]} round {r}: scores(t,ct)={seq} round_end winners={ws} "
          f"score_winner(r)={win_from_score(scm, r)} bomb(r)={bomb_side(m, r)}")
