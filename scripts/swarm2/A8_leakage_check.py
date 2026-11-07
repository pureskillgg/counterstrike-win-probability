"""A8 check 2: feature causality, label consistency and round-end / decided-state exposure.

Rebuilds the bake-off state table with xgb_primary_seed0.build_data(dedupe=True) (same code path
as the reproduction) and cross-checks each feature against the raw round events (read-only).
Checks:
  A. every state tick lies in [round_freeze_end, round_end] of its round; alive counts never
     increase inside a round; live-start state is first.
  B. bomb_planted / seconds_since_plant reproduce from the raw bomb_planted events inside the
     live window, using only plants at or before the state tick.
  C. ct_score_minus_t_score equals the cumulative (CT round wins minus T round wins) before
     the round, taken from round_end winners (label-side derivation).
  D. decided states: states where the round outcome is already fixed by the alive counts
     (no CT alive; or no T alive and no bomb planted) and their share of states, label rates,
     and effect on test log loss and AUC.
  E. label equals (round_end.winner_team_code == 3) for every state.
  F. victim side from player_death equals the side in player_info for the same round, for
     in-window deaths with a non-null team.
  G. fingerprint duplicates left among the 1,521 kept matches (round, tick, winner).
Run from the project root:
  .pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A8_leakage_check.py
"""

import collections
import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, log_loss

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    print("== A8 check 2: leakage and label consistency ==")
    xgbs = load("xgb_primary_seed0", ROOT / "scripts" / "model" / "xgb_primary_seed0.py")
    bo = xgbs.bo
    fv = bo.fv
    D = xgbs.build_data(dedupe=True)
    S, sets = D["S"], D["sets"]
    kept = D["kept_final"]
    print(f"states {S.height}; rounds {D['n_rounds']}; matches {len(kept)}")

    # raw events
    RE = fv.read("round_end", ["match_key", "round", "tick", "winner_team_code"])
    RS = fv.read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    BS = fv.read("bomb_state", ["match_key", "round", "tick", "event_type"])
    DE = fv.read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code"])
    PI = fv.read("player_info", ["match_key", "round", "player_id", "team_code"])

    end = {(m, int(r)): int(t) for m, r, t in RE.select(["match_key", "round", "tick"]).iter_rows()}
    win = {(m, int(r)): int(w) for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows()}
    frz = {}
    for m, r, t in RS.filter(pl.col("event_type") == "round_freeze_end").select(
            ["match_key", "round", "tick"]).iter_rows():
        frz.setdefault((m, int(r)), int(t))
    plants = collections.defaultdict(list)
    for m, r, t in BS.filter(pl.col("event_type") == "bomb_planted").select(
            ["match_key", "round", "tick"]).iter_rows():
        plants[(m, int(r))].append(int(t))

    Sp = S.with_columns(pl.col("match_key").alias("mk"))
    rows = Sp.to_dicts()

    # ---- A. ordering and window
    bad_window = 0
    bad_frz_first = 0
    alive_increase = 0
    by_round = collections.defaultdict(list)
    for x in rows:
        k = (x["match_key"], int(x["round"]))
        by_round[k].append(x)
        if k not in end or k not in frz:
            bad_window += 1
            continue
        if not (frz[k] <= x["tick"] <= end[k]):
            bad_window += 1
    n_at_end = 0
    for k, lst in by_round.items():
        lst.sort(key=lambda x: (x["tick"], 0 if x["kind"] == "live_start" else 1))
        if lst[0]["kind"] != "live_start" or lst[0]["tick"] != frz.get(k):
            bad_frz_first += 1
        for a, b in zip(lst[:-1], lst[1:]):
            if b["alive_ct"] > a["alive_ct"] or b["alive_t"] > a["alive_t"]:
                alive_increase += 1
        n_at_end += sum(1 for x in lst if x["tick"] == end.get(k, -1))
    print("\n== A. window and ordering ==")
    print(f"states outside [freeze_end, round_end]: {bad_window}")
    print(f"rounds whose first state is not live_start at freeze_end: {bad_frz_first}")
    print(f"consecutive state pairs where an alive count increases: {alive_increase}")
    print(f"states exactly at the round_end tick (a death at the end tick is kept): {n_at_end}")

    # ---- B. bomb plant causality
    b_bad = 0
    b_neg = 0
    b_plant_mismatch = 0
    planted_states = 0
    for k, lst in by_round.items():
        in_win = [t for t in plants.get(k, []) if frz.get(k, 0) <= t <= end.get(k, -1)]
        derived = None
        for x in lst:
            if x["bomb_planted"] == 1:
                planted_states += 1
                if x["seconds_since_plant"] < 0:
                    b_neg += 1
                if derived is None:
                    derived = x["tick"] - x["seconds_since_plant"] * bo.TICK_RATE
            else:
                if x["seconds_since_plant"] != -1.0:
                    b_bad += 1
        if in_win and derived is not None:
            if abs(derived - min(in_win)) > 1e-6:
                b_plant_mismatch += 1
        if derived is not None and not in_win:
            b_plant_mismatch += 1
    print("\n== B. bomb plant causality ==")
    print(f"planted=1 states with negative seconds_since_plant: {b_neg} (of {planted_states})")
    print(f"planted=0 states with seconds_since_plant != -1: {b_bad}")
    print(f"rounds where the derived plant tick differs from the earliest in-window plant: {b_plant_mismatch}")
    print(f"rounds with more than one plant event in window: "
          f"{sum(1 for k in by_round if len([t for t in plants.get(k, []) if frz.get(k, 0) <= t <= end.get(k, -1)]) > 1)}")

    # ---- C. score lead vs cumulative round winners
    per_match = collections.defaultdict(dict)
    for (m, r), w in win.items():
        per_match[m][r] = w
    # Team-based lead for the CT side at round start. Each completed round adds +1 (CT won) or
    # -1 (T won) to "current CT side minus current T side"; at each swap (bakeoff.swap_expected)
    # the sides exchange, so the running lead is negated. Without the negation the side-code
    # tally (side_lead) is a different quantity after halftime; both are reported.
    c_rounds = 0
    c_eq_flip = 0
    c_eq_side = 0
    c_neither = 0
    c_examples = []
    c_pre_half = [0, 0]
    c_post_half = [0, 0]
    for k, lst in by_round.items():
        m, r = k
        mr = D["H"][m]["max_rounds"]
        lead, side_lead = 0, 0
        ok = True
        for rr in range(1, r):
            w = per_match[m].get(rr)
            if w is None:
                ok = False
                break
            s = 1 if w == bo.CT else (-1 if w == bo.T else 0)
            lead += s
            side_lead += s
            if bo.swap_expected(rr, mr):
                lead = -lead
        if not ok:
            continue
        for x in lst[:1]:
            c_rounds += 1
            f = int(x["ct_score_minus_t_score"])
            post = any(bo.swap_expected(rr, mr) for rr in range(1, r))
            if f == lead:
                c_eq_flip += 1
            elif f == side_lead:
                c_eq_side += 1
            else:
                c_neither += 1
                if len(c_examples) < 5:
                    c_examples.append((m[:40], r, f, lead, side_lead))
            if post:
                c_post_half[0 if f == lead else 1] += 1
            else:
                c_pre_half[0 if f == lead else 1] += 1
    print("\n== C. score lead vs winner-derived lead (live-start state per round) ==")
    print(f"rounds checked: {c_rounds}; feature == team lead with swap flips: {c_eq_flip}; "
          f"feature == side tally without flips: {c_eq_side}; neither: {c_neither}")
    print(f"before any swap: match-equal {c_pre_half[0]}, not equal {c_pre_half[1]}; "
          f"after a swap: match-equal {c_post_half[0]}, not equal {c_post_half[1]}")
    for e in c_examples:
        print(f"  e.g. match {e[0]} round {e[1]}: feature {e[2]} vs flip-lead {e[3]} vs side tally {e[4]}")

    # ---- E. label
    lab_bad = sum(1 for x in rows if int(x["label_ct_win"]) != int(win[(x["match_key"], int(x["round"]))] == bo.CT))
    print("\n== E. label ==")
    print(f"states where label_ct_win != (winner == CT): {lab_bad}")

    # ---- F. victim side vs player_info
    side_of = {}
    for m, r, pid, tc in PI.select(["match_key", "round", "player_id", "team_code"]).iter_rows():
        side_of[(m, int(r), pid)] = tc
    f_null = 0
    f_eq = 0
    f_ne = 0
    f_no_pi = 0
    f_examples = []
    for m, r, t, pid, tc in DE.select(["match_key", "round", "tick", "player_id", "player_team_code"]).iter_rows():
        k = (m, int(r))
        if m not in kept or k not in frz or k not in end:
            continue
        if not (frz[k] <= int(t) <= end[k]):
            continue
        if tc is None:
            f_null += 1
            continue
        pside = side_of.get((m, int(r), pid))
        if pside is None:
            f_no_pi += 1
            continue
        if int(tc) == int(pside):
            f_eq += 1
        else:
            f_ne += 1
            if len(f_examples) < 5:
                f_examples.append((m[:40], int(r), int(t), int(tc), int(pside)))
    print("\n== F. in-window victim side (death event) vs player_info side ==")
    print(f"equal: {f_eq}; different: {f_ne}; null death team: {f_null}; no player_info row: {f_no_pi}")
    for e in f_examples:
        print(f"  e.g. match {e[0]} round {e[1]} tick {e[2]}: death team {e[3]} vs player_info {e[4]}")

    # ---- D. decided states and metric exposure (test split)
    print("\n== D. decided states (outcome fixed by alive counts) ==")
    Xdf = S.with_columns(
        decided=((pl.col("alive_ct") == 0) | ((pl.col("alive_t") == 0) & (pl.col("bomb_planted") == 0))).cast(pl.Int8))
    grp = (Xdf.group_by(["alive_ct", "alive_t", "bomb_planted"])
           .agg(pl.len().alias("states"), pl.col("label_ct_win").mean().alias("ct_win_rate"))
           .sort(["alive_ct", "alive_t", "bomb_planted"]))
    print("alive_ct alive_t planted states ct_win_rate  (rows with alive_ct==0 or alive_t==0 only)")
    for r in grp.iter_rows(named=True):
        if r["alive_ct"] == 0 or r["alive_t"] == 0:
            print(f"  {r['alive_ct']:>8} {r['alive_t']:>7} {r['bomb_planted']:>7} {r['states']:>7} {r['ct_win_rate']:.4f}")
    dec_all = Xdf.filter(pl.col("decided") == 1)
    print(f"decided states: {dec_all.height} of {Xdf.height} ({100 * dec_all.height / Xdf.height:.2f}%); "
          f"CT-win rate among decided: {dec_all['label_ct_win'].mean():.4f}")
    te_keys = sorted(sets["test"])
    te = Xdf.filter(pl.col("match_key").is_in(te_keys))
    tcols = ["alive_ct", "alive_t", "ct_alive_minus_t", "bomb_planted", "seconds_since_round_live",
             "seconds_since_plant", "round_number", "ct_score_minus_t_score", "is_pistol_round", "is_overtime"]
    # Test-set predictions from the saved model at the reproduced range (0,262).
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(ROOT / "models" / "xgb_primary_seed0.json"))
    Xte = te.select(tcols).to_numpy().astype(np.float64)
    yte = te["label_ct_win"].to_numpy().astype(np.int64)
    dec = te["decided"].to_numpy().astype(bool)
    p = booster.predict(xgb.DMatrix(Xte), iteration_range=(0, 262))
    p = np.clip(p, 1e-15, 1 - 1e-15)
    def ll_(y, q):
        return float(log_loss(y, q, labels=[0, 1]))
    print(f"test states {len(yte)}; decided {int(dec.sum())} ({100 * dec.mean():.2f}%)")
    print(f"test LL all states {ll_(yte, p):.5f}; decided only {ll_(yte[dec], p[dec]):.5f}; "
          f"undecided only {ll_(yte[~dec], p[~dec]):.5f}")
    print(f"test AUC all states {roc_auc_score(yte, p):.4f}; undecided only {roc_auc_score(yte[~dec], p[~dec]):.4f}")
    print(f"test accuracy all {float(((p>=0.5)==yte).mean()):.4f}; undecided {float(((p[~dec]>=0.5)==yte[~dec]).mean()):.4f}")
    print(f"mean |p - label| decided {float(np.mean(np.abs(p[dec]-yte[dec]))):.4f}; "
          f"undecided {float(np.mean(np.abs(p[~dec]-yte[~dec]))):.4f}")

    # overlap of decided states with states at the round_end tick (all states)
    ov = collections.Counter()
    rounds_with_end_state = set()
    rounds_end_decided = set()
    for x in rows:
        k = (x["match_key"], int(x["round"]))
        at_end = x["tick"] == end.get(k, -1)
        dd = bool(x["alive_ct"] == 0 or (x["alive_t"] == 0 and x["bomb_planted"] == 0))
        ov[("at_end" if at_end else "before_end", "decided" if dd else "undecided")] += 1
        if at_end:
            rounds_with_end_state.add(k)
            if dd:
                rounds_end_decided.add(k)
    print("overlap of round_end-tick states and decided states (all kept states):")
    for key in [("at_end", "decided"), ("at_end", "undecided"), ("before_end", "decided"), ("before_end", "undecided")]:
        print(f"  {key[0]:<11} {key[1]:<10} {ov[key]}")
    print(f"rounds with a state at the round_end tick: {len(rounds_with_end_state)} of {D['n_rounds']}; "
          f"of those, the end-tick state is decided: {len(rounds_end_decided)}")
    # round-level: a round whose last state is at the end tick and decided, vs others
    try:
        ep = p  # test predictions from above
        te_at_end = np.array([x["tick"] == end.get((x["match_key"], int(x["round"])), -1)
                              for x in te.to_dicts()])
        print(f"test states at round_end tick: {int(te_at_end.sum())} of {len(yte)}; "
              f"test LL at round_end tick {ll_(yte[te_at_end], ep[te_at_end]):.5f}; "
              f"test LL excluding round_end tick {ll_(yte[~te_at_end], ep[~te_at_end]):.5f}")
        print(f"test AUC excluding round_end-tick states {roc_auc_score(yte[~te_at_end], ep[~te_at_end]):.4f}")
    except Exception as exc:  # reporting only
        print(f"round_end-tick split failed: {exc!r}")

    # ---- G. fingerprint duplicates among kept matches
    fp = collections.defaultdict(list)
    per_kept = collections.defaultdict(list)
    for (m, r), w in win.items():
        if m in kept:
            per_kept[m].append((r, end[(m, r)], w))
    for m, lst in per_kept.items():
        if len(lst) >= 5:
            fp[tuple(sorted(lst))].append(m)
    groups = [v for v in fp.values() if len(v) > 1]
    print("\n== G. round-fingerprint duplicates among the kept matches ==")
    print(f"kept matches with >=5 rounds: {sum(1 for m in per_kept if len(per_kept[m]) >= 5)}; "
          f"duplicate fingerprint groups: {len(groups)}; matches in them: {sum(len(g) for g in groups)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
