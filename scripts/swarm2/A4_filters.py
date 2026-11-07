"""A4 audit: bake-off round filters (scripts/model/bakeoff.py, main() section 2 and the duplicate step).

Steps
  0. Reproduce the bake-off filter stage with the same code path (fv.read, same channel check,
     same signature grouping, same make_splits imported from bakeoff.py). Check the counts
     against the bake-off log: player_count 3663, disconnect/takeover 1767, duplicates 913,
     not 5v5 476; 29596 rounds after the first six predicates; 28683 rounds, 1521 matches,
     234265 states at the end.
  1. Order dependence: every order (720) of the six round predicates, first-failure attribution.
     The duplicate step is applied after the six (as in bakeoff.py) and also before them.
  2. Pipeline variants (full re-run of survivors, duplicate step, unresolved-victim check, states):
       baseline (bake-off order, lowest-key duplicate representative, disconnect rule 'anywhere in
       the round'); disconnect rule 'live window only' (ASSUMPTION: live = round_freeze_end to
       round_end); duplicate representative = copy with most surviving rounds (ties to lowest key);
       duplicate step first; reversed predicate order.
  3. Duplicate audit: representative with zero surviving rounds while a copy has rounds; per-round
     predicate disagreement between copies; fingerprint (round_end only) groups vs signature groups.
  4. Disconnect and bot_takeover timing relative to round_start, round_freeze_end and round_end.
  5. Breakdown of the 476 not-5v5 rounds and the 3663 player_count rounds, incl. overlap with
     disconnect and team_change events.

Read-only on the tome. Intermediate JSON goes to the scratchpad, not the project.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A4_filters.py
"""
import collections
import importlib.util
import itertools
import json
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("G:/pureskill/csds-cs2")
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
               r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fv = load("feature_validation", "scripts/model/feature_validation.py")
bk = load("bakeoff", "scripts/model/bakeoff.py")  # import only: main() is guarded by __name__
T, CT, START, TICK_RATE = fv.T, fv.CT, fv.START, fv.TICK_RATE
TOME = fv.TOME
SUFFIX = fv.SUFFIX
PRIMARY, CHRONO = bk.PRIMARY, bk.CHRONO

DENSE = ["round_end", "round_start", "round_state", "player_death", "player_info",
         "bomb_state", "team_change"]
SPARSE = ["player_disconnect", "bot_takeover"]
PRED = ["pc", "5v5", "disc", "nort", "nolive", "null"]
BAKEOFF_ORDER = ["pc", "5v5", "disc", "nort", "nolive", "null"]
EXPECT = {"pc": 3663, "disc": 1767, "dup": 913, "5v5": 476, "rounds_after_six": 29596,
          "kept_rounds": 28683, "kept_matches": 1521, "states": 234265}
RESULTS = {}


def keyset_keys(ch):
    folder = TOME / f"basic_{ch}{SUFFIX}"
    files = sorted(glob.glob(str(folder / "keyset_*")))
    return set(pl.concat([pl.read_parquet(f) for f in files])["_"].to_list())


def check(label, got, want):
    flag = "MATCH" if got == want else "MISMATCH"
    print(f"  [{flag}] {label}: got {got}, bake-off log {want}")
    return got == want


def main():
    print("== A4 round-filter audit ==")
    print(f"bakeoff module: {bk.__file__}")
    print(f"split constants: PRIMARY={PRIMARY}, CHRONO={CHRONO}, SEED={bk.SEED}")

    # ---------- load (same calls as bakeoff.py main())
    hdr = fv.read("header", ["key", "match_id", "match_date", "max_rounds", "tick_rate",
                             "t_starters_score_final", "ct_starters_score_final"])
    H = {r["key"]: r for r in hdr.iter_rows(named=True)}
    all_matches = set(H)
    RE = fv.read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                               "player_count"])
    RST = fv.read("round_start", ["match_key", "round", "tick"])
    RS = fv.read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    DE = fv.read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code",
                                  "weapon_name", "attacker_id"])
    BS = fv.read("bomb_state", ["match_key", "round", "tick", "event_type"])
    DC = fv.read("player_disconnect", ["match_key", "round", "tick", "player_id", "is_bot",
                                       "disconnect_reason"])
    BT = fv.read("bot_takeover", ["match_key", "round", "tick", "player_id", "is_controlling"])
    TC = fv.read("team_change", ["match_key", "round", "tick", "player_id", "old_team_code",
                                 "new_team_code", "is_bot"])
    PI = fv.read("player_info", ["match_key", "round", "player_id", "team_code"])
    frames = {"round_end": RE, "round_start": RST, "round_state": RS, "player_death": DE,
              "player_info": PI, "bomb_state": BS, "team_change": TC,
              "player_disconnect": DC, "bot_takeover": BT}
    print(f"loaded: header {len(all_matches)}, round_end {RE.height}, round_start {RST.height}, "
          f"round_state {RS.height}, player_death {DE.height}, player_info {PI.height}, "
          f"disconnect {DC.height}, takeover {BT.height}, team_change {TC.height}")

    # ---------- channel check (bake-off section 1)
    missing_by_match = collections.defaultdict(list)
    for ch in DENSE + SPARSE:
        df = frames[ch]
        ks = keyset_keys(ch)
        have = set(df["match_key"].unique().to_list())
        no_keyset = all_matches - ks
        zero_rows = all_matches - have
        dense = ch in DENSE
        miss = no_keyset | (zero_rows if dense else set())
        for m in miss:
            missing_by_match[m].append(ch)
    kept_matches = all_matches - set(missing_by_match)
    n_rounds_channel_dropped = sum(1 for m in RE["match_key"].to_list() if m not in kept_matches)
    print(f"channel stage: kept matches {len(kept_matches)}, dropped {len(all_matches) - len(kept_matches)} "
          f"({n_rounds_channel_dropped} rounds)")
    check("kept matches after channel stage", len(kept_matches), 1709)

    # ---------- round-level records (bake-off section 2)
    rd = {}
    for x in RE.iter_rows(named=True):
        m = x["match_key"]
        if m in kept_matches:
            rd[(m, int(x["round"]))] = {"end": int(x["tick"]), "winner": x["winner_team_code"],
                                        "pc": x["player_count"]}
    rst_keys = set()
    rst_tick = {}
    rst_rows = collections.Counter()
    for x in RST.iter_rows(named=True):
        if x["match_key"] in kept_matches:
            k = (x["match_key"], int(x["round"]))
            rst_keys.add(k)
            rst_rows[k] += 1
            rst_tick.setdefault(k, int(x["tick"]))
    live = {}
    live_dups = 0
    for x in RS.filter(pl.col("event_type") == "round_freeze_end").iter_rows(named=True):
        m = x["match_key"]
        if m not in kept_matches:
            continue
        k = (m, int(x["round"]))
        if k in live:
            live_dups += 1
            continue
        live[k] = {"tick": int(x["tick"]), "t": x["t_score"], "ct": x["ct_score"]}
    players = collections.defaultdict(dict)
    pi_dup_rows = 0
    for x in PI.iter_rows(named=True):
        m = x["match_key"]
        if m not in kept_matches:
            continue
        k = (m, int(x["round"]))
        if x["player_id"] in players[k]:
            pi_dup_rows += 1
        players[k][x["player_id"]] = x["team_code"]
    disc_any = set()
    disc_src = collections.defaultdict(set)
    for name, df in (("disconnect", DC), ("takeover", BT)):
        for x in df.iter_rows(named=True):
            if x["match_key"] in kept_matches:
                k = (x["match_key"], int(x["round"]))
                disc_any.add(k)
                disc_src[k].add(name)
    deaths = collections.defaultdict(list)
    for x in DE.sort("tick").iter_rows(named=True):
        m = x["match_key"]
        if m in kept_matches:
            deaths[(m, int(x["round"]))].append({"tick": int(x["tick"]), "pid": x["player_id"],
                                                 "team": x["player_team_code"]})
    plants = collections.defaultdict(list)
    for x in BS.filter(pl.col("event_type") == "bomb_planted").iter_rows(named=True):
        if x["match_key"] in kept_matches:
            plants[(x["match_key"], int(x["round"]))].append(int(x["tick"]))
    tc_rounds = set()
    for x in TC.iter_rows(named=True):
        if x["match_key"] in kept_matches:
            tc_rounds.add((x["match_key"], int(x["round"])))
    print(f"rounds in kept matches: {len(rd)}; live-start rows {len(live)} (duplicate freeze rows "
          f"{live_dups}); rst duplicate rows {sum(v - 1 for v in rst_rows.values() if v > 1)}; "
          f"player_info duplicate (match, round, player) rows {pi_dup_rows}")
    check("rounds in kept matches (before round filters)", len(rd), 35502)

    # live-window disconnect set: an event whose tick is in [round_freeze_end, round_end]
    def classify(m, r, tick):
        k = (m, r)
        if k not in rd:
            return "round_not_in_round_end"
        start = rst_tick.get(k)
        if start is None:
            return "no_round_start"
        if tick < start:
            return "before_round_start"
        if k not in live:
            return "no_freeze_end"
        if tick < live[k]["tick"]:
            return "freeze_time"
        if tick <= rd[k]["end"]:
            return "live"
        return "after_round_end"

    disc_live = set()
    ev_class_rows = collections.Counter()
    ev_class_round = collections.defaultdict(set)
    ev_detail = collections.defaultdict(collections.Counter)
    after_offsets = []
    before_offsets = []
    for name, df in (("disconnect", DC), ("takeover", BT)):
        for x in df.iter_rows(named=True):
            m = x["match_key"]
            if m not in kept_matches:
                continue
            r = int(x["round"])
            tick = int(x["tick"])
            cls = classify(m, r, tick)
            ev_class_rows[(name, cls)] += 1
            if (m, r) in rd:
                ev_class_round[(m, r)].add(cls)
            if cls == "live":
                disc_live.add((m, r))
            if name == "disconnect":
                ev_detail[cls][f"is_bot={x['is_bot']}"] += 1
                ev_detail[cls][f"reason={x['disconnect_reason']!r}"] += 1
            if cls == "after_round_end":
                after_offsets.append((tick - rd[(m, r)]["end"]) / TICK_RATE)
            if cls == "before_round_start":
                before_offsets.append((rst_tick[(m, r)] - tick) / TICK_RATE)

    # ---------- per-round predicates
    flags = {}
    info = {}
    for k in sorted(rd):
        rr = rd[k]
        pl_ = players.get(k, {})
        n_t = sum(1 for v in pl_.values() if v == T)
        n_ct = sum(1 for v in pl_.values() if v == CT)
        flags[k] = {
            "pc": rr["pc"] != 10,
            "5v5": not (len(pl_) == 10 and n_t == START and n_ct == START),
            "disc": k in disc_any,
            "nort": k not in rst_keys,
            "nolive": k not in live,
            "null": (k in live) and (live[k]["t"] is None or live[k]["ct"] is None),
        }
        info[k] = {"len_pl": len(pl_), "n_t": n_t, "n_ct": n_ct}

    # ---------- step 0: reproduce bake-off attribution (BAKEOFF_ORDER, duplicate step last)
    reasons = collections.Counter()
    survivors6 = []
    for k in sorted(rd):
        f = flags[k]
        for p in BAKEOFF_ORDER:
            if f[p]:
                reasons[p] += 1
                break
        else:
            survivors6.append(k)
    print("\n== 0. Reproduction (bake-off order, first failure counts) ==")
    for p in BAKEOFF_ORDER:
        print(f"  {p:<8} first-failure count {reasons[p]}")
    check("player_count != 10 rounds", reasons["pc"], EXPECT["pc"])
    check("disconnect/takeover rounds (after pc and 5v5 in order)", reasons["disc"], EXPECT["disc"])
    check("not 5v5 rounds (after pc)", reasons["5v5"], EXPECT["5v5"])
    check("rounds after the six predicates", len(survivors6), EXPECT["rounds_after_six"])

    # signature groups (bake-off duplicate step)
    sig_parts = collections.defaultdict(list)
    for x in RE.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("R", int(x["round"]), int(x["tick"]),
                                               int(x["winner_team_code"]))))
    for x in DE.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("D", int(x["round"]), int(x["tick"]),
                                               str(x["player_id"]))))
    for x in PI.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("P", int(x["round"]), str(x["player_id"]),
                                               str(x["team_code"]))))
    sigs = {m: tuple(sorted(v)) for m, v in sig_parts.items()}
    sig_groups = collections.defaultdict(list)
    for m in sorted(kept_matches):
        sig_groups[sigs.get(m, ())].append(m)
    dedup_groups = [g for g in sig_groups.values() if len(g) > 1]
    dup_drop_lowest = set()
    for g in dedup_groups:
        dup_drop_lowest.update(g[1:])
    dup_rounds_bakeoff = sum(1 for k in survivors6 if k[0] in dup_drop_lowest)
    print(f"\n  duplicate groups among channel-kept matches: {len(dedup_groups)} groups, "
          f"{sum(len(g) for g in dedup_groups)} matches; copies removed {len(dup_drop_lowest)}")
    check("duplicate rounds removed (after six predicates)", dup_rounds_bakeoff, EXPECT["dup"])

    # ---------- step 1: order dependence of the six predicates
    print("\n== 1. Order dependence (720 orders of the six predicates) ==")
    pattern = collections.Counter(tuple(flags[k][p] for p in PRED) for k in rd)

    def attribute(order, pat):
        c = collections.Counter()
        for vec, n in pat.items():
            if not any(vec):
                continue
            d = dict(zip(PRED, vec))
            for p in order:
                if d[p]:
                    c[p] += n
                    break
        return c

    vectors = collections.Counter()
    ranges = {p: [10 ** 9, -1] for p in PRED}
    sample_orders = {}
    for order in itertools.permutations(PRED):
        c = attribute(order, pattern)
        vec = tuple(c[p] for p in PRED)
        vectors[vec] += 1
        for p in PRED:
            ranges[p][0] = min(ranges[p][0], c[p])
            ranges[p][1] = max(ranges[p][1], c[p])
        sample_orders.setdefault(vec, " > ".join(order))
    print(f"  distinct count vectors (pc, 5v5, disc, nort, nolive, null): {len(vectors)}")
    for p in PRED:
        print(f"  {p:<8} min {ranges[p][0]:>6}  max {ranges[p][1]:>6}")
    base_vec = tuple(attribute(BAKEOFF_ORDER, pattern)[p] for p in PRED)
    print(f"  bake-off order vector: {dict(zip(PRED, base_vec))}")
    RESULTS["order_ranges"] = {p: ranges[p] for p in PRED}
    RESULTS["order_distinct_vectors"] = len(vectors)
    RESULTS["order_examples"] = {" | ".join(map(str, k)): v for k, v in list(sample_orders.items())[:8]}

    marg = {p: sum(1 for k in rd if flags[k][p]) for p in PRED}
    uniq = {p: sum(1 for k in rd if flags[k][p] and sum(flags[k].values()) == 1) for p in PRED}
    print("  marginal (any failure) and unique (only this predicate fails) counts:")
    for p in PRED:
        print(f"    {p:<8} marginal {marg[p]:>6}  unique {uniq[p]:>6}")
    RESULTS["marginal"] = marg
    RESULTS["unique"] = uniq

    # overlap of pc, disc and 5v5
    print("  overlap (round sets, all channel-kept rounds):")
    for a, b in [("pc", "disc"), ("pc", "5v5"), ("5v5", "disc")]:
        both = sum(1 for k in rd if flags[k][a] and flags[k][b])
        print(f"    {a} & {b}: {both}")

    # ---------- step 2: pipeline variants (full re-run)
    def pipeline(name, order=BAKEOFF_ORDER, disc_rule="any", dedup_rule="lowest", dup_first=False):
        disc_set = disc_any if disc_rule == "any" else disc_live
        pool = list(rd)
        dup_pre = set()
        if dup_first:
            # duplicate step before the round filters: choose the copy by the same rule
            dup_pre = _dup_drop(dedup_rule, pool_survivors=None)
            pool = [k for k in pool if k[0] not in dup_pre]
        counts = collections.Counter()
        surv = []
        for k in pool:
            f = dict(flags[k])
            f["disc"] = k in disc_set
            for p in order:
                if f[p]:
                    counts[p] += 1
                    break
            else:
                surv.append(k)
        if dup_first:
            counts["dup"] = sum(1 for k in rd if k[0] in dup_pre)
        else:
            dup_post = _dup_drop(dedup_rule, pool_survivors=surv)
            counts["dup"] = sum(1 for k in surv if k[0] in dup_post)
            surv = [k for k in surv if k[0] not in dup_post]
        # unresolved victim side check and states
        n_states = 0
        n_live_states = 0
        unres = 0
        kept = []
        for k in surv:
            rr = rd[k]
            frz = live[k]["tick"]
            end = rr["end"]
            pl_ = players.get(k, {})
            seen = set()
            bad = False
            ndeaths = 0
            for d in deaths.get(k, []):
                if d["tick"] < frz or d["tick"] > end:
                    continue
                side = d["team"]
                if side is None:
                    side = pl_.get(d["pid"])
                if side not in (T, CT):
                    bad = True
                    break
                if d["pid"] in seen:
                    continue
                seen.add(d["pid"])
                ndeaths += 1
            if bad:
                unres += 1
                continue
            kept.append(k)
            n_states += 1 + ndeaths
        counts["unresolved"] = unres
        kept_matches_final = {k[0] for k in kept}
        RESULTS.setdefault("pipelines", {})[name] = {
            "dropped": dict(counts), "kept_rounds": len(kept), "kept_matches": len(kept_matches_final),
            "states": n_states}
        print(f"  [{name}] dropped {dict(counts)}; kept rounds {len(kept)}, matches "
              f"{len(kept_matches_final)}, states {n_states}")
        return kept

    def _dup_drop(rule, pool_survivors):
        """Copies to drop. 'lowest': keep the lowest key per group (bake-off). 'best': keep the copy
        with the most surviving rounds (ties to lowest key). pool_survivors None -> rounds are not
        filtered yet (dup-first variant), so the copy count is by all channel-kept rounds."""
        if rule == "lowest":
            return set(dup_drop_lowest)
        per_match = collections.Counter()
        src = list(rd) if pool_survivors is None else pool_survivors
        for k in src:
            per_match[k[0]] += 1
        drop = set()
        for g in dedup_groups:
            gs = sorted(g)
            best = max(gs, key=lambda m: (per_match.get(m, 0), -gs.index(m)))
            drop.update(m for m in gs if m != best)
        return drop

    print("\n== 2. Pipeline variants ==")
    base_kept = pipeline("baseline")
    check("kept rounds (baseline)", len(base_kept), EXPECT["kept_rounds"])
    check("kept matches (baseline)", len({k[0] for k in base_kept}), EXPECT["kept_matches"])
    check("states (baseline)", RESULTS["pipelines"]["baseline"]["states"], EXPECT["states"])
    pipeline("disc_live_window_only", disc_rule="live")
    pipeline("dedup_best_representative", dedup_rule="best")
    pipeline("dup_step_first", dup_first=True)
    pipeline("reversed_order", order=list(reversed(BAKEOFF_ORDER)))
    pipeline("no_pc_rule", order=["5v5", "disc", "nort", "nolive", "null"])
    pipeline("disc_first", order=["disc", "pc", "5v5", "nort", "nolive", "null"])
    pipeline("5v5_first", order=["5v5", "pc", "disc", "nort", "nolive", "null"])

    # ---------- step 3: duplicate audit
    print("\n== 3. Duplicate audit ==")
    surv_by_match = collections.defaultdict(set)
    for (m, r) in survivors6:
        surv_by_match[m].add(r)
    rounds_by_match = collections.defaultdict(set)
    for (m, r) in rd:
        rounds_by_match[m].add(r)
    rep_empty_groups = 0
    rep_empty_rounds_lost = 0
    groups_with_flag_disagree = 0
    round_pairs_disagree = 0
    disagree_pred = collections.Counter()
    round_count_mismatch = 0
    group_sizes = collections.Counter(len(g) for g in dedup_groups)
    dropped_copy_with_rounds = 0
    for g in dedup_groups:
        rep = g[0]
        n_surv = {m: len(surv_by_match.get(m, ())) for m in g}
        if n_surv[rep] == 0 and any(n_surv[m] > 0 for m in g[1:]):
            rep_empty_groups += 1
            rep_empty_rounds_lost += sum(n_surv[m] for m in g[1:])
        dropped_copy_with_rounds += sum(1 for m in g[1:] if n_surv[m] > 0)
        if len({len(rounds_by_match[m]) for m in g}) > 1:
            round_count_mismatch += 1
        disagree_here = False
        for m in g[1:]:
            common = rounds_by_match[rep] & rounds_by_match[m]
            for r in common:
                a, b = flags[(rep, r)], flags[(m, r)]
                diffs = [p for p in PRED if a[p] != b[p]]
                if diffs:
                    disagree_here = True
                    round_pairs_disagree += 1
                    for p in diffs:
                        disagree_pred[p] += 1
        if disagree_here:
            groups_with_flag_disagree += 1
    print(f"  group sizes: {dict(sorted(group_sizes.items()))}")
    print(f"  groups where the lowest-key representative has 0 surviving rounds but a copy has some: "
          f"{rep_empty_groups} groups, {rep_empty_rounds_lost} surviving rounds of copies dropped with them")
    print(f"  groups whose members have different round counts in round_end: {round_count_mismatch}")
    print(f"  groups with per-round predicate disagreement between copies: {groups_with_flag_disagree}; "
          f"round pairs disagreeing: {round_pairs_disagree}; by predicate: {dict(disagree_pred)}")
    print(f"  copies (not representatives) that had surviving rounds: {dropped_copy_with_rounds}")
    RESULTS["dup_audit"] = {"rep_empty_groups": rep_empty_groups,
                            "rep_empty_rounds_lost": rep_empty_rounds_lost,
                            "groups_with_flag_disagree": groups_with_flag_disagree,
                            "round_pairs_disagree": round_pairs_disagree,
                            "disagree_by_pred": dict(disagree_pred),
                            "round_count_mismatch_groups": round_count_mismatch,
                            "copies_with_surviving_rounds": dropped_copy_with_rounds}

    # fingerprint (round_end only, >=5 rounds) across all matches, vs signature groups
    per_match_rounds = collections.defaultdict(list)
    for x in RE.iter_rows(named=True):
        per_match_rounds[x["match_key"]].append((int(x["round"]), int(x["tick"]), int(x["winner_team_code"])))
    fps = collections.defaultdict(list)
    for m, lst in per_match_rounds.items():
        if len(lst) >= 5:
            fps[tuple(sorted(lst))].append(m)
    fp_groups = [v for v in fps.values() if len(v) > 1]
    fp_matches = sum(len(v) for v in fp_groups)
    sig_all = collections.defaultdict(list)
    for m in all_matches:
        sig_all[sigs.get(m, ())].append(m)
    sig_all_groups = [g for g in sig_all.values() if len(g) > 1]
    fp_only = []
    for v in fp_groups:
        sigset = {sigs.get(m, ()) for m in v}
        if len(sigset) > 1:
            fp_only.append(v)
    print(f"  fingerprint groups (round_end only, >=5 rounds, all matches): {len(fp_groups)} groups, "
          f"{fp_matches} matches; signature groups (all matches): {len(sig_all_groups)} groups, "
          f"{sum(len(g) for g in sig_all_groups)} matches")
    print(f"  fingerprint groups whose members do not share one signature: {len(fp_only)}")
    for v in fp_only[:5]:
        print("    members:", [(m[:45], m in kept_matches) for m in v])
    dropped_in_sig = [g for g in sig_all_groups if any(m not in kept_matches for m in g)]
    print(f"  signature groups (all matches) with a member dropped at the channel stage: {len(dropped_in_sig)}")
    RESULTS["fingerprint"] = {"fp_groups": len(fp_groups), "fp_matches": fp_matches,
                              "sig_groups_all": len(sig_all_groups),
                              "fp_only_groups": len(fp_only),
                              "sig_groups_with_channel_dropped": len(dropped_in_sig)}

    # straddle check (bake-off computes this on the pre-duplicate match set)
    pre_final = {k[0] for k in survivors6}
    print(f"  matches with no round after the six predicates: {len(kept_matches - pre_final)} (bake-off log 141)")
    print(f"  copies removed with no surviving round: {len(dup_drop_lowest) - dropped_copy_with_rounds}")
    pre_sets = bk.make_splits(pre_final, H)
    straddle = {PRIMARY: 0, CHRONO: 0}
    for g in dedup_groups:
        members = [m for m in g if m in pre_final]
        if len(members) < 2:
            continue
        for name in (PRIMARY, CHRONO):
            parts = {part for m in members for part in ("train", "val", "test")
                     if m in pre_sets[name][part]}
            if len(parts) > 1:
                straddle[name] += 1
    print(f"  groups straddling the pre-duplicate split (primary, chrono): {straddle[PRIMARY]}, {straddle[CHRONO]}")
    RESULTS["straddle"] = straddle

    # ---------- step 4: disconnect and takeover timing
    print("\n== 4. Disconnect and bot_takeover timing (kept matches) ==")
    cls_counts = collections.Counter()
    for (name, cls), n in sorted(ev_class_rows.items()):
        cls_counts[(name, cls)] = n
        print(f"  {name:<10} {cls:<24} events {n}")
    lw = collections.Counter()
    for k in disc_any:
        if k not in rd:
            lw["disc round not in round_end"] += 1
            continue
        cls = ev_class_round[k]
        if "live" in cls:
            lw["has live-window event"] += 1
        else:
            lw["only outside live window"] += 1
    print(f"  rounds with a disconnect/takeover event (any, in round_end): {sum(1 for k in disc_any if k in rd)}")
    for n, v in lw.most_common():
        print(f"    {n}: {v}")
    pat = collections.Counter()
    for k in disc_any:
        if k in rd:
            pat[tuple(sorted(ev_class_round[k]))] += 1
    print("  class patterns per flagged round (events in the round):")
    for p, n in pat.most_common():
        print(f"    {p}: {n}")
    # bake-off rule restricted to rounds that pass pc and 5v5 (the rounds where disc is applied)
    eligible = [k for k in rd if not flags[k]["pc"] and not flags[k]["5v5"]]
    disc_on_elig = [k for k in eligible if k in disc_any]
    disc_live_on_elig = [k for k in eligible if k in disc_live]
    print(f"  rounds passing pc and 5v5: {len(eligible)}; flagged by disc (any): {len(disc_on_elig)}; "
          f"flagged by disc (live window only): {len(disc_live_on_elig)}; "
          f"flagged only by outside-window events: {len(disc_on_elig) - len(disc_live_on_elig)}")
    after_arr = sorted(after_offsets)
    before_arr = sorted(before_offsets)
    if after_arr:
        q = lambda a, p: a[min(len(a) - 1, int(p * len(a)))]
        print(f"  after_round_end: n={len(after_arr)} seconds after round_end min {after_arr[0]:.2f} "
              f"median {q(after_arr, .5):.2f} p90 {q(after_arr, .9):.2f} max {after_arr[-1]:.2f}")
    if before_arr:
        q = lambda a, p: a[min(len(a) - 1, int(p * len(a)))]
        print(f"  before_round_start: n={len(before_arr)} seconds before round_start min {before_arr[0]:.2f} "
              f"median {q(before_arr, .5):.2f} p90 {q(before_arr, .9):.2f} max {before_arr[-1]:.2f}")
    print("  disconnect event detail by class (is_bot, reason):")
    for cls in sorted(ev_detail):
        top = ev_detail[cls].most_common(6)
        print(f"    {cls}: {top}")
    RESULTS["disc_timing"] = {"rounds_flagged_any": len(disc_on_elig), "rounds_flagged_live": len(disc_live_on_elig),
                              "class_patterns": {str(p): n for p, n in pat.most_common()},
                              "event_rows": {f"{a}|{b}": n for (a, b), n in ev_class_rows.items()}}

    # ---------- step 5: not-5v5 and player_count breakdowns
    print("\n== 5. Breakdowns ==")
    not5 = [k for k in rd if not flags[k]["pc"] and flags[k]["5v5"]]
    shape = collections.Counter((info[k]["len_pl"], info[k]["n_t"], info[k]["n_ct"]) for k in not5)
    print(f"  not-5v5 rounds (pc==10): {len(not5)}; (len, n_T, n_CT) from player_info:")
    for s, n in shape.most_common(12):
        print(f"    {s}: {n}")
    tc_not5 = sum(1 for k in not5 if k in tc_rounds)
    pass_pc_5v5 = [k for k in rd if not flags[k]["pc"] and not flags[k]["5v5"]]
    tc_pass = sum(1 for k in pass_pc_5v5 if k in tc_rounds)
    print(f"  not-5v5 rounds with any team_change in round: {tc_not5} of {len(not5)}; "
          f"among rounds passing pc and 5v5: {tc_pass} of {len(pass_pc_5v5)}")
    zero_team = sum(1 for k in not5 if any(v == 0 for v in players.get(k, {}).values()))
    print(f"  not-5v5 rounds with a player_info row of team 0: {zero_team}")
    pcs = collections.Counter(rd[k]["pc"] for k in rd if flags[k]["pc"])
    print(f"  player_count values among the pc-failing rounds: {dict(sorted(pcs.items()))}")
    pc_disc = sum(1 for k in rd if flags[k]["pc"] and k in disc_any)
    print(f"  pc-failing rounds that also have a disconnect/takeover event (any): {pc_disc} of {reasons['pc']} "
          f"(player_count != 10 is first in the bake-off order, so these count as pc)")
    pc10_disc = sum(1 for k in rd if not flags[k]["pc"] and k in disc_any)
    print(f"  player_count == 10 rounds with a disconnect/takeover event (any): {pc10_disc}")
    RESULTS["breakdown"] = {"not5v5_shape": {str(s): n for s, n in shape.most_common(12)},
                            "not5v5_tc_round": tc_not5, "pc_values": dict(pcs),
                            "pc_with_disc": pc_disc, "pc10_with_disc": pc10_disc}

    out_path = SCRATCH / "A4_filters_results.json"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(RESULTS, indent=1, default=str), encoding="utf-8")
        print(f"\nwrote {out_path}")
    except OSError as e:
        print(f"\ncould not write scratch json: {e}")
    print("done")


if __name__ == "__main__":
    main()
