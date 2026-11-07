"""Leakage lens for the buy-type pilot.

Re-derives the leakage checks from data\\state_sample\\players_1hz.parquet and rounds.parquet and
compares them with data\\state_sample\\round_buy_types.parquet. The pilot and probe scripts are read
as text only (AST and regex). The only file written is logs\\buy_types\\verify_leakage.txt.

Run from the repo root:
  pixi run python -I -X utf8 scripts\\eda\\verify_buy_leakage.py
"""

import ast
import random
import re
import sys
from collections import Counter
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
STATE = ROOT / "data" / "state_sample"
PLAYERS = STATE / "players_1hz.parquet"
ROUNDS = STATE / "rounds.parquet"
BUYS = STATE / "round_buy_types.parquet"
EDA = ROOT / "scripts" / "eda"
LOG_FILE = ROOT / "logs" / "buy_types" / "verify_leakage.txt"

T, CT = 2, 3
KEY = ["match_key", "round"]
PISTOL = (1, 13)
ECO_MAX, FULL_MIN = 10_000, 20_000  # default set, typed here so this check does not import the pilot
TICK_STEP = 64
SPOT_SEED = 20261013
N_SPOT = 10
C4_TIMER_S = 40.0  # CS2 bomb timer is 40 s after plant; an explosion ends the round at about this gap
OT_FIRST_ROUND = 25
FIELDS = ["ct_equipment", "t_equipment", "ct_money", "t_money", "ct_buy", "t_buy", "round_class",
          "is_pistol", "ct_win", "split", "higher_side", "higher_side_won"]
PAIR_TO_CLASS = {
    ("eco", "eco"): "eco_vs_eco",
    ("eco", "full"): "anti_eco",
    ("eco", "force"): "force_vs_eco",
    ("force", "full"): "full_vs_force",
    ("force", "force"): "force_vs_force",
    ("full", "full"): "full_vs_full",
}
# Fields that exist only after freeze end or are outcomes. A class must not read them.
LIVE_OR_OUTCOME = {
    "plant_tick", "round_end_tick", "map_name", "player_death", "winner", "winner_team_code",
    "bomb_planted", "alive_gap_snapshots", "round_end", "x_pos", "y_pos", "z_pos", "health",
    "armor", "weapon_code", "is_defusing", "seconds_since_live", "round_number", "round_end_tick",
}

lines = []


def emit(s=""):
    print(s)
    lines.append(s)


def side_class(eq):
    if eq < ECO_MAX:
        return "eco"
    if eq < FULL_MIN:
        return "force"
    return "full"


def round_class(rnd, ct_b, t_b):
    if rnd in PISTOL:
        return "pistol"
    return PAIR_TO_CLASS[tuple(sorted((ct_b, t_b)))]


def higher(ct_eq, t_eq):
    if ct_eq > t_eq:
        return "CT"
    if t_eq > ct_eq:
        return "T"
    return "tie"


def side_table(df):
    agg = df.group_by(KEY + ["team_code"]).agg(
        pl.len().alias("n"),
        pl.col("equipment_value_calc").cast(pl.Int64).sum().alias("eq"),
        pl.col("money").cast(pl.Int64).sum().alias("mny"),
    )
    out = {}
    for r in agg.iter_rows(named=True):
        out.setdefault((r["match_key"], r["round"]), {})[int(r["team_code"])] = (
            int(r["n"]), int(r["eq"] or 0), int(r["mny"] or 0))
    return out


def side_of(tbl, k, team):
    return tbl.get(k, {}).get(team, (0, 0, 0))


def ast_string_refs(src, names):
    """(line, token) for each code-level string or identifier in `names`. Docstring skipped."""
    tree = ast.parse(src)
    doc = None
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        doc = tree.body[0].value
    hits = []
    for node in ast.walk(tree):
        if node is doc:
            continue
        tok = None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            tok = node.value
        elif isinstance(node, ast.Name):
            tok = node.id
        elif isinstance(node, ast.Attribute):
            tok = node.attr
        if tok is not None and tok in names:
            hits.append((node.lineno, tok))
    return hits


def read_parquet_calls(src):
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "read_parquet":
            strs = sorted({c.value for c in ast.walk(node)
                           if isinstance(c, ast.Constant) and isinstance(c.value, str)})
            out.append((node.lineno, strs))
    return out


def alt_snapshot_check(tbl, buy_rows, fz_tbl):
    absent = missing = changed_side = changed_round = eq_moved = 0
    for r in buy_rows:
        k = (r["match_key"], r["round"])
        if k not in tbl:
            missing += 1
            continue
        ct_n, cteq, _ = side_of(tbl, k, CT)
        t_n, teq, _ = side_of(tbl, k, T)
        if ct_n == 0 or t_n == 0:
            absent += 1
        cb, tb = side_class(cteq), side_class(teq)
        if (cb, tb) != (r["ct_buy"], r["t_buy"]):
            changed_side += 1
        if round_class(r["round"], cb, tb) != r["round_class"]:
            changed_round += 1
        if (cteq, teq) != (side_of(fz_tbl, k, CT)[1], side_of(fz_tbl, k, T)[1]):
            eq_moved += 1
    return {"no_snapshot": missing, "side_absent": absent, "side_class_changes": changed_side,
            "round_class_changes": changed_round, "equipment_differs_from_freeze": eq_moved}


def main():
    results = {}
    emit("verify_buy_leakage: leakage lens for the buy-type pilot")
    emit(f"inputs: {PLAYERS.name}, {ROUNDS.name}, {BUYS.name} under {STATE}")

    # ---- 0. Inputs --------------------------------------------------------------------
    pcols = pq.read_schema(PLAYERS).names
    rcols = pq.read_schema(ROUNDS).names
    bcols = pq.read_schema(BUYS).names
    emit(f"players_1hz columns ({len(pcols)}): {pcols}")
    emit(f"rounds columns ({len(rcols)}): {rcols}")
    emit(f"round_buy_types columns ({len(bcols)}): {bcols}")
    emit(f"rounds.parquet holds ct_score and t_score: {'ct_score' in rcols} and {'t_score' in rcols}; "
         "the pilot must not read them (checked in section 3)")

    players = pl.read_parquet(PLAYERS).with_columns(
        pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
    rounds = pl.read_parquet(ROUNDS).with_columns(pl.col("round").cast(pl.Int64))
    buys = pl.read_parquet(BUYS).with_columns(pl.col("round").cast(pl.Int64))
    emit(f"rows: players {players.height:,} | rounds {rounds.height:,} | round_buy_types {buys.height:,}")
    emit(f"duplicate (match_key, round): rounds {rounds.height - rounds.select(KEY).unique().height} | "
         f"round_buy_types {buys.height - buys.select(KEY).unique().height}")

    # ---- 1. Freeze-end snapshot only -------------------------------------------------
    emit()
    emit("== 1. Freeze-end snapshot only (no live, end, plant or death data) ==")
    fmin = players.group_by(KEY).agg(pl.col("tick").min().alias("ft"))
    rnd_small = rounds.select(KEY + ["live_start_tick", "round_end_tick"])
    pj = (players.join(rnd_small, on=KEY, how="left")
          .join(fmin, on=KEY, how="left")
          .with_columns((pl.col("tick") - pl.col("live_start_tick")).alias("d")))
    rounds_no_players = rounds.join(fmin, on=KEY, how="anti").height
    players_no_round = fmin.join(rounds.select(KEY), on=KEY, how="anti").height
    rows_no_round = pj.filter(pl.col("live_start_tick").is_null()).height
    rows_before_live = pj.filter(pl.col("d") < 0).height
    rows_off_grid = pj.filter(pl.col("d") % TICK_STEP != 0).height
    rows_at_or_after_end = pj.filter(pl.col("tick") >= pl.col("round_end_tick")).height
    ssl_bad = pj.filter(pl.col("seconds_since_live") != pl.col("d") // TICK_STEP).height
    emit(f"player rows {pj.height:,} | player rows with no round in rounds.parquet: {rows_no_round} | "
         f"rounds with no player rows: {rounds_no_players} | player rounds with no rounds row: {players_no_round}")
    emit(f"rows before live_start_tick: {rows_before_live} | rows off the live_start+64k grid: {rows_off_grid} | "
         f"rows at or after round_end_tick: {rows_at_or_after_end} | seconds_since_live mismatches: {ssl_bad}")

    rl = fmin.join(rnd_small, on=KEY, how="inner")
    n_eq = rl.filter(pl.col("ft") == pl.col("live_start_tick")).height
    n_lt = rl.filter(pl.col("ft") < pl.col("live_start_tick")).height
    n_gt = rl.filter(pl.col("ft") > pl.col("live_start_tick")).height
    emit(f"first-snapshot tick vs live_start_tick over {rl.height} rounds: equal {n_eq} | "
         f"strictly before {n_lt} | strictly after {n_gt}")

    fz_rows = pj.filter(pl.col("tick") == pl.col("ft"))
    np_dist = Counter(pj.filter(pl.col("d") == 0).group_by(KEY).agg(pl.len().alias("np"))["np"].to_list())
    emit(f"players at the freeze snapshot (players: rounds): {dict(sorted(np_dist.items()))}")
    emit(f"null equipment_value_calc at freeze: {fz_rows['equipment_value_calc'].null_count()} | "
         f"null money at freeze: {fz_rows['money'].null_count()}")

    fz_tbl = side_table(fz_rows)
    s2 = pj.filter(pl.col("tick") > pl.col("ft")).group_by(KEY).agg(pl.col("tick").min().alias("t1"))
    sec_rows = pj.join(s2, on=KEY, how="inner").filter(pl.col("tick") == pl.col("t1"))
    tl = pj.group_by(KEY).agg(pl.col("tick").max().alias("tl"))
    last_rows = pj.join(tl, on=KEY, how="inner").filter(pl.col("tick") == pl.col("tl"))
    buy_rows = buys.to_dicts()
    alt2 = alt_snapshot_check(side_table(sec_rows), buy_rows, fz_tbl)
    altL = alt_snapshot_check(side_table(last_rows), buy_rows, fz_tbl)
    emit("sensitivity (not a pass/fail check): classes recomputed from another snapshot, kept rounds")
    emit(f"  second snapshot (live second 1): {alt2}")
    emit(f"  last snapshot before round end:  {altL}")
    results["c1_counts"] = {
        "rows_before_live": rows_before_live, "rows_off_grid": rows_off_grid,
        "rows_at_or_after_end": rows_at_or_after_end, "ssl_bad": ssl_bad,
        "n_eq": n_eq, "n_gt": n_gt, "rounds": rl.height, "rows_no_round": rows_no_round,
        "rounds_no_players": rounds_no_players, "players_no_round": players_no_round,
        "np_dist": dict(np_dist),
    }
    results["alt2"] = alt2
    results["altL"] = altL

    # Code-level: what the pilot reads and whether any live or outcome token appears in code.
    pilot_src = (EDA / "buy_type_pilot.py").read_text(encoding="utf-8")
    probe_src = (EDA / "buy_type_probe.py").read_text(encoding="utf-8")
    emit()
    emit("pilot read_parquet calls (AST, code only):")
    for ln, strs in read_parquet_calls(pilot_src):
        emit(f"  line {ln}: {strs}")
    live_hits = ast_string_refs(pilot_src, LIVE_OR_OUTCOME)
    emit(f"pilot code references to live or outcome fields (plant, end, deaths, winner, positions, health, etc.): {len(live_hits)}")
    for ln, tok in live_hits:
        emit(f"  line {ln}: {tok}")
    results["pilot_live_hits"] = len(live_hits)

    # ---- 2. Freeze boundary, literal wording -------------------------------------------
    emit()
    emit("== 2. Freeze boundary, literal wording of check 2 ==")
    emit(f"literal 'freeze-end tick is before live_start_tick': true for {n_lt} of {rl.height} rounds")
    emit(f"equal for {n_eq} of {rl.height} rounds. Equality is by construction: extract_state_1hz.py sets "
         "live_start_tick = fz (line 289), the freeze-end tick (line 278), and snapshots start at fz (line 290).")
    results["c2_literal_pass_count"] = n_lt

    # ---- 3. Sides and labels --------------------------------------------------------------
    emit()
    emit("== 3. Sides from team_code; label ct_win from rounds.parquet; no score fields ==")
    tc_vals = sorted(players["team_code"].unique().to_list())
    emit(f"team_code values in players_1hz: {tc_vals} (2 = T, 3 = CT)")
    ct_fz = fz_rows.filter(pl.col("team_code") == CT)
    t_fz = fz_rows.filter(pl.col("team_code") == T)
    t_def = t_fz.filter(pl.col("has_defuser")).height
    ct_def_share = ct_fz["has_defuser"].mean()
    ct_c4 = ct_fz.filter(pl.col("inv_c4") > 0).height
    holders = t_fz.filter(pl.col("inv_c4") > 0).group_by(KEY).agg(pl.len().alias("h"))
    rl_h = rl.join(holders, on=KEY, how="left").with_columns(pl.col("h").fill_null(0))
    h_dist = dict(sorted(Counter(rl_h["h"].to_list()).items()))
    emit(f"freeze side check: T players with defuser {t_def} (expect 0) | CT players with defuser share "
         f"{ct_def_share:.3f} | CT players holding C4 {ct_c4} (expect 0) | T C4 holders per round (holders: rounds) {h_dist}")
    cw_vals = sorted(rounds["ct_win"].unique().to_list())
    emit(f"ct_win values in rounds.parquet: {cw_vals} | nulls {rounds['ct_win'].null_count()} "
         f"| CT win share {rounds['ct_win'].mean():.4f}")
    pr = rounds.filter(pl.col("plant_tick").is_not_null()).with_columns(
        ((pl.col("round_end_tick") - pl.col("plant_tick")) / TICK_STEP).alias("gap_s"))
    for w in (0, 1):
        g = pr.filter(pl.col("ct_win") == w)
        if g.height:
            emit(f"  plant rounds with ct_win={w}: {g.height} | seconds from plant to round end: "
                 f"min {g['gap_s'].min():.2f} median {g['gap_s'].median():.2f} max {g['gap_s'].max():.2f}")
    after_timer = pr.filter(pl.col("gap_s") > C4_TIMER_S)
    long_ct_win = after_timer.filter(pl.col("ct_win") == 1).height
    explode_like = pr.filter((pl.col("gap_s") >= 39.9) & (pl.col("ct_win") == 0)).height
    emit(f"  CT wins after the {C4_TIMER_S:.0f} s bomb timer (gap > {C4_TIMER_S:.0f} s): {long_ct_win} "
         f"of {after_timer.height} plant rounds past the timer (should be 0: a CT win after the timer is impossible)")
    emit(f"  T wins at a gap of 39.9 s or more after plant (explosion-like): {explode_like}")
    results["label_after_timer"] = after_timer.height
    results["label_after_timer_ct_win1"] = long_ct_win
    results["label_explode_like_t_win"] = explode_like

    src_hits = ast_string_refs(pilot_src, {"ct_score", "t_score"})
    emit(f"pilot code references to ct_score or t_score (AST, docstring skipped): {len(src_hits)}")
    for ln, tok in src_hits:
        emit(f"  line {ln}: {tok}")
    probe_score_hits = ast_string_refs(probe_src, {"ct_score", "t_score"})
    emit(f"probe code references to ct_score or t_score: {len(probe_score_hits)} (probe is not an output of this build)")
    results["pilot_score_hits"] = len(src_hits)

    # ---- 4. Recompute from players_1hz; spot check and full population ----------------------
    emit()
    emit(f"== 4. Recompute from players_1hz (first snapshot per round); spot check {N_SPOT} rounds, seed {SPOT_SEED}; all rounds ==")
    fs = fz_tbl
    rd_map = {(r["match_key"], r["round"]): r for r in rounds.select(KEY + ["ct_win", "split"]).to_dicts()}

    def expect(k):
        ctn, cteq, ctm = side_of(fs, k, CT)
        tn, teq, tm = side_of(fs, k, T)
        cb, tb = side_class(cteq), side_class(teq)
        hs = higher(cteq, teq)
        cw = rd_map[k]["ct_win"]
        hsw = None if hs == "tie" else ((cw == 1) if hs == "CT" else (cw == 0))
        return {"ct_n": ctn, "t_n": tn, "ct_equipment": cteq, "t_equipment": teq,
                "ct_money": ctm, "t_money": tm, "ct_buy": cb, "t_buy": tb,
                "round_class": round_class(k[1], cb, tb), "is_pistol": k[1] in PISTOL,
                "ct_win": cw, "split": rd_map[k]["split"], "higher_side": hs, "higher_side_won": hsw}

    detail = {}
    mism = Counter()
    rows_bad = 0
    n55 = 0
    for r in buy_rows:
        k = (r["match_key"], r["round"])
        e = expect(k)
        if e["ct_n"] == 5 and e["t_n"] == 5:
            n55 += 1
        bad = [f for f in FIELDS if r[f] != e[f]]
        for f in bad:
            mism[f] += 1
        if bad:
            rows_bad += 1
        detail[k] = (r, e, bad)
    emit(f"rounds in round_buy_types with exactly 5 players per side at freeze: {n55} of {buys.height}")
    emit(f"full population: rounds with any field mismatch {rows_bad} of {buys.height}; "
         f"per-field mismatches: {dict(mism) if mism else 'none'}")
    results["full_rows_bad"] = rows_bad
    results["full_mism"] = dict(mism)

    keys_sorted = sorted(detail.keys())
    rng = random.Random(SPOT_SEED)
    spot = rng.sample(keys_sorted, N_SPOT)
    spot_bad = 0
    emit(f"spot check ({N_SPOT} rounds, random.Random({SPOT_SEED}).sample over sorted (match_key, round)):")
    for k in spot:
        r, e, bad = detail[k]
        if bad:
            spot_bad += 1
        uuid_part = k[0].split("/")[-2][-12:]
        emit(f"  {uuid_part} r{k[1]:>2} | CT n={e['ct_n']} eq={e['ct_equipment']} money={e['ct_money']} -> {e['ct_buy']} "
             f"(file {r['ct_buy']}) | T n={e['t_n']} eq={e['t_equipment']} money={e['t_money']} -> {e['t_buy']} "
             f"(file {r['t_buy']}) | class {e['round_class']} (file {r['round_class']}) | ct_win {e['ct_win']} | "
             f"{'MATCH' if not bad else 'MISMATCH ' + str(bad)}")
    emit(f"spot check mismatches: {spot_bad} of {N_SPOT}")
    results["spot_bad"] = spot_bad

    # ---- 5. Coverage -----------------------------------------------------------------------
    emit()
    emit("== 5. Coverage: rounds.parquet vs round_buy_types ==")
    miss = rounds.join(buys.select(KEY), on=KEY, how="anti")
    for r in miss.iter_rows(named=True):
        k = (r["match_key"], r["round"])
        emit(f"  in rounds.parquet, not in round_buy_types: {r['match_key']} round {r['round']} | "
             f"CT n={side_of(fs, k, CT)[0]} | T n={side_of(fs, k, T)[0]} | live_start_tick {r['live_start_tick']}")
    extra = buys.join(rounds.select(KEY), on=KEY, how="anti").height
    emit(f"rounds in round_buy_types missing from rounds.parquet: {extra}")
    results["coverage_missing"] = miss.height
    results["coverage_extra"] = extra

    # ---- 6. Write targets ----------------------------------------------------------------
    emit()
    emit("== 6. Write targets (scripts under scripts\\eda) ==")
    E_PAT = re.compile(r"(?<![A-Za-z])[Ee]:[\\/]|csds_final|(?<![A-Za-z])/[eE]/")
    W_PAT = re.compile(r"write_parquet|write_text|write_csv|to_parquet|to_csv|\bopen\(|mkdir|json\.dump|savefig|\.write\(")
    P_PAT = re.compile(r"^(ROOT|STATE|PLAYERS|ROUNDS|OUT_PARQUET|LOG_DIR|OUT_JSON|OUT_TXT|LOG_FILE|E_CSDS)\s*=")
    e_total = 0
    for name in ["buy_type_pilot.py", "buy_type_probe.py"]:
        txt = (EDA / name).read_text(encoding="utf-8").splitlines()
        e_hits = [(i, ln.strip()) for i, ln in enumerate(txt, 1) if E_PAT.search(ln)]
        w_hits = [(i, ln.strip()) for i, ln in enumerate(txt, 1) if W_PAT.search(ln)]
        p_hits = [(i, ln.strip()) for i, ln in enumerate(txt, 1) if P_PAT.search(ln)]
        e_total += len(e_hits)
        emit(f"{name}: E: or csds_final references {len(e_hits)}")
        for i, ln in e_hits:
            emit(f"  line {i}: {ln}")
        emit(f"{name}: path constants")
        for i, ln in p_hits:
            emit(f"  line {i}: {ln}")
        emit(f"{name}: write or open calls {len(w_hits)}")
        for i, ln in w_hits:
            emit(f"  line {i}: {ln}")
    emit("verify_buy_leakage.py itself writes one file: LOG_FILE (logs\\buy_types\\verify_leakage.txt) via write_text, and reads only.")
    results["e_refs_in_scripts"] = e_total

    # ---- 7. Freeze-state anomaly vs outcome ----------------------------------------------
    emit()
    emit("== 7. Freeze-state anomaly vs outcome: T C4 holder present or absent (non-pistol kept rounds) ==")
    nonp = rl_h.filter(~pl.col("round").is_in(list(PISTOL))).join(
        rounds.select(KEY + ["ct_win"]), on=KEY, how="left")
    for lab, sub in [("T has C4 holder", nonp.filter(pl.col("h") > 0)),
                     ("T has no C4 holder", nonp.filter(pl.col("h") == 0))]:
        if sub.height:
            emit(f"  {lab}: rounds {sub.height} | CT win {100 * sub['ct_win'].mean():.1f}%")
    nonot = nonp.filter(pl.col("round") < OT_FIRST_ROUND)
    emit(f"  non-pistol, non-overtime rounds in rounds.parquet: {nonot.height} | with no T C4 holder at freeze: "
         f"{nonot.filter(pl.col('h') == 0).height} ({100 * nonot.filter(pl.col('h') == 0).height / nonot.height:.1f}%)")
    emit(f"  non-pistol rounds in rounds.parquet (incl. overtime): {nonp.height} | with no T C4 holder: "
         f"{nonp.filter(pl.col('h') == 0).height}")
    results["c4_none_rounds"] = nonp.filter(pl.col("h") == 0).height

    # ---- 8. Summary ----------------------------------------------------------------------
    emit()
    emit("== 8. Summary ==")
    c1 = (rows_before_live == 0 and rows_off_grid == 0 and rows_at_or_after_end == 0 and ssl_bad == 0
          and n_gt == 0 and n_eq == rl.height and rows_no_round == 0 and rounds_no_players == 0
          and players_no_round == 0 and live_hits == [])
    c2 = n_lt == rl.height
    c3 = (tc_vals == [T, CT] and t_def == 0 and ct_c4 == 0 and set(cw_vals) <= {0, 1}
          and len(src_hits) == 0 and long_ct_win == 0)
    c4 = (spot_bad == 0 and rows_bad == 0 and n55 == buys.height)
    c5 = (e_total == 0)
    emit(f"check 1 freeze-end only (no live, end, death, winner or plant fields; first snapshot = live start; grid and bounds hold): {'PASS' if c1 else 'FAIL'}")
    emit(f"check 2 literal 'freeze-end tick strictly before live_start_tick' for every round: {'PASS' if c2 else 'FAIL'} ({n_lt} of {rl.height} strictly before; intended boundary 'not after live start' holds: {n_gt == 0})")
    emit(f"check 3 sides from team_code, ct_win from rounds.parquet, no score fields in code: {'PASS' if c3 else 'FAIL'}")
    emit(f"check 4 spot check {N_SPOT} rounds: {spot_bad} mismatches; full recompute of {buys.height} rounds: {rows_bad} rounds with any mismatch: {'PASS' if c4 else 'FAIL'}")
    emit(f"check 5 writes only under G:, no E: references in pilot or probe: {'PASS' if c5 else 'FAIL'}")
    results["checks"] = {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit()
    emit(f"wrote {LOG_FILE}")


if __name__ == "__main__":
    main()
