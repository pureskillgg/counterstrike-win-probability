"""Buy-type pilot on the 1 Hz freeze-end sample (equipment and money only).

Freeze-end state = first snapshot of each (match_key, round). Per side (team_code 2 = T,
3 = CT): n_players, equipment_sum (sum of equipment_value_calc), money_sum (sum of money).
Kept rounds: both sides have exactly 5 players. Each side gets a buy class from its
equipment sum under three threshold sets. The round class comes from the unordered pair
of side classes, with pistol rounds (round 1 and 13) overriding. Win rates come from
rounds.parquet ct_win. 95% CIs come from a match-cluster bootstrap (1000 resamples, seed 0).

ct_score and t_score are never read.

Run from the repo root:
  pixi run python -I -X utf8 scripts\\eda\\buy_type_pilot.py
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
STATE = ROOT / "data" / "state_sample"
PLAYERS = STATE / "players_1hz.parquet"
ROUNDS = STATE / "rounds.parquet"
OUT_PARQUET = STATE / "round_buy_types.parquet"
LOG_DIR = ROOT / "logs" / "buy_types"
OUT_JSON = LOG_DIR / "summary.json"
OUT_TXT = LOG_DIR / "summary.txt"

KEY = ["match_key", "round"]
T_SIDE, CT_SIDE = 2, 3
N_PER_SIDE = 5
PISTOL_ROUNDS = (1, 13)
OT_FIRST_ROUND = 25
BOOT_B = 1000
BOOT_SEED = 0
FEW_CLUSTERS = 30  # classes with fewer matches than this get a flag (CI unreliable)

# Side equipment-sum cut-offs in dollars. Side class: eco if sum < eco_max; force if
# eco_max <= sum < full_min; full if sum >= full_min.
THRESHOLD_SETS = {
    "conservative": {
        "eco_max": 5000,
        "full_min": 25000,
        "reason": (
            "Narrow eco and full zones: 25,000 = 5 x $5,000, the top of the unverified "
            "full-buy band in the research; 5,000 = 5 x $1,000 has no research source."
        ),
    },
    "default": {
        "eco_max": 10000,
        "full_min": 20000,
        "reason": (
            "Per-player cut-offs from the opened switchblade guide (eco under $2,000, full "
            "$4,000 or more) times 5 players. The probe found no supported valley, so the "
            "research sets the cut, not the data."
        ),
    },
    "aggressive": {
        "eco_max": 10000,
        "full_min": 20000,
        "reason": (
            "Same per-player research cut-offs as default (eco under $2,000, full $4,000 "
            "or more) times 5 players."
        ),
    },
}
DEFAULT_SET = "default"
SET_ORDER = ["conservative", "default", "aggressive"]

BUY_CLASSES = ["eco", "force", "full"]
ALL_CLASSES = [
    "pistol",
    "eco_vs_eco",
    "anti_eco",
    "force_vs_eco",
    "full_vs_force",
    "force_vs_force",
    "full_vs_full",
]
PAIR_TO_CLASS = {
    ("eco", "eco"): "eco_vs_eco",
    ("eco", "full"): "anti_eco",
    ("eco", "force"): "force_vs_eco",
    ("force", "full"): "full_vs_force",
    ("force", "force"): "force_vs_force",
    ("full", "full"): "full_vs_full",
}

OUT_SCHEMA = {
    "match_key": pl.String,
    "round": pl.Int64,
    "split": pl.String,
    "is_pistol": pl.Boolean,
    "ct_equipment": pl.Int64,
    "t_equipment": pl.Int64,
    "ct_money": pl.Int64,
    "t_money": pl.Int64,
    "ct_buy": pl.String,
    "t_buy": pl.String,
    "round_class": pl.String,
    "ct_win": pl.Int8,
    "higher_side": pl.String,
    "higher_side_won": pl.Boolean,
}


def side_class(eq, eco_max, full_min):
    if eq < eco_max:
        return "eco"
    if eq < full_min:
        return "force"
    return "full"


def round_class(is_pistol, a, b):
    if is_pistol:
        return "pistol"
    key = tuple(sorted((a, b)))
    if key not in PAIR_TO_CLASS:
        raise ValueError(f"unmapped side-class pair {key}")
    return PAIR_TO_CLASS[key]


def higher_side(ct_eq, t_eq):
    if ct_eq > t_eq:
        return "CT"
    if t_eq > ct_eq:
        return "T"
    return "tie"


def boot_ci(wins, ns):
    """Match-cluster bootstrap for sum(wins)/sum(ns). Resamples clusters with replacement."""
    wins = np.asarray(wins, dtype=np.float64)
    ns = np.asarray(ns, dtype=np.float64)
    m = ns.size
    if m == 0 or ns.sum() == 0:
        return None, None, 0
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, m, size=(BOOT_B, m))
    ws = wins[idx].sum(axis=1)
    nn = ns[idx].sum(axis=1)
    keep = nn > 0
    rates = ws[keep] / nn[keep]
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return float(lo), float(hi), int(keep.sum())


def classify_rows(base_rows, eco_max, full_min):
    out = []
    for r in base_rows:
        ct_buy = side_class(r["ct_equipment"], eco_max, full_min)
        t_buy = side_class(r["t_equipment"], eco_max, full_min)
        is_pistol = r["round"] in PISTOL_ROUNDS
        rc = round_class(is_pistol, ct_buy, t_buy)
        hs = higher_side(r["ct_equipment"], r["t_equipment"])
        if hs == "CT":
            hsw = r["ct_win"] == 1
        elif hs == "T":
            hsw = r["ct_win"] == 0
        else:
            hsw = None
        out.append(
            {
                "match_key": r["match_key"],
                "round": r["round"],
                "split": r["split"],
                "is_pistol": is_pistol,
                "ct_equipment": r["ct_equipment"],
                "t_equipment": r["t_equipment"],
                "ct_money": r["ct_money"],
                "t_money": r["t_money"],
                "ct_buy": ct_buy,
                "t_buy": t_buy,
                "round_class": rc,
                "ct_win": r["ct_win"],
                "higher_side": hs,
                "higher_side_won": hsw,
            }
        )
    return out


def class_block(rows):
    matches = sorted({r["match_key"] for r in rows})
    n = len(rows)
    ct_w = dict.fromkeys(matches, 0)
    ct_n = dict.fromkeys(matches, 0)
    hs_w = dict.fromkeys(matches, 0)
    hs_n = dict.fromkeys(matches, 0)
    for r in rows:
        m = r["match_key"]
        ct_w[m] += r["ct_win"]
        ct_n[m] += 1
        if r["higher_side"] != "tie":
            hs_n[m] += 1
            hs_w[m] += 1 if r["higher_side_won"] else 0
    ct_wins = sum(ct_w.values())
    hs_nontie = sum(hs_n.values())
    hs_wins = sum(hs_w.values())
    ct_lo, ct_hi, ct_ok = boot_ci([ct_w[m] for m in matches], [ct_n[m] for m in matches])
    hs_lo, hs_hi, hs_ok = boot_ci([hs_w[m] for m in matches], [hs_n[m] for m in matches])
    return {
        "rounds": n,
        "matches": len(matches),
        "few_clusters": len(matches) < FEW_CLUSTERS,
        "overtime_rounds": sum(1 for r in rows if r["round"] >= OT_FIRST_ROUND),
        "ct_win_count": ct_wins,
        "ct_win_rate": (ct_wins / n) if n else None,
        "ct_win_ci95": [ct_lo, ct_hi],
        "ct_win_boot_usable": ct_ok,
        "higher_side_ties": n - hs_nontie,
        "higher_side_rounds": hs_nontie,
        "higher_side_wins": hs_wins,
        "higher_side_win_rate": (hs_wins / hs_nontie) if hs_nontie else None,
        "higher_side_win_ci95": [hs_lo, hs_hi],
        "higher_side_boot_usable": hs_ok,
    }


def side_buy_counts(rows, pistol_only=False):
    sel = [r for r in rows if r["is_pistol"] == pistol_only]
    ct = Counter(r["ct_buy"] for r in sel)
    t = Counter(r["t_buy"] for r in sel)
    return {
        "ct": {k: ct.get(k, 0) for k in BUY_CLASSES},
        "t": {k: t.get(k, 0) for k in BUY_CLASSES},
    }


def pct(x, nd=1):
    return "n/a" if x is None else f"{100 * x:.{nd}f}"


def ci_str(ci):
    lo, hi = ci
    return "n/a" if lo is None else f"[{100 * lo:.1f}, {100 * hi:.1f}]"


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    return o


def main():
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    pdf = pl.read_parquet(
        PLAYERS,
        columns=[
            "match_key", "round", "tick", "player_id", "team_code", "money",
            "equipment_value_calc", "has_defuser", "inv_c4",
        ],
    ).with_columns(pl.col("round").cast(pl.Int64))
    rd = pl.read_parquet(
        ROUNDS, columns=["match_key", "round", "ct_win", "split", "live_start_tick"]
    ).with_columns(pl.col("round").cast(pl.Int64))

    anomalies = []
    summary = {}

    # ---- 1. Inputs and key checks -------------------------------------------------
    emit("== 1. Inputs and key checks ==")
    kp = pdf.select(KEY).unique()
    kr = rd.select(KEY)
    assert kr.unique().height == kr.height, "duplicate keys in rounds table"
    miss_r = kp.join(kr, on=KEY, how="anti").height
    miss_p = kr.join(kp, on=KEY, how="anti").height
    dup = pdf.height - pdf.unique(["match_key", "round", "tick", "player_id"]).height
    assert miss_r == 0 and miss_p == 0 and dup == 0
    emit(f"player rows {pdf.height:,} | round rows {rd.height:,} | matches {rd['match_key'].n_unique()}")
    emit(f"player-rounds missing from rounds table {miss_r} | rounds with no players {miss_p} | duplicate player-tick rows {dup}")
    tc_vals = sorted(pdf["team_code"].unique().to_list())
    assert set(tc_vals) <= {T_SIDE, CT_SIDE}, tc_vals
    emit(f"team_code values {tc_vals} (2 = T, 3 = CT) | null equipment {pdf['equipment_value_calc'].null_count()} money {pdf['money'].null_count()}")
    cw = rd["ct_win"]
    assert cw.null_count() == 0 and set(cw.to_list()) <= {0, 1}
    emit(f"ct_win counts {dict(sorted(Counter(cw.to_list()).items()))} nulls {cw.null_count()}")
    emit(f"split counts (rounds) {dict(sorted(Counter(rd['split'].to_list()).items()))}")
    rpm = rd.group_by("match_key").len()["len"]
    short = int((rpm < 13).sum())
    emit(f"rounds per match: min {rpm.min()} max {rpm.max()} | matches with fewer than 13 rounds: {short}")
    if short:
        anomalies.append(f"{short} matches have fewer than 13 rounds (min {rpm.min()}); partial matches are kept as the task specifies")

    # Freeze-end state: first snapshot per (match, round).
    first = pdf.group_by(KEY).agg(pl.col("tick").min().alias("freeze_tick"))
    chk = first.join(rd.select(KEY + ["live_start_tick"]), on=KEY, how="inner")
    n_ge = chk.filter(pl.col("freeze_tick") >= pl.col("live_start_tick")).height
    n_eq = chk.filter(pl.col("freeze_tick") == pl.col("live_start_tick")).height
    n_gt = chk.filter(pl.col("freeze_tick") > pl.col("live_start_tick")).height
    assert n_ge == chk.height
    emit(f"Q1a first snapshot >= live_start_tick: {n_ge} of {chk.height} | exactly equal {n_eq} | strictly after {n_gt}")
    fz = pdf.join(first, on=KEY).filter(pl.col("tick") == pl.col("freeze_tick"))
    emit(f"freeze-end player rows {fz.height:,}")

    # Side mapping sanity (not used for classification): defusers belong to CT, C4 to T.
    ct_p = fz.filter(pl.col("team_code") == CT_SIDE)
    t_p = fz.filter(pl.col("team_code") == T_SIDE)
    ct_def_share = ct_p["has_defuser"].mean()
    t_def_n = t_p.filter(pl.col("has_defuser")).height
    ct_c4_n = ct_p.filter(pl.col("inv_c4") > 0).height
    t_sides_c4 = t_p.group_by(KEY).agg((pl.col("inv_c4") > 0).any().alias("has_c4"))
    t_c4_share = t_sides_c4["has_c4"].mean()
    sanity = {
        "ct_players_with_defuser_share": ct_def_share,
        "t_players_with_defuser_count_expect_0": t_def_n,
        "ct_players_holding_c4_count_expect_0": ct_c4_n,
        "t_sides_with_c4_holder_share": t_c4_share,
    }
    emit(
        "side-mapping sanity (not used for classes): CT players with defuser "
        f"{ct_def_share:.3f} | T players with defuser {t_def_n} (expect 0) | "
        f"CT players holding C4 {ct_c4_n} (expect 0) | T sides with a C4 holder {t_c4_share:.3f}"
    )
    if t_def_n or ct_c4_n:
        anomalies.append(
            f"side-mapping sanity: T players with defuser {t_def_n}, CT players with C4 {ct_c4_n}"
        )
    summary["sanity"] = sanity

    # ---- 2. Side sums and the 5-v-5 filter ----------------------------------------
    emit()
    emit("== 2. Side sums and 5-v-5 filter ==")
    side = fz.group_by(KEY + ["team_code"]).agg(
        pl.len().alias("n"),
        pl.col("equipment_value_calc").cast(pl.Int64).sum().alias("eq"),
        pl.col("money").cast(pl.Int64).sum().alias("money"),
    )

    def one_side(code, p):
        return side.filter(pl.col("team_code") == code).select(
            KEY
            + [
                pl.col("n").alias(f"{p}_n"),
                pl.col("eq").alias(f"{p}_equipment"),
                pl.col("money").alias(f"{p}_money"),
            ]
        )

    base = (
        rd.select(KEY + ["split", "ct_win"])
        .join(one_side(CT_SIDE, "ct"), on=KEY, how="left")
        .join(one_side(T_SIDE, "t"), on=KEY, how="left")
    )
    fill_cols = ["ct_n", "ct_equipment", "ct_money", "t_n", "t_equipment", "t_money"]
    base = base.with_columns([pl.col(c).fill_null(0).cast(pl.Int64) for c in fill_cols])
    side_rounds = 2 * base.height
    bad_sides = int((base["ct_n"] != N_PER_SIDE).sum() + (base["t_n"] != N_PER_SIDE).sum())
    emit(f"side-rounds {side_rounds:,} | side-rounds without exactly {N_PER_SIDE} players: {bad_sides}")
    both5 = (pl.col("ct_n") == N_PER_SIDE) & (pl.col("t_n") == N_PER_SIDE)
    dropped = base.filter(~both5).sort(KEY)
    kept = base.filter(both5).sort(KEY)
    emit(f"rounds dropped (not 5 v 5): {dropped.height} | kept: {kept.height} | input: {base.height}")
    dropped_list = []
    for r in dropped.iter_rows(named=True):
        item = {
            "match_key": r["match_key"], "round": r["round"],
            "ct_n": r["ct_n"], "t_n": r["t_n"],
            "ct_equipment": r["ct_equipment"], "t_equipment": r["t_equipment"],
            "ct_money": r["ct_money"], "t_money": r["t_money"],
        }
        dropped_list.append(item)
        emit(
            f"  dropped: {r['match_key']} round {r['round']} | CT n={r['ct_n']} eq={r['ct_equipment']:,} money={r['ct_money']:,}"
            f" | T n={r['t_n']} eq={r['t_equipment']:,} money={r['t_money']:,}"
        )
        anomalies.append(
            f"dropped round {r['round']} of {r['match_key']} (ct_n={r['ct_n']}, t_n={r['t_n']})"
        )
    kr_round = kept["round"]
    groups = {
        "round_1": int((kr_round == 1).sum()),
        "round_13": int((kr_round == 13).sum()),
        "rounds_2_12_and_14_24": int((((kr_round >= 2) & (kr_round <= 12)) | ((kr_round >= 14) & (kr_round <= 24))).sum()),
        "overtime_25_plus": int((kr_round >= OT_FIRST_ROUND).sum()),
    }
    emit(f"kept by group: round 1 {groups['round_1']} | round 13 {groups['round_13']} | "
         f"rounds 2-12 and 14-24 {groups['rounds_2_12_and_14_24']} | overtime 25+ {groups['overtime_25_plus']}")
    summary["filter"] = {
        "side_rounds": side_rounds,
        "side_rounds_not_5": bad_sides,
        "input_rounds": base.height,
        "dropped_rounds": dropped_list,
        "kept_rounds": kept.height,
        "kept_by_group": groups,
    }

    base_rows = kept.to_dicts()
    emit(f"kept rounds (base): {len(base_rows)}")

    # ---- 3. Threshold sets and classification -------------------------------------
    emit()
    emit("== 3. Classification under three threshold sets ==")
    per_set_rows = {}
    for name in SET_ORDER:
        cfg = THRESHOLD_SETS[name]
        assert cfg["eco_max"] < cfg["full_min"]
        rows = classify_rows(base_rows, cfg["eco_max"], cfg["full_min"])
        assert len(rows) == len(base_rows)
        assert all(r["round_class"] in ALL_CLASSES for r in rows)
        cnt = Counter(r["round_class"] for r in rows)
        assert sum(cnt.values()) == len(rows)
        per_set_rows[name] = rows
        emit(f"{name}: eco_max {cfg['eco_max']:,}, full_min {cfg['full_min']:,} | class counts " +
             ", ".join(f"{c} {cnt.get(c, 0)}" for c in ALL_CLASSES))

    # Pistol rows must be identical across sets; default vs aggressive comparison.
    pistol_sets = {name: [(r["match_key"], r["round"]) for r in per_set_rows[name] if r["is_pistol"]] for name in SET_ORDER}
    assert pistol_sets["conservative"] == pistol_sets["default"] == pistol_sets["aggressive"]
    cls_def = [r["round_class"] for r in per_set_rows["default"]]
    cls_agg = [r["round_class"] for r in per_set_rows["aggressive"]]
    cls_con = [r["round_class"] for r in per_set_rows["conservative"]]
    same_def_agg = cls_def == cls_agg
    diff_def_con = sum(1 for a, b in zip(cls_def, cls_con) if a != b)
    diff_def_agg = sum(1 for a, b in zip(cls_def, cls_agg) if a != b)
    emit(f"round classes identical default vs aggressive: {same_def_agg} | rounds differing default vs conservative: {diff_def_con} | default vs aggressive: {diff_def_agg}")
    if same_def_agg:
        anomalies.append("default and aggressive threshold sets give identical round classes (both are 10,000 / 20,000)")

    # ---- 4. Per-set tables --------------------------------------------------------
    sets_out = {}
    for name in SET_ORDER:
        cfg = THRESHOLD_SETS[name]
        rows = per_set_rows[name]
        classes = {c: class_block([r for r in rows if r["round_class"] == c]) for c in ALL_CLASSES}
        overall = class_block(rows)
        sb_np = side_buy_counts(rows, pistol_only=False)
        sb_p = side_buy_counts(rows, pistol_only=True)
        sets_out[name] = {
            "eco_max": cfg["eco_max"],
            "full_min": cfg["full_min"],
            "reason": cfg["reason"],
            "side_buy_nonpistol": sb_np,
            "side_buy_pistol": sb_p,
            "classes": classes,
            "overall": overall,
        }
        emit()
        emit(f"== 4. Threshold set: {name} (eco_max {cfg['eco_max']:,}, full_min {cfg['full_min']:,}) ==")
        emit(f"  why: {cfg['reason']}")
        emit("  non-pistol side buy counts: CT " + " ".join(f"{k} {sb_np['ct'][k]}" for k in BUY_CLASSES)
             + " | T " + " ".join(f"{k} {sb_np['t'][k]}" for k in BUY_CLASSES))
        emit("  pistol-round side buy counts: CT " + " ".join(f"{k} {sb_p['ct'][k]}" for k in BUY_CLASSES)
             + " | T " + " ".join(f"{k} {sb_p['t'][k]}" for k in BUY_CLASSES))
        hdr = (f"  {'class':<15}{'rounds':>7}{'matches':>9}{'OT':>5}  {'CT win %':>9} {'95% CI':>17}"
               f"  {'hi-side n':>9} {'hi-side win %':>14} {'95% CI':>17}")
        emit(hdr)
        for c in ALL_CLASSES + ["ALL"]:
            b = overall if c == "ALL" else classes[c]
            flag = "*" if b["few_clusters"] else " "
            emit(
                f"  {c:<15}{b['rounds']:>7}{b['matches']:>8}{flag}{b['overtime_rounds']:>5}"
                f"  {pct(b['ct_win_rate']):>9} {ci_str(b['ct_win_ci95']):>17}"
                f"  {b['higher_side_rounds']:>9} {pct(b['higher_side_win_rate']):>14} {ci_str(b['higher_side_win_ci95']):>17}"
            )
        emit("  (* = fewer than 30 matches; CI from match-cluster bootstrap is unreliable. hi-side = higher-equipment side, ties excluded)")
        for c in ALL_CLASSES:
            if classes[c]["rounds"] and classes[c]["few_clusters"]:
                anomalies.append(f"[{name}] class {c} has only {classes[c]['matches']} matches")

    # ---- 5. Pistol rounds ---------------------------------------------------------
    emit()
    emit("== 5. Pistol rounds (same under all sets) ==")
    pistol_rows = [r for r in per_set_rows[DEFAULT_SET] if r["is_pistol"]]
    pistol_by_round = {}
    for rn in PISTOL_ROUNDS:
        blk = class_block([r for r in pistol_rows if r["round"] == rn])
        pistol_by_round[str(rn)] = blk
        emit(f"  round {rn}: rounds {blk['rounds']} | matches {blk['matches']} | CT win {pct(blk['ct_win_rate'])}% "
             f"{ci_str(blk['ct_win_ci95'])} | hi-side n {blk['higher_side_rounds']} win {pct(blk['higher_side_win_rate'])}%")
    # Pistol rounds keep the same class in every set; confirm the pistol block matches across sets.
    for name in SET_ORDER:
        assert sets_out[name]["classes"]["pistol"]["rounds"] == len(pistol_rows)

    # ---- 6. Differences between sets ----------------------------------------------
    emit()
    emit("== 6. Round-class transitions, default vs other sets ==")
    trans = {}
    for other in ["conservative", "aggressive"]:
        pair = Counter(
            (a, b) for a, b in zip(cls_def, [r["round_class"] for r in per_set_rows[other]]) if a != b
        )
        trans[other] = {f"{a} -> {b}": n for (a, b), n in sorted(pair.items())}
        emit(f"  default vs {other}: {sum(pair.values())} rounds change class")
        for k, v in trans[other].items():
            emit(f"    {k}: {v}")

    # ---- 7. Output parquet (default set) ------------------------------------------
    default_rows = per_set_rows[DEFAULT_SET]
    out_df = pl.from_dicts(default_rows, schema=OUT_SCHEMA)
    assert out_df.height == len(base_rows)
    assert out_df["round_class"].is_in(ALL_CLASSES).all()
    assert out_df["round_class"].null_count() == 0
    out_df.write_parquet(OUT_PARQUET)
    chk_df = pl.read_parquet(OUT_PARQUET)
    assert chk_df.height == out_df.height and chk_df.columns == list(OUT_SCHEMA.keys())
    emit()
    emit("== 7. Output ==")
    emit(f"wrote {OUT_PARQUET} ({chk_df.height} rows, {chk_df.width} columns, default set)")
    emit(f"round_class counts (default): {dict(Counter(chk_df['round_class'].to_list()))}")

    # ---- 8. JSON and text summary -------------------------------------------------
    final_n = len(default_rows)
    summary["thresholds"] = {k: {"eco_max": v["eco_max"], "full_min": v["full_min"], "reason": v["reason"]}
                             for k, v in THRESHOLD_SETS.items()}
    summary["default_set"] = DEFAULT_SET
    summary["sets"] = sets_out
    summary["pistol_by_round"] = pistol_by_round
    summary["set_comparison"] = {
        "default_equals_aggressive": same_def_agg,
        "rounds_class_differs_default_vs_conservative": diff_def_con,
        "rounds_class_differs_default_vs_aggressive": diff_def_agg,
        "transitions": trans,
    }
    summary["bootstrap"] = {
        "unit": "match_key (cluster)",
        "method": "per class: resample the matches that contribute rounds to that class, with replacement, "
                  "B resamples; statistic pools the drawn matches' class rounds",
        "B": BOOT_B,
        "seed": BOOT_SEED,
        "percentiles": [2.5, 97.5],
    }
    summary["output"] = {"parquet": str(OUT_PARQUET), "rows": chk_df.height, "set": DEFAULT_SET}
    summary["anomalies"] = anomalies
    summary["caveats"] = [
        "Freeze-end equipment includes weapons carried over from earlier rounds, so a side's class describes its loadout at freeze end, not its buy decision.",
        "Sides come from team_code at freeze end (2 = T, 3 = CT). The defuser and C4 checks support the mapping but do not prove it.",
        "Overtime rounds (25+) are in the kept set and classed like other rounds; their counts are shown in the OT column.",
        "Bootstrap CIs resample matches within each class; classes with fewer than 30 matches have unreliable intervals.",
        "No score fields are used. Round win is ct_win from rounds.parquet only.",
    ]
    summary["final_round_count"] = final_n
    OUT_JSON.write_text(json.dumps(clean(summary), indent=2), encoding="utf-8")

    emit()
    emit("== 8. Anomalies ==")
    for a in anomalies:
        emit(f"  - {a}")
    if not anomalies:
        emit("  none")
    emit()
    emit(f"FINAL ROUND COUNT (kept, both sides 5 players): {final_n}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
