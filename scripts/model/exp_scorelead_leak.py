"""Score-lead leakage check on the 2026 grid (consecutive round pairs).

Question: does score_lead_diag carry information about the current round's result, beyond the previous round's result?

Inputs
  data/combined/grid.parquet    1 Hz grid. score_lead_diag and label_ct_win are constant within a round (checked below).
  data/combined/rounds.parquet  round table. use (round is in the grid), final (round winner side, 3 = CT),
                                swap_boundary (sides swap after this round), max_rounds_final.

Rules
  - Rounds: era == 2026, use == True, joined to the grid on (match_key, round). A round is present if it has grid rows.
  - Consecutive pair (r-1, r): same match_key, round(r) - round(r-1) == 1, both rounds present.
  - d = score_lead_diag(r) - score_lead_diag(r-1).
  - prev = +1 if CT won round r-1 else -1 (label_ct_win of r-1). cur = +1 if CT won round r else -1 (label_ct_win of r).
  - Shares over pairs with d != 0: sign(d) == prev, sign(d) == -prev, sign(d) == cur, sign(d) == -cur.
  - Side swap: a pair crosses a side swap iff swap_boundary(r-1) is True. The shares are recomputed without those pairs.

Output: data/combined/exp_scorelead_leak.json (the script refuses to overwrite an existing file).

Run: pixi run python -I -X utf8 scripts\\model\\exp_scorelead_leak.py
"""
import json
import os
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
STORE = os.path.join(ROOT, "data", "combined")
GRID = os.path.join(STORE, "grid.parquet")
ROUNDS = os.path.join(STORE, "rounds.parquet")
OUT = os.path.join(STORE, "exp_scorelead_leak.json")
ERA = "2026"
CT = 3  # team code of CT in rounds.final (make_grid.py: T, CT = 2, 3)

# Numbers quoted by the earlier quick check. They are compared with the recomputed values, not copied into them.
REF = {
    "consecutive_pairs": 259948,
    "prev_same_pct": 95.13,
    "prev_opp_pct": 0.37,
    "cur_same_pct": 59.93,
    "cur_opp_pct": 35.58,
}


def swap_expr(rnd, mr):
    # prepare.py swap_after: sides swap after completed round rnd (halftime at mr // 2, then after the 3rd round
    # of each 6-round overtime period).
    return (rnd == mr // 2) | ((rnd > mr) & (((rnd - mr - 3) % 6) == 0))


def shares(X):
    n = X.height
    row = X.select(
        (pl.col("sd") == pl.col("prev_s")).sum().alias("prev_same"),
        (pl.col("sd") == -pl.col("prev_s")).sum().alias("prev_opp"),
        (pl.col("sd") == pl.col("cur_s")).sum().alias("cur_same"),
        (pl.col("sd") == -pl.col("cur_s")).sum().alias("cur_opp"),
        (pl.col("prev_s") == pl.col("cur_s")).sum().alias("prev_eq_cur_winner"),
    ).row(0, named=True)
    cnt = {"pairs": n, **{k: int(v) for k, v in row.items()}}
    pct = {k: (round(100.0 * cnt[k] / n, 4) if n else None)
           for k in ("prev_same", "prev_opp", "cur_same", "cur_opp", "prev_eq_cur_winner")}
    return cnt, pct


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"refusing to overwrite existing {OUT}")

    G = pl.read_parquet(GRID, columns=["match_key", "round", "era", "score_lead_diag", "label_ct_win"])
    G = G.filter(pl.col("era") == ERA)
    grid_rows_2026 = G.height
    K = G.group_by(["match_key", "round"]).agg(
        pl.len().alias("grid_rows"),
        pl.col("score_lead_diag").n_unique().alias("n_distinct_lead"),
        pl.col("label_ct_win").n_unique().alias("n_distinct_label"),
        pl.col("score_lead_diag").first().cast(pl.Int64).alias("lead"),
        pl.col("label_ct_win").first().cast(pl.Int64).alias("ct_win"),
    )

    R = pl.read_parquet(ROUNDS, columns=["match_key", "round", "era", "use", "final", "swap_boundary", "max_rounds_final"])
    R = R.filter((pl.col("era") == ERA) & pl.col("use"))
    rounds_use_2026 = R.height

    J = K.join(R, on=["match_key", "round"], how="full", coalesce=True)
    rounds_only_in_rounds_file = J.filter(pl.col("grid_rows").is_null()).height
    grid_rounds_without_rounds_row = J.filter(pl.col("grid_rows").is_not_null() & pl.col("final").is_null()).height
    Jg = J.filter(pl.col("grid_rows").is_not_null() & pl.col("final").is_not_null())

    label_mismatch = Jg.filter(pl.col("ct_win") != (pl.col("final") == CT).cast(pl.Int64)).height
    swap_recompute_mismatch = Jg.filter(
        swap_expr(pl.col("round"), pl.col("max_rounds_final")) != pl.col("swap_boundary")).height

    S = Jg.sort(["match_key", "round"]).with_columns(
        pl.col("round").shift(1).over("match_key").alias("prev_round"),
        pl.col("lead").shift(1).over("match_key").alias("prev_lead"),
        pl.col("ct_win").shift(1).over("match_key").alias("prev_ct_win"),
        pl.col("swap_boundary").shift(1).over("match_key").alias("prev_swap"),
    )
    P = S.filter(pl.col("prev_round").is_not_null() & ((pl.col("round") - pl.col("prev_round")) == 1))
    total_pairs = P.height
    null_label_pairs = P.filter(pl.col("prev_ct_win").is_null() | pl.col("ct_win").is_null()).height
    P = P.filter(pl.col("prev_ct_win").is_not_null() & pl.col("ct_win").is_not_null())
    P = P.with_columns((pl.col("lead") - pl.col("prev_lead")).alias("d"))
    zero_d_pairs = P.filter(pl.col("d") == 0).height
    N = P.filter(pl.col("d") != 0).with_columns(
        pl.col("d").sign().alias("sd"),
        pl.when(pl.col("prev_ct_win") == 1).then(1).otherwise(-1).alias("prev_s"),
        pl.when(pl.col("ct_win") == 1).then(1).otherwise(-1).alias("cur_s"),
    )

    swap_pairs_all = P.filter(pl.col("prev_swap")).height
    swap_pairs_nonzero = N.filter(pl.col("prev_swap")).height
    NS = N.filter(~pl.col("prev_swap"))
    SW = N.filter(pl.col("prev_swap"))

    cnt_all, pct_all = shares(N)
    cnt_ns, pct_ns = shares(NS)
    cnt_sw, pct_sw = shares(SW)

    # Identity check: without side swaps, sign(d) should equal prev in every pair (score bookkeeping).
    ns_identity_prev_same = int((NS["sd"] == NS["prev_s"]).sum()) if NS.height else 0

    d_sign_counts = {str(k): int(v) for k, v in N.group_by("sd").agg(pl.len().alias("n")).iter_rows()}

    ours = {"prev_same_pct": pct_all["prev_same"], "prev_opp_pct": pct_all["prev_opp"],
            "cur_same_pct": pct_all["cur_same"], "cur_opp_pct": pct_all["cur_opp"]}
    comparison = {"consecutive_pairs": {"reference": REF["consecutive_pairs"], "ours": total_pairs,
                                        "match": total_pairs == REF["consecutive_pairs"]}}
    for k in ("prev_same_pct", "prev_opp_pct", "cur_same_pct", "cur_opp_pct"):
        comparison[k] = {"reference": REF[k], "ours_2dp": round(ours[k], 2),
                         "match": round(ours[k], 2) == REF[k], "diff_pp": round(ours[k] - REF[k], 4)}
    comparison["all_match"] = all(v["match"] for v in comparison.values())

    OUTPUT = {
        "script": "scripts/model/exp_scorelead_leak.py",
        "input_files": {"grid": GRID, "rounds": ROUNDS},
        "store": "data/combined",
        "era_filter": ERA,
        "definitions": {
            "round_present": "round has at least one row in grid.parquet with era 2026, and rounds.parquet has use == True for it",
            "consecutive_pair": "same match_key and round(r) - round(r-1) == 1, both rounds present; rounds sorted by round number within match",
            "score_lead_diag": "grid column = ct_sc - t_sc at freeze end (rounds.parquet, fill_null(0)); see scripts/data/make_grid.py line 75",
            "d": "score_lead_diag(r) - score_lead_diag(r-1)",
            "sign": "sign(x) = +1 if x > 0, -1 if x < 0; d == 0 pairs are excluded from all shares",
            "prev": "+1 if CT won round r-1 (grid label_ct_win = 1; equals rounds.final == 3), else -1",
            "cur": "+1 if CT won round r (grid label_ct_win = 1; equals rounds.final == 3), else -1",
            "shares": "share of pairs with d != 0 where sign(d) == prev, sign(d) == -prev, sign(d) == cur, sign(d) == -cur",
            "side_swap": "pair (r-1, r) crosses a side swap iff rounds.swap_boundary(r-1) is True; swap_boundary = prepare.swap_after(round, max_rounds_final) = (round == max_rounds_final // 2) or (round > max_rounds_final and (round - max_rounds_final - 3) % 6 == 0); recomputed from max_rounds_final and checked against the stored column",
            "excluding_side_swap": "same shares recomputed on pairs with prev_swap == False",
            "prev_eq_cur_winner": "diagnostic: share of pairs where the CT-win sign of round r-1 equals that of round r (a winner-repeat share, not a score-lead share)",
        },
        "checks": {
            "grid_rows_2026": grid_rows_2026,
            "rounds_use_2026": rounds_use_2026,
            "rounds_present_in_grid_2026": Jg.height,
            "grid_rounds_without_rounds_row": grid_rounds_without_rounds_row,
            "rounds_in_rounds_file_not_in_grid": rounds_only_in_rounds_file,
            "rounds_with_more_than_one_distinct_score_lead": int((K["n_distinct_lead"] > 1).sum()),
            "rounds_with_more_than_one_distinct_label": int((K["n_distinct_label"] > 1).sum()),
            "label_ct_win_vs_final_mismatch": label_mismatch,
            "swap_boundary_vs_recomputed_mismatch": swap_recompute_mismatch,
            "pairs_with_null_label_excluded": null_label_pairs,
        },
        "counts": {
            "consecutive_pairs": total_pairs,
            "pairs_d_zero": zero_d_pairs,
            "pairs_d_nonzero": N.height,
            "pairs_crossing_side_swap_all": swap_pairs_all,
            "pairs_crossing_side_swap_nonzero_d": swap_pairs_nonzero,
            "pairs_no_side_swap_nonzero_d": NS.height,
            "sign_d_counts": d_sign_counts,
            "no_swap_pairs_with_sign_d_equal_prev": ns_identity_prev_same,
        },
        "shares_pct": {
            "all_nonzero_d_pairs": {"n": cnt_all["pairs"], **pct_all},
            "excluding_side_swap_pairs": {"n": cnt_ns["pairs"], **pct_ns},
            "side_swap_pairs_only": {"n": cnt_sw["pairs"], **pct_sw},
        },
        "counts_raw": {
            "all_nonzero_d_pairs": cnt_all,
            "excluding_side_swap_pairs": cnt_ns,
            "side_swap_pairs_only": cnt_sw,
        },
        "reference": {**REF,
                      "reference_internal_sum_pct": {"prev": round(REF["prev_same_pct"] + REF["prev_opp_pct"], 2),
                                                     "cur": round(REF["cur_same_pct"] + REF["cur_opp_pct"], 2)},
                      "note": "a binary sign over one denominator sums to 100; the reference pairs sum to about 95.5"},
        "comparison": comparison,
    }

    os.makedirs(STORE, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(OUTPUT, f, indent=2, ensure_ascii=False)

    print(json.dumps(OUTPUT["checks"], indent=2, ensure_ascii=False))
    print(json.dumps(OUTPUT["counts"], indent=2, ensure_ascii=False))
    print(json.dumps(OUTPUT["shares_pct"], indent=2, ensure_ascii=False))
    print(json.dumps(OUTPUT["comparison"], indent=2, ensure_ascii=False))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
