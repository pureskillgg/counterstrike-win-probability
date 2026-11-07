"""Freeze-end buy-type probe on the 1 Hz player sample (read-only inputs under G:).

Takes the first snapshot of each (match_key, round), which is the round_freeze_end
snapshot, and computes per-side equipment and money. Uses equipment and money only;
ct_score and t_score are never read.

Pistol rounds are round 1 and round 13. Overtime is round 25 and up.
Non-pistol = rounds 2-12 and 14-24 (primary set for Q2 and Q4). Overtime (25+) is
reported separately, and Q4 also shows the counts with overtime included.

Run from the repo root, redirecting stdout:
  pixi run python -I -X utf8 scripts\\eda\\buy_type_probe.py > logs\\buy_types\\probe.txt
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
PLAYERS = ROOT / "data" / "state_sample" / "players_1hz.parquet"
ROUNDS = ROOT / "data" / "state_sample" / "rounds.parquet"

QS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
QLABELS = ["p5", "p10", "p25", "p50", "p75", "p90", "p95"]
SIDES = [2, 3]  # 2 = T, 3 = CT
THRESHOLDS = [2000, 5000, 8000, 10000, 15000, 20000, 25000]
BIN_W = 1000
BIN_MAX = 40000
SMOOTH_K = 3
PEAK_MIN_FRAC = 0.05  # a smoothed peak must reach 5% of the smoothed maximum


def qline(label, vals):
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        print(f"{label}: n=0")
        return
    qs = np.quantile(a, QS)  # numpy default (linear interpolation)
    body = "  ".join(f"{name}={v:,.1f}" for name, v in zip(QLABELS, qs))
    print(f"{label}: n={a.size}  {body}")


def bin_label(i):
    return f"{i * BIN_W:>6,}-{(i + 1) * BIN_W:<6,}"


def main():
    pdf = pl.read_parquet(
        PLAYERS,
        columns=["match_key", "round", "tick", "player_id", "team_code", "money", "equipment_value_calc"],
    ).with_columns(pl.col("round").cast(pl.Int64))
    rd = pl.read_parquet(ROUNDS, columns=["match_key", "round", "live_start_tick"]).with_columns(
        pl.col("round").cast(pl.Int64)
    )

    print("== Input checks ==")
    print("players rows:", pdf.height, " rounds rows:", rd.height)
    pk = pdf.select(["match_key", "round"]).unique()
    rk = rd.select(["match_key", "round"]).unique()
    print("distinct (match_key, round): players", pk.height, " rounds table", rk.height)
    print("rounds table duplicate keys:", rd.height - rk.height)
    print("player rounds missing from rounds table:", pk.join(rk, on=["match_key", "round"], how="anti").height)
    print("rounds table rows with no player rows:", rk.join(pk, on=["match_key", "round"], how="anti").height)
    print(
        "duplicate (match_key, round, tick, player_id) rows:",
        pdf.height - pdf.unique(["match_key", "round", "tick", "player_id"]).height,
    )
    print(
        "null counts: equipment_value_calc",
        pdf["equipment_value_calc"].null_count(),
        " money",
        pdf["money"].null_count(),
        " team_code",
        pdf["team_code"].null_count(),
    )
    print("team_code values:", sorted(pdf["team_code"].unique().to_list()))
    print("round range:", pdf["round"].min(), "to", pdf["round"].max())

    # First snapshot per round = round_freeze_end snapshot.
    first = pdf.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("first_tick"))
    chk = first.join(rd, on=["match_key", "round"], how="inner")
    at_or_after = chk.filter(pl.col("first_tick") >= pl.col("live_start_tick")).height
    strictly_after = chk.filter(pl.col("first_tick") > pl.col("live_start_tick")).height
    exactly_equal = chk.filter(pl.col("first_tick") == pl.col("live_start_tick")).height
    print()
    print("== Q1 ==")
    print(f"Q1a rounds with first snapshot tick >= live_start_tick: {at_or_after} of {chk.height}")
    print(f"Q1a   of which strictly after live_start_tick: {strictly_after}")
    print(f"Q1a   of which exactly equal to live_start_tick: {exactly_equal}")
    gap = chk["live_start_tick"] - chk["first_tick"]
    print(
        "Q1a live_start_tick - first_tick: min",
        gap.min(),
        " median",
        gap.median(),
        " max",
        gap.max(),
    )
    n_match = pdf["match_key"].n_unique()
    print("distinct matches in players table:", n_match)

    fz = pdf.join(first, on=["match_key", "round"]).filter(pl.col("tick") == pl.col("first_tick"))
    print("freeze-end player rows:", fz.height)
    print("freeze-end rows with team_code not in {2, 3}:", fz.filter(~pl.col("team_code").is_in(SIDES)).height)

    side = fz.group_by(["match_key", "round", "team_code"]).agg(
        pl.len().alias("n_players"),
        pl.col("equipment_value_calc").sum().alias("eq_sum"),
        pl.col("money").sum().alias("money_sum"),
    )
    side_df = pl.DataFrame({"team_code": pl.Series(SIDES, dtype=pdf.schema["team_code"])})
    grid = (
        rk.join(side_df, how="cross")
        .join(side, on=["match_key", "round", "team_code"], how="left")
        .with_columns(
            pl.col("n_players").fill_null(0),
            pl.col("eq_sum").fill_null(0),
            pl.col("money_sum").fill_null(0),
        )
    )
    n_sr = grid.height
    bad = grid.filter(pl.col("n_players") != 5)
    print(f"Q1b side-rounds without exactly 5 players: {bad.height} of {n_sr} (side-rounds = rounds x 2 sides)")
    for row in bad.iter_rows(named=True):
        fz_tick = first.filter(
            (pl.col("match_key") == row["match_key"]) & (pl.col("round") == row["round"])
        )["first_tick"][0]
        later = pdf.filter(
            (pl.col("match_key") == row["match_key"])
            & (pl.col("round") == row["round"])
            & (pl.col("team_code") == row["team_code"])
            & (pl.col("tick") > fz_tick)
        )
        next_tick = later["tick"].min()
        n_next = later.filter(pl.col("tick") == next_tick).height if next_tick is not None else 0
        print(
            f"  bad side-round: match_key={row['match_key']} round={row['round']} "
            f"team_code={row['team_code']} n_players={row['n_players']} eq_sum={row['eq_sum']} "
            f"money_sum={row['money_sum']} | next tick with a row for this side={next_tick} "
            f"n_players_there={n_next} fz_tick={fz_tick}"
        )
    dist = Counter(grid["n_players"].to_list())
    print("side-round player-count distribution:", dict(sorted(dist.items())))

    rnd = grid["round"].to_numpy()
    eq = grid["eq_sum"].to_numpy().astype(np.float64)
    is_r1 = rnd == 1
    is_r13 = rnd == 13
    is_np = ((rnd >= 2) & (rnd <= 12)) | ((rnd >= 14) & (rnd <= 24))
    is_ot = rnd >= 25
    rnds_per_group = {
        "r1": rk.filter(pl.col("round") == 1).height,
        "r13": rk.filter(pl.col("round") == 13).height,
        "np": rk.filter((pl.col("round") >= 2) & (pl.col("round") <= 12)).height
        + rk.filter((pl.col("round") >= 14) & (pl.col("round") <= 24)).height,
        "ot": rk.filter(pl.col("round") >= 25).height,
    }
    print("rounds per group:", rnds_per_group, " (sum", sum(rnds_per_group.values()), "of", rk.height, ")")
    print("matches containing round 1:", rk.filter(pl.col("round") == 1).height,
          " containing round 13:", rk.filter(pl.col("round") == 13).height)
    print()
    print("== Q2 side equipment sum at freeze-end (both sides pooled) ==")
    print("side-rounds per group: r1", int(is_r1.sum()), " r13", int(is_r13.sum()),
          " np(2-12,14-24)", int(is_np.sum()), " ot(25+)", int(is_ot.sum()))
    qline("Q2 round 1", eq[is_r1])
    qline("Q2 round 13", eq[is_r13])
    qline("Q2 non-pistol rounds 2-12 and 14-24", eq[is_np])
    qline("Q2 overtime rounds 25+", eq[is_ot])

    print()
    print("== Q3 per-player freeze-end values ==")
    for r in (1, 13):
        sub = fz.filter(pl.col("round") == r)
        qline(f"Q3 round {r} equipment_value_calc", sub["equipment_value_calc"].to_numpy())
        qline(f"Q3 round {r} money", sub["money"].to_numpy())

    print()
    print("== Q4 non-pistol side-round equipment sums ==")
    reg_eq = eq[is_np]
    all_eq = eq[is_np | is_ot]
    print(
        "primary set = rounds 2-12 and 14-24: side-rounds", reg_eq.size,
        " | incl. overtime 25+: side-rounds", all_eq.size,
        " (overtime alone:", int(is_ot.sum()), ")",
    )
    for thr in THRESHOLDS:
        print(f"  eq < {thr:>6,}: primary {int((reg_eq < thr).sum()):>5}   incl. OT {int((all_eq < thr).sum()):>5}")

    nb = BIN_MAX // BIN_W

    def hist_counts(v):
        inr = v[(v >= 0) & (v < BIN_MAX)]
        return np.bincount((inr // BIN_W).astype(np.int64), minlength=nb)

    counts = hist_counts(reg_eq)
    ot_counts = hist_counts(eq[is_ot])
    print(f"histogram {BIN_W}-dollar bins, 0 to {BIN_MAX:,} (primary column; OT column = rounds 25+ only):")
    print(
        f"  primary below 0: {int((reg_eq < 0).sum())}   at or above {BIN_MAX:,}: {int((reg_eq >= BIN_MAX).sum())}"
        f"   | OT below 0: {int((eq[is_ot] < 0).sum())}   at or above {BIN_MAX:,}: {int((eq[is_ot] >= BIN_MAX).sum())}"
    )
    for i in range(nb):
        print(f"  {bin_label(i)}  primary {int(counts[i]):>4}   OT {int(ot_counts[i]):>3}")
    print(f"  primary total in bins: {int(counts.sum())}   OT total in bins: {int(ot_counts.sum())}")

    padded = np.pad(counts.astype(np.float64), 1, mode="edge")
    sm = np.convolve(padded, np.ones(SMOOTH_K) / SMOOTH_K, mode="valid")
    peak_floor = PEAK_MIN_FRAC * sm.max()
    peaks = []
    for i in range(nb):
        left_ok = i == 0 or sm[i] >= sm[i - 1]
        right_ok = i == nb - 1 or sm[i] > sm[i + 1]
        if left_ok and right_ok and sm[i] >= peak_floor and sm[i] > 0:
            peaks.append(i)
    print(f"smoothed (3-bin moving average) peaks, floor {PEAK_MIN_FRAC:.0%} of max:")
    for p in peaks:
        print(f"  peak {bin_label(p)}  smoothed={sm[p]:.1f}  raw={int(counts[p])}")
    print("valleys between consecutive peaks (smoothed minimum):")
    for a, b in zip(peaks, peaks[1:]):
        v = min(range(a, b + 1), key=lambda j: sm[j])
        if a < v < b:
            depth = 1 - sm[v] / min(sm[a], sm[b])
            print(
                f"  valley {bin_label(v)}  smoothed={sm[v]:.1f}  raw={int(counts[v])}"
                f"  between peaks at bins {a} and {b}  depth={depth:.1%}"
            )
        else:
            print(f"  no valley between peaks {a} and {b}")
    raw_min = [i for i in range(1, nb - 1) if counts[i] < counts[i - 1] and counts[i] < counts[i + 1]]
    print("raw-count local minima (strictly below both neighbours); z = (lower neighbour - bin) / sqrt(lower neighbour):")
    for i in raw_min:
        lo = min(int(counts[i - 1]), int(counts[i + 1]))
        z = (lo - int(counts[i])) / np.sqrt(lo) if lo > 0 else float("nan")
        print(f"  {bin_label(i)}  raw={int(counts[i])}  neighbours={int(counts[i - 1])},{int(counts[i + 1])}  z={z:.2f}")
    raw_max = [i for i in range(1, nb - 1) if counts[i] > counts[i - 1] and counts[i] >= counts[i + 1]]
    print("raw-count local maxima (strictly above left neighbour, >= right neighbour):")
    for i in raw_max:
        print(f"  {bin_label(i)}  raw={int(counts[i])}  neighbours={int(counts[i - 1])},{int(counts[i + 1])}")

    print()
    print("== Q5 share of per-player freeze-end equipment_value_calc <= 1000 ==")
    for r in range(1, 14):
        sub = fz.filter(pl.col("round") == r)["equipment_value_calc"].to_numpy()
        if sub.size == 0:
            print(f"  round {r}: no players")
            continue
        k = int((sub <= 1000).sum())
        print(f"  round {r:>2}: {k} of {sub.size} = {k / sub.size:.4f}")
    r2_12 = fz.filter((pl.col("round") >= 2) & (pl.col("round") <= 12))["equipment_value_calc"].to_numpy()
    r13 = fz.filter(pl.col("round") == 13)["equipment_value_calc"].to_numpy()
    k2 = int((r2_12 <= 1000).sum())
    k13 = int((r13 <= 1000).sum())
    print(f"Q5 rounds 2-12 pooled: {k2} of {r2_12.size} = {k2 / r2_12.size:.4f}")
    print(f"Q5 round 13: {k13} of {r13.size} = {k13 / r13.size:.4f}")


if __name__ == "__main__":
    main()
