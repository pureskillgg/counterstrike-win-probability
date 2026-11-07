"""Parity test: win_probability() on the raw tome channels of one held-out match vs the grid predictions of the same model.

  ...python.exe -I -X utf8 scripts\\serving\\test_win_probability.py [--matches 3]

Reads the DSDK tome pages of a test match (no project code), runs the function, and compares every second it returns with the grid's
export_winprob_v3_basic_test_predictions.parquet at the same tick. Reports the largest and mean gap in percentage points, the share of seconds
that agree to 0.01 points, and the rounds whose ticks differ. Also checks that the T probability is 100 minus the CT probability.
"""
import argparse
import glob
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import win_probability as WP

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
SUFFIX = ".2026-09-18,2026-10-08"
CHANNELS = ["round_state", "round_start", "round_end", "player_info", "player_death", "bomb_state"]


def read_channel(name, match_key):
    folder = os.path.join(TOME, "header" if name == "header" else f"basic_{name}{SUFFIX}")
    files = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
    if not files:
        return None
    df = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    return df.filter(pl.col("match_key") == match_key) if "match_key" in df.columns else df


PERTICK = r"E:\pureskill_tomes\pertick_status_1001_1008\tome\csds"
EXPORTS = {
    "winprob_v6_drop_heavy": "export_winprob_v6_drop_heavy_test_predictions.parquet",
    "winprob_v8_ensemble": "export_winprob_v8_ensemble_test_predictions.parquet",
    "winprob_v7_basic_scorelead": "export_winprob_v7_basic_scorelead_test_predictions.parquet",
    "winprob_v3_basic": "export_winprob_v3_basic_test_predictions.parquet",
}


def read_player_status(match_key):
    """Per-tick player status for one match, from the per-tick tome (the tome the parity test reads does not carry it).
    Uses the per-match folders written by partition_pertick_status.py when they exist (one pass over the pages instead of one per match)."""
    import hashlib
    single = os.path.join(r"E:\pureskill_tomes\pertick_status_1001_1008\per_match", hashlib.sha1(match_key.encode("utf-8")).hexdigest()[:16] + ".parquet")
    if os.path.isfile(single):
        return pl.read_parquet(single)
    files = sorted(glob.glob(os.path.join(PERTICK, "basic_player_status", "dataframe_*")))
    if not files:
        return None
    return pl.scan_parquet(files).filter(pl.col("match_key") == match_key).collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=0, help="number of tome test matches to check (0 = all of them)")
    ap.add_argument("--model", default="winprob_v6_drop_heavy", choices=sorted(EXPORTS))
    a = ap.parse_args()
    grid = pl.read_parquet(os.path.join(ROOT, "data", "combined", EXPORTS[a.model]), columns=["match_key", "round", "tick", "p"])
    # the tome holds 1-8 October only: keep test matches that are in it
    hdr = pl.concat([pl.read_parquet(f, columns=["key"]) for f in sorted(glob.glob(os.path.join(TOME, "header", "dataframe_*")))])
    in_tome = set(hdr["key"].to_list())
    keys = [k for k in grid["match_key"].unique().sort().to_list() if k in in_tome]
    if a.matches:
        keys = keys[: a.matches]
    print(f"test matches in the tome: {len(in_tome & set(grid['match_key'].unique().to_list()))}; testing {len(keys)}; model {a.model}")
    # score lead from the tome (freeze-end round_state scores) against the grid's score_lead_diag, every round of every checked match
    lead_grid = pl.read_parquet(os.path.join(ROOT, "data", "combined", "grid.parquet"), columns=["match_key", "round", "score_lead_diag", "era"])
    lead_grid = lead_grid.filter(pl.col("era") == "2026").unique(subset=["match_key", "round"]).select(["match_key", "round", "score_lead_diag"])
    lead_checked, lead_equal = 0, 0
    total_gap, total_n = [], 0
    for key in keys:
        data = {"header": read_channel("header", key)}
        data["header"] = data["header"].filter(pl.col("key") == key) if data["header"] is not None else None
        for ch in CHANNELS:
            data[ch] = read_channel(ch, key)
        data["player_status"] = read_player_status(key)
        data = {k: v for k, v in data.items() if v is not None}
        rs = data.get("round_state")
        if rs is not None and "event_type" in rs.columns:
            fz = rs.filter(pl.col("event_type") == "round_freeze_end").select(["round", "ct_score", "t_score"]).unique(subset=["round"], keep="first")
            fz = fz.with_columns((pl.col("ct_score") - pl.col("t_score")).alias("tome_lead")).with_columns(pl.lit(key).alias("match_key"))
            cmp = fz.join(lead_grid.filter(pl.col("match_key") == key), on=["match_key", "round"], how="inner")
            lead_checked += cmp.height
            lead_equal += int((cmp["tome_lead"] == cmp["score_lead_diag"]).sum())
        WP.last_warnings = []
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ct = WP.win_probability(data, team="CT", model_name=a.model)
            tt = WP.win_probability(data, team="T", model_name=a.model)
        g = grid.filter(pl.col("match_key") == key)
        ref = {int(t): float(p) * 100.0 for t, p in zip(g["tick"].to_list(), g["p"].to_list())}
        common = sorted(set(ct) & set(ref))
        if not common:
            print(f"{key[-40:]}: no common ticks (function {len(ct)}, grid {len(ref)})")
            continue
        gaps = [abs(ct[t] - ref[t]) for t in common]
        total_gap.extend(gaps)
        agree = sum(1 for x in gaps if x <= 0.01) / len(gaps)
        flip = max(abs((ct[t] + tt[t]) - 100.0) for t in ct if t in tt) if ct else 0.0
        print(f"{key.split('/')[-2][:8]}: {len(common)} seconds in common; max gap {max(gaps):.4f} pp, mean {sum(gaps) / len(gaps):.4f} pp, "
              f"agree to 0.01 pp {agree:.1%}; CT+T max deviation from 100 {flip:.6f}")
        print(f"   warnings: {len(caught)} ({sorted({str(w.message)[:70] for w in caught})[:3]})")
    if total_gap:
        agree_all = sum(1 for x in total_gap if x <= 0.01) / len(total_gap)
        print(f"ALL {len(keys)} matches: {len(total_gap):,} seconds in common; max gap {max(total_gap):.4f} pp, mean {sum(total_gap) / len(total_gap):.5f} pp, "
              f"agree to 0.01 pp {agree_all:.2%}")
    print(f"score lead: tome freeze-end lead equals the grid's score_lead_diag in {lead_equal:,} of {lead_checked:,} rounds")
    print("done")


if __name__ == "__main__":
    main()
