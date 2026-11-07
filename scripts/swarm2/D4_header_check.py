"""D4 extra check: which record reproduces the header's final team scores in the round-13 mismatched matches?

Two per-round winner sources for each complete match (round count == header t_starters + ct_starters final):
  SRC_END  round_end.winner_team_code for every round.
  SRC_BOMB the same, but for rounds with a bomb outcome row the winner is the bomb side (defused -> 3 CT,
           exploded -> 2 T), as in e2_side_check_r13.py.
Team A = header t_starters (starts on T). Team A's side in round r is T unless an odd number of side swaps
happened before r. Swaps follow bakeoff.swap_expected(r, max_rounds) (ASSUMPTION A11: that rule is taken from
the bake-off; it reproduced 1,165 overtime rounds in score_lead_fix for 2026).
A match is OK if the win counts reproduce (t_starters, ct_starters) from the header, SWAPPED if they reproduce
the reverse, BAD otherwise.
Affected = match has >= 1 label-based round-13 outcome mismatch.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D4_header_check.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(14)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
    "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds",
}
OUTCOMES = ["bomb_defused", "bomb_exploded"]
T, CT = 2, 3


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def read(base, ch, cols):
    return read_dir(sorted(base.glob(f"basic_{ch}.*"))[0], cols)


def swap_expected(r, max_rounds):
    """Copy of bakeoff.swap_expected (ASSUMPTION A11)."""
    return r == max_rounds // 2 or (r > max_rounds and (r - max_rounds - 3) % 6 == 0)


def run_era(era, base):
    print(f"\n==================== ERA {era} ====================")
    # ASSUMPTION A12: the 2024 header has no max_rounds column; MR12 (24 regulation rounds) is assumed.
    hdr_cols = ["key", "t_starters_score_final", "ct_starters_score_final"]
    has_mr = "max_rounds" in pl.read_parquet_schema(sorted(glob.glob(str(base / "header" / "dataframe_*")))[0])
    if has_mr:
        hdr_cols.append("max_rounds")
    H = read_dir(base / "header", hdr_cols)
    if not has_mr:
        H = H.with_columns(pl.lit(24).alias("max_rounds"))
    print(f"header rows: {H.height}; max_rounds source: {'header' if has_mr else 'ASSUMED 24 (A12)'}")
    re_ = read(base, "round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(
        ["match_key", "round"], keep="first")
    bs = read(base, "bomb_state", ["match_key", "round", "event_type"]).filter(pl.col("event_type").is_in(OUTCOMES))
    outs = (bs.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
    outs = outs.with_columns(
        pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bside"))
    R = re_.join(outs.select(["match_key", "round", "bside"]), on=["match_key", "round"], how="left")
    # affected flag: label-based round-13 mismatch (player_count 20 in 2024, any round length in 2026)
    mm13 = (R.filter((pl.col("round") == 13) & pl.col("bside").is_not_null())
            .with_columns((pl.col("bside") != pl.col("winner_team_code")).alias("mm"))
            .filter(pl.col("mm")).select("match_key").unique().with_columns(pl.lit(True).alias("affected")))
    Hd = {r["key"]: r for r in H.iter_rows(named=True)}
    rows_by_match = {}
    for r in R.sort(["match_key", "round"]).iter_rows(named=True):
        rows_by_match.setdefault(r["match_key"], []).append(r)
    affected = set(mm13["match_key"].to_list())

    counts = {}

    def bump(key):
        counts[key] = counts.get(key, 0) + 1

    for m, rounds in rows_by_match.items():
        h = Hd.get(m)
        if h is None:
            bump("no_header")
            continue
        mr = int(h["max_rounds"])
        tA, tB = int(h["t_starters_score_final"]), int(h["ct_starters_score_final"])
        n = len(rounds)
        rnums = [int(x["round"]) for x in rounds]
        complete = (n == tA + tB) and rnums == list(range(1, n + 1))
        grp = "affected" if m in affected else "unaffected"
        if not complete:
            bump(f"{grp}|incomplete")
            continue
        for src in ("end", "bomb"):
            wa = wb = 0
            swaps = 0
            for x in rounds:
                r = int(x["round"])
                w = x["winner_team_code"]
                if src == "bomb" and x["bside"] is not None:
                    w = x["bside"]
                side_A = T if swaps % 2 == 0 else CT
                if int(w) == side_A:
                    wa += 1
                else:
                    wb += 1
                if swap_expected(r, mr):
                    swaps += 1
            if (wa, wb) == (tA, tB):
                status = "ok"
            elif (wa, wb) == (tB, tA):
                status = "swapped"
            else:
                status = "bad"
            bump(f"{grp}|{src}|{status}")
    print("\nmatch counts by group, source and header status:")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")
    print(f"affected matches (>=1 r13 label mismatch): {len(affected)}")


for era, base in TOMES.items():
    run_era(era, base)
