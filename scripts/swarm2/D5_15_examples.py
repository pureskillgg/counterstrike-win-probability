"""D5 step 15: worked examples for the round-8/9 side-swap question (2024 tome).

Picks the first late regime-A control match with a side-field swap at 8->9 (expected: round-9 winner wrong),
and the first regime-B match (early or late) with a side-field swap at 8->9 (expected: no wrong winner).
Prints freeze-end scores (t, ct), round_end winner/reason, and bomb rows for labels 7..11, plus the last
victim side per label (from player_death) as an independent truth for no-bomb rounds.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D5_15_examples.py
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from D5_common import read  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)

SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5")
R = pl.read_parquet(SCR / "rounds.parquet").select(["match_key", "round", "fe_t_score", "fe_ct_score", "winner",
                                                    "reason_code", "end_tick", "fe_tick"]).sort(["match_key", "round"])
M = pl.read_parquet(SCR / "matches.parquet")
mk = pl.read_parquet(SCR / "rounds.parquet").group_by("match_key").agg(
    pl.col("n_round_announce_match_start").fill_null(0).sum().alias("mstart"))
LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]
M = M.join(mk, on="match_key", how="left").with_columns(
    regime=pl.when(pl.col("mstart") > 0).then(pl.lit("A")).otherwise(pl.lit("B")),
    grp=pl.when(pl.col("aff_r13")).then(pl.lit("aff_r13"))
    .when(pl.col("day").is_in(LATE)).then(pl.lit("ctrl_late"))
    .when(pl.col("day").is_in(EARLY)).then(pl.lit("early")).otherwise(pl.lit("other")))
B = pl.read_parquet(SCR / "bomb.parquet")

N = R.with_columns(nt=pl.col("fe_t_score").shift(-1).over("match_key"),
                   nct=pl.col("fe_ct_score").shift(-1).over("match_key"),
                   nr=pl.col("round").shift(-1).over("match_key"))
N = N.filter((pl.col("nr") == pl.col("round") + 1) & (pl.col("round") == 8) & pl.col("fe_t_score").is_not_null()
             & pl.col("nt").is_not_null() & pl.col("winner").is_not_null())
a = pl.when(pl.col("winner") == 2).then(1).otherwise(0)
b = pl.when(pl.col("winner") == 3).then(1).otherwise(0)
N = N.with_columns(ns=(pl.col("nt") == pl.col("fe_t_score") + a) & (pl.col("nct") == pl.col("fe_ct_score") + b),
                   sw=(pl.col("nt") == pl.col("fe_ct_score") + b) & (pl.col("nct") == pl.col("fe_t_score") + a))
S8 = N.filter(pl.col("sw") & ~pl.col("ns")).join(M.select(["match_key", "grp", "regime", "map_name", "day"]), on="match_key")

pickA = S8.filter((pl.col("regime") == "A") & (pl.col("grp") == "ctrl_late")).sort("match_key").head(1)
pickB = S8.filter((pl.col("regime") == "B")).sort("match_key").head(1)
print("regime-A example:", pickA.select(["match_key", "map_name", "day", "grp", "regime"]).to_dicts())
print("regime-B example:", pickB.select(["match_key", "map_name", "day", "grp", "regime"]).to_dicts())

DE = read("player_death", ["match_key", "round", "tick", "player_team_code"]).filter(pl.col("player_team_code").is_in([2, 3]))
for name, pk in [("REGIME A (round 9 expected wrong)", pickA), ("REGIME B (no wrong winner expected)", pickB)]:
    if pk.height == 0:
        print("no example for", name)
        continue
    k = pk["match_key"][0]
    print("\n" + "=" * 110)
    print(name, k)
    t = R.filter((pl.col("match_key") == k) & pl.col("round").is_between(6, 11)).sort("round")
    print(t.select(["round", "fe_tick", "fe_t_score", "fe_ct_score", "end_tick", "winner", "reason_code"]))
    bb = B.filter((pl.col("match_key") == k) & pl.col("round").is_between(6, 11)).sort("tick").select(
        ["round", "tick", "event_type", "player_team_code"])
    print("bomb rows (labels 6..11):")
    print(bb)
    dd = DE.filter((pl.col("match_key") == k) & pl.col("round").is_between(6, 11))
    lastv = dd.sort("tick").group_by("round").agg(pl.col("player_team_code").last().alias("last_victim_side"),
                                                  pl.len().alias("deaths"))
    print("last victim side per label (deaths):")
    print(lastv.sort("round"))
