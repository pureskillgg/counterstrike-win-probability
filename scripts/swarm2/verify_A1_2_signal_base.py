# verify A1-2: base rates for the header degenerate-average signal and per-team engine fit for the claimed 10.
import glob, os
import polars as pl

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
J = pl.read_csv(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_A1_2_conventions.py".replace("verify_A1_2_conventions.py", "verify_A1_2_roster.csv"), infer_schema_length=0)
J = J.with_columns([pl.col(c).cast(pl.Int64, strict=False) for c in ["n", "S", "hdr_t", "hdr_ct", "A_roster", "B_roster", "A_swap", "B_swap", "max_rounds", "unresolved"]])
h = pl.read_parquet(os.path.join(TOME, "header", "dataframe_00000"),
                    columns=["key", "t_starters_avg_wins", "ct_starters_avg_wins", "t_starters_avg_rank", "ct_starters_avg_rank", "t_starters_score_final", "ct_starters_score_final"])
h = h.with_columns((pl.col("t_starters_avg_wins") == pl.col("ct_starters_avg_wins")).alias("avg_wins_eq"),
                   (pl.col("t_starters_avg_rank") == pl.col("ct_starters_avg_rank")).alias("avg_rank_eq"))
J = J.join(h.select(["key", "avg_wins_eq", "avg_rank_eq"]), on="key", how="left")
J = J.with_columns(pl.col("key").str.split("/").list.get(3).alias("day"))
J = J.with_columns(pl.when(pl.col("day").is_in(["01", "02", "03"])).then(pl.lit("d01-03")).otherwise(pl.lit("d04-08")).alias("period"))
sub = J.filter((pl.col("max_rounds") == 16) & (pl.col("n") == pl.col("S")) & (pl.col("hdr_t") != pl.col("hdr_ct")))
sub = sub.with_columns(pl.when(pl.col("cat") == "hdr_labels_swapped_vs_roster").then(pl.lit("roster-swapped"))
                       .when(pl.col("hdr_t") != pl.col("A_swap")).then(pl.lit("swap-model-only"))
                       .otherwise(pl.lit("header-ok")).alias("status"))
print("16-round, n==S, unequal header totals, by period x status x avg_wins_eq:")
print(sub.group_by(["period", "status", "avg_wins_eq"]).len().sort(["period", "status", "avg_wins_eq"]).rows())
print("16-round, n==S, unequal header totals, by period x status x avg_rank_eq:")
print(sub.group_by(["period", "status", "avg_rank_eq"]).len().sort(["period", "status", "avg_rank_eq"]).rows())
print("all 2026 matches by max_rounds x avg_wins_eq:", J.group_by(["max_rounds", "avg_wins_eq"]).len().sort(["max_rounds", "avg_wins_eq"]).rows())
print("A_flip_round for 16-round matches:", J.filter(pl.col("max_rounds") == 16).group_by("A_flip_round").len().rows())
print("16-round matches with A_flip_round None:", J.filter((pl.col("max_rounds") == 16) & pl.col("A_flip_round").is_null()).select(["key", "n", "S", "unresolved", "A_players_r1"]).rows())
E = pl.read_csv(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_A1_2_engine_entity.csv", infer_schema_length=0)
claim = ["7368f3ae", "ec8a424b", "5429f41b", "5f014ad2", "83d3ede2", "b2e50666", "d69b442a", "e60396ec", "2093742d", "cbb071c2"]
E = E.filter(pl.col("key").str.contains("|".join(claim)))
print("engine per-team fit for the claimed 10 (key, score_rows, per_entity_ok, per_side_ok, A_total_roster, B_total_roster, hdr_t, hdr_ct):")
for r in E.select(["key", "score_rows", "per_entity_ok", "per_side_ok", "A_total_roster", "B_total_roster", "hdr_t", "hdr_ct"]).iter_rows():
    print("  ", r)
K = J.filter(pl.col("key").str.contains("|".join(claim)))
print("last-kill attacker side == winner, claimed 10 (key, ok, n):", K.select(["key", "last_kill_ok", "last_kill_n"]).rows())
print("claimed-10 day counts:", K.group_by("day").len().sort("day").rows())
print("claimed-10 max_rounds:", K.group_by("max_rounds").len().rows())
