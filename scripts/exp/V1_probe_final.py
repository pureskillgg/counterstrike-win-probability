import os
import json
import numpy as np
import polars as pl

TOME = r"G:\pureskill\csds-cs2\tomes\combined_test\tome\csds\basic_round_end.COMBINED"
ROUNDS = r"G:\pureskill\csds-cs2\data\state_sample\rounds.parquet"
PLAYERS = r"G:\pureskill\csds-cs2\data\state_sample\players_1hz.parquet"
OUT_JSON = r"G:\pureskill\csds-cs2\data\viz\V1_fullfull_verify.json"

pages = [os.path.join(TOME, n) for n in sorted(os.listdir(TOME)) if n.startswith("dataframe_")]
re_df = pl.concat([pl.read_parquet(p) for p in pages], how="diagonal_relaxed")
re26 = re_df.filter(pl.col("match_key").str.starts_with("csds/2026/")).with_columns(pl.col("round").cast(pl.Int64))
rounds = pl.read_parquet(ROUNDS).with_columns(pl.col("round").cast(pl.Int64))
keys = rounds.select(["match_key", "round"])

# (a) duplicates / conflicts in round_end for the sample rounds
sub = re26.join(keys, on=["match_key", "round"], how="inner")
per_round = sub.group_by(["match_key", "round"]).agg([
    pl.len().alias("n_rows"),
    pl.col("winner_team_code").n_unique().alias("n_winners"),
])
print("round_end rows for sample rounds:", sub.height, "| distinct rounds:", per_round.height)
print("rounds with >1 row:", per_round.filter(pl.col("n_rows") > 1).height,
      "| rounds with conflicting winners:", per_round.filter(pl.col("n_winners") > 1).height)

# (b) the one round with 9 player rows at second 0
p0 = (pl.scan_parquet(PLAYERS)
      .filter(pl.col("seconds_since_live") == 0)
      .select(["match_key", "round", "player_id", "team_code", "equipment_value_calc"])
      .collect()
      .with_columns(pl.col("round").cast(pl.Int64)))
odd = p0.filter((pl.col("match_key") == "csds/2026/10/07/m_7392c5c6d537a57c2e735e98/csds") & (pl.col("round") == 16))
print("odd round rows:")
print(odd.sort("team_code"))
ct_sum = odd.filter(pl.col("team_code") == 3)["equipment_value_calc"].sum()
t_sum = odd.filter(pl.col("team_code") == 2)["equipment_value_calc"].sum()
print("odd round ct_sum", ct_sum, "t_sum (4 T players)", t_sum)
winner = re26.filter((pl.col("match_key") == "csds/2026/10/07/m_7392c5c6d537a57c2e735e98/csds") & (pl.col("round") == 16))
print("odd round tome winner rows:", winner.select(["winner_team_code", "player_count"]).to_dicts())
print("odd round in sample rounds with ct_win:",
      rounds.filter((pl.col("match_key") == "csds/2026/10/07/m_7392c5c6d537a57c2e735e98/csds") & (pl.col("round") == 16)).select(["ct_win", "map_name", "split"]).to_dicts())

# (c) pull share and alt-bootstrap numbers from my output json
with open(OUT_JSON, "r", encoding="utf-8") as fh:
    out = json.load(fh)
h = out["tables"]["headline_threshold_20000_full"]
for k in ["n_rounds", "n_matches", "ct_win_fvf", "ct_win_fvf_ci", "fvf_minus_all_rounds", "fvf_minus_all_rounds_ci",
          "spearman_diff_vs_ctwin", "spearman_ci",
          "share_ct_outspends_gt3000", "share_ct_outspends_gt3000_ci",
          "share_t_outspends_gt3000", "share_t_outspends_gt3000_ci",
          "share_within_3000", "share_within_3000_ci",
          "count_ct_gt3000", "count_t_gt3000", "count_within3000",
          "median_abs_diff_fvf", "max_abs_diff_fvf", "min_team_eq_fvf",
          "alt_boot_fvf_matches_only_ci", "fvf_rounds_per_match_median", "fvf_rounds_per_match_max",
          "n_split_test", "n_split_train", "quintile_bin_sizes"]:
    print(k, "=", h.get(k))
print("all rounds raw ct:", out["checks"]["all_rounds_ct_win_rate_raw"])
print("all rounds clean ct:", out["numbers"]["ct_win_all_rounds"], out["numbers"]["ct_win_all_rounds_ci"])
print("threshold 17000 / 23000 CT:",
      out["tables"]["threshold_17000_summary"]["ct_win_fvf"],
      out["tables"]["threshold_23000_summary"]["ct_win_fvf"])
print("threshold 23000 shares gt:", out["tables"]["threshold_23000_summary"].get("share_ct_outspends_gt3000"),
      "lt:", out["tables"]["threshold_23000_summary"].get("share_t_outspends_gt3000"))
