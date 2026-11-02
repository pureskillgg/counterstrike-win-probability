"""Validate the event-derived team HP (features_events.hp_features) against the true per-player health in the 150-match
player_status sample (data/state_sample/players_1hz.parquet; alive players only, sum of health per side per second)."""
import os
import sys

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

import features_events as FE
import wp_common as W

G = W.load_grid("combined_test")
S = pl.read_parquet(os.path.join(W.ROOT, "data", "state_sample", "players_1hz.parquet"),
                    columns=["match_key", "round", "seconds_since_live", "team_code", "health"])
truth = (S.group_by(["match_key", "round", "seconds_since_live", "team_code"]).agg(pl.col("health").sum().alias("hp"))
         .pivot(on="team_code", index=["match_key", "round", "seconds_since_live"], values="hp").rename({"2": "true_hp_t", "3": "true_hp_ct"}))
sub = G.filter(pl.col("match_key").is_in(truth["match_key"].unique().to_list()))
hp = FE.hp_features(G, "combined_test")
G2 = G.with_columns(hp)
J = G2.join(truth.rename({"seconds_since_live": "second"}), on=["match_key", "round", "second"], how="inner").fill_null(0)
print("joined rows", J.height, "matches", J["match_key"].n_unique())
for side in ("ct", "t"):
    d = (J[f"hp_{side}"] - J[f"true_hp_{side}"]).to_numpy().astype(float)
    print(f"{side}: mean err {d.mean():+.2f}  MAE {np.abs(d).mean():.2f}  share |err|<=5: {np.mean(np.abs(d) <= 5):.3f}  share |err|<=20: {np.mean(np.abs(d) <= 20):.3f}  corr {np.corrcoef(J[f'hp_{side}'], J[f'true_hp_{side}'])[0, 1]:.4f}")
print("hp at second 0 == 500 share:", float((G2.filter(pl.col("second") == 0)["hp_ct"] == 500).mean()))
print("hp_ct <= 100*alive_ct+1 share:", float((G2["hp_ct"] <= 100 * G2["alive_ct"] + 1).mean()))
