"""T3 (Meridian pin 32f55bbe; run decision pinned da8ee067 before any T3 run): does a player-to-player attention layer over the ten player slots, applied to each second before the GRU, add to the GRU on validation? (de_mirage)

Arms (de_mirage, seeds 0, 1, 2, each trained from its own initialisation by train_temporal_round_model.py with the same settings):
  GRU   --map de_mirage --loadout --seed s --suffix _t3s<s>      -> de_mirage_loadout_gru_t3s<s>_pred.npz
  GAT   --map de_mirage --loadout --gat --seed s --suffix _t3s<s> -> de_mirage_gat_loadout_gru_t3s<s>_pred.npz
The 2026-10-22 de_mirage_gat_loadout_gru outputs (no suffix) are not read here; they are described in Meridian note 3874e235.

Decision set: de_mirage validation seconds in the usable mask the training uses (train_temporal_round_model.build_table).
Rule (pin 32f55bbe): pass if for EACH seed mean(GAT minus GRU) on validation is at most -0.0005 with a 95% match-clustered CI upper
bound below 0, and the three-seed mean is at most -0.0005. Match-cluster bootstrap, 1,000 resamples, seed 0 (spatial_feature_eval.paired).
Descriptive: the same comparison on de_mirage test seconds.

Writes data/diagnostics/t3_player_attention.json. Run in the torch environment:
  G:/pureskill/csds-cs2/.pixi/envs/torch/python.exe -I -X utf8 -u scripts/model/t3_player_attention_eval.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_temporal_round_model as ttr  # noqa: E402
from spatial_feature_eval import paired  # noqa: E402

SEEDS = [0, 1, 2]
OUT = os.path.join(str(ttr.ROOT), "data", "diagnostics", "t3_player_attention.json")


def main():
    arr, X, table, label, state, usable = ttr.build_table("de_mirage", "de_mirage", "cache_snapshot_2026.parquet")
    split = arr["split"].astype(np.int64)
    keys = arr["match_keys"]
    match_idx = arr["match_idx"]
    val_rows = np.flatnonzero((split == ttr.VAL) & usable)
    test_rows = np.flatnonzero((split == ttr.TEST) & usable)
    y_va = label[val_rows].astype(np.int64)
    y_te = label[test_rows].astype(np.int64)
    mk_va = keys[match_idx[val_rows]]
    mk_te = keys[match_idx[test_rows]]
    print(f"validation rows {val_rows.size:,}, test rows {test_rows.size:,}", flush=True)

    per_seed = {}
    for s in SEEDS:
        g = np.load(os.path.join(str(ttr.OUT), f"de_mirage_gat_loadout_gru_t3s{s}_pred.npz"))
        b = np.load(os.path.join(str(ttr.OUT), f"de_mirage_loadout_gru_t3s{s}_pred.npz"))
        # both arms were scored on the same round-state rows, in the same order
        assert (g["split"].astype(np.int64) == split).all() and (b["split"].astype(np.int64) == split).all(), "row order differs"
        assert (g["match_idx"] == match_idx).all() and (b["match_idx"] == match_idx).all(), "match order differs"
        d_va = paired(mk_va, y_va, g["p"][val_rows], b["p"][val_rows])
        d_te = paired(mk_te, y_te, g["p"][test_rows], b["p"][test_rows])
        ok = bool(d_va["diff"] <= -0.0005 and d_va["ci95"][1] < 0)
        per_seed[s] = {"GAT_minus_GRU_validation": d_va, "GAT_minus_GRU_test_descriptive": d_te, "pass": ok}
        print(f"seed {s}: validation GAT minus GRU {d_va['diff']:+.6f} CI [{d_va['ci95'][0]:+.6f}, {d_va['ci95'][1]:+.6f}], pass {ok}",
              flush=True)

    diffs = [per_seed[s]["GAT_minus_GRU_validation"]["diff"] for s in SEEDS]
    mean = float(np.mean(diffs))
    passes = bool(all(per_seed[s]["pass"] for s in SEEDS) and mean <= -0.0005)
    res = {"decision_set": "de_mirage validation seconds (usable mask)", "rows_validation": int(val_rows.size),
           "rows_test": int(test_rows.size),
           "rule": "for each seed mean(GAT minus GRU) <= -0.0005 with CI upper < 0; three-seed mean <= -0.0005 (pin 32f55bbe)",
           "per_seed": per_seed, "mean_over_seeds_validation": mean, "per_seed_pass": [per_seed[s]["pass"] for s in SEEDS],
           "passes": passes}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"mean_over_seeds_validation": mean, "passes": passes}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
