import json, sys
import polars as pl

ROOT = "G:/pureskill/csds-cs2"
SCR = "C:/Users/13144/AppData/Local/Temp/claude/C--Users-13144-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
old_path = sys.argv[1] if len(sys.argv) > 1 else ROOT + "/data/combined/rounds_pre_oct26.parquet"
new_path = sys.argv[2] if len(sys.argv) > 2 else ROOT + "/data/combined/rounds.parquet"
frozen = set(json.load(open(SCR + "/frozen_test_keys.json")))
old = pl.read_parquet(old_path).filter(pl.col("match_key").is_in(list(frozen)))
new = pl.read_parquet(new_path).filter(pl.col("match_key").is_in(list(frozen)))
missing = old.join(new.select(["match_key", "round"]), on=["match_key", "round"], how="anti").height
extra = new.join(old.select(["match_key", "round"]), on=["match_key", "round"], how="anti").height
cols = [c for c in ["end", "rec", "win_reason_code", "player_count_norm", "freeze", "t_sc", "ct_sc", "n_players", "n_t", "n_ct", "era"]
        if c in old.columns and c in new.columns]
j = old.join(new, on=["match_key", "round"], how="inner", suffix="_n")
changed = j.filter(pl.any_horizontal([pl.col(c).ne_missing(pl.col(c + "_n")) for c in cols])).height
ok = missing == 0 and extra == 0 and changed == 0
print(json.dumps({"frozen_matches": len(frozen), "old_rounds": old.height, "missing": missing, "extra": extra,
                  "changed_tracked_fields": changed, "compared": j.height, "ok": ok}), flush=True)
sys.exit(0 if ok else 1)
