"""X13 probe 2: map source, fire pairing keys, join-key uniqueness."""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import polars as pl

ROOT = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
DATE = "2026-09-18,2026-10-08"
def scan(ch):
    return pl.scan_parquet(os.path.join(ROOT, f"basic_{ch}.{DATE}", "dataframe_*"))

hdr = pl.scan_parquet(os.path.join(ROOT, "header", "dataframe_*")) if False else None
# header folder lives in the channel folder; list its schema
hp = os.path.join(ROOT, "basic_round_end." + DATE, "header", "dataframe_00000")
print("header file exists:", os.path.exists(hp))
try:
    h = pl.read_parquet(hp)
    print("header cols:", list(h.schema.items())[:40])
    print("header rows:", h.height)
    print(h.head(3).to_dicts())
except Exception as e:
    print("header read err", repr(e)[:300])

mol = scan("molotov_state").collect()
print("molotov_state rows", mol.height)
sb = mol.filter(pl.col("event_type") == "inferno_startburn")
ex = mol.filter(pl.col("event_type") == "inferno_expire")
print("startburn", sb.height, "expire", ex.height)
print("startburn entity_id null", sb["entity_id"].null_count(), "entity_id_fixed null", sb["entity_id_fixed"].null_count())
print("startburn x null", sb["x_pos"].null_count(), "player_id_fixed null", sb["player_id_fixed"].null_count())
print("startburn dup (match_key,round,entity_id_fixed)", sb.height - sb.unique(subset=["match_key","round","entity_id_fixed"]).height)
print("startburn dup (match_key,round,entity_id)", sb.height - sb.unique(subset=["match_key","round","entity_id"]).height)
print("expire dup (match_key,round,entity_id_fixed)", ex.height - ex.unique(subset=["match_key","round","entity_id_fixed"]).height)
print("startburn sample:", sb.select(["round","tick","entity_id","entity_id_fixed","x_pos","y_pos","player_id_fixed","tick_throw","player_team_code","was_extinguished_by_smoke","burn_duration"]).head(4).to_dicts())
print("expire sample:", ex.select(["round","tick","entity_id","entity_id_fixed","x_pos","y_pos"]).head(3).to_dicts())
print("was_extinguished_by_smoke count (all events):", mol.filter(pl.col("was_extinguished_by_smoke")==True).height)
print("was_extinguished_by_smoke startburn:", sb.filter(pl.col("was_extinguished_by_smoke")==True).height, "expire:", ex.filter(pl.col("was_extinguished_by_smoke")==True).height)
print("startburn team_code counts:", sb["player_team_code"].value_counts().rows())
print("startburn player_team_code null:", sb["player_team_code"].null_count())

pi = scan("player_info").select(["match_key","round","player_id_fixed","team_code"]).collect()
dup = pi.height - pi.unique(subset=["match_key","round","player_id_fixed"]).height
print("player_info rows", pi.height, "dup keys", dup, "team counts", pi["team_code"].value_counts().rows(), "null pid", pi["player_id_fixed"].null_count())

bs = scan("bomb_state").select(["match_key","round","tick","event_type","player_id_fixed"]).collect()
bp = bs.filter(pl.col("event_type")=="bomb_planted")
print("bomb_planted rows", bp.height, "dup per round", bp.height - bp.unique(subset=["match_key","round"]).height, "pid null", bp["player_id_fixed"].null_count())
bd = bs.filter(pl.col("event_type")=="bomb_defused")
print("bomb_defused rows", bd.height, "dup per round", bd.height - bd.unique(subset=["match_key","round"]).height)

ba = scan("bomb_action").select(["match_key","round","tick","event_type","player_id_fixed","player_x_pos","player_y_pos","player_z_pos"]).collect()
bbp = ba.filter(pl.col("event_type")=="bomb_begin_plant")
print("begin_plant rows", bbp.height, "pid null", bbp["player_id_fixed"].null_count(), "x null", bbp["player_x_pos"].null_count())
print("begin_plant per round>1:", bbp.group_by(["match_key","round"]).agg(pl.len().alias("n")).filter(pl.col("n")>1).height)

re_ = scan("round_end").select(["match_key","round","tick","winner_team_code","win_reason_code","player_count"]).collect()
print("round_end rows", re_.height, "dup", re_.height - re_.unique(subset=["match_key","round"]).height)
print("player_count dist:", re_["player_count"].value_counts().sort("player_count").rows())
print("win_reason dist:", re_["win_reason_code"].value_counts().sort("win_reason_code").rows())
print("matches:", re_["match_key"].n_unique())

rs = scan("round_state").filter(pl.col("event_type").is_in(["round_freeze_end","round_end"])).select(["match_key","round","tick","event_type"]).collect()
print("round_freeze_end dup", rs.filter(pl.col("event_type")=="round_freeze_end").height - rs.filter(pl.col("event_type")=="round_freeze_end").unique(subset=["match_key","round"]).height)
print("round_state round_end rows", rs.filter(pl.col("event_type")=="round_end").height)

pdth = scan("player_death").select(["match_key","round","tick","player_id_fixed","player_team_code"]).collect()
print("player_death rows", pdth.height, "team dist", pdth["player_team_code"].value_counts().rows(), "pid null", pdth["player_id_fixed"].null_count())
