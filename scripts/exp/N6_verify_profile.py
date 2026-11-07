import glob, polars as pl
BASE = "G:/pureskill/csds-cs2/tomes/combined_test/tome/csds"
def files(ch):
    return sorted(glob.glob(BASE + "/basic_" + ch + ".COMBINED/dataframe_*"))
def lf26(ch, cols=None):
    lf = pl.scan_parquet(files(ch))
    lf = lf.filter(pl.col("match_key").str.starts_with("csds/2026/"))
    if cols:
        lf = lf.select(cols)
    return lf

# round_end 2026
re_ = lf26("round_end", ["match_key","round","tick","winner_team_code","win_reason_code","player_count","player_count_norm"]).collect()
print("round_end 2026 rows:", re_.height, "matches:", re_["match_key"].n_unique())
print("player_count dist:", re_.group_by("player_count").len().sort("player_count").rows())
print("player_count_norm dist:", re_.group_by("player_count_norm").len().sort("player_count_norm").rows())
dup = re_.group_by(["match_key","round"]).len().filter(pl.col("len")>1)
print("dup (match,round) in round_end:", dup.height)
clean = re_.filter(pl.col("player_count")==10)
print("clean 5v5 rounds (player_count==10):", clean.height, "matches:", clean["match_key"].n_unique())
print("rows with player_count_norm==10:", re_.filter(pl.col("player_count_norm")==10).height)

# item tables 2026: names by def_index
for ch in ["item_dropped","item_pickup"]:
    d = lf26(ch, ["match_key","round","tick","player_id","item","def_index","player_team_code","player_id_fixed"]+(["reason"] if ch=="item_dropped" else [])).collect()
    print("\n==", ch, "rows:", d.height, "matches:", d["match_key"].n_unique())
    print("null player_team_code:", d["player_team_code"].null_count(), " null player_id:", d["player_id"].null_count(), " null def_index:", d["def_index"].null_count())
    print("def_index,item combos:")
    print(d.group_by(["def_index","item"]).len().sort(["def_index"]).rows())
    if ch=="item_dropped":
        print("reason dist:", d.group_by("reason").len().rows())
        print("reason x item (top):", d.group_by(["reason","item"]).len().sort("len", descending=True).head(40).rows())
    print("team code dist:", d.group_by("player_team_code").len().rows())
    print("round range:", d["round"].min(), d["round"].max())
    d.write_parquet("C:/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/e0a18ad2-aa9e-4a3c-9946-29641b2cde82/scratchpad/N6_"+ch+"_2026.parquet")

# player_id -> player_id_fixed mapping stability per match (use player_death 2026 as sample)
pdeath = lf26("player_death", ["match_key","round","player_id","player_id_fixed","player_team_code"]).collect()
pairs = pdeath.select(["match_key","player_id","player_id_fixed"]).unique()
per_match = pairs.group_by("match_key").agg(pl.len().alias("pairs"), pl.col("player_id").n_unique().alias("n_pid"), pl.col("player_id_fixed").n_unique().alias("n_fixed")).sort("pairs", descending=True)
print("\nplayer_id->fixed pairs per match distribution:", per_match.group_by("pairs").len().sort("pairs").rows())
print("matches with n_pid>10 or pairs>10:", per_match.filter((pl.col("n_pid")>10)|(pl.col("pairs")>10)).height)
print("player_id range:", pdeath["player_id"].min(), pdeath["player_id"].max())
print("player_death 2026 rows:", pdeath.height)
