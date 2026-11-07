# A2: relate header build_num to round-label coverage, score/winner disagreement, match_date vs folder day,
# and check whether header-signature duplicates are true duplicate matches (identical round_end fingerprints).
import os, sys, json, collections
import polars as pl
TOME = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
def ch_frame(ch, cols):
    d = [x for x in os.listdir(TOME) if x.split(".", 1)[0] == "basic_" + ch]
    dp = os.path.join(TOME, d[0])
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    parts = [pl.scan_parquet(os.path.join(dp, p["dataframe"]["dataframe"])).select(cols).collect() for p in meta["pages"]]
    return pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]
hdr = pl.read_parquet(os.path.join(TOME, "header", "dataframe_00000"), columns=["key", "match_date", "map_name", "build_num", "t_starters_score_final", "ct_starters_score_final", "number_of_points", "match_id"])
hdr = hdr.rename({"key": "match_key"}).with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day"), pl.col("match_date").str.slice(0, 10).alias("mdate"))
hdr = hdr.with_columns((pl.col("mdate").str.slice(8, 2) == pl.col("day")).alias("date_eq_folder"))
print("match_date day == folder day by folder day:")
print(hdr.group_by("day").agg(pl.len().alias("n"), pl.col("date_eq_folder").sum().alias("eq")).sort("day").rows(), flush=True)
print("match_date month by folder day (first 6):", hdr.group_by(["day", pl.col("mdate").str.slice(0, 7)]).len().sort(["day"]).head(12).rows(), flush=True)
st = ch_frame("round_state", ["round", "tick", "event_type", "t_score", "ct_score", "match_key"])
fe = st.filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "t_score", "ct_score"]).unique(["match_key", "round"], keep="first").sort(["match_key", "round"])
re_ = ch_frame("round_end", ["round", "tick", "winner_team_code", "match_key"]).unique(["match_key", "round"], keep="first").sort(["match_key", "round"])
fe_min = fe.group_by("match_key").agg(pl.col("round").min().alias("fe_min"))
nxt = fe.with_columns([pl.col("round").shift(-1).over("match_key").alias("r_next"), pl.col("t_score").shift(-1).over("match_key").alias("t_next"), pl.col("ct_score").shift(-1).over("match_key").alias("ct_next")]).filter(pl.col("r_next") == pl.col("round") + 1)
nxt = nxt.with_columns([(pl.col("t_next") - pl.col("t_score")).alias("dT"), (pl.col("ct_next") - pl.col("ct_score")).alias("dCT")]).with_columns(pl.when(pl.col("dT") == 1).then(2).when(pl.col("dCT") == 1).then(3).otherwise(0).alias("w_score"))
b = nxt.join(re_.select(["match_key", "round", "winner_team_code"]).rename({"round": "rk", "winner_team_code": "w_end"}), left_on=["match_key", "round"], right_on=["match_key", "rk"], how="left").filter(pl.col("w_score") != 0)
bad = b.with_columns((pl.col("w_end") != pl.col("w_score")).alias("dis")).group_by("match_key").agg(pl.col("dis").sum().alias("n_dis"), pl.len().alias("n_pairs"))
mm = hdr.select(["match_key", "build_num", "day", "map_name", "match_id"]).join(fe_min, on="match_key", how="left").join(bad, on="match_key", how="left")
mm = mm.with_columns(pl.when(pl.col("day") == "01").then(pl.lit("01")).when(pl.col("day") <= "07").then(pl.lit("02_07")).otherwise(pl.lit("12_25")).alias("grp"))
print("build_num x group:", mm.group_by(["grp", "build_num"]).len().sort(["grp", "build_num"]).rows(), flush=True)
print("build_num x F-label-1 present:", mm.with_columns((pl.col("fe_min") == 1).alias("has_F1")).group_by(["build_num", "has_F1"]).len().sort(["build_num", "has_F1"]).rows(), flush=True)
s = mm.filter(pl.col("n_pairs").is_not_null())
print("disagreeing rounds per build (dis, pairs):", s.group_by("build_num").agg(pl.col("n_dis").sum().alias("dis"), pl.col("n_pairs").sum().alias("pairs")).rows(), flush=True)
print("matches with >=1 disagreement by build:", s.with_columns((pl.col("n_dis") > 0).alias("any")).group_by(["build_num", "any"]).len().sort(["build_num", "any"]).rows(), flush=True)
print("map by build:", mm.group_by(["build_num", "map_name"]).len().sort(["build_num", "len"], descending=[False, True]).rows()[:20], flush=True)
# header-signature duplicates -> round fingerprint
sig = hdr.with_columns(pl.concat_str([pl.col("mdate"), pl.col("map_name"), pl.col("t_starters_score_final").cast(pl.Utf8), pl.col("ct_starters_score_final").cast(pl.Utf8), pl.col("number_of_points").cast(pl.Utf8)], separator="|").alias("sig"))
dg = sig.group_by("sig").agg(pl.col("match_key").alias("mks"), pl.len().alias("n")).filter(pl.col("n") > 1)
print("signature-dup groups:", dg.height, flush=True)
fp = re_.sort(["match_key", "round"]).group_by("match_key").agg(pl.concat_str([pl.col("round").cast(pl.Utf8), pl.col("tick").cast(pl.Utf8), pl.col("winner_team_code").cast(pl.Utf8)], separator=":").alias("seq")).with_columns(pl.col("seq").list.join(",").alias("fp"))
fpm = dict(zip(fp["match_key"].to_list(), fp["fp"].to_list()))
true_dup = 0; not_dup = 0; empty_fp = 0
ex = []
for r in dg.iter_rows(named=True):
    fps = [fpm.get(k, "") for k in r["mks"]]
    if any(f == "" for f in fps): empty_fp += 1
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            if fps[i] and fps[i] == fps[j]: true_dup += 1
            else: not_dup += 1
    if len(ex) < 3 and len(set(fps)) > 1: ex.append(r["mks"][:3])
print("signature-dup pairs with identical round_end fingerprint:", true_dup, " pairs differing:", not_dup, " groups with empty fp:", empty_fp, flush=True)
print("examples of signature-dup groups with differing fingerprints:", ex, flush=True)
print("DONE", flush=True)
